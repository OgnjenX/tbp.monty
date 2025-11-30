from __future__ import annotations

"""Meta-Controller for creativity/perspective shifting in Monty.

This module implements a light-weight, rule-based meta-controller that monitors
creativity-related metrics from learning modules and adjusts behavior via:
- Temporary feature weight perturbations
- Optional transform-manager jitter for perspective switching

It subscribes to Monty's callback hooks:
    monty.before_vote_cb = meta.before_vote
    monty.after_step_cb = meta.after_step

Design goals:
- Small EMA smoothing for stability
- Stagnation detection based on entropy and MLH slope
- Cooldown windows to avoid rapid toggling
- Clean separation of observe_metrics, decide_action, apply_action
"""

import logging
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------- Helper Types ----------------------------


@dataclass
class LMStats:
    entropy_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    slope_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    ema_entropy: float = 0.0
    ema_slope: float = 0.0
    perturbed: bool = False
    cooldown: int = 0
    active_ttl: int = 0  # remaining steps for active intervention


class RandomJitterTransformManager:
    """Minimal transform manager that injects small random jitter.

    This serves as a proof-of-concept perspective shifter. The jitter is kept
    small and symmetric around identity.
    """

    def __init__(self, loc_sigma: float = 0.01, rot_sigma_deg: float = 2.0):
        self.loc_sigma = float(loc_sigma)
        self.rot_sigma_rad = float(rot_sigma_deg) * np.pi / 180.0

    def pairwise_transform(
        self,
        receiving_id: int,
        sending_id: int,
        receiving_pose: Any,
        sending_pose: Any,
        sensor_disp: np.ndarray,
        sensor_rotation: Any,
    ) -> Tuple[np.ndarray, Any]:
        # Add tiny translational jitter
        jitter_t = np.random.normal(scale=self.loc_sigma, size=3)
        # Add tiny rotational jitter: axis-angle around a random axis
        axis = np.random.normal(size=3)
        axis_norm = np.linalg.norm(axis) + 1e-12
        axis = axis / axis_norm
        angle = np.random.normal(scale=self.rot_sigma_rad)
        # Represent rotation in the same type as incoming rotation if possible
        try:
            from scipy.spatial.transform import Rotation

            jitter_r = Rotation.from_rotvec(axis * angle)
            if hasattr(sensor_rotation, "as_matrix"):
                out_rot = jitter_r * sensor_rotation
            else:
                out_rot = jitter_r.as_matrix() @ sensor_rotation
        except Exception:
            # Fallback: return original rotation unchanged if scipy not available
            out_rot = sensor_rotation

        return sensor_disp + jitter_t, out_rot

    def postprocess_votes(
        self,
        receiving_id: int,
        sending_id: int,
        object_id: str,
        locations: np.ndarray,
        rotations: Any,
    ) -> Tuple[np.ndarray, Any]:
        # Optionally broaden locations a touch to encourage exploration
        if isinstance(locations, np.ndarray) and locations.size > 0:
            noise = np.random.normal(scale=self.loc_sigma, size=locations.shape)
            locations = locations + noise
        return locations, rotations

    def postprocess_state_objects(
        self,
        receiving_id: int,
        sending_id: int,
        object_id: str,
        states: list,
    ) -> list:
        # No-op for state objects in this simple manager
        return states


# ---------------------------- Meta Controller ----------------------------


class MetaController:
    def __init__(
        self,
        monty,
        entropy_min: float = 0.2,
        slope_threshold: float = 1e-3,
        stagnation_window: int = 5,
        ema_alpha: float = 0.3,
        cooldown_steps: int = 10,
        perturb_amp: float = 0.15,
        weight_ttl: int = 20,
        transform_prob: float = 0.3,
        transform_loc_sigma: float = 0.01,
        transform_rot_sigma_deg: float = 2.0,
    ) -> None:
        """Initialize controller and default meta-parameters.

        Args:
            monty: MontyBase-compatible model instance
            entropy_min: low-entropy threshold (possible stall signal)
            slope_threshold: |EMA slope| below this considered stalled
            stagnation_window: number of consecutive steps to consider for stall
            ema_alpha: smoothing for EMA of metrics
            cooldown_steps: wait period after an intervention
            perturb_amp: amplitude for feature-weight randomization (+/- fraction)
            weight_ttl: steps to keep weight perturbations active
            transform_prob: probability to trigger a transform on intervention
            transform_loc_sigma: std of loc jitter in meters (approx.)
            transform_rot_sigma_deg: std of rot jitter in degrees
        """
        self.monty = monty
        self.entropy_min = float(entropy_min)
        self.slope_threshold = float(slope_threshold)
        self.stagnation_window = int(stagnation_window)
        self.ema_alpha = float(ema_alpha)
        self.cooldown_steps = int(cooldown_steps)
        self.perturb_amp = float(perturb_amp)
        self.weight_ttl = int(weight_ttl)
        self.transform_prob = float(transform_prob)
        self.transform_loc_sigma = float(transform_loc_sigma)
        self.transform_rot_sigma_deg = float(transform_rot_sigma_deg)

        # Per-LM runtime state
        self._lm_state: Dict[Any, LMStats] = {}
        # Telemetry buffers (simple list of dicts)
        self._log: List[Dict[str, Any]] = []
        # Lock to avoid concurrent modifications from both callbacks (defensive)
        self._lock = threading.RLock()

        # Attach by default
        self.attach()

    # --------------- Public API ---------------
    def attach(self) -> None:
        """Attach controller callbacks to Monty instance."""
        self.monty.before_vote_cb = self.before_vote
        self.monty.after_step_cb = self.after_step

    def detach(self) -> None:
        """Detach controller callbacks from Monty instance."""
        if getattr(self.monty, "before_vote_cb", None) is self.before_vote:
            self.monty.before_vote_cb = None
        if getattr(self.monty, "after_step_cb", None) is self.after_step:
            self.monty.after_step_cb = None

    def before_vote(self, sensor_module_outputs, learning_module_outputs) -> None:
        """Pre-vote hook: observe metrics and possibly act before voting."""
        with self._lock:
            metrics = self.observe_metrics()
            action = self.decide_action(metrics)
            self.apply_action(action)

    def after_step(self, step_metrics: dict) -> None:
        """After-step hook: lightweight logging and TTL/cooldown bookkeeping."""
        with self._lock:
            # Decrement TTL and cooldowns
            for lm, st in self._lm_state.items():
                if st.cooldown > 0:
                    st.cooldown -= 1
                if st.active_ttl > 0:
                    st.active_ttl -= 1
                    if st.active_ttl == 0:
                        # Passive restore once TTL expires
                        self._restore_if_perturbed(lm, st, reason="ttl_expired")

            # Append step-level info to log
            safe_metrics = dict(step_metrics)
            safe_metrics["num_lms"] = len(getattr(self.monty, "learning_modules", []))
            self._log.append({"type": "step", **safe_metrics})

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._log)

    # --------------- Core Logic ---------------
    def observe_metrics(self) -> Dict[Any, Dict[str, float]]:
        """Collect per-LM metrics with EMA smoothing and store telemetry."""
        out: Dict[Any, Dict[str, float]] = {}
        for lm in getattr(self.monty, "learning_modules", []):
            # Initialize state lazily per-LM
            st = self._lm_state.get(lm)
            if st is None:
                st = LMStats()
                self._lm_state[lm] = st

            # Query metrics from LM (with fail-safe defaults)
            try:
                ent = float(lm.get_entropy())
            except Exception:
                ent = 0.0
            try:
                slope = float(lm.get_mlh_slope())
            except Exception:
                slope = 0.0

            # Update histories and EMAs
            st.entropy_hist.append(ent)
            st.slope_hist.append(slope)
            st.ema_entropy = self._ema_update(st.ema_entropy, ent, self.ema_alpha)
            st.ema_slope = self._ema_update(st.ema_slope, slope, self.ema_alpha)

            # Log snapshot
            out[lm] = dict(entropy=ent, slope=slope, ema_entropy=st.ema_entropy, ema_slope=st.ema_slope)
            self._log.append(
                {
                    "type": "lm_metrics",
                    "lm_id": getattr(lm, "learning_module_id", None),
                    **out[lm],
                    "perturbed": st.perturbed,
                    "cooldown": st.cooldown,
                    "active_ttl": st.active_ttl,
                }
            )

        return out

    def decide_action(self, metrics: Dict[Any, Dict[str, float]]):
        """Simple rule-based policy to detect stagnation and decide interventions.

        Returns a list of (lm, action_dict) tuples. action_dict may include:
            - perturb_weights: bool
            - transform: bool
        """
        actions: List[Tuple[Any, Dict[str, Any]]] = []
        for lm, vals in metrics.items():
            st = self._lm_state[lm]
            # Skip if cooling down
            if st.cooldown > 0:
                continue

            # Check stagnation using EMA and a small window of recent values
            entropy_low = (st.ema_entropy <= self.entropy_min) or self._window_below(
                st.entropy_hist, self.entropy_min, self.stagnation_window
            )
            slope_small = (abs(st.ema_slope) <= self.slope_threshold) or self._window_abs_below(
                st.slope_hist, self.slope_threshold, self.stagnation_window
            )

            if entropy_low or slope_small:
                actions.append(
                    (
                        lm,
                        {
                            "perturb_weights": True,
                            "transform": random.random() < self.transform_prob,
                        },
                    )
                )
            else:
                # If stable and currently perturbed, restore
                if st.perturbed and st.active_ttl <= 0:
                    actions.append((lm, {"restore": True}))

        return actions

    def apply_action(self, actions: List[Tuple[Any, Dict[str, Any]]]) -> None:
        for lm, act in actions:
            st = self._lm_state[lm]

            if act.get("restore"):
                self._restore_if_perturbed(lm, st, reason="stability")
                continue

            if act.get("perturb_weights") and not st.perturbed:
                # Attempt to perturb feature weights
                try:
                    new_weights = self._perturb_feature_weights(lm)
                    lm.set_temp_feature_weights(new_weights)
                    st.perturbed = True
                    st.cooldown = self.cooldown_steps
                    st.active_ttl = self.weight_ttl
                    self._log.append(
                        {
                            "type": "action",
                            "lm_id": getattr(lm, "learning_module_id", None),
                            "action": "perturb_weights",
                            "weights_keys": list(new_weights.keys()),
                        }
                    )
                except Exception as e:
                    logger.debug(f"MetaController: failed to set temp weights: {e}")

            if act.get("transform"):
                # Toggle transform manager to a jitter variant for the TTL window
                try:
                    jitter = RandomJitterTransformManager(
                        loc_sigma=self.transform_loc_sigma,
                        rot_sigma_deg=self.transform_rot_sigma_deg,
                    )
                    setattr(self.monty, "transform_manager", jitter)
                    # Record that a transform is active at the Monty level
                    self._log.append(
                        {
                            "type": "action",
                            "action": "set_transform_manager",
                            "loc_sigma": self.transform_loc_sigma,
                            "rot_sigma_deg": self.transform_rot_sigma_deg,
                        }
                    )
                except Exception as e:
                    logger.debug(f"MetaController: failed to set transform_manager: {e}")

    # --------------- Internals ---------------
    @staticmethod
    def _ema_update(prev: float, val: float, alpha: float) -> float:
        if prev == 0.0:
            return float(val)
        return float(alpha * val + (1.0 - alpha) * prev)

    @staticmethod
    def _window_below(hist: Deque[float], thresh: float, window: int) -> bool:
        if len(hist) < window:
            return False
        arr = list(hist)[-window:]
        return all(v <= thresh for v in arr)

    @staticmethod
    def _window_abs_below(hist: Deque[float], thresh: float, window: int) -> bool:
        if len(hist) < window:
            return False
        arr = list(hist)[-window:]
        return all(abs(v) <= thresh for v in arr)

    def _perturb_feature_weights(self, lm) -> Dict[str, float]:
        # Get current weights; fall back to equal weights if missing
        try:
            base = dict(lm.feature_weights)
        except Exception:
            base = {}
        if not base:
            # Try to infer from tolerances keys if available
            keys = list(getattr(lm, "tolerances", {}).keys())
            base = {k: 1.0 for k in keys}

        amp = self.perturb_amp
        new_w = {}
        for k, v in base.items():
            scale = 1.0 + random.uniform(-amp, amp)
            nv = float(max(1e-6, v * scale))
            new_w[k] = nv
        return new_w

    def _restore_if_perturbed(self, lm, st: LMStats, reason: str) -> None:
        if not st.perturbed:
            return
        try:
            lm.restore_feature_weights()
            # Also reset any temporary transform back to default/None
            try:
                if getattr(self.monty, "transform_manager", None) is not None:
                    setattr(self.monty, "transform_manager", None)
            except Exception:
                pass
            st.perturbed = False
            st.active_ttl = 0
            st.cooldown = self.cooldown_steps  # brief cooldown after restore
            self._log.append(
                {
                    "type": "action",
                    "lm_id": getattr(lm, "learning_module_id", None),
                    "action": "restore_weights",
                    "reason": reason,
                }
            )
        except Exception as e:
            logger.debug(f"MetaController: failed to restore weights: {e}")


__all__ = ["MetaController", "RandomJitterTransformManager"]
