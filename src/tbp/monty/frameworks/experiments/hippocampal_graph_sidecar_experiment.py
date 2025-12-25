# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.models.hippocampal_graph_lm import HippocampalGraphLM

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HippocampalGraphSidecarConfig:
    enabled: bool = True
    learning_rate: float = 0.8
    decay_rate: float = 0.0
    max_observations_per_episode: int = 10_000
    enable_replay: bool = True
    max_episode_history: int = 200
    context_from_primary_target: bool = False
    dedupe_consecutive_object_ids: bool = True
    output_confidence_threshold: float = 0.0
    write_csv: bool = True


class MontyObjectRecognitionHippocampalGraphSidecarExperiment(
    MontyObjectRecognitionExperiment
):
    """Object recognition experiment with HippocampalGraphLM as a sidecar.

    This does not register HippocampalGraphLM as a Monty learning module.
    Instead, it listens to learning module outputs each step and streams
    object-level events into HippocampalGraphLM via `observe_object`.

    This is the lowest-friction way to evaluate hippocampal relational memory
    on existing Monty experiments (especially multi-object world-image scenes).
    """

    def __init__(self, config):
        super().__init__(config=config)
        self._hipp_cfg = HippocampalGraphSidecarConfig(
            **dict(config.get("hippocampal_graph", {}))
        )
        self._hipp: Optional[HippocampalGraphLM] = None
        self._hipp_csv_path: Optional[Path] = None
        self._hipp_csv_file = None
        self._hipp_csv_writer: Optional[csv.DictWriter] = None
        self._last_object_id: Optional[str] = None

    def setup_experiment(self, config: Dict[str, Any]) -> None:
        super().setup_experiment(config)
        if not self._hipp_cfg.enabled:
            return

        self._hipp = HippocampalGraphLM(
            learning_rate=self._hipp_cfg.learning_rate,
            decay_rate=self._hipp_cfg.decay_rate,
            max_observations_per_episode=self._hipp_cfg.max_observations_per_episode,
            enable_replay=self._hipp_cfg.enable_replay,
            max_episode_history=self._hipp_cfg.max_episode_history,
        )

        if self._hipp_cfg.write_csv:
            self._hipp_csv_path = self.output_dir / "hippocampal_graph_sidecar.csv"
            self._hipp_csv_file = self._hipp_csv_path.open("w", newline="")
            self._hipp_csv_writer = csv.DictWriter(
                self._hipp_csv_file,
                fieldnames=[
                    "phase",
                    "episode_index",
                    "primary_target",
                    "context",
                    "n_observations_in_episode",
                    "episode_count",
                    "stored_episodes",
                    "n_objects",
                    "n_relations",
                    "total_observations",
                    "completion_cue",
                    "completion_suggestions",
                ],
            )
            self._hipp_csv_writer.writeheader()

    def close(self):
        try:
            if self._hipp_csv_file is not None:
                self._hipp_csv_file.close()
        finally:
            super().close()

    def pre_episode(self):
        super().pre_episode()
        if self._hipp is None:
            return

        self._hipp.reset()
        self._last_object_id = None

        context = None
        if self._hipp_cfg.context_from_primary_target:
            context = getattr(self.env_interface, "primary_target", {}).get("object")
        if context:
            self._hipp.set_episode_context(str(context))

    def post_episode(self, steps):
        if self._hipp is not None:
            n_obs = len(self._hipp.current_episode_observations)
            cue = None
            context = self._hipp.current_episode_context
            if n_obs > 0:
                cue = self._hipp.current_episode_observations[0].object_id

            self._hipp.end_episode()
            stats = self._hipp.get_statistics()

            suggestions = []
            if cue is not None:
                suggestions = self._hipp.complete_from_cue(
                    [cue],
                    context_filter=context,
                    top_k_objects=5,
                )

            if self._hipp_csv_writer is not None:
                phase = (
                    "train" if self.experiment_mode.value == "train" else "eval"
                )
                episode_index = (
                    self.train_episodes
                    if phase == "train"
                    else self.eval_episodes
                )
                primary_target = getattr(self.env_interface, "primary_target", {}).get(
                    "object"
                )
                self._hipp_csv_writer.writerow(
                    {
                        "phase": phase,
                        "episode_index": episode_index,
                        "primary_target": primary_target,
                        "context": context,
                        "n_observations_in_episode": n_obs,
                        "episode_count": stats.get("episode_count"),
                        "stored_episodes": stats.get("stored_episodes"),
                        "n_objects": stats.get("n_objects"),
                        "n_relations": stats.get("n_relations"),
                        "total_observations": stats.get("total_observations"),
                        "completion_cue": cue,
                        "completion_suggestions": ";".join(
                            f"{obj}:{score:.3f}" for obj, score in suggestions
                        ),
                    }
                )

        super().post_episode(steps)

    def run_episode_steps(self):
        env_interface = self.env_interface
        if env_interface is None:
            raise RuntimeError("env_interface is not set; did you call pre_epoch()?")

        loader_step = -1
        for loader_step, observation in enumerate(env_interface):
            if self._should_terminate_episode(loader_step):
                return loader_step

            self._step_model(observation)
            self._step_hippocampus(loader_step)

            if self.model.is_done:
                return loader_step

        self.model.set_is_done()
        return loader_step

    def _should_terminate_episode(self, step: int) -> bool:
        if self.model.check_reached_max_matching_steps(self.max_steps):
            logger.info(f"Terminated due to maximum matching steps : {self.max_steps}")
            return True

        if step >= self.max_total_steps:
            logger.info(f"Terminated due to maximum episode steps : {step}")
            self.model.deal_with_time_out()
            return True

        return False

    def _step_model(self, observation: Any) -> None:
        if self.model.is_motor_only_step:
            self.model.pass_features_directly_to_motor_system(observation)
        else:
            self.model.step(observation)

    def _step_hippocampus(self, step: int) -> None:
        if self._hipp is None:
            return
        self._observe_from_lms(step_timestamp=float(step))

    def _observe_from_lms(self, step_timestamp: float) -> None:
        if self._hipp is None:
            return

        for lm in getattr(self.model, "learning_modules", []):
            observation_data = self._extract_observation_data(lm)
            if observation_data is None:
                continue

            self._hipp.observe_object(
                object_id=observation_data["object_id"],
                location=observation_data["location"],
                timestamp=step_timestamp,
                confidence=observation_data["confidence"],
                source_lm=observation_data["source_lm"],
            )
            self._last_object_id = observation_data["object_id"]

    def _extract_observation_data(self, lm) -> Optional[Dict[str, Any]]:
        """Extract and validate observation data from a learning module.

        Returns a dict with observation data if valid, None otherwise.
        """
        try:
            output = lm.get_output()
        except Exception:
            return None

        if output is None or not getattr(output, "use_state", False):
            return None

        non_morph_features = getattr(output, "non_morphological_features", {}) or {}
        object_id = non_morph_features.get("object_id")
        if not object_id:
            return None

        confidence = float(getattr(output, "confidence", 0.0))
        if confidence < self._hipp_cfg.output_confidence_threshold:
            return None

        object_id = str(object_id)
        if (
            self._hipp_cfg.dedupe_consecutive_object_ids
            and object_id == self._last_object_id
        ):
            return None

        location_raw = getattr(output, "location", None)
        if location_raw is None:
            return None
        location = np.asarray(location_raw, dtype=float)
        if np.size(location) == 0:
            return None

        return {
            "object_id": object_id,
            "location": location,
            "confidence": confidence,
            "source_lm": str(getattr(lm, "learning_module_id", "lm")),
        }
