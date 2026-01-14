# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class EntorhinalLocationIntegratorConfig:
    """Configuration for converting sensor-frame points to a shared world frame.

    This is an explicit "EC-like" adapter layer. It is intentionally simple:
    it uses motor-system pose (agent + sensor) to map a point expressed in the
    sensor frame into a world/global frame.
    """

    enabled: bool = False
    sensor_key_hint: Optional[str] = None


class EntorhinalLocationIntegrator:
    """Convert egocentric (sensor-frame) locations into world coordinates."""

    def __init__(self, config: EntorhinalLocationIntegratorConfig):
        self._cfg = config

    @staticmethod
    def _find_sensor_state(agent_state: Mapping[str, Any], hint: Optional[str]) -> Optional[Mapping[str, Any]]:
        sensors = agent_state.get("sensors")
        if not isinstance(sensors, Mapping) or not sensors:
            return None

        # Try "patch" -> "patch.depth" first (common), then any key containing hint.
        if hint:
            depth_key = f"{hint}.depth"
            if depth_key in sensors:
                return sensors[depth_key]
            if hint in sensors:
                return sensors[hint]
            for key, value in sensors.items():
                if hint in str(key):
                    return value

        # Fall back to first sensor state.
        return next(iter(sensors.values()))

    def sensor_to_world(self, location_sensor: np.ndarray, agent_state: Mapping[str, Any]) -> np.ndarray:
        """Map a 3D point from sensor frame into world frame.

        Args:
            location_sensor: Shape (3,) point in sensor coordinates.
            agent_state: Dict-like agent state with keys:
              - "position": world position of agent
              - "rotation": world rotation of agent (quaternion)
              - "sensors": mapping of sensor states, each with "position" and "rotation"

        Returns:
            Shape (3,) point in world coordinates.
        """
        loc = np.asarray(location_sensor, dtype=float).reshape(3)
        if not self._cfg.enabled:
            return loc

        try:
            import quaternion as qt  # type: ignore
        except Exception:
            # Keep import-time dependencies minimal; if quaternion support isn't
            # available, fall back to a no-op rather than breaking callers.
            return loc

        sensor_state = self._find_sensor_state(agent_state, self._cfg.sensor_key_hint)
        if sensor_state is None:
            return loc

        agent_position = np.asarray(agent_state.get("position", np.zeros(3)), dtype=float).reshape(3)
        agent_rotation = agent_state.get("rotation", None)
        sensor_position = np.asarray(sensor_state.get("position", np.zeros(3)), dtype=float).reshape(3)
        sensor_rotation = sensor_state.get("rotation", None)

        if agent_rotation is None or sensor_rotation is None:
            return loc

        # Sensor pose in world.
        agent_rot_m = qt.as_rotation_matrix(agent_rotation)
        sensor_translation_world = agent_position + (agent_rot_m @ sensor_position)
        sensor_rotation_world = agent_rotation * sensor_rotation
        sensor_rot_m_world = qt.as_rotation_matrix(sensor_rotation_world)

        return (sensor_rot_m_world @ loc) + sensor_translation_world


def observation_has_world_coordinates(observation: Any) -> bool:
    """Best-effort detection for whether an observation includes world-frame coords.

    Many env pipelines include a non-null "world_camera" modality only when
    world-coordinate conversion was applied upstream. We treat this as a signal
    that downstream locations are already in a shared world frame.
    """
    if isinstance(observation, Mapping):
        if observation.get("semantic_3d_in_world") is True:
            return True
        if observation.get("semantic_3d_frame") == "world":
            return True
        if "world_camera" in observation and observation["world_camera"] is not None:
            return True
        for value in observation.values():
            if observation_has_world_coordinates(value):
                return True
    return False


def observation_coordinate_frame(observation: Any) -> Optional[str]:
    """Return the coordinate frame for `semantic_3d` when deterministically known.

    Returns:
        "world" if the observation declares it is in world coordinates.
        "sensor" if the observation declares it is in sensor coordinates.
        None if unknown/ambiguous.
    """
    if not isinstance(observation, Mapping):
        return None

    frame = observation.get("semantic_3d_frame")
    if frame in ("world", "sensor"):
        return frame
    if observation.get("semantic_3d_in_world") is True:
        return "world"
    if observation.get("semantic_3d_in_world") is False:
        return "sensor"

    # Fall back: `world_camera` strongly implies world-frame conversion occurred,
    # but its absence is not proof of sensor-frame coordinates.
    if observation.get("world_camera") is not None:
        return "world"

    for value in observation.values():
        nested = observation_coordinate_frame(value)
        if nested is not None:
            return nested

    return None
