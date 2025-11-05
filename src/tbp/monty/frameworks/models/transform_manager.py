from __future__ import annotations

"""Identity transform manager for vote-combination hooks.

This module defines a minimal TransformManager interface used by Monty models to
optionally override or postprocess the transforms applied when combining votes
between learning modules. By default, this manager is a no-op.
"""

import numpy as np
from typing import Any, Iterable, Tuple


class IdentityTransformManager:
    """Default no-op transform manager.

    Methods accept both scipy Rotation objects and raw 3x3 rotation matrices, and
    return the inputs unchanged.
    """

    def pairwise_transform(
        self,
        receiving_id: int,
        sending_id: int,
        receiving_pose: Iterable[Any],
        sending_pose: Iterable[Any],
        sensor_disp: np.ndarray,
        sensor_rotation: Any,
    ) -> Tuple[np.ndarray, Any]:
        return sensor_disp, sensor_rotation

    def postprocess_votes(
        self,
        receiving_id: int,
        sending_id: int,
        object_id: str,
        locations: np.ndarray,
        rotations: Any,
    ) -> Tuple[np.ndarray, Any]:
        return locations, rotations

    def postprocess_state_objects(
        self,
        receiving_id: int,
        sending_id: int,
        object_id: str,
        states: list,
    ) -> list:
        return states

