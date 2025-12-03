# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Any

import numpy as np

from tbp.monty.frameworks.models.buffer import BufferEncoder


class State:
    """State class used as message packages passed in Monty using CMP.

    The cortical messaging protocol (CMP) is used to pass messages between Monty
    components and makes sure we can easily set up arbitrary configurations of them.
    This class makes it easier to define the CMP in one place and defines the content
    and structure of messages passed between Monty components. It also contains some
    helper funtions to access and modify the message content.

    States are represented in this format but can be interpreted by the receiver in
    different ways:
       Observed states: states output py sensor modules
       Hypothesized states: states output by learning modules
       Goal states: motor output of learning modules

    Attributes:
        location: 3D vector representing the location of the state
        morphological_features: dictionary of morphological features. Should include
            pose_vectors of shape (3,3) and pose_fully_defined (bool).
        non_morphological_features: dictionary of non-morphological features.
        confidence: confidence in the state. In range [0,1].
        use_state: boolean indicating whether the state should be used or not.
        sender_id: string identifying the sender of the state.
        sender_type: string identifying the type of sender. Can be "SM" or "LM".
    """

    def __init__(
        self,
        location,
        morphological_features,
        non_morphological_features,
        confidence,
        use_state,
        sender_id,
        sender_type,
    ):
        """Initialize a state."""
        # Preserve legacy `location` attribute for backward compatibility.
        # Also normalize into a `location_payload` dict that supports multiple
        # location types (metric vectors, SDRs, topo ids, etc.).
        self.location = location
        # QUESTION: Divide into pose_dependent and pose_independent features instead?
        self.morphological_features = morphological_features
        self.non_morphological_features = non_morphological_features
        self.confidence = confidence
        self.use_state = use_state
        self.sender_id = sender_id
        self.sender_type = sender_type
        self._set_allowable_sender_types()
        if self.use_state:
            self._check_all_attributes()

    # --- Location helpers -------------------------------------------------
    def is_metric(self) -> bool:
        """Return True if the stored location is a metric vector.

        Legacy `location` numeric arrays are treated as metric by default.
        """
        return self.location_payload is not None and self.location_payload.get(
            "type"
        ) == "metric"

    def is_sdr(self) -> bool:
        """Return True if the stored location is an SDR."""
        return self.location_payload is not None and self.location_payload.get(
            "type"
        ) == "sdr"

    def location_dim(self) -> int | None:
        """Return dimensionality of the location value when applicable.

        Returns None for non-vector or unknown types.
        """
        val = self.location_payload.get("value")
        if isinstance(val, np.ndarray):
            return val.shape[0]
        return None

    def get_location_value(self):
        """Return the underlying location value (payload 'value').

        For legacy callers this mirrors `self.location` when the payload is
        metric; for non-metric payloads it returns the payload value which may
        be an SDR, id, or other structure.
        """
        return self.location_payload.get("value")

    def get_location_type(self) -> str | None:
        """Return the location payload type string (e.g., 'metric', 'sdr')."""
        return self.location_payload.get("type") if self.location_payload else None

    def get_frame_id(self) -> str | None:
        """Return the reference frame id for the location."""
        return self.location_payload.get("frame_id") if self.location_payload else None

    def decode_to_metric(self, registry=None) -> np.ndarray:
        """Decode location to metric coordinates.

        If location is already metric, returns the value directly.
        If location payload contains 'metric_origin', uses that directly.
        Otherwise uses the provided (or default) registry to decode.

        Args:
            registry: Optional TransformRegistry. If None, uses default.

        Returns:
            Metric coordinates as numpy array.

        Raises:
            KeyError: If no transform is registered to decode the location type.
        """
        if self.is_metric():
            return np.asarray(self.get_location_value())

        # Check if payload contains metric_origin (e.g., SDR with preserved origin)
        if self.location_payload and "metric_origin" in self.location_payload:
            return np.asarray(self.location_payload["metric_origin"])

        # Lazy import to avoid circular dependencies
        from tbp.monty.frameworks.utils.location_transforms import get_default_registry

        if registry is None:
            registry = get_default_registry()
        loc_type = self.get_location_type()
        if loc_type is None:
            raise ValueError("Cannot decode: location payload has no type")
        decoded = registry.apply(loc_type, "metric", self.location_payload)
        return np.asarray(decoded["value"])

    def encode_to_sdr(self, registry=None) -> dict:
        """Encode metric location to SDR using the transform registry.

        If location is already SDR, returns the payload directly.
        Otherwise uses the provided (or default) registry to encode.

        Args:
            registry: Optional TransformRegistry. If None, uses default.

        Returns:
            SDR location payload dict.

        Raises:
            KeyError: If no transform is registered for metric->sdr.
        """
        if self.is_sdr():
            return self.location_payload.copy()
        if not self.is_metric():
            raise ValueError("Can only encode metric locations to SDR")
        from tbp.monty.frameworks.utils.location_transforms import get_default_registry

        if registry is None:
            registry = get_default_registry()
        return registry.apply("metric", "sdr", self.location_payload)

    def __repr__(self):
        """Return a string representation of the object."""
        repr_string = (
            f"State from {self.sender_id}:\n"
            f"   Location: {np.round(self.location, 3)}.\n"
            f"   Morphological Features: \n"
        )
        if self.morphological_features is not None:
            for feature in self.morphological_features:
                feat_val = self.morphological_features[feature]
                if isinstance(feat_val, np.ndarray):
                    feat_val = np.round(feat_val, 3)
                if feature == "pose_vectors":
                    repr_string += f"       {feature}: \n"
                    for vector in feat_val:
                        repr_string += f"           {vector}\n"
                else:
                    repr_string += f"       {feature}: {feat_val}\n"
        repr_string += "   Non-Morphological Features: \n"
        if self.non_morphological_features is not None:
            for feature in self.non_morphological_features:
                feat_val = self.non_morphological_features[feature]
                # Round numpy arrays and numeric scalars for nicer repr output.
                if isinstance(feat_val, np.ndarray):
                    feat_val = np.round(feat_val, 3)
                elif isinstance(feat_val, (float, int, np.floating, np.integer)):
                    feat_val = round(float(feat_val), 3)
                repr_string += f"       {feature}: {feat_val}\n"
        repr_string += (
            f"   Confidence: {self.confidence}\n"
            f"   Use State: {self.use_state}\n"
            f"   Sender Type: {self.sender_type}\n"
        )
        return repr_string

    def _set_allowable_sender_types(self):
        """Set the allowable sender types of this State class."""
        self.allowable_sender_types = ("SM", "LM")

    def transform_morphological_features(self, translation=None, rotation=None):
        """Apply translation and/or rotation to morphological features."""
        if translation is not None:
            self.location += translation
        if rotation is not None:
            self.morphological_features["pose_vectors"] = np.dot(
                rotation, self.morphological_features["pose_vectors"]
            )

    def set_displacement(self, displacement, ppf=None):
        """Add displacement (represented as dict) to state.

        TODO S: Add this to state or in another place?
        """
        self.displacement = {
            "displacement": displacement,
        }
        if ppf is not None:
            self.displacement["ppf"] = ppf

    def get_feature_by_name(self, feature_name):
        if feature_name in self.morphological_features.keys():
            feature_val = self.morphological_features[feature_name]
        elif feature_name in self.non_morphological_features.keys():
            feature_val = self.non_morphological_features[feature_name]
        else:
            raise ValueError(f"Feature {feature_name} not found in state.")
        return feature_val

    def get_nth_pose_vector(self, pose_vector_index):
        """Return the nth pose vector.

        When self.sender_type == "SM", the first pose vector is the surface normal and
        the second and third are the curvature directions.
        When self.sender_type == "LM", the pose vectors correspond to the rotation of
        the object relative to the model learned of it.
        """
        return self.morphological_features["pose_vectors"][pose_vector_index]

    def get_surface_normal(self):
        """Return the surface normal vector.

        Raises:
            ValueError: If `self.sender_type` is not SM
        """
        if self.sender_type == "SM":
            return self.get_nth_pose_vector(0)

        raise ValueError("Sender type must be SM to get surface normal.")

    def get_pose_vectors(self):
        """Return the pose vectors."""
        return self.morphological_features["pose_vectors"]

    def get_curvature_directions(self):
        """Return the curvature direction vectors.

        Raises:
            ValueError: If `self.sender_type` is not SM
        """
        if self.sender_type == "SM":
            return self.get_nth_pose_vector(1), self.get_nth_pose_vector(2)

        raise ValueError("Sender type must be SM to get curvature directions.")

    def get_on_object(self):
        """Return whether we think we are on the object or not.

        This is currently used in the policy to stay on the object.
        """
        if "on_object" in self.morphological_features.keys():
            return self.morphological_features["on_object"]

        # TODO: Use depth values to estimate on_object (either threshold or large
        # displacement)
        return True

    def _check_all_attributes(self):
        assert "pose_vectors" in self.morphological_features.keys(), (
            "pose_vectors should be in morphological_features but keys are "
        )
        f"{self.morphological_features.keys()}"
        # TODO S: may want to test length and angle between vectors as well
        # pose_vectors should still be a (3,3) array when present (surface
        # normals and curvature directions are 3D quantities for sensor-based
        # inputs). We keep this check but do not force the location to be
        # numeric-3D unless the payload is metric.
        assert self.morphological_features["pose_vectors"].shape == (
            3,
            3,
        ), "pose should be defined by three orthonormal unit vectors but pose_vectors "
        f"shape is {self.morphological_features['pose_vectors'].shape}"
        assert "pose_fully_defined" in self.morphological_features.keys()
        assert isinstance(self.morphological_features["pose_fully_defined"], bool), (
            "pose_fully_defined must be a boolean but type is "
        )
        f"{type(self.morphological_features['pose_fully_defined'])}"

        # Only enforce 3D location shape when the location is metric (legacy
        # numeric locations). This allows abstract LMs to attach SDRs or
        # other non-metric payloads without failing this API-level check.
        if self.is_metric():
            loc_val = self.location_payload.get("value")
            assert isinstance(loc_val, np.ndarray), (
                "metric location must be a numpy array"
            )
            assert loc_val.shape == (
                3,
            ), f"Location must be a 3D vector but shape is {loc_val.shape}"
        else:
            # Non-metric locations are permitted; no assertion on shape.
            pass
        assert self.confidence >= 0 and self.confidence <= 1, (
            f"Confidence must be in [0,1] but is {self.confidence}"
        )
        assert isinstance(self.use_state, bool), (
            f"use_state must be a boolean but is {type(self.use_state)}"
        )
        assert isinstance(self.sender_id, str), (
            f"sender_id must be string but is {type(self.sender_id)}"
        )
        assert self.sender_type in self.allowable_sender_types, (
            f"sender_type must be SM or LM but is {self.sender_type}"
        )

    # Keep location payload and legacy location attribute in sync -------------
    @property
    def location(self):
        return getattr(self, "_location_raw", None)

    @location.setter
    def location(self, value):
        self._location_raw = value
        if isinstance(value, dict) and "type" in value:
            # Already a payload dict; store directly so downstream code can access
            self.location_payload = value
        else:
            # Treat legacy numeric inputs as metric coordinates in body frame
            self.location_payload = {
                "type": "metric",
                "value": value,
                "frame_id": "body",
            }


class GoalState(State):
    """Specialization of :class:`State` for goal states with null (None) values allowed.

    Specialized form of state that still adheres to the cortical messaging protocol,
    but can have null (None) values associated with the location and morphological
    features.

    Used by goal-state generators (GSGs) to communicate goal states to other GSGs, and
    to motor actuators.

    The state variables generally have the same meaning as for the base :class:`State`
    class, and they represent the target values for the receiving system. Thus
    if a goal-state specifies a particular object ID (non-morphological feature)
    in a particular pose (location and morphological features), then the receiving
    system should attempt to achieve that state.

    Note however that for the goal-state, the confidence corresponds to the conviction
    with which a GSG believes that the current goal-state should be acted upon. Float
    bound in [0.0, 1.0].
    """

    def __init__(
        self,
        location: np.ndarray | None,
        morphological_features: dict[str, Any] | None,
        non_morphological_features: dict[str, Any] | None,
        confidence: float,
        use_state: bool,
        sender_id: str,
        sender_type: str,
        goal_tolerances: dict[str, Any] | None,
        info: dict[str, Any] | None = None,
    ):
        """Initialize a goal state.

        Args:
            location: the location to move to in global/body-centric coordinates, or
              `None` if the location is not specified as part of the goal state.
              For example, this may be a point on an object's surface or a location
              nearby from which a sensor would have a good view of the target point.
            morphological_features: dictionary of morphological features or `None`.
              For example, it may include pose vectors, whether the pose is fully
              defined, etc.
            non_morphological_features: a dictionary containing non-morphological
              features at the target location or `None`.
            confidence: a float between 0 and 1 representing the confidence in the goal
              state.
            use_state: a boolean indicating whether the goal state should be used.
            sender_id: the ID of the sender of the goal state (e.g., `"LM_0"`).
            sender_type: the type of sender of the goal state (e.g., `"GSG"`).
            goal_tolerances: Dictionary of tolerances that GSGs use when determining
                whether the current state of the LM matches the driving goal-state
                or `None`. As such, a GSG can send a goal state with more or less
                strict tolerances if certain elements of the state (e.g. the location
                of a mug vs its orientation) are more or less important.
            info: Optional metadata for logging purposes.
        """
        self.goal_tolerances = goal_tolerances
        self.info = info or {}

        super().__init__(
            location,
            morphological_features,
            non_morphological_features,
            confidence,
            use_state,
            sender_id,
            sender_type,
        )

    def _set_allowable_sender_types(self):
        """Set the allowable sender types of this State class."""
        self.allowable_sender_types = ("GSG", "SM")

    def _check_all_attributes(self):
        """Overwrite base attribute check to also allow for None values."""
        if self.morphological_features is not None:
            assert "pose_vectors" in self.morphological_features.keys(), (
                "pose_vectors should be in morphological_features but keys are "
            )
            f"{self.morphological_features.keys()}"
            assert np.any(
                np.isnan(self.morphological_features["pose_vectors"])
            ) or self.morphological_features["pose_vectors"].shape == (
                3,
                3,
            ), (
                "pose should be undefined, or defined by three orthonormal unit "
                "vectors but pose_vectors shape is "
                f"{self.morphological_features['pose_vectors'].shape}"
            )
            assert "pose_fully_defined" in self.morphological_features.keys()
            assert (
                isinstance(self.morphological_features["pose_fully_defined"], bool)
            ) or self.morphological_features["pose_fully_defined"] is None, (
                "pose_fully_defined must be a boolean or None but type is "
            )
            f"{type(self.morphological_features['pose_fully_defined'])}"
        if self.location is not None:
            # GoalState may receive legacy numeric locations or the normalized
            # `location_payload`. If a metric payload is present, validate it is
            # 3D. Otherwise, if a legacy numpy location is supplied, validate
            # that instead.
            loc_payload = getattr(self, "location_payload", None)
            if loc_payload is not None and loc_payload.get("type") == "metric":
                val = loc_payload.get("value")
                assert isinstance(val, np.ndarray) and val.shape == (
                    3,
                ), f"Location must be a 3D vector but shape is {getattr(val, 'shape', None)}"
            elif isinstance(self.location, np.ndarray):
                assert self.location.shape == (
                    3,
                ), f"Location must be a 3D vector but shape is {self.location.shape}"

        assert self.confidence >= 0 and self.confidence <= 1, (
            f"Confidence must be in [0,1] but is {self.confidence}"
        )
        assert isinstance(self.use_state, bool), (
            f"use_state must be a boolean but is {type(self.use_state)}"
        )
        assert isinstance(self.sender_id, str), (
            f"sender_id must be string but is {type(self.sender_id)}"
        )
        assert self.sender_type in self.allowable_sender_types, (
            f"sender_type must be in {self.allowable_sender_types} but is "
            f"{self.sender_type}"
        )
        # info is optional, but it must be a dictionary.
        assert isinstance(self.info, dict), "info must be a dictionary"


def encode_goal_state(goal_state: GoalState) -> dict[str, Any]:
    """Encode a goal state into a dictionary.

    Args:
        goal_state: The goal state to encode.

    Returns:
        A dictionary containing the goal state's attributes.
    """
    return {
        "location": goal_state.location,
        "morphological_features": goal_state.morphological_features,
        "non_morphological_features": goal_state.non_morphological_features,
        "confidence": goal_state.confidence,
        "use_state": goal_state.use_state,
        "sender_id": goal_state.sender_id,
        "sender_type": goal_state.sender_type,
        "goal_tolerances": goal_state.goal_tolerances,
        "info": goal_state.info,
    }


BufferEncoder.register(GoalState, encode_goal_state)
