# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Protocol, runtime_checkable

import numpy as np

from tbp.monty.frameworks.models.graph_matching import (
    MontyForGraphMatching,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_orthonormal_vectors,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class GoalStateDrivenPolicy(Protocol):
    """Protocol for motor policies that accept driving goal states."""

    use_goal_state_driven_actions: bool

    def set_driving_goal_state(self, goal_state) -> None:
        """Set the driving goal state."""



class MontyForEvidenceGraphMatching(MontyForGraphMatching):
    """Monty model for evidence based graphs.

    Customize voting and union of possible matches.
    """

    def __init__(self, *args, **kwargs):
        """Initialize and reset LM."""
        self.association_learning_enabled = kwargs.pop(
            "association_learning_enabled",
            True,
        )
        self.min_association_strength = kwargs.pop(
            "min_association_strength",
            0.3,
        )
        # {recv_lm: {recv_obj: {send_lm: {send_obj: count}}}}
        self.association_counts = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        )
        super().__init__(*args, **kwargs)

    def _get_lm_id(self, lm, idx: int) -> str:
        return getattr(lm, "learning_module_id", f"lm_{idx}")

    def _get_voted_object_ids(self, vote: dict | None) -> list[str]:
        """Return object IDs that the LM is voting for on this step.

        This intentionally uses the same "strong evidence" gating as voting itself:
        if an object ID appears in possible_states, it has already surpassed
        vote_evidence_threshold in the LM.
        """
        if not vote or not isinstance(vote, dict):
            return []
        possible_states = vote.get("possible_states")
        if not isinstance(possible_states, dict) or not possible_states:
            return []
        return [obj_id for obj_id, states in possible_states.items() if states]

    def _update_association_counts(self, votes_per_lm: list[dict | None]) -> None:
        if not self.association_learning_enabled:
            return
        active_lms = self._get_active_lms(votes_per_lm)
        for recv_lm_id, recv_objs in active_lms:
            self._update_associations_for_receiving_lm(
                recv_lm_id,
                recv_objs,
                active_lms,
            )

    def _get_active_lms(
        self,
        votes_per_lm: list[dict | None],
    ) -> list[tuple[str, list[str]]]:
        """Get list of active LMs with voted object IDs."""
        active_lms = []
        for idx, lm in enumerate(self.learning_modules):
            vote = votes_per_lm[idx] if idx < len(votes_per_lm) else None
            voted_object_ids = self._get_voted_object_ids(vote)
            if voted_object_ids:
                active_lms.append((self._get_lm_id(lm, idx), voted_object_ids))
        return active_lms

    def _update_associations_for_receiving_lm(
        self,
        recv_lm_id: str,
        recv_objs: list[str],
        active_lms: list[tuple[str, list[str]]],
    ) -> None:
        """Update association counts for a receiving LM from all sending LMs."""
        for send_lm_id, send_objs in active_lms:
            if send_lm_id == recv_lm_id:
                continue
            for recv_obj_id in recv_objs:
                for send_obj_id in send_objs:
                    self.association_counts[recv_lm_id][recv_obj_id][send_lm_id][
                        send_obj_id
                    ] += 1

    def _get_association_strength(
        self,
        recv_lm_id: str,
        recv_obj_id: str,
        send_lm_id: str,
        send_obj_id: str,
    ) -> float:
        send_map = (
            self.association_counts.get(recv_lm_id, {})
            .get(recv_obj_id, {})
            .get(send_lm_id)
        )
        if not send_map:
            return 0.0
        count = send_map.get(send_obj_id, 0)
        if count <= 0:
            return 0.0
        max_count = max(send_map.values())
        return float(count / max_count) if max_count else 0.0

    def _pass_infos_to_motor_system(self):
        """Pass processed observations and goal-states to the motor system.

        Currently there is no complex connectivity or hierarchy, and all generated
        goal-states are considered bound for the motor-system. TODO M change this.
        """
        super()._pass_infos_to_motor_system()

        # Check the motor-system can receive goal-states
        policy = self.motor_system._policy
        if (
            isinstance(policy, GoalStateDrivenPolicy)
            and policy.use_goal_state_driven_actions
        ):
            best_goal_state = None
            best_goal_confidence = -np.inf
            for current_goal_state in self.gsg_outputs:
                if (
                    current_goal_state is not None
                    and current_goal_state.confidence > best_goal_confidence
                ):
                    best_goal_state = current_goal_state
                    best_goal_confidence = current_goal_state.confidence

            policy.set_driving_goal_state(best_goal_state)

    def _combine_votes(self, votes_per_lm):
        """Combine evidence from different lms.

        Returns:
            The combined votes.
        """
        self._update_association_counts(votes_per_lm)
        combined_votes = []
        for i in range(len(self.learning_modules)):
            lm_state_votes = {}
            if votes_per_lm[i] is not None:
                receiving_lm_pose = votes_per_lm[i]["sensed_pose_rel_body"]
                receiving_lm = self.learning_modules[i]
                receiving_lm_id = self._get_lm_id(receiving_lm, i)
                receiving_object_ids = receiving_lm.get_all_known_object_ids()
                for j in self.lm_to_lm_vote_matrix[i]:
                    if votes_per_lm[j] is not None:
                        sending_lm_id = self._get_lm_id(self.learning_modules[j], j)
                        sending_lm_pose = votes_per_lm[j]["sensed_pose_rel_body"]
                        sensor_disp = np.array(receiving_lm_pose[0]) - np.array(
                            sending_lm_pose[0]
                        )
                        sensor_rotation_disp, _ = align_orthonormal_vectors(
                            sending_lm_pose[1:],
                            receiving_lm_pose[1:],
                            as_scipy=False,
                        )
                        logger.debug(
                            f"LM {j} to {i} - displacement: {sensor_disp}, "
                            f"rotation: "
                            f"{sensor_rotation_disp}"
                        )
                        for obj in votes_per_lm[j]["possible_states"].keys():
                            # Get the displacement between the sending and receiving
                            # sensor and take this into account when transmitting
                            # possible locations on the object.
                            # "If I am here, you should be there."
                            lm_states_for_object = votes_per_lm[j]["possible_states"][
                                obj
                            ]
                            # Take the location votes and transform them so they would
                            # apply to the receiving LMs sensor. Basically saying, if my
                            # sensor is here and in this pose then your sensor should be
                            # there in that pose.
                            # NOTE: rotation votes are not being used right now.
                            transformed_lm_states_for_object = []
                            for s in lm_states_for_object:
                                # need to make a copy because the same vote state may be
                                # transformed in different ways depending on the
                                # receiving LMs' pose
                                new_s = copy.deepcopy(s)
                                rotated_displacement = new_s.get_pose_vectors().dot(
                                    sensor_disp
                                )
                                new_s.transform_morphological_features(
                                    translation=rotated_displacement,
                                    rotation=sensor_rotation_disp,
                                )
                                transformed_lm_states_for_object.append(new_s)
                            if not self.association_learning_enabled:
                                if obj in lm_state_votes.keys():
                                    lm_state_votes[obj].extend(
                                        transformed_lm_states_for_object
                                    )
                                else:
                                    lm_state_votes[obj] = (
                                        transformed_lm_states_for_object
                                    )
                                continue
                            for recv_obj_id in receiving_object_ids:
                                strength = self._get_association_strength(
                                    receiving_lm_id,
                                    recv_obj_id,
                                    sending_lm_id,
                                    obj,
                                )
                                if strength < self.min_association_strength:
                                    continue
                                if recv_obj_id in lm_state_votes:
                                    lm_state_votes[recv_obj_id].extend(
                                        transformed_lm_states_for_object
                                    )
                                else:
                                    lm_state_votes[recv_obj_id] = (
                                        transformed_lm_states_for_object.copy()
                                    )
            logger.debug(f"VOTE from LMs {self.lm_to_lm_vote_matrix[i]} to LM {i}")
            vote = lm_state_votes
            combined_votes.append(vote)
        return combined_votes

    def switch_to_exploratory_step(self):
        """Switch to exploratory step.

        Also, set mlh evidence high enough to generate output during exploration.
        """
        super().switch_to_exploratory_step()
        # Make sure new object ID gets communicated to higher level LMs during
        # exploration.
        for lm in self.learning_modules:
            lm.current_mlh["evidence"] = lm.object_evidence_threshold + 1
