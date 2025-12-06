"""Adapter for integrating Monty (neocortex) with Hippocampus.

This module bridges Monty's Learning Modules with the hippocampus by
translating CMP messages (votes and outputs) into SpatialEvents.

This is the ONLY file in tbp.hippocampus that imports from tbp.monty.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, List, Protocol, Sequence, Union, runtime_checkable

import numpy as np

from tbp.hippocampus.entorhinal import EntorhinalCortex
from tbp.hippocampus.types import SpatialEvent

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
    from tbp.monty.frameworks.models.monty_base import MontyBase

logger = logging.getLogger(__name__)


@runtime_checkable
class SupportsVoteAndOutput(Protocol):
    """Protocol for objects that support send_out_vote and get_output."""

    learning_module_id: str

    def send_out_vote(self) -> Union[dict, None]:
        ...

    def get_output(self) -> Any:
        ...


def _extract_vote_events(
        lm: SupportsVoteAndOutput,
        lm_id: str,
        timestamp: float,
) -> List[SpatialEvent]:
    """Extract SpatialEvents from LM vote.

    Args:
        lm: Learning module to poll.
        lm_id: Learning module identifier.
        timestamp: Event timestamp.

    Returns:
        List of SpatialEvents from the vote.
    """
    events: List[SpatialEvent] = []

    try:
        vote = lm.send_out_vote()
    except Exception as e:
        logger.warning(f"Error getting vote from LM {lm_id}: {e}")
        return events

    if vote is None:
        return events

    possible_states = vote.get("possible_states", {})

    for graph_id, states in possible_states.items():
        for state in states:
            try:
                morph_features = getattr(state, "morphological_features", {}) or {}

                event = SpatialEvent(
                    timestamp=timestamp,
                    location=np.asarray(state.location),
                    orientation=np.asarray(
                        morph_features.get("pose_vectors", np.eye(3))
                    ),
                    source_id=lm_id,
                    confidence=float(getattr(state, "confidence", 0.0)),
                    features={},
                    event_type="vote",
                    object_id=str(graph_id),
                    extra={
                        "sensed_pose": vote.get("sensed_pose_rel_body"),
                        "pose_fully_defined": morph_features.get("pose_fully_defined"),
                    },
                )
                events.append(event)
            except Exception as e:
                logger.warning(f"Error processing vote state from LM {lm_id}: {e}")

    return events


def _extract_output_event(
        lm: SupportsVoteAndOutput,
        lm_id: str,
        timestamp: float,
) -> Union[SpatialEvent, None]:
    """Extract SpatialEvent from LM primary output.

    Args:
        lm: Learning module to poll.
        lm_id: Learning module identifier.
        timestamp: Event timestamp.

    Returns:
        SpatialEvent from the output, or None if not usable.
    """
    try:
        output = lm.get_output()
    except Exception as e:
        logger.warning(f"Error getting output from LM {lm_id}: {e}")
        return None

    if output is None:
        return None

    # Only process if use_state is True (LM has confident output)
    if not getattr(output, "use_state", False):
        return None

    try:
        morph_features = getattr(output, "morphological_features", {}) or {}
        non_morph_features = getattr(output, "non_morphological_features", {}) or {}

        event = SpatialEvent(
            timestamp=timestamp,
            location=np.asarray(output.location),
            orientation=np.asarray(
                morph_features.get("pose_vectors", np.eye(3))
            ),
            source_id=lm_id,
            confidence=float(getattr(output, "confidence", 0.0)),
            features=dict(non_morph_features),
            event_type="output",
            object_id=non_morph_features.get("object_id"),
            extra={
                "pose_fully_defined": morph_features.get("pose_fully_defined"),
                "on_object": morph_features.get("on_object"),
            },
        )
        return event
    except Exception as e:
        logger.warning(f"Error processing output from LM {lm_id}: {e}")
        return None


class MontyAdapter:
    """Adapter that translates Monty LM outputs to hippocampus SpatialEvents.

    This adapter:
    - Polls Monty Learning Modules for votes and outputs
    - Translates CMP State messages to SpatialEvents
    - Forwards events to EntorhinalCortex

    The adapter can be called each Monty step to stream LM information
    to the hippocampus in real-time.

    Attributes:
        hippocampus: EntorhinalCortex to send events to.
        include_votes: Whether to process vote messages.
        include_outputs: Whether to process LM output messages.
        step_count: Number of steps processed.
    """

    def __init__(
            self,
            hippocampus: EntorhinalCortex,
            include_votes: bool = True,
            include_outputs: bool = True,
    ) -> None:
        """Initialize the adapter.

        Args:
            hippocampus: EntorhinalCortex instance to receive events.
            include_votes: Whether to capture vote messages.
            include_outputs: Whether to capture LM primary outputs.
        """
        self.hippocampus = hippocampus
        self.include_votes = include_votes
        self.include_outputs = include_outputs
        self.step_count = 0

    def process_step(self, monty: MontyBase) -> int:
        """Process one Monty step: poll all LMs and forward to hippocampus.

        Args:
            monty: Monty instance with learning_modules attribute.

        Returns:
            Number of events forwarded.
        """
        return self.consume_from_lms(monty.learning_modules)

    def consume_from_lms(
            self, learning_modules: Sequence[Union[LearningModule, SupportsVoteAndOutput]]
    ) -> int:
        """Poll learning modules and forward events to hippocampus.

        Args:
            learning_modules: List of LearningModule instances to poll.

        Returns:
            Number of events forwarded.
        """
        self.step_count += 1
        events: List[SpatialEvent] = []
        timestamp = time.time()

        for lm in learning_modules:
            if not isinstance(lm, SupportsVoteAndOutput):
                logger.debug(f"Skipping LM {lm}: doesn't support vote/output protocol")
                continue

            lm_id = str(getattr(lm, "learning_module_id", id(lm)))

            # Process votes
            if self.include_votes:
                vote_events = _extract_vote_events(lm, lm_id, timestamp)
                events.extend(vote_events)

            # Process primary output
            if self.include_outputs:
                output_event = _extract_output_event(lm, lm_id, timestamp)
                if output_event is not None:
                    events.append(output_event)

        # Forward to hippocampus
        if events:
            self.hippocampus.receive_events(events)
            logger.debug(
                f"Forwarded {len(events)} events to hippocampus at step {self.step_count}"
            )

        return len(events)

    def reset(self) -> None:
        """Reset step counter."""
        self.step_count = 0

    def __repr__(self) -> str:
        return (
            f"MontyAdapter(steps={self.step_count}, "
            f"hippocampus={self.hippocampus}, "
            f"votes={self.include_votes}, outputs={self.include_outputs})"
        )
