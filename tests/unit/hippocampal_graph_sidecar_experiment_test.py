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

import numpy as np

from tbp.monty.frameworks.experiments.hippocampal_graph_sidecar_experiment import (
    MontyObjectRecognitionHippocampalGraphSidecarExperiment,
)


@dataclass(frozen=True)
class _State:
    confidence: float
    location: np.ndarray


class _LM:
    def __init__(self, vote):
        self._vote = vote

    def send_out_vote(self):
        return self._vote


class _Model:
    def __init__(self, learning_modules):
        self.learning_modules = learning_modules


def _make_experiment_with_model(model) -> MontyObjectRecognitionHippocampalGraphSidecarExperiment:
    exp = MontyObjectRecognitionHippocampalGraphSidecarExperiment.__new__(
        MontyObjectRecognitionHippocampalGraphSidecarExperiment
    )
    exp.model = model
    exp._semantic_id_to_label = None
    return exp


def test_process_lm_vote_includes_all_graph_ids():
    exp = _make_experiment_with_model(_Model([]))

    lm = _LM(
        {
            "possible_states": {
                "graph_a": [_State(confidence=0.9, location=np.array([1.0, 2.0, 3.0]))],
                "graph_b": [_State(confidence=0.8, location=np.array([4.0, 5.0, 6.0]))],
            }
        }
    )

    candidates = exp._process_lm_vote(lm)
    assert len(candidates) == 2
    assert {c[0] for c in candidates} == {"graph_a", "graph_b"}


def test_extract_raw_vote_candidates_aggregates_across_lms():
    lm1 = _LM(
        {
            "possible_states": {
                "a": [_State(confidence=0.9, location=np.array([1.0, 2.0, 3.0]))],
                "b": [_State(confidence=0.8, location=np.array([4.0, 5.0, 6.0]))],
            }
        }
    )
    lm2 = _LM(
        {
            "possible_states": {
                "c": [_State(confidence=0.7, location=np.array([7.0, 8.0, 9.0]))],
            }
        }
    )
    exp = _make_experiment_with_model(_Model([lm1, lm2]))

    raw = exp._extract_raw_vote_candidates()
    assert len(raw) == 3
    assert [obj_id for obj_id, _, _ in raw] == ["a", "b", "c"]

