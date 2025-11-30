from __future__ import annotations

import types

import numpy as np

from tbp.monty.frameworks.meta.meta_controller import MetaController


class DummyLM:
    def __init__(self, lm_id: str, feature_weights=None):
        self.learning_module_id = lm_id
        self._entropy = 0.0
        self._slope = 0.0
        self.feature_weights = feature_weights or {"a": 1.0, "b": 1.0}
        self._backup = None

    def get_entropy(self):
        return self._entropy

    def get_mlh_slope(self, window: int = 10):
        return self._slope

    def set_temp_feature_weights(self, weights: dict, ttl_steps: int = 0):
        if self._backup is None:
            self._backup = dict(self.feature_weights)
        self.feature_weights = dict(weights)

    def restore_feature_weights(self):
        if self._backup is not None:
            self.feature_weights = self._backup
            self._backup = None


class DummyMonty:
    def __init__(self, lms):
        self.learning_modules = lms
        self.before_vote_cb = None
        self.after_step_cb = None
        self.transform_manager = None


def test_meta_controller_triggers_and_restores():
    # Low entropy and near-zero slope to trigger stagnation
    lm = DummyLM("lm0")
    lm._entropy = 0.0
    lm._slope = 0.0
    monty = DummyMonty([lm])

    meta = MetaController(
        monty,
        entropy_min=0.2,
        slope_threshold=1e-3,
        stagnation_window=1,
        cooldown_steps=1,
        weight_ttl=2,
        transform_prob=0.0,  # deterministic: skip transform
    )

    # Simulate a step cycle where before_vote runs, then after_step
    meta.before_vote([], [])

    # Weights should have been perturbed
    assert lm.feature_weights != {"a": 1.0, "b": 1.0}

    # Advance steps via after_step to expire TTL and trigger restore
    meta.after_step(dict(step_type="matching_step", total_steps=1, episode_steps=1))
    meta.after_step(dict(step_type="matching_step", total_steps=2, episode_steps=2))
    meta.after_step(dict(step_type="matching_step", total_steps=3, episode_steps=3))

    # Now stagnation remains but TTL expired => controller should restore
    # Trigger decision pass again
    meta.before_vote([], [])
    assert lm.feature_weights == {"a": 1.0, "b": 1.0}


def test_meta_controller_sets_transform_manager():
    lm = DummyLM("lm0")
    lm._entropy = 0.0
    lm._slope = 0.0
    monty = DummyMonty([lm])
    meta = MetaController(
        monty,
        entropy_min=1.0,
        slope_threshold=1.0,
        stagnation_window=1,
        cooldown_steps=1,
        weight_ttl=1,
        transform_prob=1.0,  # always set
    )
    meta.before_vote([], [])
    assert monty.transform_manager is not None
