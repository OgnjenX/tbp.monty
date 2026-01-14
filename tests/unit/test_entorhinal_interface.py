# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
import quaternion as qt

from tbp.monty.frameworks.models.entorhinal_interface import (
    EntorhinalLocationIntegrator,
    EntorhinalLocationIntegratorConfig,
    observation_coordinate_frame,
    observation_has_world_coordinates,
)


def test_observation_has_world_coordinates_nested_mapping():
    obs = {"agent": {"patch": {"world_camera": np.eye(4)}}}
    assert observation_has_world_coordinates(obs) is True
    assert observation_coordinate_frame(obs) == "world"


def test_entorhinal_integrator_noop_when_disabled():
    integrator = EntorhinalLocationIntegrator(
        EntorhinalLocationIntegratorConfig(enabled=False)
    )
    agent_state = {
        "position": np.array([10.0, 0.0, 0.0]),
        "rotation": qt.one,
        "sensors": {"patch.depth": {"position": np.zeros(3), "rotation": qt.one}},
    }
    loc = np.array([1.0, 2.0, 3.0])
    out = integrator.sensor_to_world(loc, agent_state)
    assert np.allclose(out, loc)


def test_entorhinal_integrator_identity_pose_adds_translation():
    integrator = EntorhinalLocationIntegrator(
        EntorhinalLocationIntegratorConfig(enabled=True, sensor_key_hint="patch")
    )
    agent_state = {
        "position": np.array([10.0, 0.0, 0.0]),
        "rotation": qt.one,
        "sensors": {"patch.depth": {"position": np.array([0.5, 0.0, 0.0]), "rotation": qt.one}},
    }
    loc_sensor = np.array([1.0, 2.0, 3.0])
    out = integrator.sensor_to_world(loc_sensor, agent_state)
    assert np.allclose(out, np.array([11.5, 2.0, 3.0]))
