"""A 24-vertex synthetic model whose rest joints are its template: J_regressor is the identity.

The template is a crude but valid skeleton in the SMPL rest frame: spine up +Y, legs down -Y,
arms out along +/-X. It lets the tests say "the wrist drops when the shoulder adducts" without
the licensed 6890-vertex asset.
"""

from __future__ import annotations

import numpy as np
import pytest

from smpl18.model import Model
from smpl18.skeleton.definition import NUM_JOINTS, PARENTS

LEFT_LEG = (1, 4, 7, 10)
RIGHT_LEG = (2, 5, 8, 11)
LEFT_ARM = (16, 18, 20, 22)
RIGHT_ARM = (17, 19, 21, 23)


def tiny_template() -> np.ndarray:
    step = 0.12
    template = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
    for joint in range(1, NUM_JOINTS):
        offset = np.array([0.0, step, 0.0])
        if joint in LEFT_LEG:
            offset = np.array([0.08, -step, 0.0])
        elif joint in RIGHT_LEG:
            offset = np.array([-0.08, -step, 0.0])
        elif joint in LEFT_ARM:
            offset = np.array([step, 0.0, 0.0])
        elif joint in RIGHT_ARM:
            offset = np.array([-step, 0.0, 0.0])
        template[joint] = template[PARENTS[joint]] + offset
    return template


def tiny_model(num_betas: int = 2) -> Model:
    shapedirs = np.zeros((NUM_JOINTS, 3, num_betas), dtype=np.float64)
    shapedirs[:, 1, 0] = 0.01 * np.arange(NUM_JOINTS)         # beta0 stretches +Y
    return Model(
        v_template=tiny_template(),
        shapedirs=shapedirs,
        J_regressor=np.eye(NUM_JOINTS),
        kintree_parents=np.array(PARENTS),
    )


@pytest.fixture
def model() -> Model:
    return tiny_model()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
