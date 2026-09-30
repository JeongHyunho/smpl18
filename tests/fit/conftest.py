"""Fixtures for the fit tests: the stand-in body, a short walk, and the shipped settings with
sample sizes cut down so the suite stays quick."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from smpl18 import synthetic
from smpl18.model.demo import demo_model
from smpl18.model.load import Model
from smpl18.skeleton.kinematics import fk_batch, rest_joints

PACKAGE = Path(__file__).resolve().parents[2]
BETAS = np.array([0.5, -0.4, 0.3, 0.6, -0.5, 0.3, 0.2, -0.3, 0.4, -0.2])


def lopsided_model(symmetry: str = "skeleton") -> Model:
    """The stand-in with a template and shape directions that are not left/right symmetric, as
    SMPL's are not: a few millimetres of noise on every joint and direction, fixed by the seed."""
    base = demo_model()
    rng = np.random.default_rng(7)
    return Model(
        v_template=base.v_template + rng.normal(0.0, 0.006, base.v_template.shape),
        shapedirs=base.shapedirs + rng.normal(0.0, 0.003, base.shapedirs.shape),
        J_regressor=base.J_regressor,
        kintree_parents=base.kintree_parents,
        symmetry=symmetry,
    )


def shipped_settings() -> dict:
    settings = yaml.safe_load((PACKAGE / "configs/settings/default.yaml").read_text("utf-8"))
    settings = copy.deepcopy(settings)
    settings["shape"]["sample_frames"] = 60
    settings["reduce"]["sample_frames"] = 200
    return settings


@pytest.fixture
def settings() -> dict:
    return shipped_settings()


@pytest.fixture(scope="session")
def model():
    return demo_model()


@pytest.fixture(scope="session")
def rest(model) -> np.ndarray:
    return rest_joints(model, BETAS)


@pytest.fixture(scope="session")
def walk(rest):
    """``(local, trans, positions, world)`` of 40 frames of walking on the shaped stand-in."""
    local, trans = synthetic.walking_motion(40, 100.0, seed=3)
    positions, world = fk_batch(rest, local, trans)
    return local, trans, positions, world
