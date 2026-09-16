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
from smpl18.skeleton.kinematics import fk_batch, rest_joints

PACKAGE = Path(__file__).resolve().parents[2]
BETAS = np.array([0.5, -0.4, 0.3, 0.6, -0.5, 0.3, 0.2, -0.3, 0.4, -0.2])


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
