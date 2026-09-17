"""A stand-in body with a surface, a short walk, and a one-subject corpus of it."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from smpl18 import convert, synthetic
from smpl18.corpus import read_corpus
from smpl18.mesh import NUM_POSE_FEATURES
from smpl18.model.demo import demo_model
from smpl18.model.load import Model
from smpl18.skeleton.kinematics import rest_joints
from smpl18.sources.subject import SubjectInfo

PACKAGE = Path(__file__).resolve().parents[2]
CONFIGS = PACKAGE / "configs"


@pytest.fixture(scope="session")
def body() -> Model:
    return demo_model(with_mesh=True)


@pytest.fixture(scope="session")
def with_posedirs(body) -> Model:
    """The same body with pose blend shapes, which a mannequin of blocks otherwise has none of."""
    rng = np.random.default_rng(11)
    return Model(
        v_template=body.v_template, shapedirs=body.shapedirs, J_regressor=body.J_regressor,
        kintree_parents=body.kintree_parents, weights=body.weights, faces=body.faces,
        posedirs=rng.normal(size=(body.num_vertices, 3, NUM_POSE_FEATURES)) * 0.002,
        gender="neutral",
    )


@pytest.fixture(scope="session")
def rest(body) -> np.ndarray:
    return rest_joints(body, np.zeros(10))


@pytest.fixture(scope="session")
def walk(rest) -> tuple[np.ndarray, np.ndarray]:
    return synthetic.walking_motion(20, 50.0, seed=2)


@pytest.fixture(scope="session")
def settings() -> dict:
    quick = copy.deepcopy(yaml.safe_load((CONFIGS / "settings" / "default.yaml")
                                         .read_text(encoding="utf-8")))
    quick["shape"]["sample_frames"] = 20
    quick["reduce"]["sample_frames"] = 20
    return quick


@pytest.fixture(scope="session")
def corpus(tmp_path_factory, body, walk, settings):
    """One subject, one trial, converted from SMPL parameters so the pose is known exactly."""
    root = tmp_path_factory.mktemp("blender-corpus")
    local, trans = walk
    parameters = root / "walk.npz"
    np.savez(parameters, poses=synthetic.rotations_to_axis_angle(local), trans=trans,
             betas=np.zeros(10), mocap_framerate=50.0)
    trial = convert.parameter_trial(parameters, model=body, up_axis="y", settings=settings)
    fit = convert.fit_subject(body, [trial], settings)
    convert.write_subject_corpus(root / "corpus", subject=SubjectInfo("S01", "neutral"),
                                 model=body, trials=[trial], fit=fit, settings=settings,
                                 settings_files=[])
    return read_corpus(root / "corpus")
