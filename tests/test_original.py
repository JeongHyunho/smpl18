"""A corpus trial written back as ordinary SMPL parameters."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from smpl18 import convert, synthetic
from smpl18.corpus import read_corpus
from smpl18.model.demo import demo_model
from smpl18.original import FPS_KEY, sequence_from_trial, write_sequence
from smpl18.skeleton.definition import FROZEN_JOINTS, HAND_JOINTS, JOINT_NAMES, NUM_JOINTS
from smpl18.skeleton.rotations import axis_angle_to_matrix
from smpl18.sources.subject import SubjectInfo

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def settings() -> dict:
    quick = copy.deepcopy(yaml.safe_load((CONFIGS / "settings" / "default.yaml")
                                         .read_text(encoding="utf-8")))
    quick["shape"]["sample_frames"] = 20
    quick["reduce"]["sample_frames"] = 20
    return quick


@pytest.fixture(scope="module")
def corpus(tmp_path_factory, settings):
    root = tmp_path_factory.mktemp("original")
    model = demo_model()
    local, trans = synthetic.walking_motion(20, 50.0, seed=5)
    parameters = root / "walk.npz"
    np.savez(parameters, poses=synthetic.rotations_to_axis_angle(local), trans=trans,
             betas=np.array([0.3, -0.2, 0.1, 0, 0, 0, 0, 0, 0, 0]), mocap_framerate=50.0)
    trial = convert.parameter_trial(parameters, model=model, up_axis="y", settings=settings)
    fit = convert.fit_subject(model, [trial], settings)
    convert.write_subject_corpus(root / "corpus", subject=SubjectInfo("S01", "female"),
                                 model=model, trials=[trial], fit=fit, settings=settings,
                                 settings_files=[])
    return read_corpus(root / "corpus")


def test_the_sequence_is_the_corpus_pose_in_24_joints(corpus) -> None:
    trial = corpus.subject("S01").trial("walk")
    sequence = sequence_from_trial(trial)
    assert sequence.poses.shape == (trial.frames, NUM_JOINTS, 3)
    np.testing.assert_allclose(sequence.poses, trial.poses_24())
    np.testing.assert_allclose(sequence.trans, trial.trans)
    np.testing.assert_allclose(sequence.betas, trial.subject.betas)
    assert sequence.gender == "female" and sequence.fps == trial.fps
    assert sequence.up_axis == trial.up_axis and sequence.frames == trial.frames


def test_the_hands_are_identity_and_the_frozen_joints_the_subjects_constants(corpus) -> None:
    trial = corpus.subject("S01").trial("walk")
    sequence = sequence_from_trial(trial)
    matrices = axis_angle_to_matrix(sequence.poses)
    still = np.broadcast_to(np.eye(3), (trial.frames, 3, 3))
    for hand in HAND_JOINTS:
        np.testing.assert_allclose(matrices[:, hand], still, atol=1e-12)
    for frozen, constant in zip(FROZEN_JOINTS, trial.subject.constants):
        held = np.broadcast_to(constant, (trial.frames, 3, 3))
        np.testing.assert_allclose(matrices[:, frozen], held, atol=1e-12)
    # And the file says as much, joint by joint.
    named = dict(zip(JOINT_NAMES, sequence.joint_provenance))
    assert {named[name] for name in ("spine1", "spine2", "left_collar")} == {"constant"}
    assert named["left_hand"] == named["right_hand"] == "absent"
    assert named["left_knee"] == "measured"


def test_the_file_holds_what_a_smpl_reader_looks_for(tmp_path, corpus) -> None:
    trial = corpus.subject("S01").trial("walk")
    path = write_sequence(tmp_path / "out" / "S01_walk.npz", sequence_from_trial(trial))
    with np.load(path, allow_pickle=False) as data:
        assert data["poses"].shape == (trial.frames, NUM_JOINTS * 3)
        assert data["trans"].shape == (trial.frames, 3)
        assert data["betas"].shape == (10,)
        assert float(data[FPS_KEY]) == trial.fps
        assert str(data["gender"]) == "female" and str(data["up_axis"]) == trial.up_axis
        assert [str(name) for name in data["joint_names"]] == list(JOINT_NAMES)
        assert len(data["joint_provenance"]) == NUM_JOINTS
        assert data["frame_valid"].dtype == bool
        about = json.loads(str(data["smpl18"]))
    assert about["subject"] == "S01" and about["trial"] == "walk"
    assert about["frozen_joints"] and "identity" in about["hands"]
    assert about["reduction_residual"]["residual_rms_m"] >= 0.0


def test_grouped_poses_are_written_when_asked(tmp_path, corpus) -> None:
    trial = corpus.subject("S01").trial("walk")
    path = write_sequence(tmp_path / "grouped.npz", sequence_from_trial(trial), flat=False)
    with np.load(path, allow_pickle=False) as data:
        assert data["poses"].shape == (trial.frames, NUM_JOINTS, 3)


def test_converting_the_export_again_returns_the_same_pose(tmp_path, corpus, settings) -> None:
    """The round trip this package can check itself: out as SMPL, back in, same rotations."""
    trial = corpus.subject("S01").trial("walk")
    path = write_sequence(tmp_path / "S01_walk.npz", sequence_from_trial(trial))
    model = demo_model()
    again = convert.parameter_trial(path, model=model, up_axis=trial.up_axis, settings=settings)
    np.testing.assert_allclose(again.local, trial.local_rotations_24(), atol=1e-12)
    np.testing.assert_allclose(again.trans, trial.trans, atol=1e-12)
    np.testing.assert_allclose(again.betas, trial.subject.betas, atol=1e-12)

    # And the corpus written from it holds the same stored pose: nothing was left to be refitted,
    # so the reduction has nothing left to lose.
    fit = convert.fit_subject(model, [again], settings)
    convert.write_subject_corpus(tmp_path / "corpus", subject=SubjectInfo("S01", "female"),
                                 model=model, trials=[again], fit=fit, settings=settings,
                                 settings_files=[])
    round_trip = read_corpus(tmp_path / "corpus").subject("S01").trial("S01_walk")
    np.testing.assert_allclose(round_trip.poses, trial.poses, atol=1e-9)
    residual = read_corpus(tmp_path / "corpus").subject("S01").record["reduced_model"]["fit"]
    assert residual["residual_rms_m"] < 1e-9
