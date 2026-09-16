import json

import numpy as np
import pytest

from smpl18.corpus import (
    FORMAT_ID,
    CorpusError,
    read_corpus,
    write_subject,
    write_summary,
    write_trial,
)
from smpl18.reduce import apply, to_18
from smpl18.skeleton.definition import FROZEN_JOINT_NAMES, JOINT18_NAMES
from smpl18.skeleton.kinematics import fk_batch
from smpl18.skeleton.rotations import axis_angle_to_matrix, matrix_to_axis_angle


def record(constants):
    return {
        "subject_id": "S1", "gender": "neutral", "betas": [0.0] * 10, "trials": ["run", "walk"],
        "reduced_model": {
            "frozen_joints": list(FROZEN_JOINT_NAMES),
            "constants": matrix_to_axis_angle(constants).tolist(),
            "fit": {"residual_rms_m": 0.001, "residual_max_m": 0.002},
        },
    }


@pytest.fixture
def corpus(tmp_path):
    rng = np.random.default_rng(0)
    local = axis_angle_to_matrix(rng.normal(0, 0.3, (6, 24, 3)))
    constants = axis_angle_to_matrix(rng.normal(0, 0.2, (4, 3)))
    reduced = apply(local, constants)
    subject_dir = tmp_path / "S1"
    write_subject(subject_dir, record(constants))
    provenance = ["measured"] * 18
    for name in ("walk", "run"):
        write_trial(subject_dir, name, poses=matrix_to_axis_angle(to_18(reduced)),
                    trans=rng.normal(size=(6, 3)), fps=100.0, up_axis="y",
                    joint_provenance=provenance, frame_valid=np.ones(6, bool),
                    manifest={"source": {"kind": "joint_centres", "format": "trc"},
                              "profile": {"id": "adhoc"}, "settings_sha256": "abc"})
    write_summary(tmp_path)
    return tmp_path, local, reduced


def test_the_summary_counts_what_the_directory_holds(corpus) -> None:
    root, *_ = corpus
    summary = json.loads((root / "SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["format_id"] == FORMAT_ID
    assert (summary["subjects"], summary["trials"], summary["frames"]) == (1, 2, 12)
    assert summary["source_kind"] == "joint_centres" and summary["format"] == "trc"
    assert summary["provenance_counts"]["pelvis"] == {"measured": 2}
    assert summary["converter"]["package"] == "smpl18"


def test_a_trial_stores_exactly_the_corpus_keys(corpus) -> None:
    root, *_ = corpus
    with np.load(root / "S1" / "walk.npz") as data:
        assert sorted(data.files) == sorted(
            ["poses", "joint_names", "trans", "fps", "up_axis", "joint_provenance", "frame_valid"]
        )
        assert data["poses"].shape == (6, 18, 3)
        assert tuple(data["joint_names"]) == JOINT18_NAMES


def test_the_24_joint_pose_comes_back_with_every_orientation(corpus) -> None:
    root, local, reduced = corpus
    trial = read_corpus(root).subject("S1").trial("walk")
    rebuilt = trial.local_rotations_24()
    rest = np.random.default_rng(1).normal(0, 0.1, (24, 3))
    zeros = np.zeros((6, 3))
    _, original_world = fk_batch(rest, local, zeros)
    _, rebuilt_world = fk_batch(rest, rebuilt, zeros)
    body = list(range(22))           # the hands are not stored
    np.testing.assert_allclose(rebuilt[:, body], reduced[:, body], atol=1e-12)
    kept = [j for j in body if j not in (3, 6, 13, 14)]
    np.testing.assert_allclose(rebuilt_world[:, kept], original_world[:, kept], atol=1e-12)
    assert trial.poses_24().shape == (6, 24, 3)
    assert trial.manifest["trial_id"] == "walk"


def test_the_reader_walks_subjects_and_trials(corpus) -> None:
    root, *_ = corpus
    loaded = read_corpus(root)
    assert loaded.subject_ids() == ["S1"]
    subject = loaded.subjects()[0]
    assert subject.trial_ids() == ["run", "walk"]
    assert subject.gender == "neutral" and subject.betas.shape == (10,)
    with pytest.raises(CorpusError, match="no trial"):
        subject.trial("swim")
    with pytest.raises(CorpusError, match="no subject"):
        loaded.subject("S2")


def test_foreign_keys_and_directories_are_refused(corpus, tmp_path_factory) -> None:
    root, *_ = corpus
    extra = root / "S1" / "odd.npz"
    np.savez(extra, poses=np.zeros((1, 18, 3)), colour=np.zeros(1))
    with pytest.raises(CorpusError, match="outside the corpus format"):
        read_corpus(root).subject("S1").trial("odd")
    with pytest.raises(CorpusError, match="not a corpus"):
        read_corpus(tmp_path_factory.mktemp("empty"))


def test_writing_checks_shapes(tmp_path) -> None:
    with pytest.raises(CorpusError, match="poses"):
        write_trial(tmp_path, "t", poses=np.zeros((2, 24, 3)), trans=np.zeros((2, 3)), fps=1,
                    up_axis="y", joint_provenance=["measured"] * 18,
                    frame_valid=np.ones(2, bool), manifest={})
    with pytest.raises(CorpusError, match="trans"):
        write_trial(tmp_path, "t", poses=np.zeros((2, 18, 3)), trans=np.zeros((3, 3)), fps=1,
                    up_axis="y", joint_provenance=["measured"] * 18,
                    frame_valid=np.ones(2, bool), manifest={})


def test_a_trial_the_record_does_not_list_is_refused(corpus) -> None:
    root, *_ = corpus
    (root / "S1" / "walk.npz").rename(root / "S1" / "stale.npz")
    with pytest.raises(CorpusError, match="another run"):
        read_corpus(root).subject("S1").trial("stale")
