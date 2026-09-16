"""Every source kind, end to end, on synthetic captures of one known motion.

Each test writes the files a user would have, converts them, reads the corpus back, rebuilds
the 24-joint pose and measures it against the motion the files were made from.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from smpl18 import convert, synthetic
from smpl18.corpus import read_corpus
from smpl18.fit.correspondence import Correspondence
from smpl18.formats import trc
from smpl18.model.demo import demo_model
from smpl18.skeleton.definition import JOINT_NAMES
from smpl18.skeleton.frames import frame_change
from smpl18.skeleton.kinematics import fk_batch, rest_joints
from smpl18.sources.markers import MarkerSet
from smpl18.sources.subject import SubjectInfo

PACKAGE = Path(__file__).resolve().parents[1]
CONFIGS = PACKAGE / "configs"
FRAMES = 50
BETAS = np.array([0.4, -0.3, 0.2, 0.5, -0.4, 0.3, 0.1, -0.2, 0.3, -0.1])
Y_TO_Z = frame_change("y", "z")


@pytest.fixture(scope="module")
def settings():
    loaded = yaml.safe_load((CONFIGS / "settings" / "default.yaml").read_text("utf-8"))
    loaded = copy.deepcopy(loaded)
    loaded["shape"]["sample_frames"] = 60
    loaded["reduce"]["sample_frames"] = 200
    return loaded


@pytest.fixture(scope="module")
def body():
    model = demo_model()
    return model, rest_joints(model, BETAS)


@pytest.fixture(scope="module")
def walk(body):
    _, rest = body
    local, trans = synthetic.walking_motion(FRAMES, 100.0, seed=11)
    return local, trans, fk_batch(rest, local, trans)[0]


def stored_error(root, subject_id, model, truth, joints):
    trial = read_corpus(root).subject(subject_id).trials()[0]
    rebuilt = trial.joints_world(model)[:, joints]
    return np.linalg.norm(rebuilt - truth[:, joints], axis=2), trial


def convert_and_write(tmp_path, model, trials, settings, subject):
    fit = convert.fit_subject(model, trials, settings)
    convert.write_subject_corpus(tmp_path / "corpus", subject=subject, model=model,
                                 trials=trials, fit=fit, settings=settings,
                                 settings_files=[{"path": "default.yaml", "sha256": "x"}])
    return fit


BODY = [JOINT_NAMES.index(n) for n in (
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "neck")]


def test_markers(tmp_path, body, walk, settings) -> None:
    model, rest = body
    local, trans, truth = walk
    labels, markers = synthetic.conventional_markers(rest, local, trans)
    path = trc.write(tmp_path / "walk.trc", labels, Y_TO_Z.apply(markers) * 1000.0,
                     rate_hz=100.0, units="mm")
    subject = SubjectInfo("S1", "neutral", synthetic.MEASUREMENTS, source="file",
                          gender_from="file")
    markerset = MarkerSet.load(CONFIGS / "markersets" / "conventional_full_body.yaml")
    trial = convert.marker_trial(path, markerset=markerset, subject=subject, up_axis="z",
                                 settings=settings)
    assert trial.kind == "marker_trajectories" and trial.fps == 100.0
    convert_and_write(tmp_path, model, [trial], settings, subject)
    error, stored = stored_error(tmp_path / "corpus", "S1", model, truth, BODY)
    assert np.median(error) < 0.005 and error.max() < 0.02
    manifest = stored.manifest
    assert manifest["source"]["native_units"] == "mm"
    assert manifest["tables"]["markerset"]["id"] == "conventional_full_body"
    assert manifest["validation"]["stored_18_joint"]["rms_m"] < 0.01
    record = read_corpus(tmp_path / "corpus").subject("S1").record
    assert record["model_is_stand_in"] and record["gender_source"] == "subject_file"
    assert record["reduced_model"]["fit"]["residual_rms_m"] < 0.01


def test_markers_with_a_gap_are_bridged(tmp_path, body, walk, settings) -> None:
    _, rest = body
    local, trans, _ = walk
    labels, markers = synthetic.conventional_markers(rest, local, trans)
    valid = np.ones(markers.shape[:2], bool)
    valid[20:23, labels.index("LKNE")] = False
    path = trc.write(tmp_path / "gap.trc", labels, markers, rate_hz=100.0, units="m", valid=valid)
    subject = SubjectInfo("S1", "neutral", synthetic.MEASUREMENTS)
    markerset = MarkerSet.load(CONFIGS / "markersets" / "conventional_full_body.yaml")
    trial = convert.marker_trial(path, markerset=markerset, subject=subject, up_axis="y",
                                 settings=settings)
    assert trial.repairs["gap_fill"]["samples_bridged"] == 3
    assert trial.targets.positions.valid.all()


def test_joint_centres_with_segment_rotations(tmp_path, body, walk, settings) -> None:
    model, rest = body
    local, trans, truth = walk
    labels, centres = synthetic.joint_centres(rest, local, trans)
    _, world = fk_batch(rest, local, trans)
    segments = {"LTH": "left_hip", "RTH": "right_hip", "PV": "pelvis"}
    path = tmp_path / "centres.npz"
    np.savez(path, names=np.array(labels), positions=centres * 100.0, fps=np.float64(100.0),
             units=np.array("cm"), segment_names=np.array(list(segments)),
             segment_rotations=np.stack([world[:, JOINT_NAMES.index(j)] for j in segments.values()],
                                        axis=1))
    table = Correspondence.load(CONFIGS / "correspondence" / "joint_centre_labels.yaml")
    data = yaml.safe_load((CONFIGS / "correspondence" / "joint_centre_labels.yaml").read_text("utf-8"))
    data["joints"]["left_hip"]["segment"] = "LTH"
    data["joints"]["right_hip"]["segment"] = "RTH"
    data["joints"]["pelvis"] = {"segment": "PV"}
    table = Correspondence.from_mapping(data)
    trial = convert.centre_trial(path, correspondence=table, up_axis="y", settings=settings)
    assert trial.targets.orientations.joints == (0, 1, 2)
    subject = SubjectInfo("S2", "neutral")
    convert_and_write(tmp_path, model, [trial], settings, subject)
    error, stored = stored_error(tmp_path / "corpus", "S2", model, truth, BODY)
    assert np.median(error) < 0.005 and error.max() < 0.02
    assert stored.manifest["tables"]["use"]["orientation_targets"]["pelvis"] == "PV"


def test_joint_centres_from_a_trc(tmp_path, body, walk, settings) -> None:
    model, rest = body
    local, trans, truth = walk
    labels, centres = synthetic.joint_centres(rest, local, trans)
    path = trc.write(tmp_path / "jc.trc", labels, Y_TO_Z.apply(centres) * 1000.0, rate_hz=100.0,
                     units="mm")
    table = Correspondence.load(CONFIGS / "correspondence" / "joint_centre_labels.yaml")
    trial = convert.centre_trial(path, correspondence=table, up_axis="z", settings=settings)
    subject = SubjectInfo("S3", "neutral")
    convert_and_write(tmp_path, model, [trial], settings, subject)
    error, _ = stored_error(tmp_path / "corpus", "S3", model, truth, BODY)
    assert np.median(error) < 0.005 and error.max() < 0.02


def test_smpl_parameters(tmp_path, body, walk, settings) -> None:
    model, rest = body
    local, trans, _ = walk
    betas = np.zeros(16)
    betas[:10] = BETAS
    z_local, z_trans = Y_TO_Z.apply_to_pose(local, trans, pelvis_rest=rest[0])
    poses = np.zeros((FRAMES, 52, 3))
    poses[:, :22] = synthetic.rotations_to_axis_angle(z_local)[:, :22]
    path = tmp_path / "seq_poses.npz"
    np.savez(path, poses=poses.reshape(FRAMES, -1), trans=z_trans, betas=betas,
             mocap_framerate=np.float64(120.0))
    trial = convert.parameter_trial(path, model=model, up_axis="z", settings=settings)
    assert trial.fps == 120.0 and trial.source["joints_stored"] == 52
    assert trial.source["model_family"] == "smplh" and trial.source["betas_stored"] == 16
    assert trial.betas.shape == (model.num_betas,)
    subject = SubjectInfo("S4", "neutral")
    fit = convert_and_write(tmp_path, model, [trial], settings, subject)
    assert fit.shape is None
    truth_local = local.copy()
    truth_local[:, 22:] = np.eye(3)
    truth = fk_batch(rest, truth_local, trans)[0]
    error, stored = stored_error(tmp_path / "corpus", "S4", model, truth, list(range(22)))
    assert len(read_corpus(tmp_path / "corpus").subject("S4").betas) == model.num_betas
    # Only the freeze moves joints here, by what the subject record says it costs.
    reduced = read_corpus(tmp_path / "corpus").subject("S4").record["reduced_model"]["fit"]
    assert error.max() <= reduced["residual_max_m"] + 1e-9
    assert stored.joint_provenance == ("measured",) * 18


def test_smpl_parameters_need_a_rate(tmp_path, body, settings) -> None:
    model, _ = body
    path = tmp_path / "norate.npz"
    np.savez(path, poses=np.zeros((3, 72)), trans=np.zeros((3, 3)), betas=np.zeros(10))
    with pytest.raises(convert.ConversionError, match="frame rate"):
        convert.parameter_trial(path, model=model, up_axis="y", settings=settings)
    trial = convert.parameter_trial(path, model=model, up_axis="y", settings=settings, fps=30.0)
    assert trial.fps == 30.0


def test_opensim(tmp_path, body, settings) -> None:
    pytest.importorskip("smpl18.sources.skeleton.opensim")
    model, rest = body
    osim_path = synthetic.write_opensim_model(tmp_path / "model.osim", rest)
    mot_path = synthetic.write_mot(tmp_path / "walk.mot",
                                   synthetic.opensim_motion(FRAMES, 100.0, seed=2), 100.0,
                                   in_degrees=True)
    table = Correspondence.load(CONFIGS / "correspondence" / "opensim_rajagopal.yaml")
    trial = convert.opensim_trial(osim_path, mot_path, correspondence=table, settings=settings)
    assert trial.format == "osim_mot" and trial.fps == pytest.approx(100.0)
    assert "acromial_l" in trial.tables["use"]["position_targets"].values()
    subject = SubjectInfo("S5", "neutral")
    fit = convert_and_write(tmp_path, model, [trial], settings, subject)
    assert fit.poses[trial.id].position_error.rms < 0.01
    stored = read_corpus(tmp_path / "corpus").subject("S5").trials()[0]
    assert stored.manifest["validation"]["stored_18_joint"]["rms_m"] < 0.015
    codes = dict(zip(stored.joint_names, stored.joint_provenance))
    assert codes["neck"] == "absent" and codes["left_knee"] == "measured"


def test_bvh(tmp_path, body, walk, settings) -> None:
    pytest.importorskip("smpl18.sources.skeleton.bvh")
    model, rest = body
    local, trans, truth = walk
    path = synthetic.write_bvh(tmp_path / "clip.bvh", rest, local, trans, 100.0,
                               units_per_metre=100.0)
    table = Correspondence.load(CONFIGS / "correspondence" / "bvh_humanoid.yaml")
    trial = convert.bvh_trial(path, correspondence=table, up_axis="y", length_unit="cm",
                              settings=settings)
    assert trial.fps == pytest.approx(100.0)
    subject = SubjectInfo("S6", "neutral")
    convert_and_write(tmp_path, model, [trial], settings, subject)
    error, _ = stored_error(tmp_path / "corpus", "S6", model, truth, BODY)
    assert np.median(error) < 0.005 and error.max() < 0.02


def test_one_subject_cannot_mix_kinds(body, settings) -> None:
    model, _ = body
    first = convert.TrialInput(id="a", kind="joint_centres", format="trc", fps=1.0)
    second = convert.TrialInput(id="b", kind="marker_trajectories", format="trc", fps=1.0)
    with pytest.raises(convert.ConversionError, match="share a source kind"):
        convert.fit_subject(model, [first, second], settings)


def test_settings_files_merge_by_section(tmp_path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"schema": "smpl18_settings_v1", "pose": {"a": 1, "b": 2}}))
    over = tmp_path / "over.yaml"
    over.write_text(yaml.safe_dump({"schema": "smpl18_settings_v1", "pose": {"b": 3}}))
    merged, record = convert.load_settings([base, over])
    assert merged["pose"] == {"a": 1, "b": 3}
    assert [r["path"] for r in record] == ["base.yaml", "over.yaml"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"pose": {}}))
    with pytest.raises(convert.ConversionError, match="settings"):
        convert.load_settings([bad])


def parameters_file(path, poses, trans, *, rate=100.0, betas=None):
    frames = poses.shape[0]
    np.savez(path, poses=poses.reshape(frames, -1), trans=trans,
             betas=np.zeros(10) if betas is None else betas, mocap_framerate=np.float64(rate))
    return path


def test_smplx_parameters_are_refused(tmp_path, body, settings) -> None:
    model, _ = body
    path = parameters_file(tmp_path / "x.npz", np.zeros((4, 55, 3)), np.zeros((4, 3)))
    with pytest.raises(convert.ConversionError, match="SMPL-X"):
        convert.parameter_trial(path, model=model, up_axis="y", settings=settings)
    odd = parameters_file(tmp_path / "odd.npz", np.zeros((4, 30, 3)), np.zeros((4, 3)))
    with pytest.raises(convert.ConversionError, match="30 joints"):
        convert.parameter_trial(odd, model=model, up_axis="y", settings=settings)


def test_a_non_finite_parameter_frame_is_held_and_marked(tmp_path, body, walk, settings) -> None:
    model, _ = body
    local, trans, _ = walk
    poses = synthetic.rotations_to_axis_angle(local)
    poses[7, 4] = np.nan
    trans = trans.copy()
    trans[20] = np.inf
    path = parameters_file(tmp_path / "gaps.npz", poses, trans)
    trial = convert.parameter_trial(path, model=model, up_axis="y", settings=settings)
    assert trial.frame_valid.sum() == FRAMES - 2 and not trial.frame_valid[[7, 20]].any()
    assert np.isfinite(trial.local).all() and np.isfinite(trial.trans).all()
    subject = SubjectInfo("S7", "neutral")
    convert_and_write(tmp_path, model, [trial], settings, subject)
    stored = read_corpus(tmp_path / "corpus").subject("S7").trials()[0]
    assert stored.frame_valid.sum() == FRAMES - 2 and np.isfinite(stored.poses).all()


def test_a_failing_replacement_keeps_the_earlier_subject(tmp_path, body, walk, settings) -> None:
    model, _ = body
    local, trans, _ = walk
    path = parameters_file(tmp_path / "kept.npz", synthetic.rotations_to_axis_angle(local), trans)
    subject = SubjectInfo("S12", "neutral")
    trial = convert.parameter_trial(path, model=model, up_axis="y", settings=settings)
    convert_and_write(tmp_path, model, [trial], settings, subject)
    fit = convert.fit_subject(model, [trial], settings)
    partial = {key: value for key, value in settings.items() if key != "checks"}
    with pytest.raises(convert.ConversionError, match="checks"):
        convert.write_subject_corpus(tmp_path / "corpus", subject=subject, model=model,
                                     trials=[trial], fit=fit, settings=partial,
                                     settings_files=[], replace=True)
    assert read_corpus(tmp_path / "corpus").subject("S12").trial_ids() == ["kept"]
    with pytest.raises(convert.ConversionError, match="markers.max_gap_frames"):
        convert.validate_settings({k: v for k, v in settings.items() if k != "markers"},
                                  "marker_trajectories")
    convert.validate_settings({k: v for k, v in settings.items() if k != "markers"},
                              "joint_centres")


def test_a_subject_is_not_written_over(tmp_path, body, walk, settings) -> None:
    model, _ = body
    local, trans, _ = walk
    first = parameters_file(tmp_path / "first.npz", synthetic.rotations_to_axis_angle(local), trans)
    second = parameters_file(tmp_path / "second.npz", synthetic.rotations_to_axis_angle(local), trans)
    subject = SubjectInfo("S8", "neutral")
    trial = convert.parameter_trial(first, model=model, up_axis="y", settings=settings)
    convert_and_write(tmp_path, model, [trial], settings, subject)
    again = convert.parameter_trial(second, model=model, up_axis="y", settings=settings)
    fit = convert.fit_subject(model, [again], settings)
    with pytest.raises(convert.ConversionError, match="already in"):
        convert.write_subject_corpus(tmp_path / "corpus", subject=subject, model=model,
                                     trials=[again], fit=fit, settings=settings, settings_files=[])
    convert.write_subject_corpus(tmp_path / "corpus", subject=subject, model=model, trials=[again],
                                 fit=fit, settings=settings, settings_files=[], replace=True)
    stored = read_corpus(tmp_path / "corpus").subject("S8")
    assert stored.trial_ids() == ["second"]
    assert read_corpus(tmp_path / "corpus").summary["trials"] == 1


def test_a_wrong_length_unit_is_flagged(tmp_path, body, walk, settings) -> None:
    model, rest = body
    local, trans, _ = walk
    path = synthetic.write_bvh(tmp_path / "clip.bvh", rest, local, trans, 100.0,
                               units_per_metre=100.0)
    table = Correspondence.load(CONFIGS / "correspondence" / "bvh_humanoid.yaml")
    trial = convert.bvh_trial(path, correspondence=table, up_axis="y", length_unit="m",
                              settings=settings)
    subject = SubjectInfo("S9", "neutral")
    convert_and_write(tmp_path, model, [trial], settings, subject)
    record = read_corpus(tmp_path / "corpus").subject("S9").record
    assert any("length unit" in warning for warning in record["checks"])


def test_a_wrong_up_axis_is_flagged(tmp_path, body, walk, settings) -> None:
    model, rest = body
    local, trans, _ = walk
    labels, centres = synthetic.joint_centres(rest, local, trans)
    path = trc.write(tmp_path / "jc.trc", labels, Y_TO_Z.apply(centres), rate_hz=100.0, units="m")
    table = Correspondence.load(CONFIGS / "correspondence" / "joint_centre_labels.yaml")
    trial = convert.centre_trial(path, correspondence=table, up_axis="y", settings=settings)
    subject = SubjectInfo("S10", "neutral")
    convert_and_write(tmp_path, model, [trial], settings, subject)
    stored = read_corpus(tmp_path / "corpus").subject("S10")
    assert any("--up-axis" in warning for warning in stored.record["checks"])
    assert stored.trials()[0].manifest["checks"]
    # The same file with the right axis raises nothing.
    right = convert.centre_trial(path, correspondence=table, up_axis="z", settings=settings)
    convert_and_write(tmp_path / "right", model, [right], settings, SubjectInfo("S11", "neutral"))
    assert read_corpus(tmp_path / "right" / "corpus").subject("S11").record["checks"] == []


def test_gait2392_knees_are_found(tmp_path, body, settings) -> None:
    model, rest = body
    osim_path = synthetic.write_opensim_model(tmp_path / "model.osim", rest)
    osim_path.write_text(osim_path.read_text(encoding="utf-8").replace("walker_knee_", "knee_"),
                         encoding="utf-8")
    values = {name.replace("walker_knee_", "knee_"): value
              for name, value in synthetic.opensim_motion(FRAMES, 100.0, seed=2).items()}
    mot_path = synthetic.write_mot(tmp_path / "walk.mot", values, 100.0, in_degrees=True)
    table = Correspondence.load(CONFIGS / "correspondence" / "opensim_rajagopal.yaml")
    trial = convert.opensim_trial(osim_path, mot_path, correspondence=table, settings=settings)
    use = trial.tables["use"]
    assert use["position_targets"]["left_knee"] == "knee_l"
    assert "walker_knee_l" not in use["missing_in_source"]
    assert dict(zip(JOINT_NAMES, trial.targets.provenance))["left_hip"] == "measured"


def test_a_model_without_gravity_needs_an_up_axis(tmp_path, body, settings) -> None:
    model, rest = body
    osim_path = synthetic.write_opensim_model(tmp_path / "model.osim", rest)
    osim_path.write_text(osim_path.read_text(encoding="utf-8").replace(
        "<gravity>0 -9.80665 0</gravity>", "<gravity>0 0 0</gravity>"), encoding="utf-8")
    mot_path = synthetic.write_mot(tmp_path / "walk.mot",
                                   synthetic.opensim_motion(10, 100.0, seed=2), 100.0,
                                   in_degrees=True)
    table = Correspondence.load(CONFIGS / "correspondence" / "opensim_rajagopal.yaml")
    with pytest.raises(convert.ConversionError, match="up axis"):
        convert.opensim_trial(osim_path, mot_path, correspondence=table, settings=settings)
    trial = convert.opensim_trial(osim_path, mot_path, correspondence=table, settings=settings,
                                  up_axis="y")
    assert trial.source["native_up_axis"] == "y"


def test_a_bvh_without_a_frame_period_is_refused(tmp_path, body, walk, settings) -> None:
    _, rest = body
    local, trans, _ = walk
    path = synthetic.write_bvh(tmp_path / "clip.bvh", rest, local[:3], trans[:3], 100.0,
                               units_per_metre=100.0)
    text = path.read_text(encoding="utf-8")
    start = text.index("Frame Time:")
    end = text.index("\n", start)
    path.write_text(text[:start] + "Frame Time: 0" + text[end:], encoding="utf-8")
    table = Correspondence.load(CONFIGS / "correspondence" / "bvh_humanoid.yaml")
    with pytest.raises(convert.ConversionError, match="Frame Time"):
        convert.bvh_trial(path, correspondence=table, up_axis="y", length_unit="cm",
                          settings=settings)
