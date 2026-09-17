import json

import numpy as np
import pytest

from smpl18.blender.plan import (
    PLAN_SCHEMA,
    PlanError,
    ScenePlan,
    bone_tails,
    build_plan,
    plan_for_trial,
    read_plan,
)
from smpl18.mesh import posed_vertices, skin
from smpl18.skeleton.definition import CHILDREN, JOINT_NAMES, NUM_JOINTS, PARENTS


def test_bone_tails_follow_the_body_and_never_have_zero_length(rest) -> None:
    tails = bone_tails(rest, up_axis="y", leaf_reach=0.7)
    assert np.all(np.linalg.norm(tails - rest, axis=1) > 1e-6)
    for joint in range(NUM_JOINTS):
        if len(CHILDREN[joint]) == 1:
            np.testing.assert_allclose(tails[joint], rest[CHILDREN[joint][0]])
    # The pelvis and spine3 have several children and follow the one up the body.
    np.testing.assert_allclose(tails[0], rest[JOINT_NAMES.index("spine1")])
    np.testing.assert_allclose(tails[9], rest[JOINT_NAMES.index("neck")])
    # A leaf runs on past itself, along the bone that arrives at it.
    ankle, foot = JOINT_NAMES.index("left_ankle"), JOINT_NAMES.index("left_foot")
    np.testing.assert_allclose(tails[foot], rest[foot] + 0.7 * (rest[foot] - rest[ankle]))


def test_a_plan_carries_what_the_scene_needs_and_reads_back(tmp_path, body, walk) -> None:
    local, trans = walk
    plan = build_plan(body, betas=np.zeros(10), local=local, trans=trans, fps=50.0, up_axis="y",
                      correctives=False, sample_frames=3, tolerance=2e-4, leaf_reach=0.7,
                      about={"subject": "S01"})
    assert plan.frames == local.shape[0] and plan.num_vertices == body.num_vertices
    assert not plan.has_correctives
    assert plan.about["schema"] == PLAN_SCHEMA and plan.about["subject"] == "S01"
    assert plan.about["checked_frames"] == [0, local.shape[0] // 2, local.shape[0] - 1]

    path = plan.write(tmp_path / "scene" / "one.plan.npz")
    again = read_plan(path)
    np.testing.assert_allclose(again.transforms, plan.transforms)
    np.testing.assert_allclose(again.rest, plan.rest)
    np.testing.assert_allclose(again.vertices, plan.vertices, atol=1e-6)   # float32 on disk
    assert again.tolerance == plan.tolerance and again.about == plan.about
    assert again.sample_vertices.shape == (3, body.num_vertices, 3)


def test_the_samples_are_what_this_package_skins(body, walk) -> None:
    local, trans = walk
    betas = np.array([0.3, -0.1, 0.2, 0, 0, 0, 0, 0, 0, 0])
    plan = build_plan(body, betas=betas, local=local, trans=trans, fps=50.0, up_axis="y",
                      correctives=False, sample_frames=4, tolerance=2e-4, leaf_reach=0.7)
    posed = posed_vertices(body, betas, local, trans, with_pose_offsets=False)
    np.testing.assert_allclose(plan.sample_vertices, posed.vertices[plan.sample_frames], atol=1e-12)


def test_skinning_the_plan_reproduces_the_posed_body(body, walk) -> None:
    """The one promise the scene relies on: weights and transforms are enough."""
    local, trans = walk
    plan = build_plan(body, betas=np.zeros(10), local=local, trans=trans, fps=50.0, up_axis="y",
                      correctives=False, sample_frames=0, tolerance=0.0, leaf_reach=0.7)
    posed = posed_vertices(body, np.zeros(10), local, trans, with_pose_offsets=False)
    np.testing.assert_allclose(skin(plan.vertices, plan.weights, plan.transforms), posed.vertices,
                               atol=1e-12)


def test_correctives_are_carried_as_directions_and_amounts(body, walk, with_posedirs) -> None:
    local, trans = walk
    plan = build_plan(with_posedirs, betas=np.zeros(10), local=local, trans=trans, fps=50.0,
                      up_axis="y", correctives=True, sample_frames=2, tolerance=2e-4,
                      leaf_reach=0.7)
    assert plan.has_correctives
    assert plan.posedirs.shape == (body.num_vertices, 3, 207)
    assert plan.pose_amounts.shape == (local.shape[0], 207)
    assert plan.about["correctives"] == "shape_keys"
    assert plan.about["pose_blend_shapes_mm"] > 0
    # Applying the offsets to the rest surface and skinning is the model's own answer.
    offsets = np.einsum("vij,tj->tvi", plan.posedirs, plan.pose_amounts[plan.sample_frames])
    moved = skin(plan.vertices + offsets, plan.weights, plan.transforms[plan.sample_frames])
    np.testing.assert_allclose(plan.sample_vertices, moved, atol=1e-12)


def test_a_plan_without_correctives_says_how_far_off_it_is(body, walk, with_posedirs) -> None:
    local, trans = walk
    plan = build_plan(with_posedirs, betas=np.zeros(10), local=local, trans=trans, fps=50.0,
                      up_axis="y", correctives=False, sample_frames=2, tolerance=2e-4,
                      leaf_reach=0.7)
    assert plan.about["correctives"] == "off"
    assert plan.about["pose_blend_shapes_mm"] > 0            # the reader is told what was left out
    assert not plan.has_correctives


def test_a_trial_becomes_a_plan_with_its_subject_on_it(corpus, body) -> None:
    trial = corpus.subject("S01").trial("walk")
    plan = plan_for_trial(trial, body, correctives=False, sample_frames=2, tolerance=2e-4,
                          leaf_reach=0.7)
    assert plan.about["subject"] == "S01" and plan.about["trial"] == "walk"
    assert plan.about["gender"] == trial.subject.gender
    assert plan.about["stand_in_model"] is True
    assert plan.about["frames_taken"] == f"0:{trial.frames}"
    assert plan.frames == trial.frames
    # The transforms carry the rest joints onto the trial's own world joint centres.
    homogeneous = np.concatenate([plan.rest, np.ones((NUM_JOINTS, 1))], axis=1)
    carried = np.einsum("tjab,jb->tja", plan.transforms, homogeneous)[:, :, :3]
    np.testing.assert_allclose(carried, trial.joints_world(body), atol=1e-12)


def test_a_frame_range_takes_that_slice(corpus, body) -> None:
    trial = corpus.subject("S01").trial("walk")
    plan = plan_for_trial(trial, body, correctives=False, sample_frames=2, tolerance=2e-4,
                          leaf_reach=0.7, frames=slice(5, 15, 2))
    assert plan.frames == 5 and plan.about["frames_taken"] == "5:15:2"
    with pytest.raises(PlanError, match="selects no frame"):
        plan_for_trial(trial, body, correctives=False, sample_frames=1, tolerance=0.0,
                       leaf_reach=0.7, frames=slice(500, 600))


def test_a_file_that_is_not_a_plan_is_refused(tmp_path, body, walk) -> None:
    local, trans = walk
    plan = build_plan(body, betas=np.zeros(10), local=local, trans=trans, fps=50.0, up_axis="y",
                      correctives=False, sample_frames=1, tolerance=0.0, leaf_reach=0.7)
    other = tmp_path / "other.npz"
    np.savez(other, poses=local)
    with pytest.raises(PlanError, match="schema"):
        read_plan(other)
    thin = tmp_path / "thin.npz"
    arrays = plan.arrays()
    del arrays["weights"]
    np.savez(thin, **arrays)
    with pytest.raises(PlanError, match="weights"):
        read_plan(thin)


def test_the_plan_is_json_and_numbers_only(tmp_path, body, walk) -> None:
    """Nothing pickled: Blender reads it with allow_pickle=False, as this package does."""
    local, trans = walk
    path = build_plan(body, betas=np.zeros(10), local=local, trans=trans, fps=50.0, up_axis="y",
                      correctives=False, sample_frames=1, tolerance=0.0,
                      leaf_reach=0.7).write(tmp_path / "plan.npz")
    with np.load(path, allow_pickle=False) as data:
        assert json.loads(str(data["about"]))["written_by"].startswith("smpl18 ")
        assert [str(name) for name in data["joint_names"]] == list(JOINT_NAMES)
        assert [int(parent) for parent in data["parents"]] == list(PARENTS)


def test_a_plan_can_be_built_by_hand(body, walk) -> None:
    """The dataclass is the interface; a caller with its own pose need not go through a corpus."""
    local, trans = walk
    plan = ScenePlan(rest=np.zeros((NUM_JOINTS, 3)), tails=np.ones((NUM_JOINTS, 3)),
                     transforms=np.tile(np.eye(4), (2, NUM_JOINTS, 1, 1)),
                     vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]),
                     weights=np.zeros((3, NUM_JOINTS)), fps=30.0)
    assert plan.frames == 2 and plan.num_vertices == 3 and not plan.has_correctives
    assert plan.arrays()["fps"] == 30.0
