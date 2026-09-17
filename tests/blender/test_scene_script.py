"""The script Blender runs, checked without Blender.

``scene.py`` imports ``bpy`` and ``mathutils``, which exist only inside Blender, so the module is
imported here with those two stubbed. That is enough to reach everything in it that is arithmetic
rather than scene building -- above all :func:`channel_basis`, which is the one place this package
has to know how Blender composes a posed bone.

The test for it does not trust the algebra: it implements Blender's composition rule
(``M_j = M_parent @ (L_parent^-1 @ L_j) @ B_j``, deformation ``M_j @ L_j^-1``) over *random* bone rest
matrices, and checks that the deformation comes out as the plan's transform whatever those rest
matrices are. That is exactly the claim the scene rests on: how the bones are drawn cannot change
the result. What remains unproven without Blender -- that Blender's own rule is this one -- is what
the scene's self-check against the plan's sampled frames catches at run time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smpl18.blender import launch
from smpl18.blender.plan import build_plan
from smpl18.skeleton.definition import NUM_JOINTS, PARENTS


@pytest.fixture(scope="module")
def scene():
    """``smpl18.blender.scene`` with ``bpy`` and ``mathutils`` stubbed out."""
    path = launch.scene_script()
    specification = importlib.util.spec_from_file_location("smpl18_scene_under_test", path)
    module = importlib.util.module_from_spec(specification)
    stubs = {"bpy": mock.MagicMock(name="bpy"), "mathutils": mock.MagicMock(name="mathutils")}
    with mock.patch.dict(sys.modules, stubs):
        specification.loader.exec_module(module)
    return module


def random_rest_matrices(seed: int) -> list[np.ndarray]:
    """One invertible rest matrix per bone, as unlike an identity as Blender's ever are."""
    generator = np.random.default_rng(seed)
    matrices = []
    for _ in range(NUM_JOINTS):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.random(random_state=generator.integers(1 << 30)).as_matrix()
        matrix[:3, 3] = generator.normal(size=3)
        matrices.append(matrix)
    return matrices


def blender_deformation(basis: list[np.ndarray], rest_local: list[np.ndarray]) -> list[np.ndarray]:
    """What Blender would deform each vertex group by, given those channel transforms."""
    posed = [None] * NUM_JOINTS
    for joint in range(NUM_JOINTS):
        parent = PARENTS[joint]
        if parent < 0:
            posed[joint] = rest_local[joint] @ basis[joint]
        else:
            offset = np.linalg.inv(rest_local[parent]) @ rest_local[joint]
            posed[joint] = posed[parent] @ offset @ basis[joint]
    return [posed[joint] @ np.linalg.inv(rest_local[joint]) for joint in range(NUM_JOINTS)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_channel_transforms_make_blender_carry_the_plan(scene, body, walk, seed) -> None:
    local, trans = walk
    plan = build_plan(body, betas=np.zeros(10), local=local, trans=trans, fps=50.0, up_axis="y",
                      correctives=False, sample_frames=0, tolerance=0.0, leaf_reach=0.7)
    rest_local = random_rest_matrices(seed)
    inverses = [np.linalg.inv(matrix) for matrix in rest_local]
    for frame in (0, plan.frames // 2, plan.frames - 1):
        wanted = plan.transforms[frame]
        basis = [
            scene.channel_basis(rest_local[joint], inverses[joint], wanted[joint],
                                None if PARENTS[joint] < 0 else wanted[PARENTS[joint]])
            for joint in range(NUM_JOINTS)
        ]
        for joint, deformation in enumerate(blender_deformation(basis, rest_local)):
            np.testing.assert_allclose(deformation, wanted[joint], atol=1e-10)


def test_the_root_is_the_only_bone_with_no_parent_above_it(scene) -> None:
    rest = np.eye(4)
    rest[:3, 3] = [0.1, 0.2, 0.3]
    wanted = np.eye(4)
    wanted[:3, :3] = Rotation.from_rotvec([0.2, 0.0, 0.0]).as_matrix()
    alone = scene.channel_basis(rest, np.linalg.inv(rest), wanted, None)
    np.testing.assert_allclose(alone, np.linalg.inv(rest) @ wanted @ rest, atol=1e-12)
    # A bone already carrying its parent's transform needs no channel of its own.
    np.testing.assert_allclose(scene.channel_basis(rest, np.linalg.inv(rest), wanted, wanted),
                               np.eye(4), atol=1e-12)


def test_joint_positions_apply_the_transforms_to_the_rest_joints(scene, body, walk) -> None:
    local, trans = walk
    plan = build_plan(body, betas=np.zeros(10), local=local, trans=trans, fps=50.0, up_axis="y",
                      correctives=False, sample_frames=0, tolerance=0.0, leaf_reach=0.7)
    from smpl18.skeleton.kinematics import fk_batch

    expected, _ = fk_batch(plan.rest, local, trans)
    np.testing.assert_allclose(scene.joint_positions(plan.transforms, plan.rest), expected,
                               atol=1e-12)


def test_it_reads_its_arguments_after_blenders_own(scene) -> None:
    assert scene.script_arguments(["blender", "--background", "--python", "scene.py"]) == []
    argv = ["blender", "-b", "--python", "scene.py", "--", "--plan", "p.npz", "--settings", "s.json"]
    assert scene.script_arguments(argv) == ["--plan", "p.npz", "--settings", "s.json"]
    arguments = scene.parse_arguments(["--plan", "p.npz", "--settings", "s.json", "--render", "out"])
    assert arguments.plan == Path("p.npz") and arguments.render == Path("out")
    assert arguments.blend is None and not arguments.quiet
    with pytest.raises(SystemExit):
        scene.parse_arguments(["--settings", "s.json"])          # --plan is required


def test_it_refuses_a_file_that_is_not_a_plan(scene, tmp_path, body, walk) -> None:
    local, trans = walk
    good = build_plan(body, betas=np.zeros(10), local=local, trans=trans, fps=50.0, up_axis="y",
                      correctives=False, sample_frames=1, tolerance=1e-4,
                      leaf_reach=0.7).write(tmp_path / "good.plan.npz")
    loaded = scene.load_plan(good)
    assert loaded["transforms"].shape == (local.shape[0], NUM_JOINTS, 4, 4)
    assert float(loaded["tolerance"]) == 1e-4

    other = tmp_path / "other.npz"
    np.savez(other, poses=local)
    with pytest.raises(SystemExit, match="schema"):
        scene.load_plan(other)


def test_a_missing_render_setting_is_named(scene) -> None:
    with pytest.raises(SystemExit, match="render.samples"):
        scene.setting({"render": {"engine": "cycles"}}, "render", "samples")
    with pytest.raises(SystemExit, match="scene.floor"):
        scene.setting({}, "scene", "floor")
    assert scene.setting({"render": {"samples": 8}}, "render", "samples") == 8


def test_the_script_names_the_schema_the_plan_writes(scene) -> None:
    from smpl18.blender.plan import PLAN_SCHEMA

    assert scene.PLAN_SCHEMA == PLAN_SCHEMA


def test_the_script_imports_nothing_from_this_package() -> None:
    """Blender's Python has no smpl18 installed, so the script may not reach for it."""
    import ast

    tree = ast.parse(launch.scene_script().read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert {"bpy", "numpy", "mathutils"} <= imported
    assert not [name for name in imported if name.split(".")[0] == "smpl18"]
