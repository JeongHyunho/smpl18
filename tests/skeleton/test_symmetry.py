"""The mirror operator S, and rest joints on the three symmetry modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from smpl18.model.demo import demo_model
from smpl18.model.load import Model
from smpl18.skeleton.definition import JOINT_NAMES, NUM_JOINTS, PARENTS
from smpl18.skeleton.kinematics import rest_joints, shaped_vertices
from smpl18.skeleton.symmetry import (
    DEFAULT_SYMMETRY,
    MIRROR_PARTNER,
    SYMMETRY_MODES,
    antisymmetric_part,
    mirror,
    mirror_pairs,
    symmetric_part,
    symmetry_setting,
)

PACKAGE = Path(__file__).resolve().parents[2]
BONES = [(PARENTS[joint], joint) for joint in range(1, NUM_JOINTS)]


def lopsided_surface_model(symmetry: str) -> Model:
    """The stand-in with its blocky surface, whose regressor averages vertices as SMPL's does,
    and a template and shape directions pushed off their mirror images by a few millimetres."""
    base = demo_model(with_mesh=True)
    rng = np.random.default_rng(11)
    return Model(
        v_template=base.v_template + rng.normal(0.0, 0.006, base.v_template.shape),
        shapedirs=base.shapedirs + rng.normal(0.0, 0.003, base.shapedirs.shape),
        J_regressor=base.J_regressor,
        kintree_parents=base.kintree_parents,
        symmetry=symmetry,
    )


def side_differences(rest: np.ndarray) -> np.ndarray:
    return np.array([
        np.linalg.norm(rest[a[1]] - rest[a[0]]) - np.linalg.norm(rest[b[1]] - rest[b[0]])
        for a, b in mirror_pairs(BONES)
    ])


def test_the_partners_are_the_left_right_joints() -> None:
    swaps = {(a, b) for a, b in enumerate(MIRROR_PARTNER) if a < b}
    assert swaps == {(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
                     (22, 23)}
    for joint, partner in enumerate(MIRROR_PARTNER):
        assert MIRROR_PARTNER[partner] == joint
        if partner == joint:
            assert not JOINT_NAMES[joint].startswith(("left_", "right_"))


def test_s_is_idempotent_and_keeps_what_is_already_symmetric() -> None:
    rng = np.random.default_rng(0)
    joints = rng.normal(0.0, 0.3, (NUM_JOINTS, 3))
    once = symmetric_part(joints)
    np.testing.assert_array_equal(symmetric_part(once), once)
    np.testing.assert_array_equal(mirror(once), once)
    np.testing.assert_array_equal(antisymmetric_part(once), np.zeros_like(once))
    np.testing.assert_allclose(once + antisymmetric_part(joints), joints, atol=1e-15)
    midline = [joint for joint, partner in enumerate(MIRROR_PARTNER) if joint == partner]
    np.testing.assert_array_equal(once[midline, 0], 0.0)
    # The stand-in is built mirror-symmetric, so S leaves it exactly as it is.
    plain = demo_model()
    np.testing.assert_array_equal(symmetric_part(plain.v_template), plain.v_template)
    np.testing.assert_array_equal(symmetric_part(plain.shapedirs), plain.shapedirs)


def test_shape_directions_are_mirrored_one_by_one() -> None:
    directions = np.random.default_rng(1).normal(0.0, 0.01, (NUM_JOINTS, 3, 4))
    together = symmetric_part(directions)
    for column in range(4):
        np.testing.assert_array_equal(together[:, :, column],
                                      symmetric_part(directions[:, :, column]))
    with pytest.raises(ValueError, match="24, 3"):
        mirror(np.zeros((23, 3)))


def test_the_skeleton_mode_gives_equal_sides_for_any_betas() -> None:
    model = lopsided_surface_model("skeleton")
    raw = model.with_symmetry("none")
    assert np.abs(side_differences(rest_joints(raw, np.zeros(10)))).max() > 1e-3
    for betas in np.random.default_rng(3).normal(0.0, 2.0, (10, 10)):
        rest = rest_joints(model, betas)
        assert np.abs(side_differences(rest)).max() < 1e-9
        np.testing.assert_allclose(rest, symmetric_part(rest_joints(raw, betas)), atol=1e-15)


def test_the_template_mode_mirrors_the_template_only() -> None:
    model = lopsided_surface_model("template")
    raw = model.with_symmetry("none")
    betas = np.random.default_rng(5).normal(0.0, 1.0, 10)
    template = raw.J_regressor @ raw.v_template
    np.testing.assert_allclose(rest_joints(model, np.zeros(10)), symmetric_part(template),
                               atol=1e-15)
    np.testing.assert_allclose(rest_joints(model, betas) - rest_joints(model, np.zeros(10)),
                               rest_joints(raw, betas) - template, atol=1e-15)


def test_none_is_smpls_own_skeleton_bit_for_bit() -> None:
    model = lopsided_surface_model("none")
    betas = np.random.default_rng(6).normal(0.0, 1.0, 10)
    np.testing.assert_array_equal(rest_joints(model, betas),
                                  model.J_regressor @ shaped_vertices(model, betas))


def test_the_surface_is_never_mirrored() -> None:
    betas = np.random.default_rng(8).normal(0.0, 1.0, 10)
    surfaces = [shaped_vertices(lopsided_surface_model(mode), betas) for mode in SYMMETRY_MODES]
    for surface in surfaces[1:]:
        np.testing.assert_array_equal(surface, surfaces[0])


def test_mirror_pairs_are_the_couples_of_the_two_sides() -> None:
    couples = mirror_pairs([(1, 2), (1, 3), (2, 3), (1, 4), (2, 5), (7, 10), (0, 3)])
    assert couples == [((1, 3), (2, 3)), ((1, 4), (2, 5))]      # hips, spine, one-sided: no


def test_the_default_is_the_symmetric_skeleton() -> None:
    assert DEFAULT_SYMMETRY == "skeleton"
    assert demo_model().symmetry == "skeleton"
    settings = yaml.safe_load((PACKAGE / "configs/settings/default.yaml").read_text("utf-8"))
    assert symmetry_setting(settings) == "skeleton"
    assert settings["shape"]["lr_equality_weight"] == 100.0


def test_the_setting_is_required_and_checked() -> None:
    with pytest.raises(ValueError, match="model.symmetry"):
        symmetry_setting({"shape": {}})
    with pytest.raises(ValueError, match="none, template, skeleton"):
        symmetry_setting({"model": {"symmetry": "mirror"}})
    with pytest.raises(ValueError, match="symmetry must be"):
        demo_model().with_symmetry("left")
