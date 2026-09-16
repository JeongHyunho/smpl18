"""OpenSim forward kinematics, checked against arithmetic done by hand.

Every convention here is a place a plausible wrong answer hides: the order three rotations
compose in, whether a translation rides those rotations, whether an offset frame's orientation
is body fixed, how a spline behaves at and beyond its ends, and which coordinates a motion file
stores in degrees. The fixtures are small enough to work out on paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from smpl18.formats import mot, osim
from smpl18.formats.osim import OsimFunction
from smpl18.sources import SegmentPlacement, SkeletonModel
from smpl18.sources.skeleton import (
    OpenSimSkeleton,
    UnsupportedModelError,
    coordinates_from_mot,
    opensim,
)

ATOL = 1e-12


# --- building models -----------------------------------------------------------------------------


def document(joints: str, bodies=("pelvis", "thigh", "shank"), constraints: str = "") -> str:
    body_set = "".join(f'<Body name="{b}"/>' for b in bodies)
    constraint_set = (
        f"<ConstraintSet><objects>{constraints}</objects></ConstraintSet>" if constraints else ""
    )
    return (
        '<?xml version="1.0"?><OpenSimDocument Version="40500"><Model name="m">'
        "<gravity>0 -9.80665 0</gravity>"
        f"<BodySet><objects>{body_set}</objects></BodySet>"
        f"<JointSet><objects>{joints}</objects></JointSet>"
        f"{constraint_set}</Model></OpenSimDocument>"
    )


def offset(name: str, body: str, translation="0 0 0", orientation="0 0 0") -> str:
    socket = "/ground" if body == "ground" else f"/bodyset/{body}"
    return (
        f'<PhysicalOffsetFrame name="{name}"><socket_parent>{socket}</socket_parent>'
        f"<translation>{translation}</translation><orientation>{orientation}</orientation>"
        "</PhysicalOffsetFrame>"
    )


def linear(slope=1.0, intercept=0.0) -> str:
    return f"<LinearFunction><coefficients>{slope} {intercept}</coefficients></LinearFunction>"


def spline(x: str, y: str, tag: str = "SimmSpline") -> str:
    return f"<{tag}><x>{x}</x><y>{y}</y></{tag}>"


def axis(name: str, coordinate: str, direction: str, function: str) -> str:
    return (
        f'<TransformAxis name="{name}"><coordinates>{coordinate}</coordinates>'
        f"<axis>{direction}</axis>{function}</TransformAxis>"
    )


def coordinates(*names: str, defaults: dict | None = None) -> str:
    defaults = defaults or {}
    return "<coordinates>" + "".join(
        f'<Coordinate name="{n}">'
        + (f"<default_value>{defaults[n]}</default_value>" if n in defaults else "")
        + "</Coordinate>"
        for n in names
    ) + "</coordinates>"


def custom(name, parent, child, coords, axes, frames="", defaults=None) -> str:
    return (
        f'<CustomJoint name="{name}">'
        f"<socket_parent_frame>{parent}</socket_parent_frame>"
        f"<socket_child_frame>{child}</socket_child_frame>"
        f"{coordinates(*coords, defaults=defaults)}"
        f"<SpatialTransform>{''.join(axes)}</SpatialTransform>"
        f"<frames>{frames}</frames></CustomJoint>"
    )


def fixed(kind, name, parent, child, coords=(), frames="") -> str:
    return (
        f'<{kind} name="{name}">'
        f"<socket_parent_frame>{parent}</socket_parent_frame>"
        f"<socket_child_frame>{child}</socket_child_frame>"
        f"{coordinates(*coords)}<frames>{frames}</frames></{kind}>"
    )


FREE_PELVIS = custom(
    "ground_pelvis", "ground_offset", "pelvis_offset",
    ("tilt", "list", "rotation", "tx", "ty", "tz"),
    [
        axis("rotation1", "tilt", "0 0 1", linear()),
        axis("rotation2", "list", "1 0 0", linear()),
        axis("rotation3", "rotation", "0 1 0", linear()),
        axis("translation1", "tx", "1 0 0", linear()),
        axis("translation2", "ty", "0 1 0", linear()),
        axis("translation3", "tz", "0 0 1", linear()),
    ],
    frames=offset("ground_offset", "ground") + offset("pelvis_offset", "pelvis"),
)

HIP_OFFSET = np.array([0.1, -0.05, 0.07])
PIN_HIP = fixed(
    "PinJoint", "hip", "pelvis_offset", "thigh_offset", ("hip_flexion",),
    frames=offset("pelvis_offset", "pelvis", translation="0.1 -0.05 0.07")
    + offset("thigh_offset", "thigh"),
)

KNOTS_X = "-2 -1 0 1 2"
KNEE = custom(
    "knee", "thigh_offset", "shank_offset", ("knee_angle",),
    [
        axis("rotation1", "knee_angle", "0 0 1", linear()),
        axis("rotation2", "knee_angle", "1 0 0", spline(KNOTS_X, "0.02 0.01 0 0.01 0.03")),
        axis("translation1", "knee_angle", "1 0 0",
             "<MultiplierFunction><function>" + spline(KNOTS_X, "0.004 0.002 0 -0.001 -0.003")
             + "</function><scale>1.5</scale></MultiplierFunction>"),
        axis("translation2", "knee_angle", "0 1 0", spline(KNOTS_X, "-0.39 -0.395 -0.4 -0.402 -0.41")),
    ],
    frames=offset("thigh_offset", "thigh", translation="0 -0.02 0", orientation="0.05 0 0")
    + offset("shank_offset", "shank", translation="0 0.01 0", orientation="0 0 0.1"),
)


def model_of(*joints: str, **kwargs) -> OpenSimSkeleton:
    return OpenSimSkeleton(osim.parse(document("".join(joints), **kwargs)))


@pytest.fixture(scope="module")
def leg() -> OpenSimSkeleton:
    return model_of(FREE_PELVIS, PIN_HIP, KNEE)


def one(value: float) -> np.ndarray:
    return np.array([value])


def rot(vector) -> np.ndarray:
    return Rotation.from_rotvec(vector).as_matrix()


# --- the model's shape ---------------------------------------------------------------------------


def test_it_is_a_skeleton_model_with_the_model_tree(leg):
    assert isinstance(leg, SkeletonModel)
    assert leg.bodies == ("pelvis", "thigh", "shank")
    assert leg.joints == ("ground_pelvis", "hip", "knee")
    assert leg.parents == {"pelvis": None, "thigh": "pelvis", "shank": "thigh"}
    assert leg.joint_child_bodies == {"ground_pelvis": "pelvis", "hip": "thigh", "knee": "shank"}
    assert leg.coordinate_names == (
        "tilt", "list", "rotation", "tx", "ty", "tz", "hip_flexion", "knee_angle",
    )
    assert leg.gravity == pytest.approx((0.0, -9.80665, 0.0))


def test_motion_places_ground_and_every_body_and_joint(leg):
    placed = leg.motion({"tilt": np.zeros(4)})
    assert isinstance(placed, SegmentPlacement)
    assert list(placed.rotations) == ["ground", "pelvis", "thigh", "shank"]
    assert list(placed.joint_centres) == ["ground_pelvis", "hip", "knee"]
    assert placed.rotations["shank"].shape == (4, 3, 3)
    assert placed.positions["shank"].shape == (4, 3)
    np.testing.assert_array_equal(placed.rotations["ground"], np.tile(np.eye(3), (4, 1, 1)))
    np.testing.assert_array_equal(placed.positions["ground"], np.zeros((4, 3)))


def test_read_builds_the_skeleton_from_a_file(tmp_path: Path):
    path = tmp_path / "leg.osim"
    path.write_text(document(FREE_PELVIS + PIN_HIP + KNEE), encoding="utf-8")
    skeleton = opensim.read(path)
    assert skeleton.joints == ("ground_pelvis", "hip", "knee")
    assert skeleton.model == osim.read(path)


# --- custom joints -------------------------------------------------------------------------------


def test_the_three_rotations_compose_in_the_order_the_model_lists_them(leg):
    placed = leg.motion({"tilt": one(0.3), "list": one(-0.2), "rotation": one(0.5)})
    expected = rot([0, 0, 0.3]) @ rot([-0.2, 0, 0]) @ rot([0, 0.5, 0])
    np.testing.assert_allclose(placed.rotations["pelvis"][0], expected, atol=ATOL)
    reversed_order = rot([0, 0.5, 0]) @ rot([-0.2, 0, 0]) @ rot([0, 0, 0.3])
    assert not np.allclose(placed.rotations["pelvis"][0], reversed_order, atol=1e-3)


def test_the_root_translation_lands_where_the_coordinates_say(leg):
    placed = leg.motion({"tx": one(1.0), "ty": one(2.0), "tz": one(3.0), "tilt": one(0.7)})
    np.testing.assert_allclose(placed.joint_centres["ground_pelvis"][0], [1, 2, 3], atol=ATOL)
    np.testing.assert_allclose(placed.positions["pelvis"][0], [1, 2, 3], atol=ATOL)


def test_a_translation_axis_slides_along_the_parent_frame_not_the_turned_one():
    """A joint that turns and slides on one coordinate: a rotated slide would leave the X axis."""
    slider = custom(
        "slide", "/ground", "/bodyset/pelvis", ("q",),
        [axis("rotation1", "q", "0 0 1", linear()), axis("translation1", "q", "1 0 0", linear())],
    )
    placed = model_of(slider, bodies=("pelvis",)).motion({"q": one(0.5)})
    np.testing.assert_allclose(placed.joint_centres["slide"][0], [0.5, 0.0, 0.0], atol=ATOL)
    np.testing.assert_allclose(placed.rotations["pelvis"][0], rot([0, 0, 0.5]), atol=ATOL)


def test_a_child_joint_centre_rides_on_its_parents_rotation(leg):
    placed = leg.motion({"tilt": one(0.4), "tx": one(1.0)})
    np.testing.assert_allclose(
        placed.joint_centres["hip"][0], [1.0, 0.0, 0.0] + rot([0, 0, 0.4]) @ HIP_OFFSET,
        atol=ATOL,
    )


def test_axes_are_normalised_the_way_opensim_normalises_them():
    def placed(rotation_axis, slide_axis):
        joint = custom(
            "j", "/ground", "/bodyset/pelvis", ("a", "s"),
            [axis("rotation1", "a", rotation_axis, linear()),
             axis("translation1", "s", slide_axis, linear())],
        )
        return model_of(joint, bodies=("pelvis",)).motion({"a": one(0.3), "s": one(0.5)})

    unit, scaled = placed("0 0 1", "1 0 0"), placed("0 0 2", "3 0 0")
    np.testing.assert_allclose(scaled.rotations["pelvis"], unit.rotations["pelvis"], atol=ATOL)
    np.testing.assert_allclose(scaled.positions["pelvis"], [[0.5, 0.0, 0.0]], atol=ATOL)


def test_a_constant_axis_with_no_coordinate_is_a_fixed_offset():
    joint = custom(
        "j", "/ground", "/bodyset/pelvis", ("a",),
        [axis("rotation1", "a", "0 0 1", linear()),
         axis("rotation2", "", "1 0 0", "<Constant><value>0.25</value></Constant>"),
         axis("translation2", "", "0 1 0",
              "<MultiplierFunction><function><Constant><value>0.2</value></Constant>"
              "</function><scale>2</scale></MultiplierFunction>"),
         axis("translation3", "", "0 0 1", "")],
    )
    placed = model_of(joint, bodies=("pelvis",)).motion({"a": one(0.1)})
    np.testing.assert_allclose(
        placed.rotations["pelvis"][0], rot([0, 0, 0.1]) @ rot([0.25, 0, 0]), atol=ATOL
    )
    np.testing.assert_allclose(placed.positions["pelvis"][0], [0.0, 0.4, 0.0], atol=ATOL)


def test_a_spline_driven_joint_is_evaluated_through_its_splines(leg):
    angle = 0.5
    rolled = opensim.compile_function(
        OsimFunction("SimmSpline", {"x": (-2, -1, 0, 1, 2), "y": (0.02, 0.01, 0, 0.01, 0.03)}), "t"
    )(np.array(angle))
    slide_x = 1.5 * opensim.compile_function(
        OsimFunction("SimmSpline", {"x": (-2, -1, 0, 1, 2), "y": (0.004, 0.002, 0, -0.001, -0.003)}),
        "t",
    )(np.array(angle))
    slide_y = opensim.compile_function(
        OsimFunction("SimmSpline", {"x": (-2, -1, 0, 1, 2), "y": (-0.39, -0.395, -0.4, -0.402, -0.41)}),
        "t",
    )(np.array(angle))
    placed = leg.motion({"knee_angle": one(angle)})

    thigh_rotation = Rotation.from_euler("XYZ", [0.05, 0, 0]).as_matrix()
    frame_rotation = thigh_rotation @ rot([0, 0, angle]) @ rot([rolled, 0, 0])
    centre = HIP_OFFSET + np.array([0, -0.02, 0]) + thigh_rotation @ [slide_x, slide_y, 0]
    shank_rotation = frame_rotation @ Rotation.from_euler("XYZ", [0, 0, 0.1]).as_matrix().T
    np.testing.assert_allclose(placed.joint_centres["knee"][0], centre, atol=ATOL)
    np.testing.assert_allclose(placed.rotations["shank"][0], shank_rotation, atol=ATOL)
    np.testing.assert_allclose(
        placed.positions["shank"][0], centre - shank_rotation @ [0, 0.01, 0], atol=ATOL
    )


# --- offset frames -------------------------------------------------------------------------------


def test_an_offset_frame_orientation_is_body_fixed_xyz():
    orientation = [0.3, -0.2, 0.5]
    pin = fixed(
        "PinJoint", "hip", "pelvis_offset", "thigh_offset", ("hip_flexion",),
        frames=offset("pelvis_offset", "pelvis", orientation="0.3 -0.2 0.5")
        + offset("thigh_offset", "thigh"),
    )
    placed = model_of(FREE_PELVIS, pin, bodies=("pelvis", "thigh")).motion({"tilt": one(0.0)})
    intrinsic = Rotation.from_euler("XYZ", orientation).as_matrix()
    extrinsic = Rotation.from_euler("xyz", orientation).as_matrix()
    np.testing.assert_allclose(placed.rotations["thigh"][0], intrinsic, atol=ATOL)
    assert not np.allclose(placed.rotations["thigh"][0], extrinsic, atol=1e-3)


def test_the_child_frame_is_undone_on_the_child_body():
    """child body = joint frame composed with the inverse of the child offset frame."""
    child_rotation = Rotation.from_euler("XYZ", [0.2, 0.1, -0.3]).as_matrix()
    child_translation = np.array([0.0, 0.2, 0.01])
    pin = fixed(
        "PinJoint", "hip", "pelvis_offset", "thigh_offset", ("hip_flexion",),
        frames=offset("pelvis_offset", "pelvis", translation="0.1 -0.05 0.07")
        + offset("thigh_offset", "thigh", translation="0 0.2 0.01", orientation="0.2 0.1 -0.3"),
    )
    skeleton = model_of(FREE_PELVIS, pin, bodies=("pelvis", "thigh"))
    placed = skeleton.motion({"hip_flexion": one(0.6), "tx": one(1.0)})
    frame_rotation = rot([0, 0, 0.6])
    thigh = frame_rotation @ child_rotation.T
    centre = np.array([1.0, 0.0, 0.0]) + HIP_OFFSET
    np.testing.assert_allclose(placed.joint_centres["hip"][0], centre, atol=ATOL)
    np.testing.assert_allclose(placed.rotations["thigh"][0], thigh, atol=ATOL)
    np.testing.assert_allclose(
        placed.positions["thigh"][0], centre - thigh @ child_translation, atol=ATOL
    )


# --- the other joint types -----------------------------------------------------------------------


def _single(kind: str, coords: tuple[str, ...], values: dict) -> SegmentPlacement:
    joint = fixed(kind, "j", "/ground", "/bodyset/pelvis", coords)
    return model_of(joint, bodies=("pelvis",)).motion({k: one(v) for k, v in values.items()})


def test_a_pin_joint_turns_about_z_and_moves_nothing(leg):
    still = leg.motion({"hip_flexion": one(0.0)})
    turned = leg.motion({"hip_flexion": one(0.7)})
    np.testing.assert_allclose(still.joint_centres["hip"], turned.joint_centres["hip"], atol=ATOL)
    np.testing.assert_allclose(turned.rotations["thigh"][0], rot([0, 0, 0.7]), atol=ATOL)


def test_a_slider_joint_slides_along_x():
    placed = _single("SliderJoint", ("s",), {"s": 0.4})
    np.testing.assert_allclose(placed.positions["pelvis"][0], [0.4, 0, 0], atol=ATOL)
    np.testing.assert_allclose(placed.rotations["pelvis"][0], np.eye(3), atol=ATOL)


def test_a_universal_joint_turns_about_x_then_y():
    placed = _single("UniversalJoint", ("a", "b"), {"a": 0.3, "b": -0.6})
    np.testing.assert_allclose(
        placed.rotations["pelvis"][0], rot([0.3, 0, 0]) @ rot([0, -0.6, 0]), atol=ATOL
    )


@pytest.mark.parametrize("kind", ["GimbalJoint", "BallJoint"])
def test_gimbal_and_ball_joints_turn_body_fixed_xyz(kind):
    angles = [0.3, -0.6, 1.1]
    placed = _single(kind, ("a", "b", "c"), dict(zip("abc", angles)))
    np.testing.assert_allclose(
        placed.rotations["pelvis"][0], Rotation.from_euler("XYZ", angles).as_matrix(), atol=ATOL
    )
    np.testing.assert_allclose(placed.positions["pelvis"][0], np.zeros(3), atol=ATOL)


def test_a_free_joint_turns_body_fixed_xyz_and_slides_in_the_parent_frame():
    values = dict(zip(("rx", "ry", "rz", "px", "py", "pz"), (0.3, -0.6, 1.1, 1.0, 2.0, 3.0)))
    placed = _single("FreeJoint", tuple(values), values)
    np.testing.assert_allclose(
        placed.rotations["pelvis"][0],
        Rotation.from_euler("XYZ", [0.3, -0.6, 1.1]).as_matrix(), atol=ATOL,
    )
    np.testing.assert_allclose(placed.positions["pelvis"][0], [1.0, 2.0, 3.0], atol=ATOL)


def test_a_planar_joint_turns_about_z_and_slides_along_x_and_y():
    placed = _single("PlanarJoint", ("rz", "px", "py"), {"rz": 0.8, "px": 0.5, "py": -0.25})
    np.testing.assert_allclose(placed.rotations["pelvis"][0], rot([0, 0, 0.8]), atol=ATOL)
    np.testing.assert_allclose(placed.positions["pelvis"][0], [0.5, -0.25, 0], atol=ATOL)


def test_a_weld_joint_holds_the_child_at_its_frames():
    weld = fixed(
        "WeldJoint", "weld", "pelvis_offset", "/bodyset/thigh",
        frames=offset("pelvis_offset", "pelvis", translation="0 -0.1 0", orientation="0 0 0.2"),
    )
    placed = model_of(FREE_PELVIS, weld, bodies=("pelvis", "thigh")).motion({"tx": one(1.0)})
    np.testing.assert_allclose(placed.positions["thigh"][0], [1.0, -0.1, 0.0], atol=ATOL)
    np.testing.assert_allclose(placed.rotations["thigh"][0], rot([0, 0, 0.2]), atol=ATOL)


def test_an_unsupported_joint_type_is_refused_by_name():
    joint = fixed("EllipsoidJoint", "odd", "/ground", "/bodyset/pelvis", ("a", "b", "c"))
    with pytest.raises(UnsupportedModelError, match="'odd' is a EllipsoidJoint"):
        model_of(joint, bodies=("pelvis",))


def test_a_fixed_joint_with_the_wrong_number_of_coordinates_is_refused():
    joint = fixed("UniversalJoint", "u", "/ground", "/bodyset/pelvis", ("a",))
    with pytest.raises(ValueError, match="takes 2 coordinates; it declares 1"):
        model_of(joint, bodies=("pelvis",))


# --- functions -----------------------------------------------------------------------------------


def evaluate(kind: str, values: dict, at) -> np.ndarray:
    return opensim.compile_function(OsimFunction(kind, values), "test")(np.asarray(at, float))


CUBIC_X = (0.0, 1.0, 2.0, 3.0, 4.0)


def cubic(x):
    return np.asarray(x) ** 3 - 2.0 * np.asarray(x)


def test_a_simm_spline_is_the_fmm_spline_and_reproduces_a_cubic_exactly():
    """FMM end conditions are exact for a cubic; a natural spline's are not."""
    values = {"x": CUBIC_X, "y": tuple(cubic(CUBIC_X))}
    inside = np.array([0.0, 0.5, 1.25, 2.0, 3.7, 4.0])
    np.testing.assert_allclose(evaluate("SimmSpline", values, inside), cubic(inside), atol=1e-12)
    natural = CubicSpline(CUBIC_X, cubic(CUBIC_X), bc_type="natural")(0.5)
    assert abs(natural - cubic(0.5)) > 1e-2


def test_a_simm_spline_continues_straight_along_its_end_slopes():
    values = {"x": CUBIC_X, "y": tuple(cubic(CUBIC_X))}
    # slopes of x^3 - 2x at the ends: -2 at 0 and 46 at 4
    np.testing.assert_allclose(
        evaluate("SimmSpline", values, [-1.0, 5.0, 6.0]), [2.0, 56.0 + 46.0, 56.0 + 92.0],
        atol=1e-9,
    )


def test_short_simm_splines_are_a_parabola_and_a_line():
    parabola = {"x": (0.0, 1.0, 2.0), "y": (0.0, 1.0, 4.0)}
    np.testing.assert_allclose(
        evaluate("SimmSpline", parabola, [0.5, 1.5, 3.0, -1.0]), [0.25, 2.25, 8.0, 0.0],
        atol=1e-12,
    )
    line = {"x": (1.0, 3.0), "y": (2.0, 6.0)}
    np.testing.assert_allclose(evaluate("SimmSpline", line, [0.0, 2.0, 4.0]), [0.0, 4.0, 8.0])


def test_the_older_spline_names_are_read_as_simm_splines():
    values = {"x": (0.0, 0.5, 1.5, 2.0, 3.5), "y": (0.0, 0.3, -0.2, 0.4, 0.1)}
    at = np.linspace(-1.0, 4.5, 23)
    expected = evaluate("SimmSpline", values, at)
    for kind in ("NaturalCubicSpline", "natCubicSpline"):
        np.testing.assert_allclose(evaluate(kind, values, at), expected, atol=0.0)


def test_a_spline_answers_with_its_own_knots():
    values = {"x": (0.0, 0.5, 1.5, 2.0, 3.5), "y": (0.0, 0.3, -0.2, 0.4, 0.1)}
    np.testing.assert_allclose(
        evaluate("SimmSpline", values, values["x"]), values["y"], atol=1e-15
    )


def test_a_cubic_gcv_spline_without_smoothing_is_a_natural_spline_straight_beyond_its_ends():
    x = (0.0, 1.0, 2.0, 3.0, 4.0)
    y = (0.0, 0.8, 0.1, -0.5, 0.3)
    values = {"x": x, "y": y, "half_order": 2}
    natural = CubicSpline(x, y, bc_type="natural")
    inside = np.linspace(0.0, 4.0, 17)
    np.testing.assert_allclose(evaluate("GCVSpline", values, inside), natural(inside), atol=1e-12)
    np.testing.assert_allclose(
        evaluate("GCVSpline", values, [-1.0, 5.0]),
        [y[0] - natural(0.0, 1), y[-1] + natural(4.0, 1)],
        atol=1e-12,
    )


def test_gcv_splines_that_are_not_cubic_or_that_smooth_are_refused():
    base = {"x": (0.0, 1.0, 2.0, 3.0), "y": (0.0, 1.0, 0.0, 1.0)}
    with pytest.raises(UnsupportedModelError, match="half_order 3"):
        evaluate("GCVSpline", {**base, "half_order": 3}, 0.5)
    with pytest.raises(UnsupportedModelError, match="half_order None"):
        evaluate("GCVSpline", base, 0.5)
    with pytest.raises(UnsupportedModelError, match="error_variance 0.01"):
        evaluate("GCVSpline", {**base, "half_order": 2, "error_variance": 0.01}, 0.5)


def test_a_piecewise_linear_function_interpolates_and_extends_its_end_segments():
    values = {"x": (0.0, 1.0, 3.0), "y": (0.0, 2.0, 3.0)}
    np.testing.assert_allclose(
        evaluate("PiecewiseLinearFunction", values, [-1.0, 0.5, 2.0, 5.0]),
        [-2.0, 1.0, 2.5, 4.0],
    )
    single = {"x": (1.0,), "y": (7.0,)}
    np.testing.assert_allclose(evaluate("PiecewiseLinearFunction", single, [-3.0, 9.0]), [7, 7])


def test_linear_polynomial_constant_and_multiplier_functions():
    np.testing.assert_allclose(evaluate("LinearFunction", {"coefficients": (2.0, 0.5)}, 3.0), 6.5)
    # highest order first: 2 x^2 + 1
    np.testing.assert_allclose(
        evaluate("PolynomialFunction", {"coefficients": (2.0, 0.0, 1.0)}, [0.0, 2.0]), [1.0, 9.0]
    )
    np.testing.assert_allclose(evaluate("Constant", {"value": 0.25}, [1.0, 99.0]), [0.25, 0.25])
    inner = OsimFunction("LinearFunction", {"coefficients": (1.0, 0.0)})
    np.testing.assert_allclose(
        evaluate("MultiplierFunction", {"function": inner, "scale": 2.5}, 4.0), 10.0
    )


def test_omitted_function_properties_take_opensims_defaults():
    np.testing.assert_allclose(evaluate("LinearFunction", {}, [3.0]), [3.0])
    np.testing.assert_allclose(evaluate("Constant", {}, [3.0]), [0.0])
    np.testing.assert_allclose(evaluate("PolynomialFunction", {}, [3.0]), [1.0])
    inner = OsimFunction("Constant", {"value": 2.0})
    np.testing.assert_allclose(evaluate("MultiplierFunction", {"function": inner}, [3.0]), [2.0])


@pytest.mark.parametrize(
    ("kind", "values", "message"),
    [
        ("SimmSpline", {"x": (0.0,), "y": (1.0,)}, "at least 2 knots"),
        ("SimmSpline", {"x": (0.0, 1.0), "y": (1.0,)}, "2 x values against 1 y values"),
        ("SimmSpline", {"x": (0.0, 0.0, 1.0), "y": (1.0, 1.0, 2.0)}, "strictly increase"),
        ("PiecewiseLinearFunction", {}, "at least 1 knots"),
        ("LinearFunction", {"coefficients": (1.0,)}, "two coefficients"),
        ("PolynomialFunction", {"coefficients": ()}, "no coefficients"),
        ("MultiplierFunction", {"scale": 2.0}, "wraps no function"),
    ],
)
def test_malformed_functions_are_refused_by_name(kind, values, message):
    with pytest.raises(ValueError, match=message):
        evaluate(kind, values, 0.0)


def test_an_unknown_function_type_is_refused():
    with pytest.raises(UnsupportedModelError, match="Sine is not a function type"):
        evaluate("Sine", {}, 0.0)


# --- batches -------------------------------------------------------------------------------------


def test_a_whole_trial_at_once_gives_exactly_what_frame_by_frame_gives(leg):
    rng = np.random.default_rng(11)
    names = leg.coordinate_names
    block = rng.normal(0.0, 1.2, (17, len(names)))  # steps outside the spline knots too
    batched = leg.motion(dict(zip(names, block.T)))
    for frame in range(block.shape[0]):
        single = leg.motion({n: block[frame:frame + 1, i] for i, n in enumerate(names)})
        for name, centre in single.joint_centres.items():
            np.testing.assert_allclose(batched.joint_centres[name][frame], centre[0], atol=ATOL)
        for body, rotation in single.rotations.items():
            np.testing.assert_allclose(batched.rotations[body][frame], rotation[0], atol=ATOL)
            np.testing.assert_allclose(
                batched.positions[body][frame], single.positions[body][0], atol=ATOL
            )


def test_ragged_unknown_or_empty_coordinates_are_refused(leg):
    with pytest.raises(ValueError, match="disagree on the number of frames"):
        leg.motion({"tx": np.zeros(9), "ty": np.zeros(8)})
    with pytest.raises(ValueError, match="does not declare.*'nope'"):
        leg.motion({"nope": np.zeros(3)})
    with pytest.raises(ValueError, match="nothing says how many frames"):
        leg.motion({})
    with pytest.raises(ValueError, match=r"must be a \(T,\) array"):
        leg.motion({"tx": np.zeros((3, 1))})


# --- coordinates ---------------------------------------------------------------------------------


def test_a_missing_coordinate_takes_the_models_default_value():
    joint = custom(
        "j", "/ground", "/bodyset/pelvis", ("a", "b"),
        [axis("rotation1", "a", "0 0 1", linear()), axis("translation1", "b", "1 0 0", linear())],
        defaults={"a": 0.3},
    )
    skeleton = model_of(joint, bodies=("pelvis",))
    implicit = skeleton.motion({"b": np.array([0.1, 0.2])})
    explicit = skeleton.motion({"a": np.full(2, 0.3), "b": np.array([0.1, 0.2])})
    np.testing.assert_array_equal(implicit.rotations["pelvis"], explicit.rotations["pelvis"])
    # "b" declares no default: OpenSim's is zero
    placed = skeleton.motion({"a": one(0.0)})
    np.testing.assert_allclose(placed.positions["pelvis"][0], np.zeros(3), atol=ATOL)


COUPLED = custom(
    "follower", "/bodyset/pelvis", "/bodyset/thigh", ("beta",),
    [axis("rotation1", "beta", "0 0 1", linear())],
)
LEADER = custom(
    "leader", "/ground", "/bodyset/pelvis", ("alpha",),
    [axis("translation1", "alpha", "1 0 0", linear())],
)
COUPLER = (
    '<CoordinateCouplerConstraint name="c">'
    "<coupled_coordinates_function>"
    "<PolynomialFunction><coefficients>0.5 0 0.1</coefficients></PolynomialFunction>"
    "</coupled_coordinates_function>"
    "<independent_coordinate_names>alpha</independent_coordinate_names>"
    "<dependent_coordinate_name>beta</dependent_coordinate_name>"
    "<scale_factor>2</scale_factor>"
    "</CoordinateCouplerConstraint>"
)


def test_a_dependent_coordinate_is_computed_from_its_coupler_when_not_given():
    skeleton = model_of(LEADER, COUPLED, bodies=("pelvis", "thigh"), constraints=COUPLER)
    assert skeleton.coordinate_names == ("alpha",)
    assert skeleton.dependent_coordinate_names == ("beta",)
    alpha = np.array([0.0, 0.4, -1.0])
    placed = skeleton.motion({"alpha": alpha})
    beta = 2.0 * (0.5 * alpha**2 + 0.1)
    np.testing.assert_allclose(
        placed.rotations["thigh"], Rotation.from_rotvec(np.outer(beta, [0, 0, 1])).as_matrix(),
        atol=ATOL,
    )
    given = skeleton.motion({"alpha": alpha, "beta": np.zeros(3)})
    np.testing.assert_allclose(given.rotations["thigh"], np.tile(np.eye(3), (3, 1, 1)), atol=ATOL)


def test_a_coupler_without_a_scale_factor_uses_one():
    unscaled = COUPLER.replace("<scale_factor>2</scale_factor>", "")
    skeleton = model_of(LEADER, COUPLED, bodies=("pelvis", "thigh"), constraints=unscaled)
    placed = skeleton.motion({"alpha": one(1.0)})
    np.testing.assert_allclose(placed.rotations["thigh"][0], rot([0, 0, 0.6]), atol=ATOL)


def test_couplers_that_cannot_be_evaluated_are_refused():
    two = COUPLER.replace(
        "<independent_coordinate_names>alpha</independent_coordinate_names>",
        "<independent_coordinate_names>alpha beta</independent_coordinate_names>",
    )
    with pytest.raises(UnsupportedModelError, match="couples 2 independent coordinates"):
        model_of(LEADER, COUPLED, bodies=("pelvis", "thigh"), constraints=two)
    unknown = COUPLER.replace(
        "<independent_coordinate_names>alpha", "<independent_coordinate_names>gamma"
    )
    with pytest.raises(ValueError, match="'gamma', which no joint declares"):
        model_of(LEADER, COUPLED, bodies=("pelvis", "thigh"), constraints=unknown)
    cycle = COUPLER + COUPLER.replace('name="c"', 'name="d"').replace(
        "<independent_coordinate_names>alpha", "<independent_coordinate_names>beta"
    ).replace("<dependent_coordinate_name>beta", "<dependent_coordinate_name>alpha")
    with pytest.raises(ValueError, match="cycle"):
        model_of(LEADER, COUPLED, bodies=("pelvis", "thigh"), constraints=cycle)


def test_rest_transform_is_the_child_in_its_parent_at_default_values():
    hip = custom(
        "hip", "pelvis_offset", "/bodyset/thigh", ("flex",),
        [axis("rotation1", "flex", "0 0 1", linear())],
        frames=offset("pelvis_offset", "pelvis", translation="0.1 -0.05 0.07",
                      orientation="0 0.2 0"),
        defaults={"flex": 0.4},
    )
    skeleton = model_of(FREE_PELVIS.replace(
        '<Coordinate name="tilt">', '<Coordinate name="tilt"><default_value>0.9</default_value>'
    ), hip, bodies=("pelvis", "thigh"))
    rotation, translation = skeleton.rest_transform("thigh")
    np.testing.assert_allclose(
        rotation, Rotation.from_euler("XYZ", [0, 0.2, 0]).as_matrix() @ rot([0, 0, 0.4]),
        atol=ATOL,
    )
    np.testing.assert_allclose(translation, HIP_OFFSET, atol=ATOL)
    pelvis_rotation, _ = skeleton.rest_transform("pelvis")
    np.testing.assert_allclose(pelvis_rotation, rot([0, 0, 0.9]), atol=ATOL)
    with pytest.raises(KeyError, match="ground"):
        skeleton.rest_transform("ground")


# --- motion types and motion files ---------------------------------------------------------------


TYPED = custom(
    "typed", "/ground", "/bodyset/pelvis",
    ("pure_turn", "pure_slide", "scaled", "splined", "mixed", "idle"),
    [
        axis("rotation1", "pure_turn", "0 0 1", linear(-1.0)),
        axis("rotation2", "scaled", "1 0 0", linear(2.0)),
        axis("rotation3", "mixed", "0 1 0", linear()),
        axis("translation1", "pure_slide", "1 0 0", linear()),
        axis("translation2", "splined", "0 1 0", spline("0 1 2", "0 0.1 0.3")),
        axis("translation3", "mixed", "0 0 1", spline("0 1 2", "0 0.1 0.3")),
    ],
)


def test_motion_types_follow_opensims_rule():
    skeleton = model_of(TYPED, bodies=("pelvis",))
    assert dict(skeleton.motion_types) == {
        "pure_turn": "rotational",
        "pure_slide": "translational",
        "scaled": "coupled",
        "splined": "coupled",
        "mixed": "rotational",
        "idle": None,
    }
    # a coupled coordinate is an angle when it drives a rotation axis
    assert skeleton.rotational_coordinates == ("pure_turn", "scaled", "mixed")


def test_a_declared_pure_motion_type_settles_a_coupled_coordinate():
    declared = TYPED.replace(
        '<Coordinate name="splined">',
        '<Coordinate name="splined"><motion_type>rotational</motion_type>',
    ).replace(
        '<Coordinate name="scaled">',
        '<Coordinate name="scaled"><motion_type>coupled</motion_type>',
    )
    skeleton = model_of(declared, bodies=("pelvis",))
    assert skeleton.motion_types["splined"] == "coupled"
    assert skeleton.rotational_coordinates == ("pure_turn", "scaled", "splined", "mixed")


def test_fixed_joints_have_fixed_motion_types():
    skeleton = model_of(
        fixed("FreeJoint", "free", "/ground", "/bodyset/pelvis", ("a", "b", "c", "x", "y", "z")),
        fixed("PlanarJoint", "planar", "/bodyset/pelvis", "/bodyset/thigh", ("r", "u", "v")),
        bodies=("pelvis", "thigh"),
    )
    assert skeleton.rotational_coordinates == ("a", "b", "c", "r")
    assert skeleton.motion_types["u"] == "translational"


def write_mot(path: Path, columns: dict, in_degrees: str | None = "yes") -> mot.MotTable:
    names = ["time", *columns]
    rows = zip(*([0.0, 0.01], *columns.values()))
    header = ([f"inDegrees={in_degrees}"] if in_degrees is not None else []) + ["endheader"]
    lines = header + ["\t".join(names)] + ["\t".join(repr(float(v)) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mot.read(path)


def test_degrees_are_converted_only_for_rotational_coordinates(tmp_path):
    skeleton = model_of(TYPED, bodies=("pelvis",))
    columns = {name: [90.0, 180.0] for name in skeleton.coordinate_names}
    columns["unrelated"] = [5.0, 6.0]
    table = write_mot(tmp_path / "ik.mot", columns)
    values, time = coordinates_from_mot(skeleton, table)
    np.testing.assert_allclose(time, [0.0, 0.01])
    assert set(values) == set(skeleton.coordinate_names)
    for name in ("pure_turn", "mixed"):
        np.testing.assert_allclose(values[name], [np.pi / 2, np.pi])
    # OpenSim writes coupled coordinates as they are, even in a file declared in degrees
    for name in ("pure_slide", "scaled", "splined", "idle"):
        np.testing.assert_allclose(values[name], [90.0, 180.0])
    placed = skeleton.motion(values)
    assert placed.rotations["pelvis"].shape == (2, 3, 3)


def test_a_table_in_radians_is_left_alone(tmp_path):
    skeleton = model_of(TYPED, bodies=("pelvis",))
    table = write_mot(
        tmp_path / "rad.mot", {n: [0.5, 0.25] for n in skeleton.coordinate_names}, "no"
    )
    values, _ = coordinates_from_mot(skeleton, table)
    np.testing.assert_allclose(values["pure_turn"], [0.5, 0.25])


def test_an_undeclared_unit_must_be_stated_and_a_declared_one_not_contradicted(tmp_path):
    skeleton = model_of(TYPED, bodies=("pelvis",))
    columns = {n: [90.0, 45.0] for n in skeleton.coordinate_names}
    undeclared = write_mot(tmp_path / "states.sto", columns, None)
    with pytest.raises(ValueError, match="does not declare inDegrees"):
        coordinates_from_mot(skeleton, undeclared)
    values, _ = coordinates_from_mot(skeleton, undeclared, angle_unit="deg")
    np.testing.assert_allclose(values["pure_turn"], [np.pi / 2, np.pi / 4])
    values, _ = coordinates_from_mot(skeleton, undeclared, angle_unit="rad")
    np.testing.assert_allclose(values["pure_turn"], [90.0, 45.0])
    declared = write_mot(tmp_path / "ik.mot", columns, "yes")
    with pytest.raises(ValueError, match="contradicts angle_unit='rad'"):
        coordinates_from_mot(skeleton, declared, angle_unit="rad")
    with pytest.raises(ValueError, match="'deg' or 'rad'"):
        coordinates_from_mot(skeleton, declared, angle_unit="degrees")


def test_a_missing_independent_coordinate_is_an_error_a_missing_dependent_one_is_not(tmp_path):
    skeleton = model_of(LEADER, COUPLED, bodies=("pelvis", "thigh"), constraints=COUPLER)
    table = write_mot(tmp_path / "ik.mot", {"alpha": [0.1, 0.2]})
    values, _ = coordinates_from_mot(skeleton, table)
    assert set(values) == {"alpha"}
    with_dependent = write_mot(tmp_path / "both.mot", {"alpha": [0.1, 0.2], "beta": [30.0, 60.0]})
    values, _ = coordinates_from_mot(skeleton, with_dependent)
    np.testing.assert_allclose(values["beta"], np.deg2rad([30.0, 60.0]))
    missing = write_mot(tmp_path / "missing.mot", {"beta": [0.1, 0.2]})
    with pytest.raises(ValueError, match=r"no column for coordinates \['alpha'\]"):
        coordinates_from_mot(skeleton, missing)


def test_a_states_table_names_coordinates_by_their_value_path(tmp_path):
    skeleton = model_of(LEADER, COUPLED, bodies=("pelvis", "thigh"), constraints=COUPLER)
    table = write_mot(
        tmp_path / "states.sto",
        {"/jointset/leader/alpha/value": [0.1, 0.2], "/jointset/leader/alpha/speed": [9.0, 9.0]},
        "no",
    )
    values, _ = coordinates_from_mot(skeleton, table)
    np.testing.assert_allclose(values["alpha"], [0.1, 0.2])
    doubled = write_mot(
        tmp_path / "doubled.sto", {"alpha": [0.1, 0.2], "/jointset/leader/alpha/value": [0, 0]}, "no"
    )
    with pytest.raises(ValueError, match="'alpha' has two columns"):
        coordinates_from_mot(skeleton, doubled)


# --- topology and malformed joints ---------------------------------------------------------------


def test_joints_are_evaluated_parents_first_whatever_the_file_order(leg):
    shuffled = model_of(KNEE, PIN_HIP, FREE_PELVIS)
    values = {"tilt": one(0.3), "hip_flexion": one(-0.4), "knee_angle": one(0.9)}
    expected, placed = leg.motion(values), shuffled.motion(values)
    for body in leg.bodies:
        np.testing.assert_allclose(placed.rotations[body], expected.rotations[body], atol=ATOL)
    assert shuffled.joints == ("knee", "hip", "ground_pelvis")


def test_topologies_that_cannot_be_placed_are_refused():
    with pytest.raises(UnsupportedModelError, match="no path to ground: knee"):
        model_of(FREE_PELVIS, KNEE)
    with pytest.raises(UnsupportedModelError, match="bodies no joint places: \\['shank'\\]"):
        model_of(FREE_PELVIS, PIN_HIP)
    twice = PIN_HIP.replace('name="hip"', 'name="hip2"').replace(
        '"hip_flexion"', '"hip_flexion2"'
    )
    with pytest.raises(UnsupportedModelError, match="'thigh' is moved by more than one joint"):
        model_of(FREE_PELVIS, PIN_HIP, twice, bodies=("pelvis", "thigh"))
    with pytest.raises(ValueError, match="moves 'shank', which the body set does not declare"):
        model_of(FREE_PELVIS, PIN_HIP, KNEE, bodies=("pelvis", "thigh"))


def test_a_frame_that_does_not_resolve_is_refused_with_its_socket():
    lost = PIN_HIP.replace(
        "<socket_parent_frame>pelvis_offset</socket_parent_frame>",
        "<socket_parent_frame>elsewhere</socket_parent_frame>",
    )
    with pytest.raises(UnsupportedModelError, match="'hip': its parent frame 'elsewhere'"):
        model_of(FREE_PELVIS, lost, bodies=("pelvis", "thigh"))
    unsocketed = PIN_HIP.replace(
        "<socket_child_frame>thigh_offset</socket_child_frame>", ""
    )
    with pytest.raises(ValueError, match="'hip' declares no child frame"):
        model_of(FREE_PELVIS, unsocketed, bodies=("pelvis", "thigh"))


@pytest.mark.parametrize(
    ("axes", "message"),
    [
        ([axis("rotation1", "a b", "0 0 1", linear())], "driven by 2 coordinates"),
        ([axis("rotation1", "z", "0 0 1", linear())], "'z', which the joint does not declare"),
        ([axis("rotation1", "a", "0 0 1", "")], "names coordinate 'a' but no function"),
        ([axis("rotation1", "", "0 0 1", linear())], "LinearFunction and no coordinate"),
        ([axis("rotation1", "a", "0 0 0", linear())], "zero-length axis"),
        ([axis("rotation4", "a", "0 0 1", linear())], "TransformAxis named 'rotation4'"),
        ([axis("rotation1", "a", "0 0 1", linear())] * 2, "'rotation1' twice"),
    ],
)
def test_transform_axes_that_cannot_be_evaluated_are_refused(axes, message):
    joint = custom("j", "/ground", "/bodyset/pelvis", ("a", "b"), axes)
    error = UnsupportedModelError if "driven by" in message else ValueError
    with pytest.raises(error, match=message):
        model_of(joint, bodies=("pelvis",))


def test_a_coordinate_declared_by_two_joints_is_refused():
    first = fixed("PinJoint", "a", "/ground", "/bodyset/pelvis", ("q",))
    second = fixed("PinJoint", "b", "/bodyset/pelvis", "/bodyset/thigh", ("q",))
    with pytest.raises(ValueError, match=r"more than one joint: \['q'\]"):
        model_of(first, second, bodies=("pelvis", "thigh"))


def test_two_joints_with_one_name_are_refused():
    first = fixed("PinJoint", "a", "/ground", "/bodyset/pelvis", ("q",))
    second = fixed("PinJoint", "a", "/bodyset/pelvis", "/bodyset/thigh", ("r",))
    with pytest.raises(ValueError, match=r"repeated: \['a'\]"):
        model_of(first, second, bodies=("pelvis", "thigh"))
