"""Forward kinematics of an OpenSim model, evaluated from the model's own description.

Everything a pose needs is in the model file: each joint's parent and child frames, the axes of
its degrees of freedom, and the functions that turn a coordinate into a turn or a slide --
including the splines of a knee whose one coordinate drives three rotations and two
translations. This module evaluates that description the way OpenSim does, a whole trial at
once, and nothing more; what a pose means for another skeleton is a correspondence's business.

Each convention below is a place where a plausible wrong answer hides, and each is OpenSim's:

* A ``CustomJoint`` turns its child frame by its three rotations in the order the model lists
  them, ``R = R(a1, f1) @ R(a2, f2) @ R(a3, f3)``, each about its normalised axis in the frame
  the previous turns left. Its translation ``sum(a_i * f_i)`` (axes normalised too) is read in
  the parent frame and is not carried by those rotations.
* The other joint types are fixed sequences of the same two motions, driven by their
  coordinates in order: a pin turns about Z; a slider slides along X; a universal joint turns
  about X then Y; gimbal and ball joints about X, Y, Z, body fixed; a free joint does that and
  then slides along the parent's X, Y, Z; a planar joint turns about Z and slides along X and Y;
  a weld does nothing.
* An offset frame's ``orientation`` is body-fixed X-Y-Z Euler angles.
* ``SimmSpline`` -- and ``NaturalCubicSpline``, which OpenSim reads as a ``SimmSpline`` -- is
  the Forsythe-Malcolm-Moler cubic spline: its end conditions match the third derivative of the
  cubic through the four end knots. It is not a natural spline, and the two part company near
  the ends of the range. Beyond the knots it continues straight along its end slopes, as does a
  ``PiecewiseLinearFunction``.
* A ``GCVSpline`` with zero error variance interpolates its knots with a natural spline of
  degree ``2 * half_order - 1``. The cubic one is a natural cubic spline, straight beyond its
  end knots, and is evaluated as such; other degrees, and fits that smooth, are refused rather
  than approximated.
* A coordinate's motion type is what ``CustomJoint`` makes it: driving an axis through a
  ``LinearFunction`` of slope +/-1 makes it rotational or translational by that axis; driving
  one any other way makes it coupled, unless another axis already made it pure. OpenSim
  converts only rotational coordinates when it writes a motion file in degrees, so a coupled
  coordinate sits in such a file in radians.

A property the file leaves out takes the value OpenSim gives it (``OPENSIM_DEFAULTS``). Those
values are part of the format, not choices made here.
"""

from __future__ import annotations

import os
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from ...formats import osim
from ...formats.mot import MotTable
from ...formats.osim import OsimFunction, OsimJoint, OsimModel
from ..base import SegmentPlacement, SkeletonModel

__all__ = [
    "OPENSIM_DEFAULTS",
    "OpenSimSkeleton",
    "UnsupportedModelError",
    "compile_function",
    "coordinates_from_mot",
    "read",
]

GROUND = "ground"

#: What OpenSim gives a property that a model file leaves out.
OPENSIM_DEFAULTS: Mapping[str, object] = types.MappingProxyType(
    {
        "Coordinate.default_value": 0.0,
        "CoordinateCouplerConstraint.scale_factor": 1.0,
        "TransformAxis.axis": (1.0, 0.0, 0.0),
        "Constant.value": 0.0,
        "LinearFunction.coefficients": (1.0, 0.0),
        "PolynomialFunction.coefficients": (1.0,),
        "MultiplierFunction.scale": 1.0,
        "GCVSpline.error_variance": 0.0,
    }
)

_ROTATIONS = ("rotation1", "rotation2", "rotation3")
_TRANSLATIONS = ("translation1", "translation2", "translation3")
_SPLINES = frozenset({"SimmSpline", "NaturalCubicSpline", "natCubicSpline"})
_X, _Y, _Z = np.eye(3)
#: Each joint type with a fixed motion, as ``(turns, axis)`` per coordinate in coordinate order.
_FIXED_JOINTS: Mapping[str, tuple[tuple[bool, np.ndarray], ...]] = {
    "PinJoint": ((True, _Z),),
    "SliderJoint": ((False, _X),),
    "UniversalJoint": ((True, _X), (True, _Y)),
    "GimbalJoint": ((True, _X), (True, _Y), (True, _Z)),
    "BallJoint": ((True, _X), (True, _Y), (True, _Z)),
    "FreeJoint": (
        (True, _X), (True, _Y), (True, _Z), (False, _X), (False, _Y), (False, _Z),
    ),
    "PlanarJoint": ((True, _Z), (False, _X), (False, _Y)),
    "WeldJoint": (),
}
_ANGLE_UNITS = {"deg": True, "rad": False}

Function = Callable[[np.ndarray], np.ndarray]


class UnsupportedModelError(ValueError):
    """The model uses a joint type, function type or topology this module does not evaluate."""


# --- functions -----------------------------------------------------------------------------------


def _knots(function: OsimFunction, what: str, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(function.values.get("x", ()), dtype=np.float64)
    y = np.asarray(function.values.get("y", ()), dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"{what} has {x.size} x values against {y.size} y values")
    if x.size < minimum:
        raise ValueError(f"{what} needs at least {minimum} knots, has {x.size}")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError(f"{what} has knots that do not strictly increase: {x.tolist()}")
    return x, y


def _straight_beyond(
    x: np.ndarray, y: np.ndarray, inside: Function, slope_low: float, slope_high: float
) -> Function:
    """``inside`` over the knots, continued along the given end slopes outside them."""

    def evaluate(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        return np.where(
            q < x[0],
            y[0] + (q - x[0]) * slope_low,
            np.where(q > x[-1], y[-1] + (q - x[-1]) * slope_high, inside(q)),
        )

    return evaluate


def _piecewise_linear(x: np.ndarray, y: np.ndarray) -> Function:
    if x.size == 1:
        # OpenSim gives a single knot a zero slope: the function is that knot's value.
        return lambda q: np.full(np.shape(q), y[0])
    slopes = np.diff(y) / np.diff(x)
    return _straight_beyond(x, y, lambda q: np.interp(q, x, y), slopes[0], slopes[-1])


def _fmm_coefficients(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    """``b, c, d`` of ``y[k] + t (b[k] + t (c[k] + t d[k]))``, as OpenSim's ``SimmSpline``.

    A line-for-line port of the Forsythe-Malcolm-Moler routine OpenSim uses, so that the two
    agree to rounding; the variable names and the reuse of ``b``/``c``/``d`` as scratch space
    are that routine's.
    """
    n = x.size
    b, c, d = np.zeros(n), np.zeros(n), np.zeros(n)
    if n == 2:
        b[:] = (y[1] - y[0]) / (x[1] - x[0])
        return b, c, d
    nm1, nm2 = n - 1, n - 2
    # Tridiagonal system: b is the diagonal, d the off-diagonal, c the right-hand side.
    d[0] = x[1] - x[0]
    c[1] = (y[1] - y[0]) / d[0]
    for i in range(1, nm1):
        d[i] = x[i + 1] - x[i]
        b[i] = 2.0 * (d[i - 1] + d[i])
        c[i + 1] = (y[i + 1] - y[i]) / d[i]
        c[i] = c[i + 1] - c[i]
    # End conditions: third derivatives at both ends from divided differences.
    b[0] = -d[0]
    b[nm1] = -d[nm2]
    c[0] = 0.0
    c[nm1] = 0.0
    if n > 3:
        d31 = x[3] - x[1]
        d20 = x[2] - x[0]
        d1 = x[nm1] - x[n - 3]
        d2 = x[nm2] - x[n - 4]
        d30 = x[3] - x[0]
        d3 = x[nm1] - x[n - 4]
        c[0] = c[2] / d31 - c[1] / d20
        c[nm1] = c[nm2] / d1 - c[n - 3] / d2
        c[0] = c[0] * d[0] * d[0] / d30
        c[nm1] = -c[nm1] * d[nm2] * d[nm2] / d3
    for i in range(1, n):
        t = d[i - 1] / b[i - 1]
        b[i] -= t * d[i - 1]
        c[i] -= t * c[i - 1]
    c[nm1] /= b[nm1]
    for i in range(nm2, -1, -1):
        c[i] = (c[i] - d[i] * c[i + 1]) / b[i]
    b[nm1] = (y[nm1] - y[nm2]) / d[nm2] + d[nm2] * (c[nm2] + 2.0 * c[nm1])
    for i in range(nm1):
        b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
        d[i] = (c[i + 1] - c[i]) / d[i]
        c[i] *= 3.0
    c[nm1] *= 3.0
    d[nm1] = d[nm2]
    return b, c, d


def _fmm_spline(x: np.ndarray, y: np.ndarray) -> Function:
    b, c, d = _fmm_coefficients(x, y)
    last = x.size - 2

    def inside(q: np.ndarray) -> np.ndarray:
        k = np.clip(np.searchsorted(x, q, side="right") - 1, 0, last)
        t = q - x[k]
        value = y[k] + t * (b[k] + t * (c[k] + t * d[k]))
        # The end knots answer with their own value, as OpenSim's evaluation does.
        return np.where(q == x[-1], y[-1], np.where(q == x[0], y[0], value))

    return _straight_beyond(x, y, inside, b[0], b[-1])


def _natural_cubic(x: np.ndarray, y: np.ndarray) -> Function:
    spline = CubicSpline(x, y, bc_type="natural")
    return _straight_beyond(
        x, y,
        lambda q: spline(np.clip(q, x[0], x[-1])),
        float(spline(x[0], 1)), float(spline(x[-1], 1)),
    )


def _is_constant(function: OsimFunction) -> bool:
    if function.kind == "Constant":
        return True
    inner = function.values.get("function")
    return function.kind == "MultiplierFunction" and inner is not None and _is_constant(inner)


def _slope(function: OsimFunction) -> float | None:
    """The slope of a ``LinearFunction``; ``None`` for every other type."""
    if function.kind != "LinearFunction":
        return None
    return float(_coefficients(function)[0])


def _coefficients(function: OsimFunction) -> tuple[float, ...]:
    default = OPENSIM_DEFAULTS[f"{function.kind}.coefficients"]
    return tuple(function.values.get("coefficients", default))


def compile_function(function: OsimFunction, where: str) -> Function:
    """A vectorised callable for one function description; ``where`` names it in errors."""
    kind, values = function.kind, function.values
    what = f"{where}: {kind}"
    if kind == "Constant":
        value = float(values.get("value", OPENSIM_DEFAULTS["Constant.value"]))
        return lambda q: np.full(np.shape(q), value)
    if kind == "LinearFunction":
        coefficients = _coefficients(function)
        if len(coefficients) != 2:
            raise ValueError(f"{what} needs two coefficients (slope, intercept), has {coefficients}")
        slope, intercept = coefficients
        return lambda q: slope * np.asarray(q, dtype=np.float64) + intercept
    if kind == "PolynomialFunction":
        coefficients = np.asarray(_coefficients(function), dtype=np.float64)
        if coefficients.size == 0:
            raise ValueError(f"{what} declares no coefficients")
        # Highest order first, as OpenSim writes them.
        return lambda q: np.polyval(coefficients, np.asarray(q, dtype=np.float64))
    if kind == "PiecewiseLinearFunction":
        return _piecewise_linear(*_knots(function, what, minimum=1))
    if kind in _SPLINES:
        return _fmm_spline(*_knots(function, what, minimum=2))
    if kind == "GCVSpline":
        half_order = values.get("half_order")
        error_variance = values.get("error_variance", OPENSIM_DEFAULTS["GCVSpline.error_variance"])
        if half_order != 2:
            raise UnsupportedModelError(
                f"{what} has half_order {half_order}; only the cubic spline (half_order 2) "
                "is evaluated"
            )
        if error_variance != 0.0:
            raise UnsupportedModelError(
                f"{what} has error_variance {error_variance}, so it smooths its knots rather "
                "than passing through them, which is not evaluated"
            )
        # OpenSim's fitter needs at least 2 * half_order knots.
        return _natural_cubic(*_knots(function, what, minimum=4))
    if kind == "MultiplierFunction":
        inner = values.get("function")
        if inner is None:
            raise ValueError(f"{what} wraps no function")
        scale = float(values.get("scale", OPENSIM_DEFAULTS["MultiplierFunction.scale"]))
        wrapped = compile_function(inner, where)
        return lambda q: scale * wrapped(q)
    raise UnsupportedModelError(f"{what} is not a function type this model evaluates")


# --- joints --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Motion:
    """One elementary motion of a joint: a turn about, or a slide along, a unit axis."""

    turns: bool
    axis: np.ndarray
    #: The coordinate driving it; ``None`` for a constant.
    coordinate: str | None
    function: Function


@dataclass(frozen=True, eq=False)
class _Joint:
    name: str
    parent_body: str
    child_body: str
    parent_rotation: np.ndarray
    parent_translation: np.ndarray
    child_rotation: np.ndarray
    child_translation: np.ndarray
    motions: tuple[_Motion, ...]

    def local(
        self, values: Mapping[str, np.ndarray], frames: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """The joint's own transform, child frame in parent frame: ``[T, 3, 3]`` and ``[T, 3]``."""
        rotation = np.broadcast_to(np.eye(3), (frames, 3, 3))
        translation = np.zeros((frames, 3))
        for motion in self.motions:
            argument = np.zeros(frames) if motion.coordinate is None else values[motion.coordinate]
            magnitude = np.broadcast_to(motion.function(argument), (frames,))
            if motion.turns:
                turn = Rotation.from_rotvec(motion.axis[None, :] * magnitude[:, None]).as_matrix()
                rotation = rotation @ turn
            else:
                translation = translation + motion.axis[None, :] * magnitude[:, None]
        return rotation, translation


def _unit(axis: tuple[float, float, float] | None, where: str) -> np.ndarray:
    vector = np.asarray(
        OPENSIM_DEFAULTS["TransformAxis.axis"] if axis is None else axis, dtype=np.float64
    )
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{where} has a zero-length axis")
    return vector / norm


def _custom_motions(joint: OsimJoint) -> tuple[_Motion, ...]:
    by_name: dict[str, osim.OsimTransformAxis] = {}
    for axis in joint.transform_axes:
        if axis.name not in _ROTATIONS + _TRANSLATIONS:
            raise ValueError(f"joint {joint.name!r} has a TransformAxis named {axis.name!r}")
        if axis.name in by_name:
            raise ValueError(f"joint {joint.name!r} declares TransformAxis {axis.name!r} twice")
        by_name[axis.name] = axis
    motions: list[_Motion] = []
    for name in _ROTATIONS + _TRANSLATIONS:
        axis = by_name.get(name)
        if axis is None:
            continue
        where = f"joint {joint.name!r} {name}"
        if len(axis.coordinates) > 1:
            raise UnsupportedModelError(
                f"{where} is driven by {len(axis.coordinates)} coordinates; functions of more "
                "than one coordinate are not evaluated"
            )
        coordinate = axis.coordinates[0] if axis.coordinates else None
        if coordinate is not None and coordinate not in joint.coordinates:
            raise ValueError(
                f"{where} names coordinate {coordinate!r}, which the joint does not declare"
            )
        if axis.function is None:
            if coordinate is not None:
                raise ValueError(f"{where} names coordinate {coordinate!r} but no function")
            continue  # no coordinate, no function: the axis contributes nothing
        if coordinate is None and not _is_constant(axis.function):
            raise ValueError(
                f"{where} has a {axis.function.kind} and no coordinate to evaluate it at"
            )
        motions.append(
            _Motion(
                turns=name in _ROTATIONS,
                axis=_unit(axis.axis, where),
                coordinate=coordinate,
                function=compile_function(axis.function, where),
            )
        )
    return tuple(motions)


def _identity(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64)


def _fixed_motions(joint: OsimJoint) -> tuple[_Motion, ...]:
    sequence = _FIXED_JOINTS[joint.kind]
    if len(joint.coordinates) != len(sequence):
        raise ValueError(
            f"joint {joint.name!r} is a {joint.kind}, which takes {len(sequence)} coordinates; "
            f"it declares {len(joint.coordinates)}"
        )
    return tuple(
        _Motion(turns=turns, axis=axis, coordinate=coordinate, function=_identity)
        for (turns, axis), coordinate in zip(sequence, joint.coordinates)
    )


def _frame(joint: OsimJoint, which: str) -> tuple[str, np.ndarray, np.ndarray]:
    frame = getattr(joint, f"{which}_frame")
    if frame is None:
        socket = getattr(joint, f"{which}_socket")
        if socket is None:
            raise ValueError(f"joint {joint.name!r} declares no {which} frame")
        raise UnsupportedModelError(
            f"joint {joint.name!r}: its {which} frame {socket!r} does not resolve to a frame "
            "fixed directly in a body"
        )
    rotation = Rotation.from_euler("XYZ", frame.orientation).as_matrix()
    return frame.body, rotation, np.asarray(frame.translation, dtype=np.float64)


def _compile_joint(joint: OsimJoint) -> _Joint:
    if joint.kind == "CustomJoint":
        motions = _custom_motions(joint)
    elif joint.kind in _FIXED_JOINTS:
        motions = _fixed_motions(joint)
    else:
        raise UnsupportedModelError(
            f"joint {joint.name!r} is a {joint.kind or 'joint of no type'}, which is not "
            "evaluated"
        )
    parent_body, parent_rotation, parent_translation = _frame(joint, "parent")
    child_body, child_rotation, child_translation = _frame(joint, "child")
    return _Joint(
        name=joint.name,
        parent_body=parent_body,
        child_body=child_body,
        parent_rotation=parent_rotation,
        parent_translation=parent_translation,
        child_rotation=child_rotation,
        child_translation=child_translation,
        motions=motions,
    )


def _motion_types(joint: OsimJoint) -> dict[str, str | None]:
    """OpenSim's motion type for each coordinate of a joint; ``None`` where it has none."""
    if joint.kind in _FIXED_JOINTS:
        return {
            coordinate: "rotational" if turns else "translational"
            for (turns, _), coordinate in zip(_FIXED_JOINTS[joint.kind], joint.coordinates)
        }
    axes = {axis.name: axis for axis in joint.transform_axes}
    types_: dict[str, str | None] = {}
    for coordinate in joint.coordinates:
        kind = None
        for index, name in enumerate(_ROTATIONS + _TRANSLATIONS):
            axis = axes.get(name)
            if axis is None or coordinate not in axis.coordinates or axis.function is None:
                continue
            if _slope(axis.function) in (1.0, -1.0):
                kind = "rotational" if index < 3 else "translational"
            elif kind is None:
                kind = "coupled"
        types_[coordinate] = kind
    return types_


def _drives_a_turn(joint: OsimJoint, coordinate: str) -> bool:
    return any(
        axis.name in _ROTATIONS and coordinate in axis.coordinates
        for axis in joint.transform_axes
    )


def _in_tree_order(joints: list[_Joint], bodies: tuple[str, ...]) -> tuple[_Joint, ...]:
    """Parents before children, without trusting the order the file lists them in."""
    declared = set(bodies)
    for joint in joints:
        if joint.child_body not in declared:
            raise ValueError(
                f"joint {joint.name!r} moves {joint.child_body!r}, which the body set does not "
                "declare"
            )
    placed = {GROUND}
    remaining = list(joints)
    ordered: list[_Joint] = []
    while remaining:
        ready = [joint for joint in remaining if joint.parent_body in placed]
        if not ready:
            names = ", ".join(f"{j.name} (on {j.parent_body})" for j in remaining)
            raise UnsupportedModelError(f"joints with no path to ground: {names}")
        for joint in ready:
            if joint.child_body in placed:
                raise UnsupportedModelError(
                    f"body {joint.child_body!r} is moved by more than one joint; closed "
                    "kinematic loops are not evaluated"
                )
            placed.add(joint.child_body)
            ordered.append(joint)
        remaining = [joint for joint in remaining if joint not in ready]
    orphans = [body for body in bodies if body not in placed]
    if orphans:
        raise UnsupportedModelError(f"bodies no joint places: {orphans}")
    return tuple(ordered)


@dataclass(frozen=True, eq=False)
class _Coupler:
    independent: str
    function: Function
    scale: float


# --- the model -----------------------------------------------------------------------------------


class OpenSimSkeleton(SkeletonModel):
    """An OpenSim model's bodies and joints, placed by the model's own forward kinematics.

    Coordinates are radians and metres. ``motion`` takes any model coordinate, independent or
    dependent: an independent coordinate left out takes the model's ``default_value``, and a
    dependent one left out is computed from its coupler constraint.
    """

    def __init__(self, model: OsimModel):
        self._model = model
        names = [joint.name for joint in model.joints]
        repeated_joints = sorted({name for name in names if names.count(name) > 1})
        if repeated_joints:
            raise ValueError(f"joint names must be unique; repeated: {repeated_joints}")
        compiled = [_compile_joint(joint) for joint in model.joints]
        self._ordered = _in_tree_order(compiled, model.bodies)
        self._joint_of = {joint.child_body: joint for joint in self._ordered}

        declared = model.coordinate_names
        repeated = sorted({name for name in declared if declared.count(name) > 1})
        if repeated:
            raise ValueError(f"coordinates declared by more than one joint: {repeated}")
        self._defaults = {
            detail.name: (
                OPENSIM_DEFAULTS["Coordinate.default_value"]
                if detail.default_value is None
                else detail.default_value
            )
            for joint in model.joints
            for detail in joint.coordinate_details
        }

        self._motion_types: dict[str, str | None] = {}
        rotational: set[str] = set()
        declared_types = {
            detail.name: detail.motion_type
            for joint in model.joints
            for detail in joint.coordinate_details
        }
        pure = ("rotational", "translational")
        for joint in model.joints:
            for coordinate, kind in _motion_types(joint).items():
                self._motion_types[coordinate] = kind
                if kind not in pure:
                    # Coupled or undefined: a pure declaration settles it, else what it drives.
                    declared_type = declared_types.get(coordinate)
                    if declared_type in pure:
                        kind = declared_type
                    elif _drives_a_turn(joint, coordinate):
                        kind = "rotational"
                if kind == "rotational":
                    rotational.add(coordinate)
        self._rotational = tuple(name for name in declared if name in rotational)

        self._couplers = self._compile_couplers(model, set(declared))

    @staticmethod
    def _compile_couplers(model: OsimModel, declared: set[str]) -> dict[str, _Coupler]:
        couplers: dict[str, _Coupler] = {}
        for constraint in model.couplers:
            where = f"constraint {constraint.name!r}"
            for name in (*constraint.independent, constraint.dependent):
                if name not in declared:
                    raise ValueError(f"{where} names coordinate {name!r}, which no joint declares")
            if len(constraint.independent) != 1:
                raise UnsupportedModelError(
                    f"{where} couples {len(constraint.independent)} independent coordinates; "
                    "only functions of one coordinate are evaluated"
                )
            if constraint.function is None:
                raise ValueError(f"{where} declares no coupled_coordinates_function")
            if constraint.dependent in couplers:
                raise ValueError(
                    f"coordinate {constraint.dependent!r} is the dependent of two couplers"
                )
            scale = constraint.scale_factor
            couplers[constraint.dependent] = _Coupler(
                independent=constraint.independent[0],
                function=compile_function(constraint.function, where),
                scale=(
                    OPENSIM_DEFAULTS["CoordinateCouplerConstraint.scale_factor"]
                    if scale is None
                    else scale
                ),
            )
        for dependent in couplers:
            chain = [dependent]
            while chain[-1] in couplers:
                following = couplers[chain[-1]].independent
                if following in chain:
                    raise ValueError(f"coupler constraints form a cycle: {chain + [following]}")
                chain.append(following)
        return couplers

    # --- SkeletonModel ---------------------------------------------------------------------------

    @property
    def model(self) -> OsimModel:
        return self._model

    @property
    def bodies(self) -> tuple[str, ...]:
        return self._model.bodies

    @property
    def joints(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self._model.joints)

    @property
    def parents(self) -> Mapping[str, str | None]:
        """Parent body per body; ``None`` for a body that hangs off ground."""
        parents: dict[str, str | None] = {}
        for body in self.bodies:
            parent = self._joint_of[body].parent_body
            parents[body] = None if parent == GROUND else parent
        return parents

    @property
    def joint_child_bodies(self) -> Mapping[str, str]:
        return {joint.name: joint.child_body for joint in self._ordered}

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        return self._model.independent_coordinate_names

    @property
    def dependent_coordinate_names(self) -> tuple[str, ...]:
        return self._model.dependent_coordinates

    @property
    def rotational_coordinates(self) -> tuple[str, ...]:
        """Every model coordinate that is an angle, dependent ones included, in model order.

        Rotational and translational coordinates are what OpenSim makes them; a coupled one
        is an angle when the file declares it rotational or, declaring nothing, when it drives
        a rotation axis.
        """
        return self._rotational

    @property
    def motion_types(self) -> Mapping[str, str | None]:
        """OpenSim's own motion type per model coordinate (``rotational``, ``translational``,
        ``coupled``, or ``None`` for a coordinate that drives nothing)."""
        return types.MappingProxyType(self._motion_types)

    @property
    def gravity(self) -> tuple[float, float, float] | None:
        return self._model.gravity

    def rest_transform(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """``(rotation, translation)`` of a body in its parent body with every coordinate at its
        default value (dependent ones through their couplers)."""
        if body not in self._joint_of:
            raise KeyError(f"no body named {body!r}")
        parent = self._joint_of[body].parent_body
        rest = self._rest
        parent_rotation = rest.rotations[parent][0]
        rotation = parent_rotation.T @ rest.rotations[body][0]
        translation = parent_rotation.T @ (rest.positions[body][0] - rest.positions[parent][0])
        return rotation, translation

    def motion(self, coordinates: Mapping[str, np.ndarray]) -> SegmentPlacement:
        """World rotations ``[T, 3, 3]`` and origins ``[T, 3]`` per body (``ground`` included),
        and the world position of every joint's frame origin ``[T, 3]``."""
        values, frames = self._values(coordinates)
        return self._place(values, frames)

    # --- evaluation ------------------------------------------------------------------------------

    @cached_property
    def _rest(self) -> SegmentPlacement:
        values, frames = self._values({}, frames=1)
        return self._place(values, frames)

    def _values(
        self, coordinates: Mapping[str, np.ndarray], frames: int | None = None
    ) -> tuple[dict[str, np.ndarray], int]:
        known = self._model.coordinate_names
        unknown = sorted(set(coordinates) - set(known))
        if unknown:
            raise ValueError(f"coordinates the model does not declare: {unknown}")
        values: dict[str, np.ndarray] = {}
        for name, value in coordinates.items():
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 1:
                raise ValueError(f"coordinate {name!r} must be a (T,) array, got {array.shape}")
            values[name] = array
        lengths = sorted({array.shape[0] for array in values.values()})
        if len(lengths) > 1:
            raise ValueError(f"coordinates disagree on the number of frames: {lengths}")
        if frames is None:
            if not lengths:
                raise ValueError("no coordinate was given, so nothing says how many frames to place")
            frames = lengths[0]

        dependent = set(self._model.dependent_coordinates)
        for name in known:
            if name not in values and name not in dependent:
                values[name] = np.full(frames, self._defaults[name], dtype=np.float64)

        def resolve(name: str) -> np.ndarray:
            if name not in values:
                coupler = self._couplers.get(name)
                if coupler is None:
                    raise ValueError(
                        f"dependent coordinate {name!r} was not given and no coupler "
                        "constraint defines it"
                    )
                source = resolve(coupler.independent)
                values[name] = coupler.scale * np.broadcast_to(coupler.function(source), (frames,))
            return values[name]

        for name in known:
            resolve(name)
        return values, frames

    def _place(self, values: Mapping[str, np.ndarray], frames: int) -> SegmentPlacement:
        rotations = {GROUND: np.broadcast_to(np.eye(3), (frames, 3, 3))}
        origins = {GROUND: np.zeros((frames, 3))}
        centres: dict[str, np.ndarray] = {}
        for joint in self._ordered:
            body_rotation = rotations[joint.parent_body]
            parent_rotation = body_rotation @ joint.parent_rotation
            parent_origin = origins[joint.parent_body] + body_rotation @ joint.parent_translation
            local_rotation, local_translation = joint.local(values, frames)
            frame_rotation = parent_rotation @ local_rotation
            frame_origin = parent_origin + (parent_rotation @ local_translation[:, :, None])[:, :, 0]
            centres[joint.name] = frame_origin
            child_rotation = frame_rotation @ joint.child_rotation.T
            rotations[joint.child_body] = child_rotation
            origins[joint.child_body] = frame_origin - child_rotation @ joint.child_translation
        order = (GROUND, *self.bodies)
        return SegmentPlacement(
            rotations={body: np.ascontiguousarray(rotations[body]) for body in order},
            positions={body: origins[body] for body in order},
            joint_centres={name: centres[name] for name in self.joints},
        )


# --- motion files --------------------------------------------------------------------------------


def _in_degrees(table: MotTable, angle_unit: str | None) -> bool:
    if angle_unit is not None and angle_unit not in _ANGLE_UNITS:
        raise ValueError(f"angle_unit must be 'deg' or 'rad', got {angle_unit!r}")
    if table.in_degrees is None:
        if angle_unit is None:
            raise ValueError("the table does not declare inDegrees; pass angle_unit='deg' or 'rad'")
        return _ANGLE_UNITS[angle_unit]
    if angle_unit is not None and _ANGLE_UNITS[angle_unit] != table.in_degrees:
        declared = "yes" if table.in_degrees else "no"
        raise ValueError(
            f"the table declares inDegrees={declared}, which contradicts angle_unit={angle_unit!r}"
        )
    return table.in_degrees


def _coordinate_columns(skeleton: OpenSimSkeleton, table: MotTable) -> dict[str, int]:
    """Column index per model coordinate. A states table names a coordinate's value by its
    component path, ``.../<coordinate>/value``; a motion table by the bare name."""
    known = set(skeleton.model.coordinate_names)
    columns: dict[str, int] = {}
    for index, label in enumerate(table.columns):
        parts = label.split("/")
        if label in known:
            name = label
        elif len(parts) >= 3 and parts[-1] == "value" and parts[-2] in known:
            name = parts[-2]
        else:
            continue
        if name in columns:
            raise ValueError(
                f"coordinate {name!r} has two columns: {table.columns[columns[name]]!r} and "
                f"{label!r}"
            )
        columns[name] = index
    return columns


def coordinates_from_mot(
    skeleton: OpenSimSkeleton, table: MotTable, *, angle_unit: str | None = None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Model coordinates in radians and metres from a ``.mot`` / ``.sto`` table, and its time.

    Degrees are converted for the coordinates OpenSim converts, the rotational ones by
    ``motion_types``. The unit is the table's ``inDegrees``; a table that does not declare it
    needs ``angle_unit`` (``"deg"`` or ``"rad"``), and one that does must not be contradicted.
    Every independent coordinate must have a column; dependent ones may be missing, and columns
    that name no coordinate are ignored.
    """
    in_degrees = _in_degrees(table, angle_unit)
    columns = _coordinate_columns(skeleton, table)
    missing = [name for name in skeleton.coordinate_names if name not in columns]
    if missing:
        raise ValueError(f"the table has no column for coordinates {missing}")
    values: dict[str, np.ndarray] = {}
    for name in skeleton.model.coordinate_names:
        if name not in columns:
            continue
        column = np.array(table.values[:, columns[name]], dtype=np.float64)
        if in_degrees and skeleton.motion_types.get(name) == "rotational":
            column = np.deg2rad(column)
        values[name] = column
    return values, np.array(table.time_s, dtype=np.float64)


def read(source: str | os.PathLike[str]) -> OpenSimSkeleton:
    """The skeleton of a ``.osim`` file (or of ``.osim`` XML text)."""
    return OpenSimSkeleton(osim.read(source))
