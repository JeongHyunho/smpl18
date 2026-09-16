"""Read an OpenSim ``.osim`` model: bodies, joints, coordinates, markers, and two derived orders.

A container that stores coordinate values or joint centres as bare arrays (``.b3d`` does) gives
them no names, and both arrays are shorter than the model's own lists: coordinates by the
constraint-dependent ones, centres by the joints those coordinates belong to. The orders are
therefore derived from the model itself, never hardcoded, so a model whose skeleton differs
cannot silently shift every downstream joint by one.

Serialisation note: a ``TransformAxis`` may carry its function as a direct child whose tag *is*
the concrete type (``LinearFunction``, ``SimmSpline``, ``MultiplierFunction``), or wrapped in a
``<function>`` element. Both are read. Looking only for the wrapper finds nothing in the first
style and makes every joint look rotation-only, which in turn claims the femur and tibia hold a
constant distance across a walker knee. They do not.

The kinematic description (joint frames, transform axes, functions, coordinate details, coupler
constraints) is kept exactly as written: numbers are parsed, nothing is evaluated, and a property
the file leaves out stays out (``None``, or a missing key) so that the skeleton model evaluating
it is the one place OpenSim's own defaults are applied. Offset frames are the one exception: an
omitted ``translation`` or ``orientation`` is OpenSim's zero, which is also what a socket naming
a body directly means, so both read as zeros.
"""

from __future__ import annotations

import os
import pathlib
import types
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field

from .errors import FormatError

__all__ = [
    "OsimCoordinate",
    "OsimCouplerConstraint",
    "OsimFrame",
    "OsimFunction",
    "OsimJoint",
    "OsimMarker",
    "OsimModel",
    "OsimTransformAxis",
    "parse",
    "read",
]

_AXIS_METADATA_TAGS = frozenset({"coordinates", "axis"})
_ZERO = (0.0, 0.0, 0.0)
_GROUND = "ground"
#: Function elements whose knots are ``x`` and ``y``. The two spline spellings besides
#: ``SimmSpline`` are older names OpenSim still accepts for it.
_KNOT_FUNCTIONS = frozenset(
    {"SimmSpline", "NaturalCubicSpline", "natCubicSpline", "PiecewiseLinearFunction"}
)

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class OsimFrame:
    """A joint's parent or child frame, expressed in the body it is fixed to.

    ``orientation`` is OpenSim's body-fixed X-Y-Z Euler angles in radians, as written. A socket
    that names a body (or ``ground``) directly is that body's own frame and reads as zeros.
    """

    body: str
    translation: Vector3
    orientation: Vector3
    #: The offset frame's own name; ``None`` when the socket names the body itself.
    name: str | None = None


@dataclass(frozen=True)
class OsimFunction:
    """One OpenSim function element as data: its type tag and the properties it wrote.

    ``values`` holds only what the file declares: ``value`` (``Constant``), ``coefficients``
    (``LinearFunction``, ``PolynomialFunction``), ``x`` and ``y`` (the splines and
    ``PiecewiseLinearFunction``), ``half_order`` and ``error_variance`` besides (``GCVSpline``),
    ``scale`` and the nested ``function`` (``MultiplierFunction``). Sequences are tuples of
    floats. A type not listed keeps its tag and no values.
    """

    kind: str
    #: Read-only; left out of the hash so that joints and models stay hashable.
    values: Mapping[str, object] = field(hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", types.MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class OsimTransformAxis:
    """One axis of a ``CustomJoint``'s spatial transform (``rotation1`` .. ``translation3``)."""

    name: str
    coordinates: tuple[str, ...]
    #: The axis as written, not normalised; ``None`` when the file leaves it out.
    axis: Vector3 | None
    function: OsimFunction | None


@dataclass(frozen=True)
class OsimCoordinate:
    """A coordinate a joint declares, with the properties that say how to evaluate it."""

    name: str
    #: ``None`` when the file does not declare one.
    default_value: float | None
    #: True only when the file says so.
    locked: bool
    #: ``rotational`` / ``translational`` / ``coupled`` when the file declares one (files older
    #: than OpenSim 4 do), else ``None``.
    motion_type: str | None


@dataclass(frozen=True)
class OsimCouplerConstraint:
    """A ``CoordinateCouplerConstraint``: ``dependent = scale_factor * function(independent)``."""

    name: str
    independent: tuple[str, ...]
    dependent: str
    #: ``None`` when the constraint declares no function.
    function: OsimFunction | None
    #: ``None`` when the file does not declare one.
    scale_factor: float | None


@dataclass(frozen=True)
class OsimJoint:
    """One joint of the model: its two bodies, its coordinates, and whether it translates.

    The kinematic fields describe the joint without evaluating it. ``parent_frame`` and
    ``child_frame`` are ``None`` when the socket is absent or names something that does not
    resolve to a frame fixed in a body (a chain of offset frames, a frame this reader cannot
    find); ``parent_socket`` and ``child_socket`` keep the paths as written for that case.
    """

    name: str
    parent_body: str | None
    child_body: str | None
    coordinates: tuple[str, ...]
    translates: bool
    #: The joint's XML tag: ``CustomJoint``, ``PinJoint``, ``WeldJoint`` ...
    kind: str = ""
    parent_frame: OsimFrame | None = None
    child_frame: OsimFrame | None = None
    #: The spatial transform's axes in file order; empty for joints that have none.
    transform_axes: tuple[OsimTransformAxis, ...] = ()
    #: One entry per name in ``coordinates``, in the same order.
    coordinate_details: tuple[OsimCoordinate, ...] = ()
    parent_socket: str | None = None
    child_socket: str | None = None

    def fixed_in(self) -> frozenset[str]:
        """Bodies in which this joint's centre is a fixed point.

        The centre is the joint frame origin: always fixed in the child body, and also fixed
        in the parent body when the joint contributes no translation. ``ground`` is not a
        body of the subject and never anchors anything.
        """
        bodies = {self.child_body}
        if not self.translates:
            bodies.add(self.parent_body)
        bodies.discard(None)
        bodies.discard("ground")
        return frozenset(bodies)


@dataclass(frozen=True)
class OsimMarker:
    """A marker (station) the model declares: its name, the body it sits on, its location there."""

    name: str
    body: str | None
    location: tuple[float, float, float]


@dataclass(frozen=True)
class OsimModel:
    """The model's bodies, joints and markers, plus the coordinate coupling that shrinks arrays."""

    name: str
    bodies: tuple[str, ...]
    joints: tuple[OsimJoint, ...]
    dependent_coordinates: tuple[str, ...]
    markers: tuple[OsimMarker, ...] = ()
    #: The model's own gravity vector, which is what says which way is up. ``None`` when the
    #: model does not declare one -- callers must refuse rather than assume a convention.
    gravity: tuple[float, float, float] | None = None
    #: Every ``CoordinateCouplerConstraint`` in the constraint set, in file order.
    couplers: tuple[OsimCouplerConstraint, ...] = ()

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        """Every coordinate the model declares, in model order."""
        return tuple(c for j in self.joints for c in j.coordinates)

    @property
    def independent_coordinate_names(self) -> tuple[str, ...]:
        """Model coordinate order minus the constraint-dependent coordinates."""
        dependent = set(self.dependent_coordinates)
        return tuple(c for c in self.coordinate_names if c not in dependent)

    @property
    def centre_joints(self) -> tuple[OsimJoint, ...]:
        """Model joint order minus joints whose coordinates are all constraint-dependent.

        Such a joint carries no independent degree of freedom, and a container that stores
        one centre per independent joint leaves it out.
        """
        dependent = set(self.dependent_coordinates)
        return tuple(
            j for j in self.joints
            if not (j.coordinates and all(c in dependent for c in j.coordinates))
        )

    @property
    def joint_centre_names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.centre_joints)

    @property
    def body_parents(self) -> dict[str, str | None]:
        """Each child body to the body its joint hangs off (``ground`` or ``None`` at the top)."""
        return {j.child_body: j.parent_body for j in self.joints if j.child_body is not None}

    def rigid_centre_pairs(self) -> list[list[bool]]:
        """Which pairs of stored centres must hold a constant distance, by construction.

        A self-check against frame data: the prediction is one-directional -- a predicted-rigid
        pair that measures non-rigid is a real defect, while the converse only means a degree
        of freedom did not move in the frames looked at.
        """
        homes = [j.fixed_in() for j in self.centre_joints]
        return [[bool(a & b) for b in homes] for a in homes]


def _body_from_socket(path: str | None, local_frames: dict[str, str]) -> str | None:
    """'/bodyset/femur_r/femur_r_offset' -> 'femur_r'; '/ground' -> 'ground'."""
    if not path:
        return None
    parts = path.strip().strip("/").split("/")
    if parts[0] == "ground":
        return "ground"
    if parts[0] == "bodyset":
        return parts[1] if len(parts) >= 2 else None
    return local_frames.get(parts[-1])


def _local_offset_frames(joint: ET.Element) -> dict[str, str]:
    """Offset frames declared inside the joint, mapped to the body they hang off."""
    frames: dict[str, str] = {}
    for offset in joint.iter("PhysicalOffsetFrame"):
        name = offset.get("name")
        socket = offset.find("socket_parent")
        if not name or socket is None or not socket.text:
            continue
        parts = socket.text.strip().strip("/").split("/")
        if parts[0] == "ground":
            frames[name] = "ground"
        elif parts[0] == "bodyset" and len(parts) >= 2:
            frames[name] = parts[1]
        else:
            frames[name] = parts[-1]
    return frames


def _axis_function(axis: ET.Element) -> ET.Element | None:
    """The function element of a TransformAxis, wrapped or not."""
    for child in axis:
        if child.tag in _AXIS_METADATA_TAGS:
            continue
        if child.tag == "function":
            return next(iter(child), None)
        return child
    return None


def _joint_translates(joint: ET.Element) -> bool:
    """True when a translation axis is driven by a coordinate through a non-constant function."""
    spatial = joint.find(".//SpatialTransform")
    if spatial is None:
        return False
    for axis in spatial:
        if not (axis.get("name") or "").startswith("translation"):
            continue
        coords = axis.find("coordinates")
        driven = coords is not None and (coords.text or "").strip() != ""
        if not driven:
            continue
        function = _axis_function(axis)
        if function is not None and function.tag != "Constant":
            return True
    return False


def _joint_coordinates(joint: ET.Element) -> tuple[str, ...]:
    listing = joint.find("./coordinates")
    if listing is None:
        return ()
    return tuple(c.get("name") for c in listing if c.get("name"))


# --- the kinematic description, as written -------------------------------------------------------


def _numbers(text: str, what: str) -> tuple[float, ...]:
    try:
        return tuple(float(v) for v in text.split())
    except ValueError as exc:
        raise FormatError(f"{what}: {text.strip()!r} is not a list of numbers") from exc


def _sequence(element: ET.Element, tag: str, what: str) -> tuple[float, ...] | None:
    """The numbers of a child element; ``None`` when the element is absent or empty."""
    text = element.findtext(tag)
    if text is None or not text.strip():
        return None
    return _numbers(text, f"{what} {tag}")


def _number(element: ET.Element, tag: str, what: str) -> float | None:
    values = _sequence(element, tag, what)
    if values is None:
        return None
    if len(values) != 1:
        raise FormatError(f"{what} {tag}: expected one number, found {len(values)}")
    return values[0]


def _vector(element: ET.Element, tag: str, what: str) -> Vector3 | None:
    values = _sequence(element, tag, what)
    if values is None:
        return None
    if len(values) != 3:
        raise FormatError(f"{what} {tag}: expected three numbers, found {len(values)}")
    return values  # type: ignore[return-value]


def _function(element: ET.Element | None, what: str) -> OsimFunction | None:
    """A function element as data; a ``<function>`` wrapper is looked through."""
    if element is None:
        return None
    if element.tag == "function":
        return _function(next(iter(element), None), what)
    kind = element.tag
    where = f"{what} {kind}"
    values: dict[str, object] = {}

    def keep(key: str, value: object) -> None:
        if value is not None:
            values[key] = value

    if kind == "Constant":
        keep("value", _number(element, "value", where))
    elif kind in ("LinearFunction", "PolynomialFunction"):
        keep("coefficients", _sequence(element, "coefficients", where))
    elif kind in _KNOT_FUNCTIONS or kind == "GCVSpline":
        keep("x", _sequence(element, "x", where))
        keep("y", _sequence(element, "y", where))
        if kind == "GCVSpline":
            half_order = _number(element, "half_order", where)
            if half_order is not None and not half_order.is_integer():
                raise FormatError(f"{where} half_order: {half_order} is not an integer")
            keep("half_order", None if half_order is None else int(half_order))
            keep("error_variance", _number(element, "error_variance", where))
    elif kind == "MultiplierFunction":
        keep("scale", _number(element, "scale", where))
        inner = element.find("function")
        if inner is None:
            inner = next((child for child in element if child.tag != "scale"), None)
        keep("function", _function(inner, where))
    return OsimFunction(kind=kind, values=values)


def _socket_path(reference: str) -> tuple[str, ...]:
    """A socket's component names, with slashes and ``.``/``..`` steps dropped."""
    return tuple(part for part in reference.strip().split("/") if part not in ("", ".", ".."))


def _declared_offset_frames(root: ET.Element) -> dict[tuple[str, ...], ET.Element]:
    """Every named offset frame a body, a joint or ground owns, by its component path."""
    owners: list[tuple[tuple[str, ...], ET.Element]] = []
    for set_tag, prefix in (("BodySet", "bodyset"), ("JointSet", "jointset")):
        objects = root.find(f".//{set_tag}/objects")
        for owner in objects if objects is not None else []:
            if owner.get("name"):
                owners.append(((prefix, owner.get("name")), owner))
    ground = root.find(".//Ground")
    if ground is not None:
        owners.append(((_GROUND,), ground))
    frames: dict[tuple[str, ...], ET.Element] = {}
    for path, owner in owners:
        for element in owner.iter("PhysicalOffsetFrame"):
            if element.get("name"):
                frames[(*path, element.get("name"))] = element
    return frames


def _body_frame(
    path: tuple[str, ...], frames: dict[tuple[str, ...], ET.Element]
) -> OsimFrame | None:
    """The frame a path names, when it is a body or an offset frame fixed directly in one."""
    if path == (_GROUND,):
        return OsimFrame(_GROUND, _ZERO, _ZERO)
    if len(path) == 2 and path[0] == "bodyset":
        return OsimFrame(path[1], _ZERO, _ZERO)
    element = frames.get(path)
    if element is None:
        return None
    parent = _socket_path(element.findtext("socket_parent") or "")
    if parent == (_GROUND,):
        body = _GROUND
    elif len(parent) == 2 and parent[0] == "bodyset":
        body = parent[1]
    else:
        # Hangs off another offset frame (or off nothing): not a frame fixed in a body as
        # written, and composing the chain would be evaluating it.
        return None
    where = f"offset frame {'/'.join(path)!r}"
    return OsimFrame(
        body=body,
        translation=_vector(element, "translation", where) or _ZERO,
        orientation=_vector(element, "orientation", where) or _ZERO,
        name=element.get("name"),
    )


def _joint_frame(
    socket: str | None, joint: str, frames: dict[tuple[str, ...], ET.Element]
) -> OsimFrame | None:
    if socket is None or not socket.strip():
        return None
    path = _socket_path(socket)
    if len(path) == 1 and path != (_GROUND,):
        # A bare name is a frame the joint itself declares.
        path = ("jointset", joint, path[0])
    return _body_frame(path, frames)


def _transform_axes(joint: ET.Element, where: str) -> tuple[OsimTransformAxis, ...]:
    spatial = joint.find(".//SpatialTransform")
    axes: list[OsimTransformAxis] = []
    for element in spatial if spatial is not None else []:
        if element.tag != "TransformAxis":
            continue
        name = element.get("name") or ""
        what = f"{where} TransformAxis {name!r}"
        axes.append(
            OsimTransformAxis(
                name=name,
                coordinates=tuple((element.findtext("coordinates") or "").split()),
                axis=_vector(element, "axis", what),
                function=_function(_axis_function(element), what),
            )
        )
    return tuple(axes)


def _coordinate_details(joint: ET.Element, where: str) -> tuple[OsimCoordinate, ...]:
    listing = joint.find("./coordinates")
    details: list[OsimCoordinate] = []
    for element in listing if listing is not None else []:
        name = element.get("name")
        if not name:
            continue
        motion_type = (element.findtext("motion_type") or "").strip().lower()
        details.append(
            OsimCoordinate(
                name=name,
                default_value=_number(element, "default_value", f"{where} coordinate {name!r}"),
                locked=(element.findtext("locked") or "").strip().lower() == "true",
                motion_type=motion_type or None,
            )
        )
    return tuple(details)


def _joints(root: ET.Element) -> tuple[OsimJoint, ...]:
    joints: list[OsimJoint] = []
    frames = _declared_offset_frames(root)
    joint_objects = root.find(".//JointSet/objects")
    for element in joint_objects if joint_objects is not None else []:
        local_frames = _local_offset_frames(element)
        name = element.get("name") or ""
        where = f"joint {name!r}"
        parent_socket = element.findtext("socket_parent_frame")
        child_socket = element.findtext("socket_child_frame")
        joints.append(
            OsimJoint(
                name=name,
                parent_body=_body_from_socket(parent_socket, local_frames),
                child_body=_body_from_socket(child_socket, local_frames),
                coordinates=_joint_coordinates(element),
                translates=_joint_translates(element),
                kind=element.tag,
                parent_frame=_joint_frame(parent_socket, name, frames),
                child_frame=_joint_frame(child_socket, name, frames),
                transform_axes=_transform_axes(element, where),
                coordinate_details=_coordinate_details(element, where),
                parent_socket=parent_socket.strip() if parent_socket else None,
                child_socket=child_socket.strip() if child_socket else None,
            )
        )
    return tuple(joints)


def _bodies(root: ET.Element) -> tuple[str, ...]:
    body_objects = root.find(".//BodySet/objects")
    if body_objects is None:
        return ()
    return tuple(b.get("name") for b in body_objects if b.get("name"))


def _dependent_coordinates(root: ET.Element) -> tuple[str, ...]:
    dependent: list[str] = []
    constraint_objects = root.find(".//ConstraintSet/objects")
    for element in constraint_objects if constraint_objects is not None else []:
        name = element.find("dependent_coordinate_name")
        if name is not None and (name.text or "").strip():
            dependent.append(name.text.strip())
    return tuple(dependent)


def _couplers(root: ET.Element) -> tuple[OsimCouplerConstraint, ...]:
    couplers: list[OsimCouplerConstraint] = []
    constraint_objects = root.find(".//ConstraintSet/objects")
    for element in constraint_objects if constraint_objects is not None else []:
        if element.tag != "CoordinateCouplerConstraint":
            continue
        name = element.get("name") or ""
        where = f"constraint {name!r}"
        function = element.find("coupled_coordinates_function")
        couplers.append(
            OsimCouplerConstraint(
                name=name,
                independent=tuple((element.findtext("independent_coordinate_names") or "").split()),
                dependent=(element.findtext("dependent_coordinate_name") or "").strip(),
                function=_function(
                    next(iter(function), None) if function is not None else None, where
                ),
                scale_factor=_number(element, "scale_factor", where),
            )
        )
    return tuple(couplers)


def _markers(root: ET.Element) -> tuple[OsimMarker, ...]:
    """Every named ``Marker`` with a location; a location that is not three numbers is a defect."""
    markers: list[OsimMarker] = []
    for element in root.iter("Marker"):
        name = element.get("name")
        location = (element.findtext("location") or "").split()
        if not name or not location:
            continue
        if len(location) != 3:
            raise FormatError(f"marker {name!r} declares a location of {len(location)} numbers")
        try:
            xyz = tuple(float(v) for v in location)
        except ValueError as exc:
            raise FormatError(f"marker {name!r} declares a non-numeric location") from exc
        markers.append(
            OsimMarker(
                name=name,
                body=_body_from_socket(element.findtext("socket_parent_frame"), {}),
                location=xyz,  # type: ignore[arg-type]
            )
        )
    return tuple(markers)


def _gravity(root: ET.Element) -> tuple[float, float, float] | None:
    """The model's gravity vector -- the only statement in the file about which way is up."""
    element = root.find(".//Model/gravity")
    if element is None:
        element = root.find(".//gravity")
    if element is None or not (element.text or "").strip():
        return None
    parts = element.text.split()
    if len(parts) != 3:
        return None
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def parse(xml_text: str) -> OsimModel:
    """Parse ``.osim`` XML text."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise FormatError(f"not well-formed .osim XML: {exc}") from exc
    model = root.find(".//Model")
    return OsimModel(
        name=(model.get("name") if model is not None else None) or "",
        bodies=_bodies(root),
        joints=_joints(root),
        dependent_coordinates=_dependent_coordinates(root),
        markers=_markers(root),
        gravity=_gravity(root),
        couplers=_couplers(root),
    )


def read(source: str | os.PathLike[str]) -> OsimModel:
    """Parse a ``.osim`` file, or XML text handed over directly (a ``str`` starting with ``<``)."""
    if isinstance(source, str) and source.lstrip().startswith("<"):
        return parse(source)
    return parse(pathlib.Path(source).read_text(encoding="utf-8"))
