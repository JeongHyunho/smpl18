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
"""

from __future__ import annotations

import os
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .errors import FormatError

__all__ = ["OsimJoint", "OsimMarker", "OsimModel", "parse", "read"]

_AXIS_METADATA_TAGS = frozenset({"coordinates", "axis"})


@dataclass(frozen=True)
class OsimJoint:
    """One joint of the model: its two bodies, its coordinates, and whether it translates."""

    name: str
    parent_body: str | None
    child_body: str | None
    coordinates: tuple[str, ...]
    translates: bool

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


def _joints(root: ET.Element) -> tuple[OsimJoint, ...]:
    joints: list[OsimJoint] = []
    joint_objects = root.find(".//JointSet/objects")
    for element in joint_objects if joint_objects is not None else []:
        local_frames = _local_offset_frames(element)
        parent = element.find("socket_parent_frame")
        child = element.find("socket_child_frame")
        joints.append(
            OsimJoint(
                name=element.get("name") or "",
                parent_body=_body_from_socket(
                    parent.text if parent is not None else None, local_frames
                ),
                child_body=_body_from_socket(
                    child.text if child is not None else None, local_frames
                ),
                coordinates=_joint_coordinates(element),
                translates=_joint_translates(element),
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
    )


def read(source: str | os.PathLike[str]) -> OsimModel:
    """Parse a ``.osim`` file, or XML text handed over directly (a ``str`` starting with ``<``)."""
    if isinstance(source, str) and source.lstrip().startswith("<"):
        return parse(source)
    return parse(pathlib.Path(source).read_text(encoding="utf-8"))
