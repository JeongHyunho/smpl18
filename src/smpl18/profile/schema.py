"""The profile schema, ``smpl18_profile_v1``, and its validator.

A profile is data: which kind and format a dataset is, how its files are laid out, which field
means what, its units and frame, its correspondence, its repairs and its settings. The schema is
declarative on purpose. Policies are enumerated (``root.placement``, ``shape.method``, fill
rules), never expressed; a policy that cannot be enumerated is a code feature of a kind.

Validation refuses unknown keys anywhere and reports the dotted path of the offending key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from smpl18.sources.base import Format, SourceKind

__all__ = [
    "FILL_RULES",
    "ROOT_ALIGNMENTS",
    "ROOT_PLACEMENTS",
    "SCHEMA_ID",
    "SHAPE_METHODS",
    "ProfileSchemaError",
    "kind_requirements",
    "validate_profile",
]

SCHEMA_ID = "smpl18_profile_v1"

ROOT_PLACEMENTS = ("pelvis_centre", "source_translation", "none")
ROOT_ALIGNMENTS = ("child_offsets", "directions")
SHAPE_METHODS = ("bone_lengths", "parameters")
FILL_RULES = ("weld", "distribute", "estimate")
POSE_METHODS = ("segment_rotation_transfer", "position_ik", "parameters")
UP_AXES = ("y", "z")
LENGTH_UNITS = ("m", "mm")
ANGLE_UNITS = ("rad", "deg")
WRAP_FILTERS = ("declared_by_source", "from_settings")
DISCONTINUITY_ACTIONS = ("flag",)
RESAMPLE_ROTATIONS = ("slerp",)
RESAMPLE_TRANSLATIONS = ("linear",)
TABLE_FORMATS = ("csv", "json", "xlsx")
OCCLUSION_SENTINELS = ("zero", "nan")
ROTATION_LAYOUTS = ("T,3,3", "3,3,T")
POSE_LAYOUTS = ("T,J,3", "T,J*3")
BETAS_FRAMES = ("first", "all")
REFERENCE_AGGREGATES = ("mean_rotation",)
DRIVEN_SOURCES = ("trial_metadata",)


class ProfileSchemaError(ValueError):
    """A profile does not match the schema. ``path`` is the dotted key path at fault."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path or '<root>'}: {message}")


# --------------------------------------------------------------------------------------------
# A small declarative validator. Each node checks one value and reports the dotted path.
# --------------------------------------------------------------------------------------------


class Node:
    def check(self, value: Any, path: str) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class Str(Node):
    def check(self, value: Any, path: str) -> None:
        if not isinstance(value, str) or not value:
            raise ProfileSchemaError(path, f"expected a non-empty string, got {value!r}")

    def describe(self) -> str:
        return "string"


@dataclass(frozen=True)
class Bool(Node):
    def check(self, value: Any, path: str) -> None:
        if not isinstance(value, bool):
            raise ProfileSchemaError(path, f"expected true or false, got {value!r}")

    def describe(self) -> str:
        return "boolean"


@dataclass(frozen=True)
class Num(Node):
    positive: bool = False

    def check(self, value: Any, path: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProfileSchemaError(path, f"expected a number, got {value!r}")
        if self.positive and value <= 0:
            raise ProfileSchemaError(path, f"expected a positive number, got {value!r}")

    def describe(self) -> str:
        return "number"


@dataclass(frozen=True)
class Const(Node):
    value: Any

    def check(self, value: Any, path: str) -> None:
        if value != self.value:
            raise ProfileSchemaError(path, f"expected {self.value!r}, got {value!r}")

    def describe(self) -> str:
        return repr(self.value)


@dataclass(frozen=True)
class Enum(Node):
    values: tuple[str, ...]

    def check(self, value: Any, path: str) -> None:
        if not isinstance(value, str) or value not in self.values:
            raise ProfileSchemaError(
                path, f"expected one of {list(self.values)}, got {value!r}"
            )

    def describe(self) -> str:
        return "one of " + ", ".join(self.values)


@dataclass(frozen=True)
class ListOf(Node):
    item: Node
    min_items: int = 0

    def check(self, value: Any, path: str) -> None:
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise ProfileSchemaError(path, f"expected a list, got {value!r}")
        if len(value) < self.min_items:
            raise ProfileSchemaError(
                path, f"expected at least {self.min_items} item(s), got {len(value)}"
            )
        for index, item in enumerate(value):
            self.item.check(item, f"{path}[{index}]")

    def describe(self) -> str:
        return f"list of {self.item.describe()}"


@dataclass(frozen=True)
class MapOf(Node):
    """A mapping with arbitrary string keys, every value checked by one node."""

    value: Node

    def check(self, value: Any, path: str) -> None:
        if not isinstance(value, Mapping):
            raise ProfileSchemaError(path, f"expected a mapping, got {value!r}")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProfileSchemaError(path, f"mapping keys must be strings, got {key!r}")
            self.value.check(item, f"{path}.{key}" if path else key)

    def describe(self) -> str:
        return f"mapping of {self.value.describe()}"


@dataclass(frozen=True)
class Map(Node):
    """A mapping with a fixed set of keys. Unknown keys are refused with their path."""

    fields: Mapping[str, Node]
    required: tuple[str, ...] = ()

    def check(self, value: Any, path: str) -> None:
        if not isinstance(value, Mapping):
            raise ProfileSchemaError(path, f"expected a mapping, got {value!r}")
        for key in value:
            if key not in self.fields:
                raise ProfileSchemaError(
                    f"{path}.{key}" if path else str(key),
                    f"unknown key; allowed keys are {sorted(self.fields)}",
                )
        for key in self.required:
            if key not in value:
                raise ProfileSchemaError(path, f"missing required key {key!r}")
        for key, item in value.items():
            self.fields[key].check(item, f"{path}.{key}" if path else key)

    def describe(self) -> str:
        return "mapping with keys " + ", ".join(sorted(self.fields))


@dataclass(frozen=True)
class OneOf(Node):
    """The first alternative that accepts the value wins; if none does, every failure is listed."""

    alternatives: tuple[Node, ...]

    def check(self, value: Any, path: str) -> None:
        failures: list[str] = []
        for alternative in self.alternatives:
            try:
                alternative.check(value, path)
                return
            except ProfileSchemaError as error:
                failures.append(f"as {alternative.describe()}: {error.message}")
        raise ProfileSchemaError(path, "no alternative accepted the value; " + "; ".join(failures))

    def describe(self) -> str:
        return " | ".join(a.describe() for a in self.alternatives)


# --------------------------------------------------------------------------------------------
# The schema itself.
# --------------------------------------------------------------------------------------------

#: A field reference: a top-level key, or a path of keys into a nested container.
FieldPath = OneOf((Str(), ListOf(Str(), min_items=1)))
Pattern = Str()
NamePair = ListOf(Str(), min_items=2)

LAYOUT = Map(
    fields={
        # Subjects: a pattern over paths under --input with {subject} and any grouping
        # placeholders ({study}, {split}, {variant}); * and ** are globs. It may match a file
        # (one container per subject) or a directory.
        "subject": Pattern,
        # Trials: a pattern with {trial} (and the subject/grouping placeholders), or
        # "within_container" when the trials live inside the subject's file.
        "trial": OneOf((Const("within_container"), Pattern)),
        # Companion files per trial for multi-file formats, by role, as patterns.
        "files": MapOf(Pattern),
        # File names (fnmatch globs) never treated as a trial or subject.
        "exclude": ListOf(Str()),
        # Directories whose name starts with one of these are not subjects.
        "skip_dirs_starting_with": ListOf(Str()),
        # Zero-byte files are not payload.
        "skip_empty_files": Bool(),
        # A table of subject demographics under --input, by column name.
        "subject_table": Map(
            fields={
                "file": Str(),
                "format": Enum(TABLE_FORMATS),
                "sheet": Str(),
                "id": Str(),
                "columns": Map(
                    fields={"gender": Str(), "mass_kg": Str(), "stature_m": Str()},
                ),
            },
            required=("file", "format", "id", "columns"),
        ),
    },
    required=("subject", "trial"),
)

GENDER = Map(
    fields={
        "field": FieldPath,
        "constant": Str(),
        # Source value (stripped, lower-cased) -> female | male | neutral.
        "map": MapOf(Enum(("female", "male", "neutral"))),
        # Used when the value is not in the map; without it an unmapped value is an error.
        "default": Enum(("female", "male", "neutral")),
        # Used when the field itself is missing; without it a missing field is an error.
        "when_absent": Enum(("female", "male", "neutral")),
        # A second field the value must agree with (first character), as a consistency check.
        "cross_check": FieldPath,
    },
)

FPS = Map(
    fields={
        "field": FieldPath,
        "aliases": ListOf(FieldPath),
        # Fallback table keyed by the value of a layout placeholder, e.g. {study}.
        "fallback": MapOf(Num(positive=True)),
        "fallback_key": Str(),
        "constant": Num(positive=True),
    },
)

BINDINGS = Map(
    fields={
        "poses": Map(
            fields={"field": FieldPath, "layout": Enum(POSE_LAYOUTS)},
            required=("field",),
        ),
        "betas": Map(
            fields={"field": FieldPath, "frame": Enum(BETAS_FRAMES)},
            required=("field",),
        ),
        "trans": Map(fields={"field": FieldPath}, required=("field",)),
        "fps": FPS,
        "gender": GENDER,
        "stature_m": Map(fields={"field": FieldPath}, required=("field",)),
        "mass_kg": Map(fields={"field": FieldPath}, required=("field",)),
        # Root translation coordinates of a skeleton (not rotations).
        "root_translation": ListOf(Str(), min_items=1),
        # Container passes (b3d): which pass supplies the frames, and which the unfiltered
        # values a wrap repair reads its branch cuts from.
        "frames": Map(
            fields={"pass": Str(), "unfiltered_pass": Str()},
            required=("pass",),
        ),
        # Joint centres from a table container: a struct whose fields are all centres, plus
        # named extras taken from other paths.
        "centres": Map(
            fields={"all_fields_of": FieldPath, "extra": MapOf(FieldPath)},
        ),
        # Segment world rotations per body: a path with {body}, its array layout, and aliases
        # (alias -> source body) so the correspondence can name a body the source does not.
        "rotations": Map(
            fields={"field": FieldPath, "layout": Enum(ROTATION_LAYOUTS),
                    "bodies": ListOf(Str()), "aliases": MapOf(Str())},
            required=("field", "layout"),
        ),
        "time": Map(fields={"field": FieldPath}, required=("field",)),
        "markers": Map(fields={"occlusion_sentinel": Enum(OCCLUSION_SENTINELS)}),
        # Which trials of a subject are static calibration poses, by trial id.
        "static_trials": ListOf(Str()),
    },
)

CONVENTIONS = Map(
    fields={
        "up_axis": Enum(UP_AXES),
        "length_unit": Enum(LENGTH_UNITS),
        "angle_unit": Enum(ANGLE_UNITS),
        # The frame is checked against the gravity a model declares, when it declares one.
        "up_axis_check": Enum(("model_gravity", "none")),
    },
    required=("up_axis", "length_unit", "angle_unit"),
)

RESCALE = Map(
    fields={
        # SMPL edge (parent, child) -> two source centres whose distance sets its length.
        "bones": MapOf(NamePair),
        # (ancestor, descendant) -> two centres; every offset on the path scales together.
        "chains": MapOf(NamePair),
        # (a, b) -> two centres; both move about their midpoint along the axis joining them.
        "spans": MapOf(NamePair),
    },
)

SHAPE = Map(
    fields={
        "method": Enum(SHAPE_METHODS),
        "landmark_offsets": OneOf((Const("none"), Str())),
        "rescale_to_measured": OneOf((Const(False), RESCALE)),
        "pool_per_subject": Bool(),
        # Non-adjacent joint pairs added to the bone-length targets, by SMPL joint name.
        "extra_spans": ListOf(NamePair),
        # Trial ids whose frames the shape is measured on; "all" pools every eligible trial.
        "from_trials": OneOf((Const("all"), ListOf(Str(), min_items=1))),
        "use_stature": Bool(),
        "use_mass": Bool(),
    },
    required=("method",),
)

ROOT = Map(
    fields={
        "placement": Enum(ROOT_PLACEMENTS),
        "alignment": Enum(ROOT_ALIGNMENTS),
        # Anatomical directions pairing an SMPL rest vector with a source vector.
        "directions": ListOf(
            Map(fields={"smpl": NamePair, "source": NamePair}, required=("smpl", "source"))
        ),
        # The SMPL joints whose centroid lands on the centroid of the named source centres.
        "anchor": Map(fields={"smpl": ListOf(Str(), min_items=1),
                              "source": ListOf(Str(), min_items=1)},
                      required=("smpl", "source")),
        # SMPL joints the translation is solved against every frame (their centres come from
        # the correspondence). Empty keeps the constant offset.
        "tracked": ListOf(Str()),
        # Recover a frozen root translation from a marker cluster rigid in one body.
        "recover_frozen": OneOf((
            Const(False),
            Map(fields={"markers": ListOf(Str(), min_items=3), "body": Str(),
                        "coordinates": ListOf(Str(), min_items=3)},
                required=("markers", "body", "coordinates")),
        )),
        # Directions and anchor are read from this static trial rather than from the trial.
        "from_trial": Str(),
    },
    required=("placement",),
)

POSE = Map(
    fields={
        "method": Enum(POSE_METHODS),
        # A static trial whose mean segment rotations define the neutral configuration.
        "reference": OneOf((
            Const("none"),
            Map(fields={"trial": Str(), "aggregate": Enum(REFERENCE_AGGREGATES)},
                required=("trial", "aggregate")),
        )),
        "place_unfitted_from_reference": Bool(),
        "lumbar_zero_from_reference": Bool(),
    },
    required=("method",),
)

PROVENANCE = Map(
    fields={
        "restrict_to_driven": OneOf((
            Const("none"),
            ListOf(Str(), min_items=1),
            Map(fields={"from": Enum(DRIVEN_SOURCES), "file": Str(), "field": FieldPath},
                required=("from", "file", "field")),
        )),
    },
)

REPAIRS = Map(
    fields={
        "wrap": OneOf((
            Const("none"),
            Map(
                fields={
                    "coordinates": OneOf((Const("any"), ListOf(Str(), min_items=1))),
                    "filter": Enum(WRAP_FILTERS),
                    "declared_pass": Str(),
                    "recompute_centres": Bool(),
                },
                required=("coordinates", "filter"),
            ),
        )),
        "resample": OneOf((
            Const("none"),
            Map(
                fields={"fps": Num(positive=True), "rotations": Enum(RESAMPLE_ROTATIONS),
                        "translation": Enum(RESAMPLE_TRANSLATIONS)},
                required=("fps",),
            ),
        )),
        "discontinuity": OneOf((
            Const("none"),
            Map(fields={"settings_key": Str(), "action": Enum(DISCONTINUITY_ACTIONS)},
                required=("settings_key", "action")),
        )),
        # Keep only the longest run of frames whose root was solved, so the time base stays
        # uniform; a gap ends the take.
        "longest_solved_span": Bool(),
        # Measure declared joint ranges and record the result; never drop a frame.
        "joint_ranges": OneOf((Const("none"), Map(fields={"settings_key": Str()},
                                                 required=("settings_key",)))),
    },
)

SKIP = Map(
    fields={
        "trials_without_pass": OneOf((Const("none"), Str())),
        "empty_trials": Bool(),
        "subjects_with_unresolved_gender": Bool(),
        "trials_with_nonfinite_input": Bool(),
        "trials_with_fps_other_than": OneOf((Const("none"), Num(positive=True))),
        "trials_shorter_than": OneOf((Const("none"), Map(fields={"settings_key": Str()},
                                                          required=("settings_key",)))),
    },
)

PROFILE = Map(
    fields={
        "schema": Const(SCHEMA_ID),
        "id": Str(),
        "description": Str(),
        "source_kind": Enum(tuple(k.value for k in SourceKind)),
        "format": Enum(tuple(f.value for f in Format)),
        "layout": LAYOUT,
        "bindings": BINDINGS,
        "conventions": CONVENTIONS,
        "correspondence": Str(),
        "markerset": Str(),
        "shape": SHAPE,
        "root": ROOT,
        "pose": POSE,
        "provenance": PROVENANCE,
        "repairs": REPAIRS,
        "skip": SKIP,
        "settings": OneOf((Str(), ListOf(Str(), min_items=1))),
    },
    required=("schema", "id", "source_kind", "format", "layout", "bindings",
              "conventions", "settings"),
)


@dataclass(frozen=True)
class KindRequirements:
    """Which top-level keys and bindings a source kind needs, beyond the common ones."""

    required_keys: tuple[str, ...] = ()
    required_bindings: tuple[str, ...] = ()
    forbidden_keys: tuple[str, ...] = ()


_BY_KIND = {
    SourceKind.SMPL_PARAMETERS: KindRequirements(
        required_keys=("shape",),
        required_bindings=("poses", "betas", "trans", "fps", "gender"),
        forbidden_keys=("correspondence", "markerset", "provenance"),
    ),
    SourceKind.SKELETON_MOTION: KindRequirements(
        required_keys=("correspondence", "shape", "root", "pose"),
        required_bindings=("gender",),
        forbidden_keys=("markerset",),
    ),
    SourceKind.JOINT_CENTRES: KindRequirements(
        required_keys=("correspondence", "shape", "root", "pose"),
        required_bindings=("gender", "centres", "fps"),
        forbidden_keys=("markerset",),
    ),
    SourceKind.MARKER_TRAJECTORIES: KindRequirements(
        required_keys=("markerset", "correspondence", "shape", "root", "pose"),
        required_bindings=("gender", "fps"),
    ),
}


def kind_requirements(kind: SourceKind | str) -> KindRequirements:
    return _BY_KIND[SourceKind(kind)]


def validate_profile(data: Any) -> None:
    """Raise ``ProfileSchemaError`` naming the offending path, or return ``None``."""
    PROFILE.check(data, "")
    kind = SourceKind(data["source_kind"])
    needs = kind_requirements(kind)
    for key in needs.required_keys:
        if key not in data:
            raise ProfileSchemaError(key, f"required for source_kind {kind.value!r}")
    for key in needs.forbidden_keys:
        if key in data:
            raise ProfileSchemaError(key, f"not allowed for source_kind {kind.value!r}")
    for key in needs.required_bindings:
        if key not in data["bindings"]:
            raise ProfileSchemaError(
                f"bindings.{key}", f"required for source_kind {kind.value!r}"
            )
    gender = data["bindings"].get("gender", {})
    if ("field" in gender) == ("constant" in gender):
        raise ProfileSchemaError(
            "bindings.gender", "give exactly one of 'field' or 'constant'"
        )
    if "constant" in gender and any(k in gender for k in ("map", "default", "when_absent")):
        raise ProfileSchemaError(
            "bindings.gender", "'map', 'default' and 'when_absent' apply only to a 'field' binding"
        )
    fps = data["bindings"].get("fps")
    if fps is not None:
        if ("field" in fps) == ("constant" in fps):
            raise ProfileSchemaError("bindings.fps", "give exactly one of 'field' or 'constant'")
        if ("fallback" in fps) != ("fallback_key" in fps):
            raise ProfileSchemaError(
                "bindings.fps", "'fallback' and 'fallback_key' go together"
            )
    root = data.get("root")
    if root is not None:
        alignment = root.get("alignment", "child_offsets")
        if alignment == "directions" and len(root.get("directions", ())) < 2:
            raise ProfileSchemaError(
                "root.directions", "alignment 'directions' needs at least two direction pairs"
            )
        if alignment == "child_offsets" and "directions" in root:
            raise ProfileSchemaError(
                "root.directions", "given, but root.alignment is 'child_offsets'"
            )
    pose = data.get("pose")
    if pose is not None:
        reference = pose.get("reference", "none")
        for key in ("place_unfitted_from_reference", "lumbar_zero_from_reference"):
            if pose.get(key) and reference == "none":
                raise ProfileSchemaError(f"pose.{key}", "needs pose.reference to name a trial")
    layout = data["layout"]
    if layout["trial"] == "within_container" and "files" in layout:
        raise ProfileSchemaError(
            "layout.files", "companion files do not apply when trials live within a container"
        )
