"""Load a profile: find it, substitute the environment, validate, resolve and hash what it
references, merge its settings.

Resolution order for ``Profile.load(name_or_path)``:

1. an existing path, as given;
2. ``<package>/configs/profiles/<name>.yaml`` (the examples shipped with the source tree);
3. ``$SHARED_DATASET_PATH/smpl18/profiles/<name>.yaml`` (profiles a project publishes to its
   shared drive).

Files a profile references (correspondence, marker set, landmark offsets, settings) resolve in
the same spirit: an absolute path; relative to the profile's own directory; relative to the
package ``configs/`` directory; relative to ``$SHARED_DATASET_PATH/smpl18/``. Every resolved
file is hashed so the corpus manifests can say exactly which data produced them.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml

from smpl18.profile.layout import Layout
from smpl18.profile.schema import ProfileSchemaError, validate_profile
from smpl18.sources.base import Format, SourceKind

__all__ = [
    "SHARED_ENV",
    "EnvironmentVariableUnset",
    "Profile",
    "ProfileLoadError",
    "ProfileNotFound",
    "ReferenceNotFound",
    "ReferencedFile",
    "package_configs_dir",
    "resolve_profile_path",
    "sha256_of",
    "shared_configs_dir",
    "substitute_env",
]

SHARED_ENV = "SHARED_DATASET_PATH"
_SHARED_SUBDIR = "smpl18"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ProfileLoadError(ValueError):
    """The profile could not be loaded; the message says what was tried."""


class ProfileNotFound(ProfileLoadError):
    pass


class ReferenceNotFound(ProfileLoadError):
    pass


class EnvironmentVariableUnset(ProfileLoadError):
    pass


def package_configs_dir() -> pathlib.Path | None:
    """The ``configs/`` directory of the source tree, or ``None`` in an installation without it."""
    candidate = pathlib.Path(__file__).resolve().parents[3] / "configs"
    return candidate if candidate.is_dir() else None


def shared_configs_dir(*, required: bool = False) -> pathlib.Path | None:
    """``$SHARED_DATASET_PATH/smpl18``, or ``None`` when the variable is unset."""
    value = os.environ.get(SHARED_ENV)
    if not value:
        if required:
            raise EnvironmentVariableUnset(
                f"the environment variable {SHARED_ENV} is not set; it names the shared drive "
                f"where a project publishes its profiles under {_SHARED_SUBDIR}/"
            )
        return None
    return pathlib.Path(value) / _SHARED_SUBDIR


def resolve_profile_path(name_or_path: str | os.PathLike[str]) -> pathlib.Path:
    """Where a profile named on the command line actually is, or a ``ProfileNotFound``."""
    given = pathlib.Path(name_or_path)
    tried: list[str] = []
    if given.is_file():
        return given.resolve()
    tried.append(f"as a path: {given}")

    name = given.name if given.suffix == ".yaml" else f"{given.name}.yaml"
    package = package_configs_dir()
    if package is not None:
        candidate = package / "profiles" / name
        if candidate.is_file():
            return candidate.resolve()
        tried.append(f"in the package examples: {candidate}")
    else:
        tried.append("in the package examples: no configs/ directory in this installation")

    shared = shared_configs_dir()
    if shared is not None:
        candidate = shared / "profiles" / name
        if candidate.is_file():
            return candidate.resolve()
        tried.append(f"on the shared drive: {candidate}")
    else:
        tried.append(f"on the shared drive: {SHARED_ENV} is not set")
    raise ProfileNotFound(
        f"no profile {str(name_or_path)!r}; tried " + "; ".join(tried)
    )


def substitute_env(value: Any, path: str = "") -> Any:
    """Replace ``${VAR}`` inside every string value; an unset variable is an error."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            variable = match.group(1)
            resolved = os.environ.get(variable)
            if resolved is None or resolved == "":
                raise EnvironmentVariableUnset(
                    f"{path or '<root>'} refers to ${{{variable}}}, which is not set"
                )
            return resolved
        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, Mapping):
        return {key: substitute_env(item, f"{path}.{key}" if path else str(key))
                for key, item in value.items()}
    if isinstance(value, list):
        return [substitute_env(item, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_reference(given: str, profile_dir: pathlib.Path, role: str) -> pathlib.Path:
    """A referenced file: absolute, or relative to the profile, the package, the shared drive."""
    raw = pathlib.Path(given)
    tried: list[str] = []
    if raw.is_absolute():
        if raw.is_file():
            return raw.resolve()
        tried.append(str(raw))
    else:
        candidates: list[tuple[str, pathlib.Path]] = [("the profile", profile_dir / raw)]
        package = package_configs_dir()
        if package is not None:
            candidates.append(("the package configs", package / raw))
        shared = shared_configs_dir()
        if shared is not None:
            candidates.append(("the shared drive", shared / raw))
        for label, candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
            tried.append(f"relative to {label}: {candidate}")
        if shared is None:
            tried.append(f"relative to the shared drive: {SHARED_ENV} is not set")
    raise ReferenceNotFound(f"{role}: no file {given!r}; tried " + "; ".join(tried))


@dataclass(frozen=True)
class ReferencedFile:
    role: str
    given: str
    path: pathlib.Path
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "given": self.given, "path": str(self.path),
                "sha256": self.sha256}


def _deep_merge(base: dict, override: Mapping) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: pathlib.Path, what: str) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ProfileLoadError(f"{what} {path} is not valid YAML: {error}") from error


@dataclass(frozen=True)
class Profile:
    """A validated profile with every referenced file resolved and hashed."""

    path: pathlib.Path
    sha256: str
    data: Mapping[str, Any]
    references: tuple[ReferencedFile, ...]
    settings: Mapping[str, Any]

    @classmethod
    def load(cls, name_or_path: str | os.PathLike[str]) -> Profile:
        path = resolve_profile_path(name_or_path)
        raw = _load_yaml(path, "profile")
        if not isinstance(raw, Mapping):
            raise ProfileLoadError(f"profile {path} must be a mapping at the top level")
        data = substitute_env(raw)
        try:
            validate_profile(data)
        except ProfileSchemaError as error:
            raise ProfileSchemaError(error.path, f"{error.message} (in {path})") from error
        return cls.from_validated(path, data)

    @classmethod
    def from_validated(cls, path: pathlib.Path, data: Mapping[str, Any]) -> Profile:
        directory = path.parent
        references: list[ReferencedFile] = []

        def refer(role: str, given: str) -> pathlib.Path:
            resolved = resolve_reference(given, directory, role)
            references.append(ReferencedFile(role, given, resolved, sha256_of(resolved)))
            return resolved

        for role in ("correspondence", "markerset"):
            if role in data:
                refer(role, data[role])
        offsets = data.get("shape", {}).get("landmark_offsets", "none")
        if offsets != "none":
            refer("shape.landmark_offsets", offsets)

        settings_given = data["settings"]
        if isinstance(settings_given, str):
            settings_given = [settings_given]
        merged: dict[str, Any] = {}
        for index, given in enumerate(settings_given):
            resolved = refer(f"settings[{index}]", given)
            loaded = _load_yaml(resolved, "settings file")
            if not isinstance(loaded, Mapping):
                raise ProfileLoadError(f"settings file {resolved} must be a mapping")
            merged = _deep_merge(merged, substitute_env(loaded))

        for reference in references:
            if reference.role.startswith("settings"):
                continue
            loaded = _load_yaml(reference.path, reference.role)
            if not isinstance(loaded, Mapping) or "schema" not in loaded:
                raise ProfileLoadError(
                    f"{reference.role}: {reference.path} must be a YAML mapping with a "
                    "'schema' key"
                )

        return cls(
            path=path,
            sha256=sha256_of(path),
            data=data,
            references=tuple(references),
            settings=merged,
        )

    # ---- typed accessors ---------------------------------------------------------------

    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def source_kind(self) -> SourceKind:
        return SourceKind(self.data["source_kind"])

    @property
    def format(self) -> Format:
        return Format(self.data["format"])

    @property
    def directory(self) -> pathlib.Path:
        return self.path.parent

    @property
    def layout(self) -> Layout:
        return Layout.from_mapping(self.data["layout"])

    @property
    def bindings(self) -> Mapping[str, Any]:
        return self.data["bindings"]

    @property
    def conventions(self) -> Mapping[str, Any]:
        return self.data["conventions"]

    def section(self, name: str) -> Mapping[str, Any]:
        """A top-level section (``shape``, ``root``, ``repairs`` ...), empty when absent."""
        return self.data.get(name, {})

    def reference(self, role: str) -> ReferencedFile | None:
        for entry in self.references:
            if entry.role == role:
                return entry
        return None

    def referenced_files(self) -> tuple[ReferencedFile, ...]:
        """Every file the profile refers to, with its sha256, settings files included."""
        return self.references

    def resolved(self) -> dict[str, Any]:
        """A plain dictionary of the profile as loaded, for ``profile show`` and manifests."""
        return {
            "profile": {"path": str(self.path), "sha256": self.sha256},
            **{key: _plain(value) for key, value in self.data.items()},
            "referenced_files": [entry.as_dict() for entry in self.references],
            "settings_resolved": _plain(self.settings),
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
