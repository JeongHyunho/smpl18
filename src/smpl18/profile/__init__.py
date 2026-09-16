"""Profiles: the only place a dataset is named.

``schema`` says what a profile may say, ``load`` finds and validates one and hashes what it
references, ``layout`` discovers subjects and trials on disk, ``bind`` turns bindings plus a
reader's tables into a source-kind dataclass.
"""

from .bind import BindingError, MissingField, UnresolvedGender, bind_parameters
from .layout import Layout, LayoutError, SubjectEntry, TrialEntry
from .load import (
    EnvironmentVariableUnset,
    Profile,
    ProfileLoadError,
    ProfileNotFound,
    ReferencedFile,
    ReferenceNotFound,
    resolve_profile_path,
)
from .schema import SCHEMA_ID, ProfileSchemaError, validate_profile

__all__ = [
    "SCHEMA_ID",
    "BindingError",
    "EnvironmentVariableUnset",
    "Layout",
    "LayoutError",
    "MissingField",
    "Profile",
    "ProfileLoadError",
    "ProfileNotFound",
    "ProfileSchemaError",
    "ReferenceNotFound",
    "ReferencedFile",
    "SubjectEntry",
    "TrialEntry",
    "UnresolvedGender",
    "bind_parameters",
    "resolve_profile_path",
    "validate_profile",
]
