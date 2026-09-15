"""Find subjects and trials under an input root from the patterns a profile declares.

A pattern is a relative path with ``{placeholder}`` names and the globs ``*`` and ``**``.
``{subject}`` and ``{trial}`` are the identities; any other placeholder (``{study}``,
``{split}``, ``{variant}``) is a grouping value kept on the subject and trial for provenance and
for keyed fallbacks. Results are sorted by relative path, so a run is deterministic.
"""

from __future__ import annotations

import fnmatch
import os
import pathlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

__all__ = ["Layout", "LayoutError", "SubjectEntry", "TrialEntry", "compile_pattern"]

WITHIN_CONTAINER = "within_container"
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class LayoutError(ValueError):
    pass


def compile_pattern(pattern: str) -> re.Pattern[str]:
    """Turn a pattern into a regular expression over posix relative paths.

    ``{name}`` matches one path segment (a repeated name must match the same text), ``*`` any
    run of characters inside a segment, ``**`` any number of whole segments.
    """
    text = pattern.replace("\\", "/").strip("/")
    if not text:
        raise LayoutError("an empty pattern matches nothing")
    out: list[str] = []
    seen: set[str] = set()
    position = 0
    for match in _PLACEHOLDER.finditer(text):
        out.append(_globs_to_regex(text[position:match.start()]))
        name = match.group(1)
        out.append(f"(?P={name})" if name in seen else f"(?P<{name}>[^/]+)")
        seen.add(name)
        position = match.end()
    out.append(_globs_to_regex(text[position:]))
    return re.compile("".join(out))


def _globs_to_regex(literal: str) -> str:
    parts: list[str] = []
    index = 0
    while index < len(literal):
        if literal.startswith("**/", index):
            parts.append("(?:[^/]+/)*")
            index += 3
        elif literal.startswith("**", index):
            parts.append(".*")
            index += 2
        elif literal[index] == "*":
            parts.append("[^/]*")
            index += 1
        else:
            parts.append(re.escape(literal[index]))
            index += 1
    return "".join(parts)


def fill_pattern(pattern: str, values: Mapping[str, str]) -> str:
    """Substitute known placeholders; unknown ones stay as placeholders."""
    return _PLACEHOLDER.sub(
        lambda m: values[m.group(1)] if m.group(1) in values else m.group(0), pattern
    )


@dataclass(frozen=True)
class SubjectEntry:
    id: str
    path: pathlib.Path
    relative: str
    groups: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialEntry:
    id: str
    subject: str
    path: pathlib.Path
    relative: str
    groups: Mapping[str, str] = field(default_factory=dict)
    #: Companion files by role; a path that does not exist is reported, not hidden.
    files: Mapping[str, pathlib.Path] = field(default_factory=dict)
    #: True when the trial is one of several inside ``path`` and the format reader must list them.
    within_container: bool = False

    def missing_files(self) -> tuple[str, ...]:
        return tuple(role for role, path in self.files.items() if not path.is_file())


@dataclass(frozen=True)
class Layout:
    subject: str
    trial: str
    files: Mapping[str, str] = field(default_factory=dict)
    exclude: tuple[str, ...] = ()
    skip_dirs_starting_with: tuple[str, ...] = ()
    skip_empty_files: bool = False
    subject_table: Mapping[str, object] | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Layout:
        return cls(
            subject=str(data["subject"]),
            trial=str(data["trial"]),
            files=dict(data.get("files", {})),
            exclude=tuple(data.get("exclude", ())),
            skip_dirs_starting_with=tuple(data.get("skip_dirs_starting_with", ())),
            skip_empty_files=bool(data.get("skip_empty_files", False)),
            subject_table=data.get("subject_table"),
        )

    @property
    def trials_within_container(self) -> bool:
        return self.trial == WITHIN_CONTAINER

    # ---- discovery ---------------------------------------------------------------------

    def _entries(self, root: pathlib.Path) -> list[str]:
        """Every file and directory under ``root`` as a sorted posix relative path."""
        root = pathlib.Path(root)
        if not root.is_dir():
            raise LayoutError(f"input root is not a directory: {root}")
        found: list[str] = []
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(
                d for d in dirs if not d.startswith(tuple(self.skip_dirs_starting_with))
            )
            base = pathlib.Path(current).relative_to(root).as_posix()
            prefix = "" if base == "." else base + "/"
            for name in dirs:
                found.append(prefix + name)
            for name in sorted(files):
                if any(fnmatch.fnmatch(name, glob) for glob in self.exclude):
                    continue
                if self.skip_empty_files and (pathlib.Path(current) / name).stat().st_size == 0:
                    continue
                found.append(prefix + name)
        return sorted(found)

    def subjects(self, root: pathlib.Path) -> tuple[SubjectEntry, ...]:
        root = pathlib.Path(root)
        regex = compile_pattern(self.subject)
        if "subject" not in regex.groupindex:
            raise LayoutError("the subject pattern must contain {subject}")
        out: dict[str, SubjectEntry] = {}
        for relative in self._entries(root):
            match = regex.fullmatch(relative)
            if match is None:
                continue
            groups = match.groupdict()
            identity = groups["subject"]
            if identity in out:
                raise LayoutError(
                    f"subject {identity!r} matched twice: {out[identity].relative} and "
                    f"{relative}; make the subject pattern more specific"
                )
            out[identity] = SubjectEntry(
                id=identity, path=root / relative, relative=relative, groups=groups
            )
        return tuple(out[key] for key in sorted(out, key=lambda k: out[k].relative))

    def trials(self, root: pathlib.Path, subject: SubjectEntry) -> tuple[TrialEntry, ...]:
        root = pathlib.Path(root)
        if self.trials_within_container:
            return (
                TrialEntry(
                    id=subject.id, subject=subject.id, path=subject.path,
                    relative=subject.relative, groups=dict(subject.groups),
                    within_container=True,
                ),
            )
        pattern = fill_pattern(self.trial, subject.groups)
        regex = compile_pattern(pattern)
        if "trial" not in regex.groupindex:
            raise LayoutError("the trial pattern must contain {trial}")
        out: list[TrialEntry] = []
        for relative in self._entries(root):
            match = regex.fullmatch(relative)
            if match is None:
                continue
            groups = {**subject.groups, **match.groupdict()}
            values = {**groups, "subject": subject.id}
            files = {
                role: root / fill_pattern(file_pattern, values)
                for role, file_pattern in self.files.items()
            }
            out.append(
                TrialEntry(
                    id=groups["trial"], subject=subject.id, path=root / relative,
                    relative=relative, groups=groups, files=files,
                )
            )
        return tuple(sorted(out, key=lambda t: t.relative))

    def discover(self, root: pathlib.Path) -> Iterable[tuple[SubjectEntry, tuple[TrialEntry, ...]]]:
        for subject in self.subjects(root):
            yield subject, self.trials(root, subject)
