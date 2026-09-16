"""Command-line entry point.

``--version`` and the ``profile`` group exist. The remaining subcommands the README specifies
(``extract-model``, ``convert``, ``fbx2bvh``, ``info``, ``validate``) are added with the
migration phases in docs/plan.md, each together with the module it fronts, so that no command
ever exists without an implementation.
"""

from __future__ import annotations

import argparse
import sys

import yaml

from smpl18 import __version__
from smpl18.profile import Profile, ProfileLoadError, ProfileSchemaError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smpl18",
        description="Convert motion capture into an SMPL-24 pose corpus.",
    )
    parser.add_argument("--version", action="version", version=f"smpl18 {__version__}")
    commands = parser.add_subparsers(dest="command")

    profile = commands.add_parser(
        "profile", help="validate or show a dataset profile (a name or a path)"
    )
    actions = profile.add_subparsers(dest="action", required=True)
    validate = actions.add_parser(
        "validate", help="check a profile against the schema and resolve every file it references"
    )
    validate.add_argument("profile", help="a shipped profile name or a path to a YAML file")
    show = actions.add_parser(
        "show", help="print the resolved profile as YAML, with referenced files and hashes"
    )
    show.add_argument("profile", help="a shipped profile name or a path to a YAML file")
    return parser


def _load(name_or_path: str) -> Profile | None:
    try:
        return Profile.load(name_or_path)
    except ProfileSchemaError as error:
        print(f"error: {error.path or '<root>'}: {error.message}", file=sys.stderr)
    except ProfileLoadError as error:
        print(f"error: {error}", file=sys.stderr)
    return None


def profile_validate(name_or_path: str) -> int:
    profile = _load(name_or_path)
    if profile is None:
        return 1
    print(f"OK {profile.id} ({profile.source_kind.value}/{profile.format.value}) {profile.path}")
    for entry in profile.referenced_files():
        print(f"   {entry.role}: {entry.path} sha256={entry.sha256[:16]}")
    return 0


def profile_show(name_or_path: str) -> int:
    profile = _load(name_or_path)
    if profile is None:
        return 1
    sys.stdout.write(yaml.safe_dump(profile.resolved(), sort_keys=False, allow_unicode=True,
                                    default_flow_style=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "profile":
        if args.action == "validate":
            return profile_validate(args.profile)
        if args.action == "show":
            return profile_show(args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
