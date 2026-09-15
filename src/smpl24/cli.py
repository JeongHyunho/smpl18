"""Command-line entry point.

Only ``--version`` exists in the seed. The subcommands the README specifies (``extract-model``,
``convert --profile`` / ``convert --kind --format``, ``profile validate`` / ``profile show``,
``fbx2bvh``, ``info``, ``validate``) are added with the migration phases in docs/plan.md, each
together with the module it fronts, so that no command ever exists without an implementation.
"""

from __future__ import annotations

import argparse
import sys

from smpl24 import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smpl24",
        description="Convert motion capture into an SMPL-24 pose corpus.",
    )
    parser.add_argument("--version", action="version", version=f"smpl24 {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
