"""Render settings, and running Blender on a scene plan.

Blender is not a dependency of this package and is never imported here: it is an executable a
caller names. This module resolves the numbers a scene needs (from ``configs/render/*.yaml``, in
the same style as conversion settings: merged in order, nothing defaulted in code), builds the
command, and runs it, passing the plan and the resolved settings as two files.

The script Blender runs is :mod:`smpl18.blender.scene`, which imports ``bpy`` and nothing from this
package, so Blender's own Python needs no installation of ``smpl18``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from smpl18.profile.load import package_configs_dir

__all__ = [
    "RENDER_SCHEMA",
    "BlenderError",
    "blender_command",
    "load_render_settings",
    "render_setting",
    "run_blender",
    "scene_script",
    "shipped_render_settings",
    "validate_render_settings",
    "write_render_settings",
]

RENDER_SCHEMA = "smpl18_render_v1"
#: Every value the scene script reads, as ``section.key``; checked before Blender is started.
REQUIRED_SETTINGS: tuple[tuple[str, str], ...] = (
    ("scene", "leaf_reach"), ("scene", "check_frames"), ("scene", "check_tolerance_m"),
    ("scene", "floor"), ("scene", "smooth_shading"),
    ("render", "engine"), ("render", "samples"), ("render", "resolution"),
    ("render", "frame_step"), ("render", "film_transparent"),
    ("camera", "distance_m"), ("camera", "height_m"), ("camera", "azimuth_deg"),
    ("camera", "lens_mm"),
    ("light", "sun_energy"), ("light", "sun_angle_deg"), ("light", "world_light"),
    ("material", "colour"), ("material", "roughness"),
)
ENGINES = ("cycles", "eevee")


class BlenderError(RuntimeError):
    """Blender is missing, or refused the scene, or the settings are incomplete."""


def shipped_render_settings(name: str = "default") -> Path:
    """Path of a render settings file shipped with the source tree."""
    configs = package_configs_dir()
    if configs is None:
        raise BlenderError(
            "this installation has no configs/ directory, so it ships no render settings; pass "
            "your own with --render-settings"
        )
    path = configs / "render" / f"{name}.yaml"
    if not path.is_file():
        raise BlenderError(f"no render settings named {name!r} in {path.parent}")
    return path


def load_render_settings(paths: Sequence[str | Path]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Merge render settings files in order (later sections update earlier ones) and hash them."""
    if not paths:
        raise BlenderError("no render settings file given")
    merged: dict[str, Any] = {}
    record = []
    for path in paths:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping) or data.get("schema") != RENDER_SCHEMA:
            raise BlenderError(f"{path} is not a {RENDER_SCHEMA} file")
        for key, value in data.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record.append({"path": path.name, "sha256": digest})
    return merged, record


def render_setting(settings: Mapping[str, Any], section: str, key: str) -> Any:
    """One setting, or an error naming what is missing."""
    try:
        return settings[section][key]
    except (KeyError, TypeError):
        raise BlenderError(f"render settings are missing {section}.{key}") from None


def validate_render_settings(settings: Mapping[str, Any]) -> None:
    """Check every value the scene will need, before Blender is started."""
    for section, key in REQUIRED_SETTINGS:
        render_setting(settings, section, key)
    engine = str(render_setting(settings, "render", "engine")).lower()
    if engine not in ENGINES:
        raise BlenderError(f"render.engine is {engine!r}, expected one of {ENGINES}")
    width, height = render_setting(settings, "render", "resolution")
    if int(width) < 1 or int(height) < 1:
        raise BlenderError(f"render.resolution is {[width, height]}, which is not a picture")
    if len(render_setting(settings, "material", "colour")) != 3:
        raise BlenderError("material.colour takes three numbers (linear RGB)")
    if int(render_setting(settings, "render", "frame_step")) < 1:
        raise BlenderError("render.frame_step must be at least 1")


def write_render_settings(path: str | Path, settings: Mapping[str, Any]) -> Path:
    """Write the resolved settings next to the plan, as the JSON the scene script reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, sort_keys=True, ensure_ascii=False),
                    encoding="utf-8")
    return path


def scene_script() -> Path:
    """Path of the script Blender runs."""
    return Path(__file__).resolve().with_name("scene.py")


def blender_command(blender: str | Path, *, plan: str | Path, settings: str | Path,
                    blend: str | Path | None = None, render: str | Path | None = None,
                    quiet: bool = False) -> list[str]:
    """The command that builds the scene: ``blender --background ... -- --plan ...``.

    ``--factory-startup`` keeps a teammate's add-ons and preferences out of the result, so the same
    plan gives the same scene on every machine.
    """
    if blend is None and render is None:
        raise BlenderError("nothing to do: ask for a .blend with --blend, pictures with --render")
    command = [str(blender), "--background", "--factory-startup",
               "--python", str(scene_script()), "--",
               "--plan", str(plan), "--settings", str(settings)]
    if blend is not None:
        command += ["--blend", str(blend)]
    if render is not None:
        command += ["--render", str(render)]
    if quiet:
        command.append("--quiet")
    return command


def run_blender(blender: str | Path, **kwargs: Any) -> subprocess.CompletedProcess:
    """Run :func:`blender_command`, raising with Blender's own output when it fails.

    Blender writes a great deal to stdout; on success the last lines are what the scene script
    reported, which is what a caller wants to show.
    """
    command = blender_command(blender, **kwargs)
    try:
        finished = subprocess.run(command, capture_output=True, text=True, check=False,
                                  encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise BlenderError(
            f"no Blender at {blender!r}. Install Blender and pass its executable: on Windows "
            r'"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe", on macOS '
            "/Applications/Blender.app/Contents/MacOS/Blender, on Linux usually just `blender`"
        ) from None
    if finished.returncode:
        tail = "\n".join((finished.stdout or "").splitlines()[-25:])
        raise BlenderError(
            f"Blender exited with {finished.returncode}.\n{tail}\n{(finished.stderr or '').strip()}"
        )
    return finished
