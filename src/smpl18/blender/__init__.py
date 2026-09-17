"""Rendering a corpus in Blender: a plan built here, a scene built there.

Two halves that never meet in one interpreter:

* :mod:`smpl18.blender.plan` writes everything about a trial into one npz -- the rest surface, the
  skinning weights and, per frame, each joint's rest-to-posed transform -- and
  :mod:`smpl18.blender.launch` resolves the render settings and starts Blender on it.
* :mod:`smpl18.blender.scene` is the script Blender runs. It imports ``bpy`` and is therefore *not*
  imported from here; ``launch.scene_script()`` gives its path.

The plan carries a few frames of vertices computed by :mod:`smpl18.mesh`, and the scene checks its
own deformation against them, so the two halves agree or the run fails.
"""

from __future__ import annotations

from .launch import (
    RENDER_SCHEMA,
    BlenderError,
    blender_command,
    load_render_settings,
    render_setting,
    run_blender,
    scene_script,
    shipped_render_settings,
    validate_render_settings,
    write_render_settings,
)
from .plan import PLAN_SCHEMA, PlanError, ScenePlan, build_plan, plan_for_trial, read_plan

__all__ = [
    "PLAN_SCHEMA",
    "RENDER_SCHEMA",
    "BlenderError",
    "PlanError",
    "ScenePlan",
    "blender_command",
    "build_plan",
    "load_render_settings",
    "plan_for_trial",
    "read_plan",
    "render_setting",
    "run_blender",
    "scene_script",
    "shipped_render_settings",
    "validate_render_settings",
    "write_render_settings",
]
