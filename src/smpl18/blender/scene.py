"""The script Blender runs: a scene plan in, a body on an armature out.

Run it through Blender, not through Python::

    blender --background --factory-startup --python scene.py -- \
        --plan plan.npz --settings render.json --blend body.blend --render frames/

It imports ``bpy`` and ``numpy`` (both come with Blender) and nothing from ``smpl18``, so Blender's
own Python needs no installation of this package. Everything it needs was computed when the plan
was written (:mod:`smpl18.blender.plan`).

What it builds:

* the rest surface as a mesh, with one vertex group per SMPL joint from the skinning weights;
* an armature whose bones sit on the rest joints, the mesh bound to it;
* per frame, each bone's channel transform set so the bone ends up carrying the plan's rest-to-posed
  transform. Blender's skinning is ``pose.matrix @ bone.matrix_local^-1``, so the bone's rest matrix
  cancels and how the bones are drawn cannot change the result;
* SMPL's pose blend shapes as 207 shape keys, when the plan carries them;
* a camera, a sun and a floor, from the render settings.

Then it checks itself: for each frame the plan sampled, it evaluates the mesh Blender actually
deforms and compares it with the vertices computed by this package. Beyond the tolerance it exits
non-zero with the worst distance, so a wrong convention or a Blender change fails the run instead of
quietly bending the body.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

#: The plan schema this script understands.
PLAN_SCHEMA = "smpl18_blender_plan_v1"
OBJECT_NAMES = {"mesh": "smpl18_body", "armature": "smpl18_armature", "floor": "floor",
                "camera": "camera", "sun": "sun", "material": "smpl18_body"}


# --- arguments and inputs -------------------------------------------------------------------------


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse the arguments after Blender's own ``--``."""
    parser = argparse.ArgumentParser(
        prog="blender --background --python scene.py --",
        description="Build a smpl18 body in Blender from a scene plan.",
    )
    parser.add_argument("--plan", required=True, type=Path, help="the plan npz to build")
    parser.add_argument("--settings", required=True, type=Path,
                        help="the resolved render settings, as JSON")
    parser.add_argument("--blend", type=Path, help="save the scene here")
    parser.add_argument("--render", type=Path,
                        help="render the frames into this directory (or path prefix)")
    parser.add_argument("--quiet", action="store_true", help="only report the self-check")
    return parser.parse_args(argv)


def script_arguments(argv: list[str]) -> list[str]:
    """What follows ``--`` in Blender's command line, which is what this script was given."""
    return argv[argv.index("--") + 1:] if "--" in argv else []


def load_plan(path: Path) -> dict[str, np.ndarray]:
    """Read the plan npz, refusing anything that is not one."""
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema"]) if "schema" in data.files else "(none)"
        if schema != PLAN_SCHEMA:
            raise SystemExit(f"{path}: schema is {schema!r}, not {PLAN_SCHEMA!r}")
        return {key: data[key] for key in data.files}


def setting(settings: dict, section: str, key: str):
    """One render setting, or an error naming what is missing."""
    try:
        return settings[section][key]
    except (KeyError, TypeError):
        raise SystemExit(f"render settings are missing {section}.{key}") from None


# --- the body ------------------------------------------------------------------------------------


def clear_scene() -> None:
    """Empty the factory scene, so only what the plan describes is in the file."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials,
                       bpy.data.cameras, bpy.data.lights, bpy.data.actions):
        for item in list(collection):
            collection.remove(item)


def build_mesh(vertices: np.ndarray, faces: np.ndarray, weights: np.ndarray,
               joint_names: list[str], *, smooth: bool) -> bpy.types.Object:
    """The rest surface as a mesh object, with one vertex group per joint."""
    mesh = bpy.data.meshes.new(OBJECT_NAMES["mesh"])
    mesh.from_pydata(vertices.astype(float).tolist(), [], faces.astype(int).tolist())
    mesh.validate()
    if smooth:
        for polygon in mesh.polygons:
            polygon.use_smooth = True
    obj = bpy.data.objects.new(OBJECT_NAMES["mesh"], mesh)
    bpy.context.collection.objects.link(obj)

    for joint, name in enumerate(joint_names):
        bound = np.nonzero(weights[:, joint])[0]
        if not bound.size:
            continue
        group = obj.vertex_groups.new(name=name)
        for vertex in bound:
            group.add([int(vertex)], float(weights[vertex, joint]), "REPLACE")
    return obj


def build_armature(rest: np.ndarray, tails: np.ndarray, parents: np.ndarray,
                   joint_names: list[str]) -> bpy.types.Object:
    """An armature with one bone per joint, heads on the rest joints, tails as the plan drew them."""
    armature = bpy.data.armatures.new(OBJECT_NAMES["armature"])
    obj = bpy.data.objects.new(OBJECT_NAMES["armature"], armature)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bones = []
    for joint, name in enumerate(joint_names):
        bone = armature.edit_bones.new(name)
        bone.head = Vector(rest[joint])
        bone.tail = Vector(tails[joint])
        if (bone.tail - bone.head).length < 1e-6:      # Blender drops a bone of no length
            bone.tail = bone.head + Vector((0.0, 0.0, 0.02))
        bones.append(bone)
    for joint, bone in enumerate(bones):
        parent = int(parents[joint])
        if parent >= 0:
            bone.parent = bones[parent]
            bone.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def bind(mesh_obj: bpy.types.Object, armature_obj: bpy.types.Object) -> None:
    """Deform the mesh with the armature, both left at the origin."""
    modifier = mesh_obj.modifiers.new(name="armature", type="ARMATURE")
    modifier.object = armature_obj
    mesh_obj.parent = armature_obj


def channel_basis(rest_local: np.ndarray, rest_local_inverse: np.ndarray, wanted: np.ndarray,
                  parent_wanted: np.ndarray | None) -> np.ndarray:
    """The channel transform that makes a bone carry ``wanted``, whatever its rest matrix is.

    Blender composes a posed bone in armature space as ``M_j = M_parent @ (L_parent^-1 @ L_j) @ B_j``,
    where ``L`` is the bone's rest matrix (``bone.matrix_local``) and ``B_j`` the channel transform
    (``pose_bone.matrix_basis``); it then deforms the mesh by ``M_j @ L_j^-1``. Asking for that
    deformation to be the plan's transform ``A_j`` means ``M_j = A_j @ L_j``, and substituting
    ``M_parent = A_parent @ L_parent`` leaves

        ``B_j = L_j^-1 @ A_parent^-1 @ A_j @ L_j``,

    with ``A_parent`` the identity at the root. Both ``L``'s cancel out of the deformation, which is
    why nothing here depends on where a bone's tail points or how it is rolled. Solving it in closed
    form also means no dependency-graph update between bones, so a long trial keyframes quickly.
    """
    above = np.eye(4) if parent_wanted is None else np.linalg.inv(parent_wanted)
    return rest_local_inverse @ above @ wanted @ rest_local


def animate(armature_obj: bpy.types.Object, transforms: np.ndarray, parents: np.ndarray,
            joint_names: list[str]) -> None:
    """Keyframe the bones so bone ``j`` carries ``transforms[frame, j]`` at each frame."""
    rest_local = [np.array(armature_obj.data.bones[name].matrix_local) for name in joint_names]
    rest_inverse = [np.linalg.inv(matrix) for matrix in rest_local]
    pose_bones = [armature_obj.pose.bones[name] for name in joint_names]
    for bone in pose_bones:
        bone.rotation_mode = "QUATERNION"

    for frame in range(transforms.shape[0]):
        posed = transforms[frame]
        for joint, bone in enumerate(pose_bones):
            parent = int(parents[joint])
            basis = channel_basis(rest_local[joint], rest_inverse[joint], posed[joint],
                                  None if parent < 0 else posed[parent])
            bone.matrix_basis = Matrix(basis.tolist())
            bone.keyframe_insert("location", frame=frame + 1)
            bone.keyframe_insert("rotation_quaternion", frame=frame + 1)


def add_correctives(mesh_obj: bpy.types.Object, posedirs: np.ndarray,
                    amounts: np.ndarray) -> int:
    """SMPL's pose blend shapes as shape keys, their amounts keyframed per frame.

    ``posedirs`` is ``(V, 3, 207)`` and ``amounts`` is ``(T, 207)``: the same linear combination the
    model applies, expressed as something Blender can carry on a static mesh.
    """
    rest = np.array([vertex.co for vertex in mesh_obj.data.vertices], dtype=np.float64)
    mesh_obj.shape_key_add(name="rest", from_mix=False)
    keys = []
    for direction in range(posedirs.shape[2]):
        key = mesh_obj.shape_key_add(name=f"pose_{direction:03d}", from_mix=False)
        moved = rest + posedirs[:, :, direction]
        key.data.foreach_set("co", moved.ravel())
        key.slider_min, key.slider_max = -10.0, 10.0
        keys.append(key)
    for frame in range(amounts.shape[0]):
        for direction, key in enumerate(keys):
            key.value = float(amounts[frame, direction])
            key.keyframe_insert("value", frame=frame + 1)
    return len(keys)


# --- what it is looked at with --------------------------------------------------------------------


def up_rotation(up_axis: str) -> Matrix:
    """Object rotation that stands a body whose up axis is ``up_axis`` up in Blender's Z-up world."""
    return Matrix.Rotation(math.radians(90.0), 4, "X") if up_axis == "y" else Matrix.Identity(4)


def joint_positions(transforms: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """``(T, 24, 3)`` world joint centres, applying each frame's transforms to the rest joints."""
    homogeneous = np.concatenate([rest, np.ones((rest.shape[0], 1))], axis=1)
    return np.einsum("tjab,jb->tja", transforms, homogeneous)[:, :, :3]


def add_camera_and_light(settings: dict, positions: np.ndarray, lowest: float,
                         rotation: Matrix) -> None:
    """A camera looking at the body, a sun, a world light and (optionally) a floor under it."""
    centre = Vector((positions[:, 0].mean(axis=0)).tolist())
    centre = rotation @ centre
    distance = float(setting(settings, "camera", "distance_m"))
    azimuth = math.radians(float(setting(settings, "camera", "azimuth_deg")))
    height = float(setting(settings, "camera", "height_m"))

    camera_data = bpy.data.cameras.new(OBJECT_NAMES["camera"])
    camera_data.lens = float(setting(settings, "camera", "lens_mm"))
    camera = bpy.data.objects.new(OBJECT_NAMES["camera"], camera_data)
    bpy.context.collection.objects.link(camera)
    eye = Vector((centre.x + distance * math.sin(azimuth),
                  centre.y - distance * math.cos(azimuth),
                  lowest + height))
    camera.location = eye
    direction = centre - eye
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera

    sun_data = bpy.data.lights.new(OBJECT_NAMES["sun"], type="SUN")
    sun_data.energy = float(setting(settings, "light", "sun_energy"))
    sun = bpy.data.objects.new(OBJECT_NAMES["sun"], sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(float(setting(settings, "light", "sun_angle_deg"))),
                          0.0, azimuth)
    sun.location = eye + Vector((0.0, 0.0, 2.0))

    world = bpy.data.worlds.new("world") if not bpy.data.worlds else bpy.data.worlds[0]
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    ambient = float(setting(settings, "light", "world_light"))
    background.inputs[0].default_value = (ambient, ambient, ambient, 1.0)

    if setting(settings, "scene", "floor"):
        bpy.ops.mesh.primitive_plane_add(size=8.0, location=(centre.x, centre.y, lowest))
        bpy.context.active_object.name = OBJECT_NAMES["floor"]


def add_material(mesh_obj: bpy.types.Object, settings: dict) -> None:
    """A matte material, so the render is not the default grey."""
    material = bpy.data.materials.new(OBJECT_NAMES["material"])
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    red, green, blue = (float(value) for value in setting(settings, "material", "colour"))
    shader.inputs["Base Color"].default_value = (red, green, blue, 1.0)
    shader.inputs["Roughness"].default_value = float(setting(settings, "material", "roughness"))
    mesh_obj.data.materials.append(material)


def configure_render(settings: dict, frames: int, fps: float) -> None:
    """Engine, samples, resolution and the frame range."""
    scene = bpy.context.scene
    engine = str(setting(settings, "render", "engine")).lower()
    available = {item.identifier for item in
                 bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    wanted = "CYCLES" if engine == "cycles" else "BLENDER_EEVEE_NEXT"
    if wanted not in available:                      # EEVEE was renamed across Blender versions
        wanted = next((name for name in available if name.startswith("BLENDER_EEVEE")), "CYCLES")
    scene.render.engine = wanted
    samples = int(setting(settings, "render", "samples"))
    engine_settings = getattr(scene, "cycles" if wanted == "CYCLES" else "eevee", None)
    if engine_settings is not None:                  # absent if the engine's add-on is disabled
        attribute = "samples" if wanted == "CYCLES" else "taa_render_samples"
        setattr(engine_settings, attribute, samples)
    width, height = setting(settings, "render", "resolution")
    scene.render.resolution_x, scene.render.resolution_y = int(width), int(height)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = bool(setting(settings, "render", "film_transparent"))
    scene.render.fps = max(1, int(round(fps)))
    scene.frame_start, scene.frame_end = 1, max(1, frames)
    scene.frame_step = int(setting(settings, "render", "frame_step"))


# --- checking the result -------------------------------------------------------------------------


def evaluated_vertices(mesh_obj: bpy.types.Object) -> np.ndarray:
    """``(V, 3)`` vertices of the mesh as Blender deforms it at the current frame."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = mesh_obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", vertices)
    result = vertices.reshape(-1, 3)
    evaluated.to_mesh_clear()
    return result


def check_against_plan(mesh_obj: bpy.types.Object, plan: dict) -> tuple[float, int]:
    """Compare Blender's own deformation with the plan's vertices. Returns the worst distance."""
    frames = plan.get("sample_frames")
    wanted = plan.get("sample_vertices")
    if frames is None or wanted is None or not len(frames):
        return 0.0, 0
    worst = 0.0
    for index, frame in enumerate(frames):
        bpy.context.scene.frame_set(int(frame) + 1)
        found = evaluated_vertices(mesh_obj)
        distance = np.linalg.norm(found - wanted[index].astype(np.float64), axis=1).max()
        worst = max(worst, float(distance))
    return worst, len(frames)


# --- putting it together -------------------------------------------------------------------------


def build(arguments: argparse.Namespace) -> int:
    plan = load_plan(arguments.plan)
    settings = json.loads(arguments.settings.read_text(encoding="utf-8"))
    about = json.loads(str(plan["about"])) if "about" in plan else {}
    joint_names = [str(name) for name in plan["joint_names"]]
    transforms = plan["transforms"]
    rest = plan["rest_joints"]
    def say(message: str) -> None:
        if not arguments.quiet:
            print(message)

    clear_scene()
    mesh_obj = build_mesh(plan["vertices"].astype(np.float64), plan["faces"],
                          plan["weights"].astype(np.float64), joint_names,
                          smooth=bool(setting(settings, "scene", "smooth_shading")))
    armature_obj = build_armature(rest, plan["bone_tails"], plan["parents"], joint_names)
    bind(mesh_obj, armature_obj)
    add_material(mesh_obj, settings)
    keys = 0
    if "posedirs" in plan and "pose_amounts" in plan:
        keys = add_correctives(mesh_obj, plan["posedirs"].astype(np.float64), plan["pose_amounts"])
    animate(armature_obj, transforms, plan["parents"], joint_names)

    rotation = up_rotation(str(about.get("up_axis", "z")))
    armature_obj.matrix_world = rotation
    positions = joint_positions(transforms, rest)
    turn = np.array(rotation.to_3x3())
    # The floor goes under the lowest vertex of the frames the plan carries, which is below the
    # lowest joint centre (a sole is below its ankle); with no samples, the joints must do.
    lowest_of = plan["sample_vertices"] if "sample_vertices" in plan else positions
    lowest = float(np.einsum("ab,...b->...a", turn, lowest_of)[..., 2].min())
    add_camera_and_light(settings, positions, lowest, rotation)
    configure_render(settings, transforms.shape[0], float(plan["fps"]))

    say(f"built {about.get('subject', '?')}/{about.get('trial', '?')}: "
        f"{transforms.shape[0]} frames, {plan['vertices'].shape[0]} vertices, "
        f"{len(joint_names)} bones"
        + (f", {keys} pose shape keys" if keys else ", no pose blend shapes"))

    worst, checked = check_against_plan(mesh_obj, plan)
    tolerance = float(plan.get("tolerance", 0.0))
    if checked:
        say(f"self-check: {checked} frames, worst vertex {worst * 1000:.4f} mm from this "
            f"package's own skinning (tolerance {tolerance * 1000:.4f} mm)")
        if worst > tolerance:
            print(f"ERROR: Blender's deformation is {worst * 1000:.3f} mm from the plan, beyond "
                  f"the {tolerance * 1000:.3f} mm allowed. The scene was not saved.", file=sys.stderr)
            return 2
    else:
        say("self-check: the plan carries no sample frames, so nothing was compared")

    bpy.context.scene.frame_set(1)
    if arguments.blend:
        arguments.blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(arguments.blend.resolve()))
        say(f"saved {arguments.blend}")
    if arguments.render:
        target = arguments.render
        directory = target if target.suffix == "" else target.parent
        directory.mkdir(parents=True, exist_ok=True)
        prefix = str((target / "frame_") if target.suffix == "" else target)
        bpy.context.scene.render.filepath = prefix
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(animation=True, write_still=False)
        rendered = len(sorted(directory.glob("*.png")))
        say(f"rendered {rendered} frames into {directory}")
    return 0


def main() -> int:
    return build(parse_arguments(script_arguments(sys.argv)))


if __name__ == "__main__":
    sys.exit(main())
