"""Command-line entry point: ``smpl18 <command>``.

Commands::

    convert markers   labelled markers (.trc/.c3d) + a marker set
    convert centres   joint-centre trajectories (.trc/.c3d/.npz) + a correspondence
    convert opensim   an OpenSim model (.osim) + coordinate files (.mot/.sto) + a correspondence
    convert bvh       BVH clips + a correspondence
    convert smpl      SMPL / SMPL-H parameter files (.npz)
    extract-model     licensed SMPL .pkl -> clean .npz the converters read
    demo-models       write the stand-in body model, for trying the pipeline
    info              what a corpus holds and how well each trial was reproduced
    export-smpl       a corpus trial back as ordinary SMPL parameters (24 joints)
    blender           build a Blender scene from a corpus trial, and render it
    fbx2bvh           FBX -> BVH through an installed Blender
    profile           validate or show a dataset profile

Every ``convert`` writes one subject: its trials, fitted together, into ``--out``. Numbers come
from ``--settings``; nothing numeric is defaulted here, and neither is any number a render needs
(``--render-settings``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from smpl18 import __version__
from smpl18.model.select import GENDERS
from smpl18.profile import Profile, ProfileLoadError, ProfileSchemaError

AXES = ("x", "y", "z")
#: The source kind each convert subcommand reads.
KINDS = {"markers": "marker_trajectories", "centres": "joint_centres",
         "opensim": "skeleton_motion", "bvh": "skeleton_motion", "smpl": "smpl_parameters"}
LENGTH_UNITS = ("m", "cm", "mm")


# --- parser --------------------------------------------------------------------------------------


def _subject_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("subject")
    group.add_argument("--subject", type=Path,
                       help="subject file (smpl18_subject_v1): id, gender, measurements")
    group.add_argument("--subject-id", help="the subject's id (overrides the file's)")
    group.add_argument("--gender", choices=GENDERS,
                       help="selects the body model (overrides the file's)")
    group.add_argument("--measurement", action="append", default=[], metavar="NAME=METRES",
                       help="a subject measurement (repeatable; overrides the file's)")


def _common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, type=Path, help="corpus directory to write into")
    parser.add_argument("--settings", required=True, action="append", type=Path,
                        help="settings file(s); later files update earlier ones")
    parser.add_argument("--models", type=Path,
                        help="directory with SMPL_<GENDER>_clean.npz (else $SMPL18_MODELS)")
    parser.add_argument("--quiet", action="store_true", help="print only the final summary")
    parser.add_argument("--replace", action="store_true",
                        help="convert a subject already in the corpus anew, removing its trials")
    _subject_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smpl18",
        description="Convert motion capture into an 18-joint reduced SMPL pose corpus.",
    )
    parser.add_argument("--version", action="version", version=f"smpl18 {__version__}")
    commands = parser.add_subparsers(dest="command")

    convert = commands.add_parser("convert", help="convert one subject's trials into a corpus")
    kinds = convert.add_subparsers(dest="kind", required=True)

    markers = kinds.add_parser("markers", help="labelled surface markers (.trc / .c3d)")
    markers.add_argument("--input", required=True, nargs="+", type=Path, help="marker files, one per trial")
    markers.add_argument("--markerset", required=True, type=Path, help="marker-set description (YAML)")
    markers.add_argument("--up-axis", required=True, choices=AXES, help="the files' vertical axis")
    markers.add_argument("--occlusion-sentinel", choices=("none", "zero"), default="none",
                         help="'zero' reads an exact (0, 0, 0) sample as a lost marker")
    _common_options(markers)

    centres = kinds.add_parser("centres", help="joint-centre trajectories (.trc / .c3d / .npz)")
    centres.add_argument("--input", required=True, nargs="+", type=Path)
    centres.add_argument("--correspondence", required=True, type=Path,
                         help="table naming which centre is which SMPL joint (names: centres)")
    centres.add_argument("--up-axis", required=True, choices=AXES)
    _common_options(centres)

    opensim = kinds.add_parser("opensim", help="an OpenSim model with coordinate files")
    opensim.add_argument("--osim", required=True, type=Path, help="the (scaled) model")
    opensim.add_argument("--mot", required=True, nargs="+", type=Path,
                         help="coordinate files (.mot / .sto), one per trial")
    opensim.add_argument("--correspondence", required=True, type=Path)
    opensim.add_argument("--up-axis", choices=AXES,
                         help="only for a model that declares no gravity")
    opensim.add_argument("--angle-unit", choices=("deg", "rad"),
                         help="only for a file whose header does not say inDegrees")
    _common_options(opensim)

    bvh = kinds.add_parser("bvh", help="BVH clips")
    bvh.add_argument("--input", required=True, nargs="+", type=Path)
    bvh.add_argument("--correspondence", required=True, type=Path)
    bvh.add_argument("--up-axis", required=True, choices=AXES)
    bvh.add_argument("--length-unit", required=True, choices=LENGTH_UNITS,
                     help="the unit of the file's offsets and positions")
    _common_options(bvh)

    smpl = kinds.add_parser("smpl", help="SMPL / SMPL-H parameters (.npz)")
    smpl.add_argument("--input", required=True, nargs="+", type=Path)
    smpl.add_argument("--up-axis", required=True, choices=AXES)
    smpl.add_argument("--fps", type=float, help="frame rate, when the files store none")
    smpl.add_argument("--poses-key", default="poses")
    smpl.add_argument("--trans-key", default="trans")
    smpl.add_argument("--betas-key", default="betas")
    _common_options(smpl)

    extract = commands.add_parser("extract-model",
                                  help="licensed SMPL .pkl -> clean .npz (no pickle code runs)")
    extract.add_argument("--pkl", required=True)
    extract.add_argument("--gender", required=True, choices=GENDERS)
    extract.add_argument("--out", required=True)
    extract.add_argument("--num-betas", type=int, required=True,
                         help="shape directions to keep (the SMPL files carry 10 or 300)")
    extract.add_argument("--with-mesh", action="store_true",
                         help="also keep the skinning weights, pose blend shapes and faces")

    demo = commands.add_parser("demo-models",
                               help="write the stand-in body (not SMPL) to try the pipeline")
    demo.add_argument("--out", required=True, type=Path)
    demo.add_argument("--with-mesh", action="store_true",
                      help="also give it a blocky surface, so a scene can be rendered without SMPL")

    info = commands.add_parser("info", help="summarise a corpus")
    info.add_argument("corpus", type=Path)
    info.add_argument("--json", action="store_true", help="print the summary as JSON")

    export = commands.add_parser(
        "export-smpl", help="a corpus trial back as ordinary SMPL parameters (24 joints)"
    )
    export.add_argument("--corpus", required=True, type=Path)
    export.add_argument("--subject", help="subject id in the corpus (default: every subject)")
    export.add_argument("--trial", action="append",
                        help="trial id (repeatable; default: every trial of the subject)")
    export.add_argument("--out", required=True, type=Path,
                        help="directory for <subject>_<trial>.npz, or a .npz for a single trial")
    export.add_argument("--poses", choices=("flat", "grouped"), default="flat",
                        help="write poses as (T, 72), which most readers expect, or (T, 24, 3)")

    scene = commands.add_parser(
        "blender", help="build a Blender scene from a corpus trial, and render it"
    )
    scene.add_argument("--corpus", required=True, type=Path)
    scene.add_argument("--subject", required=True, help="subject id in the corpus")
    scene.add_argument("--trial", help="trial id (default: the subject's first)")
    scene.add_argument("--out", required=True, type=Path,
                       help="directory for the plan, the .blend and the frames")
    scene.add_argument("--models", type=Path,
                       help="directory with SMPL_<GENDER>_clean.npz (else $SMPL18_MODELS); the "
                            "model must carry the mesh (extract-model --with-mesh)")
    scene.add_argument("--render-settings", action="append", type=Path, metavar="FILE",
                       help="render settings file(s); defaults to the shipped configs/render/default.yaml")
    scene.add_argument("--correctives", action="store_true",
                       help="carry SMPL's pose blend shapes in as 207 shape keys")
    scene.add_argument("--frames", metavar="START:STOP[:STEP]",
                       help="a slice of the trial, to keep a long capture to a scene one can open")
    scene.add_argument("--blender", help="the Blender executable; without it the plan is written "
                                         "and the command to run printed")
    scene.add_argument("--blend", action="store_true", help="save <out>/scene.blend")
    scene.add_argument("--render", action="store_true", help="render PNG frames into <out>/frames")
    scene.add_argument("--quiet", action="store_true", help="print only what was written")

    fbx = commands.add_parser("fbx2bvh", help="export an FBX animation to BVH with Blender")
    fbx.add_argument("--input", required=True, type=Path)
    fbx.add_argument("--out", required=True, type=Path)
    fbx.add_argument("--blender", required=True, help="the Blender executable")

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


# --- convert -------------------------------------------------------------------------------------


class CommandError(Exception):
    pass


def _measurements(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise CommandError(f"--measurement expects NAME=METRES, got {item!r}")
        try:
            out[name.strip()] = float(value)
        except ValueError:
            raise CommandError(f"--measurement {name}: {value!r} is not a number") from None
    return out


def _subject(args):
    from smpl18.sources.subject import SubjectInfo

    extra = _measurements(args.measurement)
    if args.subject is not None:
        return SubjectInfo.load(args.subject).with_overrides(
            id=args.subject_id, gender=args.gender, measurements=extra
        )
    if not args.subject_id or not args.gender:
        raise CommandError("give --subject FILE, or both --subject-id and --gender")
    return SubjectInfo(id=args.subject_id, gender=args.gender, measurements=extra)


def _unique_ids(trials) -> None:
    seen = set()
    for trial in trials:
        if trial.id in seen:
            raise CommandError(f"two inputs share the trial name {trial.id!r}; rename one")
        seen.add(trial.id)


def _arguments(args) -> dict:
    skip = {"command", "kind", "quiet", "settings", "models", "out", "replace"}
    out = {}
    for key, value in vars(args).items():
        if key in skip or value is None:
            continue
        if isinstance(value, Path):
            value = value.name
        elif isinstance(value, list):
            value = [v.name if isinstance(v, Path) else v for v in value]
        out[key] = value
    return out


def convert(args) -> int:
    from smpl18 import convert as pipeline
    from smpl18.fit.correspondence import Correspondence
    from smpl18.model.load import Model
    from smpl18.sources.markers import MarkerSet

    say = (lambda message: None) if args.quiet else (lambda message: print(message, flush=True))
    settings, settings_files = pipeline.load_settings(args.settings)
    pipeline.validate_settings(settings, KINDS[args.kind])
    subject = _subject(args)
    pipeline.check_subject_free(args.out, subject.id, replace=args.replace)
    model = Model.for_gender(subject.gender, root=args.models)
    if model.stand_in:
        print("note: the body model is the stand-in from `smpl18 demo-models`, not SMPL",
              file=sys.stderr)
    trials = []
    if args.kind == "markers":
        markerset = MarkerSet.load(args.markerset)
        for path in args.input:
            say(f"read {path.name}")
            trials.append(pipeline.marker_trial(
                path, markerset=markerset, subject=subject, up_axis=args.up_axis,
                settings=settings, occlusion_sentinel=args.occlusion_sentinel,
            ))
    elif args.kind == "centres":
        table = Correspondence.load(args.correspondence)
        for path in args.input:
            say(f"read {path.name}")
            trials.append(pipeline.centre_trial(path, correspondence=table,
                                                up_axis=args.up_axis, settings=settings))
    elif args.kind == "opensim":
        table = Correspondence.load(args.correspondence)
        for path in args.mot:
            say(f"read {args.osim.name} + {path.name}")
            trials.append(pipeline.opensim_trial(
                args.osim, path, correspondence=table, settings=settings,
                up_axis=args.up_axis, angle_unit=args.angle_unit,
            ))
    elif args.kind == "bvh":
        table = Correspondence.load(args.correspondence)
        for path in args.input:
            say(f"read {path.name}")
            trials.append(pipeline.bvh_trial(path, correspondence=table, up_axis=args.up_axis,
                                             length_unit=args.length_unit, settings=settings))
    else:
        for path in args.input:
            say(f"read {path.name}")
            trials.append(pipeline.parameter_trial(
                path, model=model, up_axis=args.up_axis, settings=settings, fps=args.fps,
                poses_key=args.poses_key, trans_key=args.trans_key, betas_key=args.betas_key,
            ))
    _unique_ids(trials)
    fit = pipeline.fit_subject(model, trials, settings, progress=say)
    profile = {"id": "adhoc", "command": f"convert {args.kind}", "arguments": _arguments(args)}
    summary = pipeline.write_subject_corpus(
        args.out, subject=subject, model=model, trials=trials, fit=fit, settings=settings,
        settings_files=settings_files, profile=profile, replace=args.replace,
    )
    print(f"wrote subject {subject.id}: {len(trials)} trial(s) -> {args.out}")
    if fit.shape is not None:
        print(f"  shape: bone RMS {fit.shape.bone_rms_m * 1000:.1f} mm over "
              f"{len(fit.shape.bones)} bones")
    for trial in trials:
        pose = fit.poses[trial.id]
        if pose is not None:
            print(f"  {trial.id}: {int(pose.frame_valid.sum())}/{pose.frame_valid.size} frames, "
                  f"joint-centre RMS {pose.position_error.rms * 1000:.1f} mm")
    constants = fit.constants
    print(f"  reduction: freeze cost RMS {constants.fitted.rms_m * 1000:.1f} mm "
          f"(mean-rotation guess {constants.initial.rms_m * 1000:.1f} mm)")
    print(f"  corpus now holds {summary['subjects']} subject(s), {summary['trials']} trial(s)")
    record = json.loads((args.out / subject.id / "subject.json").read_text(encoding="utf-8"))
    for warning in record.get("checks", []):
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _mm(value) -> str:
    return "n/a" if value is None else f"{value * 1000:.1f} mm"


# --- other commands ------------------------------------------------------------------------------


def info(args) -> int:
    from smpl18.corpus import read_corpus

    corpus = read_corpus(args.corpus)
    if args.json:
        print(json.dumps(corpus.summary, indent=2))
        return 0
    summary = corpus.summary
    print(f"{args.corpus}: {summary['subjects']} subject(s), {summary['trials']} trial(s), "
          f"{summary['frames']} frames; source {summary['source_kind']} / {summary['format']}")
    for subject in corpus.subjects():
        record = subject.record
        reduced = record["reduced_model"]["fit"]
        stand_in = "  [stand-in body, not SMPL]" if record.get("model_is_stand_in") else ""
        print(f"  {subject.id}: {record['gender']}, model {record['model_file']}{stand_in}")
        if record.get("fit"):
            print(f"    shape: bone RMS {_mm(record['fit']['bone_rms_m'])}")
        print(f"    reduction: freeze cost RMS {_mm(reduced['residual_rms_m'])}, "
              f"max {_mm(reduced['residual_max_m'])}")
        for warning in record.get("checks", []):
            print(f"    warning: {warning}")
        for trial in subject.trials():
            validation = trial.manifest.get("validation") or {}
            stored = validation.get("stored_18_joint") or {}
            error = (f", stored-pose joint-centre RMS {_mm(stored['rms_m'])}"
                     if stored.get("rms_m") is not None else "")
            print(f"    {trial.id}: {trial.frames} frames at {trial.fps:g} Hz, "
                  f"{int(trial.frame_valid.sum())} valid{error}")
    return 0


def demo_models(args) -> int:
    from smpl18.model.demo import write_models

    for path in write_models(args.out, with_mesh=args.with_mesh):
        print(f"wrote {path}")
    surface = " with a blocky surface" if args.with_mesh else ""
    print(f"these are a stand-in body{surface} with the SMPL-24 tree, for trying the pipeline only")
    return 0


# --- out of the corpus again ----------------------------------------------------------------------


def _subjects_and_trials(args):
    """The corpus trials a command was pointed at, refusing a name the corpus does not have."""
    from smpl18.corpus import read_corpus

    corpus = read_corpus(args.corpus)
    subject_ids = [args.subject] if args.subject else corpus.subject_ids()
    if not subject_ids:
        raise CommandError(f"{args.corpus} holds no subject")
    for subject_id in subject_ids:
        subject = corpus.subject(subject_id)
        wanted = getattr(args, "trial", None)
        wanted = [wanted] if isinstance(wanted, str) else wanted
        for trial_id in wanted or subject.trial_ids():
            yield subject.trial(trial_id)


def export_smpl(args) -> int:
    from smpl18.original import sequence_from_trial, write_sequence

    trials = list(_subjects_and_trials(args))
    single = args.out.suffix == ".npz"
    if single and len(trials) > 1:
        raise CommandError(f"--out is one file but {len(trials)} trials were selected; give a "
                           "directory, or name one subject and one trial")
    for trial in trials:
        sequence = sequence_from_trial(trial)
        path = args.out if single else args.out / f"{trial.subject.id}_{trial.id}.npz"
        write_sequence(path, sequence, flat=args.poses == "flat")
        measured = sum(1 for p in sequence.joint_provenance if p == "measured")
        print(f"wrote {path}: {sequence.frames} frames at {sequence.fps:g} Hz, "
              f"{sequence.gender}, poses {'(T, 72)' if args.poses == 'flat' else '(T, 24, 3)'}, "
              f"{measured}/24 joints measured")
    return 0


def _frame_slice(text: str | None) -> slice | None:
    """``START:STOP[:STEP]`` as a slice; an empty part means the end it stands for."""
    if not text:
        return None
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise CommandError(f"--frames expects START:STOP[:STEP], got {text!r}")
    try:
        start, stop, *rest = (int(part) if part.strip() else None for part in parts)
    except ValueError:
        raise CommandError(f"--frames expects whole numbers, got {text!r}") from None
    return slice(start, stop, rest[0] if rest else None)


def blender(args) -> int:
    from smpl18.blender import launch
    from smpl18.blender import plan as scene_plan
    from smpl18.model.load import Model

    trials = list(_subjects_and_trials(args))
    trial = trials[0]
    if not args.trial and len(trials) > 1:
        print(f"note: subject {trial.subject.id} has {len(trials)} trials; building {trial.id}. "
              "Name another with --trial", file=sys.stderr)
    files = args.render_settings or [launch.shipped_render_settings()]
    settings, settings_files = launch.load_render_settings(files)
    launch.validate_render_settings(settings)

    model = Model.for_gender(trial.subject.gender, root=args.models)
    if not model.has_mesh:
        raise CommandError(
            f"{model.path} carries no surface, only the skeleton. Extract the model again with "
            "`smpl18 extract-model --with-mesh`, or write the stand-in body with "
            "`smpl18 demo-models --with-mesh` to try this without SMPL"
        )
    if model.stand_in:
        print("note: the body is the stand-in from `smpl18 demo-models`, a mannequin of blocks, "
              "not SMPL", file=sys.stderr)

    built = scene_plan.plan_for_trial(
        trial, model,
        correctives=args.correctives,
        sample_frames=int(launch.render_setting(settings, "scene", "check_frames")),
        tolerance=float(launch.render_setting(settings, "scene", "check_tolerance_m")),
        leaf_reach=float(launch.render_setting(settings, "scene", "leaf_reach")),
        frames=_frame_slice(args.frames),
    )
    built.about["render_settings"] = settings_files
    plan_path = built.write(args.out / f"{trial.subject.id}_{trial.id}.plan.npz")
    settings_path = launch.write_render_settings(plan_path.with_suffix(".render.json"), settings)
    print(f"wrote {plan_path}: {built.frames} frames, {built.num_vertices} vertices, "
          f"{'207 pose shape keys' if built.has_correctives else 'no pose blend shapes'}")
    if not built.has_correctives and built.about["pose_blend_shapes_mm"] > 0:
        print(f"note: leaving the pose blend shapes out moves the surface by up to "
              f"{built.about['pose_blend_shapes_mm']:.1f} mm; pass --correctives to carry them in")

    blend = args.out / "scene.blend" if args.blend else None
    frames = args.out / "frames" if args.render else None
    if args.blender is None:
        command = launch.blender_command(
            "blender", plan=plan_path, settings=settings_path,
            blend=blend or args.out / "scene.blend", render=frames,
        )
        print("no --blender given, so nothing was built. Run:\n  " + " ".join(f'"{part}"'
              if " " in part else part for part in command))
        return 0
    if blend is None and frames is None:
        raise CommandError("with --blender, ask for --blend, --render, or both")
    finished = launch.run_blender(args.blender, plan=plan_path, settings=settings_path,
                                  blend=blend, render=frames, quiet=args.quiet)
    for line in (finished.stdout or "").splitlines():
        if line.startswith(("built ", "self-check", "saved ", "rendered ")):
            print(line)
    return 0


def extract_model(args) -> int:
    from smpl18.model.extract import extract_clean

    result = extract_clean(args.pkl, args.out, gender=args.gender, num_betas=args.num_betas,
                           with_mesh=args.with_mesh)
    print(f"wrote {result.path}")
    for key, shape in result.shapes.items():
        print(f"    {key:16s} {shape}")
    return 0


#: Run inside Blender: import the FBX into an empty scene and export its first armature to BVH
#: over the action's frame range, in Blender's frame (Z up) and the scene's units.
BLENDER_SCRIPT = """
import sys
import bpy
source, target = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=source)
armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
if not armatures:
    raise SystemExit("no armature in " + source)
armature = armatures[0]
bpy.context.view_layer.objects.active = armature
armature.select_set(True)
action = armature.animation_data.action if armature.animation_data else None
if action is None:
    raise SystemExit("the armature in " + source + " has no animation")
start, end = (int(round(f)) for f in action.frame_range)
bpy.ops.export_anim.bvh(filepath=target, frame_start=start, frame_end=end)
"""


def blender_command(blender: str, source: Path, target: Path) -> list[str]:
    return [str(blender), "--background", "--factory-startup", "--python-expr", BLENDER_SCRIPT,
            "--", str(source), str(target)]


def fbx2bvh(args) -> int:
    command = blender_command(args.blender, args.input.resolve(), args.out.resolve())
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise CommandError(f"cannot run Blender at {args.blender}: {error}") from error
    if result.returncode != 0 or not args.out.exists():
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-15:])
        raise CommandError(f"Blender did not write {args.out}:\n{tail}")
    print(f"wrote {args.out}. Blender writes Z up (--up-axis z); read one OFFSET line to "
          "choose --length-unit, since the file keeps the FBX's scale")
    return 0


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
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    handlers = {
        "convert": convert,
        "info": info,
        "demo-models": demo_models,
        "extract-model": extract_model,
        "export-smpl": export_smpl,
        "blender": blender,
        "fbx2bvh": fbx2bvh,
    }
    try:
        if args.command in handlers:
            return handlers[args.command](args)
        if args.command == "profile":
            if args.action == "validate":
                return profile_validate(args.profile)
            return profile_show(args.profile)
    except CommandError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (ValueError, LookupError, OSError) as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
