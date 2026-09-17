# smpl18

Turn whatever motion capture you have — **labelled markers only**, **joint kinematics only**,
joint centres, an animation skeleton, or SMPL parameters — into one representation: an
**18-joint reduced SMPL pose corpus**. You get one body shape per subject, one axis-angle pose
sequence per trial, and a record of where every joint's motion came from and how well the
result reproduces the source.

The conversion fits the full 24-joint SMPL skeleton first. It then freezes four joints
(`spine1`, `spine2`, both collars) to per-subject constants and drops the two hand joints. The
constants are **fitted** to minimise the joint-centre error the freeze causes, and that error is
recorded. The world orientation of every stored segment survives the reduction exactly
([`docs/primer.md`](docs/primer.md) §5).

The repository contains no body-model files and no motion data; see
[Body models and licences](#body-models-and-licences).

---

## Which command fits your data

| You have | Command | You also need |
|---|---|---|
| Labelled surface markers (`.trc`, `.c3d`) | `smpl18 convert markers` | a marker set describing your labels, and the subject's measurements |
| OpenSim kinematics (`.osim` + `.mot`/`.sto` from IK) | `smpl18 convert opensim` | a correspondence naming your model's joints (Rajagopal/gait2392 ships) |
| An animation skeleton (`.bvh`; `.fbx` through Blender) | `smpl18 convert bvh` | a correspondence naming your rig's joints (a humanoid one ships), the file's length unit and up axis |
| Joint-centre trajectories (`.trc`, `.c3d`, `.npz`) | `smpl18 convert centres` | a correspondence naming your centre labels (a common set ships) |
| SMPL / SMPL-H parameters (`.npz`) | `smpl18 convert smpl` | nothing else; nothing is fitted (SMPL-X: see [below](#smpl-parameters)) |

Every command takes the same subject, settings, model and output options. One invocation
converts one subject, and its trials are fitted together: give all of a subject's trials at
once. Several invocations can write different subjects into the same corpus; a subject that is
already there is refused unless `--replace` is given. [`examples/`](examples/README.md) has a
runnable script for each row.

---

## Quick start

### 1. Install

Python 3.12 or newer.

```bash
git clone https://github.com/JeongHyunho/smpl18.git
cd smpl18
python -m pip install -e ".[dev]"
smpl18 --version
```

### 2. Get a body model

SMPL is licensed by the Max Planck Institute and is **not redistributed here**. Download the
`.pkl` models from the SMPL website under your own licence, then extract the arrays this package
needs into a licence-free `.npz` per gender. No pickle code runs while reading them.

```bash
smpl18 extract-model --pkl path/to/basicmodel_m_lbs_10_207_0_v1.1.0.pkl --gender male --num-betas 10 --out path/to/smpl18-models
smpl18 extract-model --pkl path/to/basicmodel_f_lbs_10_207_0_v1.1.0.pkl --gender female --num-betas 10 --out path/to/smpl18-models
smpl18 extract-model --pkl path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl --gender neutral --num-betas 10 --out path/to/smpl18-models
```

Point the converters at that directory with `--models`, or set `SMPL18_MODELS` once. There is no
default location.

**Just trying it out?** `smpl18 demo-models --out demo-models` writes a *stand-in body*. It has
the SMPL-24 tree and ordinary adult proportions, so every command runs without the licensed
files. It is not a human shape model, and a corpus made with it says so.

### 3. Try the examples

```bash
python examples/01_markers_to_smpl18.py     # markers only
python examples/02_opensim_to_smpl18.py     # OpenSim kinematics only
python examples/03_bvh_to_smpl18.py         # BVH
python examples/04_joint_centres_to_smpl18.py
python examples/05_smpl_parameters_to_smpl18.py
python examples/06_read_the_corpus.py example-output/01_markers/corpus
```

Each example writes synthetic input files and prints the exact `smpl18` command it runs. It then
reports how far the stored poses are from the motion the files were made from.

---

## Converting your data

The options below are shared by every `convert` command.

| Option | Meaning |
|---|---|
| `--out DIR` | the corpus directory; created or added to |
| `--settings FILE` | the engine settings; [`configs/settings/default.yaml`](configs/settings/default.yaml) is a documented starting point. Repeat the option to layer a file of your own over it |
| `--models DIR` | the extracted model directory (or `SMPL18_MODELS`) |
| `--subject FILE` | a subject file: id, gender, measurements (see below) |
| `--subject-id ID --gender G` | instead of, or overriding, the file; `G` is `male`, `female` or `neutral` and selects the body model |
| `--measurement NAME=METRES` | a subject measurement, repeatable, overriding the file |
| `--quiet` | print only the summary |
| `--replace` | convert a subject that is already in the corpus anew; its earlier trials are removed, because they were fitted with the old shape and constants |

A **subject file** is a small YAML file:

```yaml
schema: smpl18_subject_v1
id: S01
gender: female
measurements:            # metres; only marker sets read them
  marker_radius: 0.007
  leg_length_left: 0.86
  leg_length_right: 0.86
  knee_width_left: 0.10
  knee_width_right: 0.10
  ankle_width_left: 0.07
  ankle_width_right: 0.07
  elbow_width_left: 0.065
  elbow_width_right: 0.065
  shoulder_offset_left: 0.035
  shoulder_offset_right: 0.035
```

### Markers only

```bash
smpl18 convert markers \
    --input S01/walk01.c3d S01/walk02.c3d \
    --markerset configs/markersets/conventional_full_body.yaml \
    --up-axis z \
    --subject S01.yaml \
    --settings configs/settings/default.yaml \
    --models path/to/smpl18-models --out corpus
```

- **Marker set.** It says how each joint centre follows from your labels.
  [`conventional_full_body.yaml`](configs/markersets/conventional_full_body.yaml) covers the
  Plug-in Gait-style full-body labels (`LASI … RFIN`):
  - hip centres by the Davis regression;
  - knees and ankles by the chord through the thigh and shank wands;
  - elbows on the flexion axis through the epicondyle marker;
  - wrists between the wrist-bar markers.

  It also builds segment frames (pelvis, thorax, head, thighs, shanks, feet, arms, hands), which
  steer each segment's twist. An elbow straighter than 10° has no well-defined flexion axis, so
  that frame goes without an elbow centre (a static trial with straight arms has none). The
  upper-arm marker must sit directly laterally on the arm, not towards its back, and further
  out than the line from the shoulder centre to the elbow marker, or the elbow rule cannot
  tell the joint centre from its mirror image; a trial where the elbow centre does not hold
  still on the upper arm is flagged. For another protocol, copy the file and edit the labels and rules
  ([Writing your own tables](#writing-your-own-tables)).
- **Measurements.** The marker set declares which ones it needs. A missing one is an error that
  names it.
- **Up axis.** Give the files' vertical axis; `.trc` and `.c3d` do not state it. Length units are
  read from the file (`mm`, `cm`, `m`).
- **Gaps.** Short interior gaps are bridged linearly (`markers.max_gap_frames` in the settings).
  A frame with too few joint centres left is marked invalid and holds its nearest solved pose.
  `--occlusion-sentinel zero` treats exact `(0, 0, 0)` samples as lost markers, for writers that
  park them at the origin.

### Kinematics only (OpenSim)

```bash
smpl18 convert opensim \
    --osim S01/scaled_model.osim \
    --mot S01/walk01_ik.mot S01/walk02_ik.mot \
    --correspondence configs/correspondence/opensim_rajagopal.yaml \
    --subject-id S01 --gender female \
    --settings configs/settings/default.yaml \
    --models path/to/smpl18-models --out corpus
```

- **Forward kinematics.** The model's own FK places every body and joint centre from the
  coordinates:
  - OpenSim 4 custom, pin, slider, universal, gimbal, ball, free, planar and weld joints;
  - spline and linear coordinate functions;
  - coupled coordinates.

  An OpenSim 3 model must be opened and saved once in OpenSim 4. The coordinate file must hold
  every independent coordinate; coupled ones follow their couplers.
- **Up axis.** It comes from the model's gravity. Degrees or radians come from the file's
  `inDegrees` header; `--angle-unit` covers a file that does not say.
- **Correspondence.** The table names which model joint gives each SMPL joint its centre and
  which body gives it its rotation. The shipped
  [`opensim_rajagopal.yaml`](configs/correspondence/opensim_rajagopal.yaml) fits the Rajagopal
  and gait2392/gait2354 families (the latter through an alias for their knee names). A joint
  your model lacks is reported and left unobserved.

### Animation skeleton (BVH, FBX)

```bash
smpl18 convert bvh \
    --input clip01.bvh clip02.bvh \
    --correspondence configs/correspondence/bvh_humanoid.yaml \
    --up-axis y --length-unit cm \
    --subject-id actor01 --gender male \
    --settings configs/settings/default.yaml \
    --models path/to/smpl18-models --out corpus
```

**The file's frame and units.** BVH files do not state their length unit or up axis, so both are
required.

**Joint names.** [`bvh_humanoid.yaml`](configs/correspondence/bvh_humanoid.yaml) matches the
common humanoid names (`Hips`, `Spine`, `LeftUpLeg`, `LeftArm`, …) and strips namespace prefixes
such as `mixamorig:`. Add `aliases` for a rig that spells a joint differently.

**What FBX is, and why it goes through Blender.** FBX is Autodesk's binary format for 3D scenes
and skeletal animation. Reading it needs the proprietary SDK, but Blender (free) imports FBX and
exports BVH:

```bash
smpl18 fbx2bvh --input clip.fbx --out clip.bvh --blender path/to/blender
smpl18 convert bvh --input clip.bvh --up-axis z --length-unit <unit> ...
```

Blender writes its own Z-up frame. The offsets keep the scale the FBX was imported at, so read
one `OFFSET` line (a thigh is about 0.4 m) before choosing `--length-unit`. A wrong choice is
reported, as below.

### Joint centres

```bash
smpl18 convert centres \
    --input S01/walk01_centres.trc \
    --correspondence configs/correspondence/joint_centre_labels.yaml \
    --up-axis z --subject-id S01 --gender male \
    --settings configs/settings/default.yaml \
    --models path/to/smpl18-models --out corpus
```

- **Input files.** A `.trc` or `.c3d` whose "markers" are joint centres works as it is.
- **The `.npz` form.** An `.npz` holds `names` [K], `positions` [T, K, 3], `fps` and `units`. It
  may also hold segment rotations (`segment_names` [M], `segment_rotations` [T, M, 3, 3]), which
  a table entry names with `segment:`.
- **Placement differences.** Lower the weight of centres your software places differently from
  SMPL (trunk, neck, head); the shipped table does.

### SMPL parameters

```bash
smpl18 convert smpl --input seq01_poses.npz seq02_poses.npz --up-axis z \
    --subject-id 50002 --gender male \
    --settings configs/settings/default.yaml --models path/to/smpl18-models --out corpus
```

- **What is kept.** The 22 body joints are kept and the hands are dropped. `betas` are taken as
  stored, cut to the model's width. A frame with a non-finite pose or translation is marked
  invalid and holds its nearest valid frame.
- **Frame and rate.** The up axis is changed and the frame rate is read from
  `mocap_framerate`/`fps` (or `--fps`).
- **Array names.** `--poses-key`, `--trans-key` and `--betas-key` rename the arrays.
- **SMPL-X is refused.** Its betas and rest pelvis belong to another template, so they cannot
  be carried onto SMPL. Compute an SMPL-X sequence's joint positions with the SMPL-X model and
  convert them with `smpl18 convert centres`.

### Settings

Every number the engine uses is in the settings file, with a comment saying what it does;
nothing numeric is defaulted in code. The sections the converters read:

| Section | What it sets |
|---|---|
| `output` | the corpus's up axis |
| `shape` | how many betas, their prior, the refinement rounds and sample |
| `pose` | the per-frame solve: iterations and tolerance; the weights of positions, orientations and the pose prior (including the heavier prior on the four joints the corpus freezes and on the knees' and elbows' off-axis turn); optional smoothing; frames per batch; the least number of joint centres a frame needs |
| `markers` | how long a marker gap may be bridged |
| `checks` | the plausibility warnings: a bone-length range, a ceiling on the stored pose's distance from the source, and how upright the trunk should mostly be |
| `reduce` | the 18-joint reduction's fit |

The values used are copied into every trial's manifest.

**Plausibility warnings.** A wrong length unit or up axis converts without an error but gives a
body no one has: bones metres long, or a trunk lying on its back. After writing, each command
prints a `warning:` when measured bones fall outside a human range, when the stored pose is far
from the source's joint centres, or when the trunk is seldom upright. The warnings are also
recorded (`checks` in the subject record and each manifest) and shown by `smpl18 info`. A trial
genuinely spent lying down trips the last check too.

---

## What you get

```
corpus/
  SUMMARY.json               counts, converter version, source kinds, provenance counts
  S01/
    subject.json             gender, model file + sha256, betas, shape-fit residuals, the four
                             frozen constants and what the freeze cost
    walk01.npz               poses [T,18,3], joint_names [18], trans [T,3], fps, up_axis,
                             joint_provenance [18], frame_valid [T]
    walk01.manifest.json     source files + sha256, tables used, repairs, settings, validation
```

`joint_provenance` says, per stored joint:
- `measured`: the source observed it;
- `derived`: its turn was shared out to it;
- `absent`: nothing observed it, so it is held at rest.

Each manifest's `validation` gives the joint-centre error against the source's own targets,
before and after the reduction. The full definition is in
[`docs/corpus-format.md`](docs/corpus-format.md).

```bash
smpl18 info corpus           # per subject and trial: frames, residuals, the freeze's cost
```

Reading it in Python:

```python
from smpl18.corpus import read_corpus
from smpl18.model import Model

corpus = read_corpus("corpus")
subject = corpus.subject("S01")
trial = subject.trial("walk01")
trial.poses                    # (T, 18, 3) axis-angle, in trial.joint_names order
local = trial.local_rotations_24()          # (T, 24, 3, 3): frozen joints at their constants
model = Model.for_gender(subject.gender, root="path/to/smpl18-models")
joints = trial.joints_world(model)          # (T, 24, 3) world joint centres
```

The same pipeline is available without the command line (`smpl18.convert`):

```python
from smpl18 import convert
from smpl18.model import Model
from smpl18.sources.markers import MarkerSet
from smpl18.sources.subject import SubjectInfo

settings, settings_files = convert.load_settings(["configs/settings/default.yaml"])
subject = SubjectInfo.load("S01.yaml")
model = Model.for_gender(subject.gender, root="path/to/smpl18-models")
markerset = MarkerSet.load("configs/markersets/conventional_full_body.yaml")
trials = [convert.marker_trial(path, markerset=markerset, subject=subject, up_axis="z",
                               settings=settings)
          for path in ["S01/walk01.c3d", "S01/walk02.c3d"]]
fit = convert.fit_subject(model, trials, settings)
convert.write_subject_corpus("corpus", subject=subject, model=model, trials=trials, fit=fit,
                             settings=settings, settings_files=settings_files)
```

---

## Out again: plain SMPL, and Blender

The corpus stores 18 joints. Everything else that reads bodies reads the original 24-joint SMPL
structure, so there are two ways back out, and neither re-fits anything —
[`docs/blender.md`](docs/blender.md) is the full guide.

**SMPL parameters**, for any SMPL reader:

```bash
smpl18 export-smpl --corpus corpus --out smpl/
```

`poses [T, 72]` axis-angle, `betas`, `trans`, `mocap_framerate`, `gender`, with the four frozen
joints at the subject's constants and the hands at identity. `joint_provenance` comes along, now
over all 24 joints (`constant` for a frozen one, `absent` for a hand), so what was observed is
still distinguishable from what was inferred. Reading such a file back with `smpl18 convert smpl`
returns the same rotations.

**A body moving in Blender**:

```bash
smpl18 extract-model --pkl basicmodel_neutral_*.pkl --gender neutral --num-betas 10 \
    --out smpl18-models --with-mesh          # the weights and pose blend shapes a render needs
smpl18 blender --corpus corpus --subject S01 --trial walk01 \
    --models smpl18-models --out scene/ --blend --render \
    --blender "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
```

You get an ordinary Blender rig: the shaped surface skinned to 24 bones, keyframed, with a camera,
a sun and a floor — plus a `.blend` and PNG frames if you asked for them. Without `--blender` the
scene plan is written and the command to run is printed, which is what to do on a machine that has
no Blender. Without any SMPL model, `smpl18 demo-models --with-mesh` writes a mannequin of blocks
so the whole path runs anyway.

Two things worth knowing:

- **The bone convention cannot be got wrong.** The plan carries each joint's rest-to-posed
  transform, not rotations to interpret; Blender's own rest matrices cancel out of its skinning, so
  nothing depends on how the bones are drawn. The scene then *checks itself* against vertices
  computed here and refuses to save if it is off by more than a fifth of a millimetre.
- **Pose blend shapes** (SMPL's 207 `posedirs`) are left out by default, and the command prints how
  far that moves the surface; `--correctives` carries them in as shape keys.

Every number a render uses lives in [`configs/render/default.yaml`](configs/render/default.yaml),
as with conversion settings; nothing is defaulted in code.

---

## How it works

1. **Targets.** Every source is brought to the same form in the corpus frame:
   - **positions** of SMPL joint centres;
   - optionally **orientations** of the segments SMPL joints move.

   A marker set derives both from markers. A skeleton's forward kinematics gives both.
   Joint-centre files give positions, and orientations if they store segment rotations.
2. **Shape.**
   - Betas are fitted to the bone lengths the targets show (median over frames).
   - They are then refined against every target, alternating with pose solves on a sample of
     frames, so that the trunk and other chains count too.
   - All of a subject's trials are pooled.
3. **Pose.** Every frame is solved by Levenberg–Marquardt with an analytic Jacobian:
   - joint centres to their targets;
   - segment orientations to theirs, after a per-segment constant that a first positions-only
     pass calibrates;
   - a pose prior that holds unobserved twist at rest. The prior is heavier on the joints the
     corpus will freeze, and on the knees' and elbows' rotation off their hinge axis.

   The constants between source and SMPL segment frames are calibrated once per subject, on
   all its trials, so a segment seen only through its own frame (a head, a hand) keeps one
   neutral twist across them. That assumes the markers stay where they were put: convert
   sessions with re-applied markers as separate subjects.
4. **Reduction.**
   - The four frozen joints get the per-subject constants that least displace the joints below
     them, fitted on the subject's pooled poses.
   - `spine3` and the shoulders absorb what was removed.
   - The hands are dropped.

[`docs/primer.md`](docs/primer.md) explains the skeleton and each step;
[`docs/plan.md`](docs/plan.md) explains the design.

---

## Writing your own tables

Tables are data, validated when loaded (unknown keys are refused).

- **Marker set** (`smpl18_markerset_v1`, [`configs/markersets/`](configs/markersets/)). It
  declares:
  - `measurements` the subject must supply;
  - `lengths` measured from markers;
  - `frames` built from two directions each;
  - `centres`, each by one rule:
    - `point`: a marker or a mean of markers;
    - `offset`: in a frame, each coordinate a linear combination of lengths and measurements;
    - `chord`: the wand construction;
    - `hinge`: on a flexion axis through a lateral marker;
  - `segments`, which frame orients which SMPL joint.

  A point may be a label, a list of labels (their mean), or `{centre: <joint>}`. A `hinge`
  also names `min_flexion_deg`, below which the frame goes without that centre, and
  `drift_tolerance`, how far the centre may move on the proximal segment.
- **Correspondence** (`smpl18_correspondence_v1`,
  [`configs/correspondence/`](configs/correspondence/)).
  - `names: opensim_joints` / `bvh_joints`: an entry lists source joints. The first gives the
    position, the body the last one moves gives the orientation; `position: false` or
    `orientation: false` drops either.
  - `names: centres`: an entry names a `centre` and optionally a `segment`.
  - For every table: `fill: weld | distribute` for joints without a source, `weight` per entry,
    `prefixes` to strip and `aliases` (`{name in the file: name in the table}`) for naming
    variants, and a `lumbar` block for a single trunk body spanning SMPL's three spine joints.

---

## Dataset profiles

Whole public datasets are described by **profiles**, YAML files that bind a source kind and a
file format to a dataset's layout, field names, units and repairs. The code never names a
dataset. Example profiles ship for AddBiomechanics, GAITEX and AMASS in
[`configs/profiles/`](configs/profiles/). A profile that cannot be published is found through
the `SHARED_DATASET_PATH` environment variable.

```bash
smpl18 profile validate configs/profiles/gaitex.yaml   # schema check, referenced files hashed
smpl18 profile show amass                              # the resolved profile
```

Converting a whole dataset directly from its profile (`smpl18 convert --profile …`) is on the
roadmap ([`docs/plan.md`](docs/plan.md) §5). Today, convert a dataset's subjects with the
commands above. The schema is in [`docs/profile-schema.md`](docs/profile-schema.md).

---

## Body models and licences

- SMPL body models are licensed by MPI and must be obtained by each user. This repository never
  contains them, and the extracted `.npz` files must not be committed or shared.
- Motion data is never stored in this repository either. Converters read from paths you give
  them and write only to `--out`. The examples and tests make their own synthetic inputs.
- The code is under the MIT licence (`LICENSE`). SMPL and every dataset keep their own licences.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check src tests examples
```

The tests use synthetic fixtures and the stand-in body; nothing needs the licensed models or
motion data. They also run every example on a few frames, so the examples stay in step with the
code. A test searches `src/` for the shipped dataset ids, so that no dataset name can creep into
code.
