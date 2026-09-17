# smpl18 — design and roadmap

| | |
|---|---|
| Status | in progress: every source kind converts file by file (`smpl18 convert ...`) into the corpus; profile-driven dataset conversion is next (§5) |
| Last updated | 2026-09-17 |
| Scope | this document states what the package is for, how it is arranged, and what remains. The skeleton itself is explained in [`primer.md`](primer.md); the on-disk result in [`corpus-format.md`](corpus-format.md); a profile field by field in [`profile-schema.md`](profile-schema.md) |

---

## 0. What this package is for

Motion capture arrives in incompatible shapes: SMPL-family parameter files, articulated skeletons
with joint angles, joint-centre trajectories, raw surface markers. Downstream work wants one
representation. `smpl18` converts any of them into an **18-joint reduced-model pose corpus**: one
body shape per subject, one axis-angle pose sequence per trial, with a record of where every
joint's motion came from.

Every path goes through the full 24-joint SMPL pose first, because that is what the sources and
the body model speak, and is reduced at the end (section 3.6). The reduction freezes four joints
to per-subject constants chosen by optimisation, not by assumption.

One rule shapes the code:

> **Code implements source kinds and file formats. A dataset is a profile.**

A *source kind* is what the source observes. A *format* is how it is stored. A *profile* is a YAML
file that says which kind and format a dataset is, how its files are laid out, which field means
what, its units and frame, its joint correspondence, its repairs and its settings. Adding a dataset
means writing a profile, never a module.

---

## 1. Goals and non-goals

**Goals**

1. Generic source kinds and formats, each implemented once, tested on synthetic fixtures.
2. Datasets as profiles, validated against a schema.
3. One corpus format with per-joint provenance (`measured` / `derived` / `absent`), fit residuals,
   and the profile and settings that produced it.
4. The body model handled honestly: selected per subject by gender, located explicitly, hashed,
   never bundled.
5. Documentation with a quick start a new user can follow in ten minutes, and a runnable
   example for every kind of input.

**Non-goals**

- Sensor synthesis, dataset-specific bundle formats, and dataset licensing or governance: a
  consumer's concern, not this package's.
- Mesh-level (marker-to-vertex) fitting in 0.1; marker input goes through joint centres and
  segment frames first (§3.5).
- Anything that reads or writes body-model files or motion data inside this repository.

---

## 2. Repository shape

```
smpl18/
├── README.md  CHANGELOG.md  LICENSE  pyproject.toml
├── docs/
│   ├── plan.md               this document
│   ├── primer.md             SMPL-24, reaching it from SMPL or markers, the 18-joint reduction
│   ├── corpus-format.md      the on-disk corpus, versioned
│   ├── blender.md            out again: SMPL parameters, and a scene in Blender
│   ├── profile-schema.md     what a profile may say, field by field
│   └── kinds/                one page per source kind
├── configs/                  DATA, never code
│   ├── profiles/             one file per dataset
│   ├── correspondence/       source joint/centre names -> SMPL-24 joints, fill rules, lumbar block
│   ├── markersets/           marker labels -> joint-centre rules and segment frames
│   ├── offsets/              landmark offset tables
│   ├── render/               every number a Blender scene uses
│   └── settings/             every number the engine uses
├── examples/                 one runnable script per kind of input, on synthetic captures
├── src/smpl18/
│   ├── __init__.py  cli.py  convert.py  corpus.py  synthetic.py  mesh.py  original.py
│   ├── blender/     plan.py  launch.py  scene.py (runs inside Blender; imports bpy, not smpl18)
│   ├── model/        extract.py  load.py  select.py  demo.py (the stand-in body)
│   ├── skeleton/     definition.py  rotations.py  kinematics.py  frames.py
│   ├── formats/      npz.py  pickle_safe.py  jsonfile.py  osim.py  mot.py  trc.py  c3d.py  b3d.py  bvh.py  mat.py  _vendor/
│   ├── sources/      base.py (the four kinds)  markers.py (marker sets)  subject.py
│   │   └── skeleton/     opensim.py  bvh.py (forward kinematics of articulated skeletons)
│   ├── fit/          targets.py  correspondence.py  shape.py  pose.py
│   ├── reduce/       fixation.py  fit.py  residual.py
│   └── profile/      schema.py  load.py  bind.py  layout.py
└── tests/            mirrors src/; synthetic fixtures and the stand-in body only
```

The dependency direction is fixed: `formats` know nothing about kinds; `sources` know nothing
about datasets; `fit`/`repair`/`corpus` know nothing about formats; `profile` binds them; only
`convert` and `cli` see the whole picture. `synthetic` serves the examples and tests only. No module imports a profile by name, and a test greps
`src/` for the shipped profile ids so that no dataset name can creep into code.

---

## 3. Design

### 3.1 Source kinds (`smpl18.sources`)

| Kind | Observes | Data | Converter |
|---|---|---|---|
| `smpl_parameters` | SMPL-family `poses`, `betas`, `trans`, gender, fps | per trial: poses `[T, J, 3]` (J = 24/52/55), trans, fps, up axis; per subject: betas, gender | trim to 24 (hands identity), betas to model width, frame change, resample |
| `skeleton_motion` | an articulated skeleton (bodies, joints, parents, rest transforms) and its joint angles per frame | `SkeletonModel` + `[T]` coordinate values → segment world rotations and positions by the model's own forward kinematics | shape from bone lengths; pose by segment-rotation transfer with a calibration alignment |
| `joint_centres` | world positions of anatomical joint centres per frame | `[T, K, 3]` with K named centres, validity mask | shape from bone lengths; pose by position IK; root from the pelvis centre |
| `marker_trajectories` | labelled surface markers per frame | `[T, M, 3]` with labels, validity mask | marker set → joint centres → the `joint_centres` converter |

Skeleton models are themselves generic: an OpenSim-style model (custom joints, spline-coupled
coordinates) and a BVH-style one (offset hierarchy with Euler channels). FBX is not a skeleton
model here: an external Blender step exports BVH, which this package reads.

### 3.2 Formats (`smpl18.formats`)

Readers return **tables with the file's own names** and never interpret them: `npz` and the
whitelisting pickle reader (no code execution) return arrays by key; JSON returns objects; `osim`
returns a skeleton description; `mot`/`sto` return columns; `trc` and `c3d` return labelled
trajectories with rates and units; `b3d` returns the embedded `.osim`, the processing passes and
the subject fields as stored; `bvh` returns hierarchy plus channels; `mat` returns variables. A
format has no idea which dataset it serves, and it converts no units and no frames.

### 3.3 Profiles (`smpl18.profile`)

A profile is the only place a dataset appears. Schema `smpl18_profile_v1`, in outline:

```yaml
schema: smpl18_profile_v1
id: <dataset id>
source_kind: smpl_parameters | skeleton_motion | joint_centres | marker_trajectories
format: npz | pickle | json | osim_mot | b3d | trc | c3d | bvh | mat
layout:                      # how subjects and trials are found under --input
  subject: "<pattern with {study} {subject}>"
  trial:   "<pattern with {trial}>" | within_container
bindings:                    # which field means what, in the file's own names
  poses: ...  betas: ...  trans: ...  fps: {field: ..., fallback: {<group>: <value>}}
  gender: {field: ..., map: {female: female, f: female, male: male, unknown: neutral}}
  root_translation: [<coordinate names>]
  frames: {pass: <name>, fallback: <name>}
conventions: {up_axis: y|z, length_unit: m|mm, angle_unit: rad|deg}
correspondence: ../correspondence/<file>.yaml   # names -> SMPL joints, fill rules, trunk body, lumbar split, aliases
markerset: ../markersets/<file>.yaml            # marker_trajectories only
shape: {method: bone_lengths | parameters, landmark_offsets: <file>|none, rescale_to_measured: ..., pool_per_subject: ...}
root: {placement: pelvis_centre | source_translation | none, recover_frozen: ...}
provenance: {restrict_to_driven: [<source joints>]}
repairs:
  wrap: {coordinates: [...], filter: {order: ..., cutoff_hz: ...}}
  resample: {fps: ...}
  discontinuity: {settings_key: ...}
skip: {trials_without: <pass name>, subjects_with_unresolved_gender: ...}
settings: ../settings/<file>.yaml
```

The loader refuses unknown keys and names the offending path, resolves a profile name as a path,
then as a shipped example, then under `$SHARED_DATASET_PATH/smpl18/profiles/`, resolves relative
references against the profile's own directory, substitutes `${SHARED_DATASET_PATH}` in values,
and hashes every referenced file into the corpus manifests. `bind.py` turns a profile plus format
tables into the kind's data; `layout.py` discovers subjects and trials on disk.

Datasets that cannot be published keep their profiles outside this repository and are found
through `SHARED_DATASET_PATH`, which mirrors the `configs/` layout.

### 3.4 The shipped profiles

| Dataset | kind | format | What the profile carries |
|---|---|---|---|
| AddBiomechanics | `skeleton_motion` | `b3d` | study and subject layout; the `biological_sex` map with `unknown → neutral`; dynamics pass first with the kinematics pass as the fallback; the OpenSim naming table; the trunk body; the lumbar split over spine1/2/3; landmark offsets; wrap repair on the coordinates that wrap, with the filter the source declares; skip a trial without a dynamics pass |
| GAITEX | `skeleton_motion` | `osim_mot` (+ `trc`) | per-trial scaled skeletons pooled per subject; root recovery for the frozen translation the published IK left; gender declared neutral; provenance restricted to the joints the worn sensors drive; joint-range settings |
| AMASS | `smpl_parameters` | `npz` | the first 66 pose parameters plus identity hands; betas truncated to the model width; the `gender` field; `mocap_framerate` with a per-sub-dataset fallback table; Z-up |

None of these needed a Python file. That is the test of the design: if a dataset needs code, the
code belongs to a kind or a format, and the profile only points at it.

### 3.5 Targets, fit, corpus

Every kind but SMPL parameters is first brought to **targets** in the corpus frame
(`smpl18.fit.targets`): positions of SMPL joint centres, and optionally orientations of the
segments SMPL joints move, each column with a weight and a validity mask. A marker set
(`sources.markers`) derives both from markers; a skeleton's forward kinematics
(`sources.skeleton`) and a correspondence table (`fit.correspondence`) derive both from a
skeleton; a correspondence table maps a joint-centre file's labels. Provenance follows from which
joints the targets reach: `measured` when the joint's bone is seen at both ends or its segment is
oriented, `derived` when only something below it is seen, `absent` otherwise (held at identity);
a table's `fill` rules override the last two.

- `fit.shape`: betas from the median lengths of the observed rigid pairs (bones and siblings),
  then refined by alternating a pose solve on sampled frames with a closed-form beta solve against
  every target (each frame's translation eliminated), so the trunk counts too. Both objectives
  are mean squared distances plus one beta prior.
- `fit.pose`: per frame, Levenberg-Marquardt with an analytic Jacobian over the translation, the
  root's world rotation (updated on the group) and a rotation vector per free joint (updated
  additively, which keeps the prior linear). Residuals: joint centres; segment orientations after
  a per-target constant calibrated from positions-only passes over all of the subject's trials; a pose prior, heavier on the four
  joints the corpus freezes (so a turn the targets cannot place goes to the joints that keep it)
  and on the knees' and elbows' off-hinge rotation (which positions cannot see); an optional
  smoothing pass. Frames are solved in batches; one with too few targets holds its neighbour
  and is marked invalid.
- `corpus` writes and reads the format in `corpus-format.md`, rebuilds the 24-joint pose from a
  trial and its subject record, and keeps `SUMMARY.json` in step with the directory. A subject is
  written once with all its trials; writing it again needs `--replace`, and a reader refuses a
  trial its subject record does not list.
- Plausibility checks (`checks` settings) warn about what converts without an error but cannot
  be a person: bones outside a human range (a wrong length unit), a stored pose far from the
  source, a trunk that is seldom upright (a wrong up axis).
- Every manifest carries the source files and hashes, the tables used and what they missed, the
  repairs, the settings, the converter version and commit, and the joint-centre error against the
  source's own targets before and after the reduction.

Still to come with profile-driven conversion: the wrap repair, resampling and the discontinuity
scan (`repair`), and the profile features that use them.

### 3.6 Reduction to 18 joints (`smpl18.reduce`)

The corpus stores 18 joints. `spine1`, `spine2` and both collars are frozen to per-subject
constants; the two hand joints are dropped; `spine3` and the two shoulders absorb the frozen
joints' rotation exactly, so every segment's world orientation is unchanged by the reduction.

What the freeze does change is joint positions, and by how much depends on the constants. They are
therefore **fitted**: twelve parameters (four rotation vectors) by nonlinear least squares, from a
starting guess of each frozen joint's mean rotation, minimising the joint-centre difference between
the reduced and the original pose over the affected joints (spine3, neck, head, shoulders, elbows,
wrists). Frames are subsampled to a count the settings name; the solver, its iteration cap and that
count are settings, never code defaults.

The subject record keeps the constants, which joint absorbed each, the fitted residual and the
starting guess's residual (RMS and maximum, in metres), the per-joint RMS, the frames used and
whether the solver converged. A reader can rebuild the 24-joint pose from that, and can see what
the reduction cost before trusting it. `primer.md` section 5 states the mathematics and says when
the reduction is the wrong thing to do.

### 3.7 Settings discipline

No numeric limit has a default in library code. Functions take a `settings` mapping resolved by the
profile and fail when a required key is missing; `configs/settings/` ships documented examples. A
threshold that lives in configuration is visible in the corpus manifest afterwards, and two runs
cannot disagree silently about it.

### 3.8 CLI

```
smpl18 convert markers  --input <trc/c3d>... --markerset <yaml> --up-axis <x|y|z> ...
smpl18 convert centres  --input <trc/c3d/npz>... --correspondence <yaml> --up-axis <x|y|z> ...
smpl18 convert opensim  --osim <model> --mot <mot/sto>... --correspondence <yaml> ...
smpl18 convert bvh      --input <bvh>... --correspondence <yaml> --up-axis <x|y|z> --length-unit <m|cm|mm> ...
smpl18 convert smpl     --input <npz>... --up-axis <x|y|z> [--fps] ...
    common: --settings <yaml>... --models <dir> --out <corpus>
            (--subject <yaml> | --subject-id <id> --gender <g>) [--measurement NAME=METRES]...
smpl18 extract-model --pkl <file> --gender <g> --num-betas <n> --out <dir> [--with-mesh]
smpl18 demo-models --out <dir> [--with-mesh]
smpl18 info <corpus> [--json]
smpl18 export-smpl --corpus <corpus> [--subject <id>] [--trial <id>]... --out <dir|npz> [--poses flat|grouped]
smpl18 blender --corpus <corpus> --subject <id> [--trial <id>] --out <dir> [--models <dir>]
    [--render-settings <yaml>]... [--correctives] [--frames START:STOP[:STEP]]
    [--blender <exe>] [--blend] [--render]
smpl18 fbx2bvh --input <clip.fbx> --blender <exe> --out <clip.bvh>
smpl18 profile validate <name-or-path>
smpl18 profile show <name-or-path>
```

The `convert` subcommands are named after the data a user has, not after the source kinds,
because that is the question a user starts from; the manifest records the kind. A subcommand is
added only together with the module it fronts, so no command exists without an implementation.

### 3.9 Out again: the surface, SMPL parameters, Blender

A corpus that nothing can read is of no use, and what other tools read is the 24-joint structure.
Two exits, neither of which re-fits anything (`docs/blender.md`):

- **`smpl18.original`** writes a trial as ordinary SMPL parameters. The pose is
  `CorpusTrial.poses_24()`, so the frozen joints carry the subject's constants and the hands
  identity; `joint_provenance` widens from 18 to 24 entries, a frozen joint reading `constant` and a
  hand `absent`, so the file still distinguishes what was observed. `smpl18 convert smpl` reads its
  own output back to the same rotations, which is the round trip the tests check.
- **`smpl18.mesh`** adds the two steps between a pose and a surface that a conversion never needed:
  the pose blend shapes and linear blend skinning. Its `lbs_transforms` is the piece a renderer
  wants — one rigid transform per joint per frame.
- **`smpl18.blender`** splits in two so that nothing has to be true of both interpreters.
  `plan.py` writes one npz holding the rest surface, the weights and those transforms; `launch.py`
  resolves `configs/render/*.yaml` and runs Blender on it; `scene.py` runs *inside* Blender,
  importing `bpy` and nothing from this package.

  The plan carries transforms rather than rotations on purpose. Blender deforms a vertex group by
  `pose.matrix @ bone.matrix_local⁻¹`, so asking for `pose.matrix = A_j @ bone.matrix_local` cancels
  the bone's rest matrix out of the result: no convention about bone directions, rolls or axes has
  to be agreed between the two sides, and the rig can be drawn along the body without a correction.
  A test proves the cancellation over random rest matrices. Since that is a proof about algebra and
  not about Blender, the plan also carries a few frames of vertices computed here and the scene
  compares its own deformation with them, saving nothing if it is off.

---

## 4. Decisions

| Decision | Outcome |
|---|---|
| Name | `smpl18`: the skeleton's name, one word for the distribution, the import and the command |
| Licence | MIT (`LICENSE`), covering this code only. SMPL body models and every dataset keep their own licences |
| Repository | private while the work is in progress |
| `.c3d` reading | `ezc3d`, a regular dependency |
| FBX | a Blender bridge (`smpl18 fbx2bvh`); native FBX reading is out of scope |
| Rendering | Blender, driven as an executable, never imported. The scene is described by a plan of per-joint transforms, so Blender's bone conventions cancel; the script it runs checks its own skinning against `smpl18.mesh` and refuses to save when it differs |
| Profiles that cannot be published | kept outside this repository and found through `SHARED_DATASET_PATH` |

---

## 5. Roadmap

| Step | State |
|---|---|
| Model handling: clean-model extraction, loading with hashes, selection by gender with no default location | **done** |
| Skeleton: joint definition, rotations and Kabsch, forward kinematics, frame changes | **done** |
| Formats: osim, mot, trc, c3d, b3d, bvh, npz, json, mat, whitelisted pickle | **done** |
| Profiles: schema, loader, layout discovery, parameter binding, `profile validate` / `show` | **done** |
| Reduction to 18 joints with the fitted constants (`smpl18.reduce`) | **done** |
| Sources: OpenSim and BVH forward kinematics, marker sets, joint-centre files, subject files | **done** |
| Fit: targets, correspondence tables, shape, per-frame pose with calibrated orientations | **done** |
| Corpus: write, read, summary, 24-joint rebuild | **done** |
| `convert markers / centres / opensim / bvh / smpl`, `info`, `extract-model`, `demo-models`, `fbx2bvh` | **done** |
| Examples for every kind, run as tests; round trips per format on synthetic captures | **done** |
| Out again: `export-smpl` (the 24-joint SMPL structure) and `blender` (a scene, checked against `smpl18.mesh` inside Blender) | **done** |
| Repair: wrap, resample, discontinuity scan | next |
| `convert --profile`: whole datasets through their profiles (layout, bindings, pooling, repairs) | next |
| Validation against real captures with independent joint centres | with the first real corpus |
| 0.1.0: semantic versioning, changelog, release checklist | when a corpus can be produced end to end |

---

## 6. Facts worth keeping in view

- **The frame change needs the rest pelvis.** Turning the up axis is `R_0' = C R_0` and
  `t' = C (j_0 + t) − j_0`, not `t' = C t`: the root rotation turns the body about the pelvis
  `j_0`, while `t` moves the model origin. On a test skeleton the short form landed 0.248 m off;
  the exact form is at 4e-16. See `primer.md` §3.4.
- **A format reader that interprets is a bug.** Frames, units, pass selection and fill policies are
  profile decisions. A reader that quietly turns a Y-up world into Z-up makes two datasets
  indistinguishable in the corpus.
- **The body-model set matters, not one file.** Generators pick per subject by gender, so a corpus
  of mixed-sex subjects came from more than one model file; recording the set and its hashes is the
  honest account.
- **Betas cannot reach every skeleton.** Ten shape coefficients miss some cohorts' hip and shoulder
  separations. Either accept and record the residual, or rescale the rest skeleton and ship the
  scaled skeleton, since the pose was fitted against it.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| A dataset name creeps into code (`if profile.id == ...`) | the grep test over `src/`; anything dataset-specific must be expressible in the schema, or the schema grows |
| The profile schema becomes a second programming language | keep it declarative: enumerated policies, no expressions. A policy that cannot be enumerated is a code feature of a kind |
| A threshold sneaks in as a default | tests assert every settings key is required |
| Model-path convenience returns as a hardcoded path | `Model.for_gender` has no default root, and a test asserts it fails without one |
| `filterwarnings = error` makes a dependency's deprecation fatal | dependency majors are pinned |
