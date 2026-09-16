# smpl24 — design and roadmap

| | |
|---|---|
| Status | in progress: formats, profiles, model and skeleton are in; the fitting, repair and corpus layers are next (§5) |
| Last updated | 2026-09-16 |
| Scope | this document states what the package is for, how it is arranged, and what remains. The skeleton itself is explained in [`primer.md`](primer.md); the on-disk result in [`corpus-format.md`](corpus-format.md); a profile field by field in [`profile-schema.md`](profile-schema.md) |

---

## 0. What this package is for

Motion capture arrives in incompatible shapes: SMPL-family parameter files, articulated skeletons
with joint angles, joint-centre trajectories, raw surface markers. Downstream work wants one
representation. `smpl24` converts any of them into an **SMPL-24 pose corpus**: one body shape per
subject, one axis-angle pose sequence per trial on the 24-joint SMPL skeleton, with a record of
where every joint's motion came from.

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
5. Documentation with a quick start a new user can follow in ten minutes.

**Non-goals**

- Sensor synthesis, dataset-specific bundle formats, and dataset licensing or governance: a
  consumer's concern, not this package's.
- Mesh-level (marker-to-vertex) fitting in 0.1; marker input goes through joint centres first (§4.3).
- Anything that reads or writes body-model files or motion data inside this repository.

---

## 2. Repository shape

```
smpl24/
├── README.md  CHANGELOG.md  LICENSE  pyproject.toml
├── docs/
│   ├── plan.md               this document
│   ├── primer.md             SMPL-24 and how to reach it from SMPL or markers
│   ├── corpus-format.md      the on-disk corpus, versioned
│   ├── profile-schema.md     what a profile may say, field by field
│   └── kinds/                one page per source kind
├── configs/                  DATA, never code
│   ├── profiles/             one file per dataset
│   ├── correspondence/       source joint/body names -> SMPL-24 joints, fill rules, trunk body, lumbar split
│   ├── markersets/           marker labels -> joint-centre rules
│   ├── offsets/              landmark offset tables
│   └── settings/             filter cut-offs, joint-rate limits, IK weights, joint ranges
├── src/smpl24/
│   ├── __init__.py  cli.py
│   ├── model/        extract.py  load.py  select.py
│   ├── skeleton/     definition.py  rotations.py  kinematics.py  frames.py
│   ├── formats/      npz.py  pickle_safe.py  jsonfile.py  osim.py  mot.py  trc.py  c3d.py  b3d.py  bvh.py  mat.py  _vendor/
│   ├── sources/      base.py (the four kind dataclasses)
│   │   ├── parameters/   SMPL-family parameter streams
│   │   ├── skeleton/     articulated-skeleton motion (OpenSim-style and BVH-style FK)
│   │   ├── centres/      joint-centre trajectories
│   │   └── markers/      surface markers -> joint centres
│   ├── profile/      schema.py  load.py  bind.py  layout.py
│   ├── fit/          shape.py  correspondence.py  pose_transfer.py  pose_ik.py  root.py
│   ├── repair/       wrap.py  resample.py  discontinuity.py
│   ├── corpus/       schema.py  write.py  read.py  summary.py
│   ├── validate/     fk_reproduction.py  bone_lengths.py  joint_ranges.py  roundtrip.py
│   └── convert.py    kind -> SMPL-24 pipelines, one per kind, driven by a bound profile
└── tests/            mirrors src/; profile fixtures are tiny YAML files over synthetic data
```

The dependency direction is fixed: `formats` know nothing about kinds; `sources` know nothing
about datasets; `fit`/`repair`/`corpus` know nothing about formats; `profile` binds them; only
`convert` and `cli` see the whole picture. No module imports a profile by name, and a test greps
`src/` for the shipped profile ids so that no dataset name can creep into code.

---

## 3. Design

### 3.1 Source kinds (`smpl24.sources`)

| Kind | Observes | Data | Converter |
|---|---|---|---|
| `smpl_parameters` | SMPL-family `poses`, `betas`, `trans`, gender, fps | per trial: poses `[T, J, 3]` (J = 24/52/55), trans, fps, up axis; per subject: betas, gender | trim to 24 (hands identity), betas to model width, frame change, resample |
| `skeleton_motion` | an articulated skeleton (bodies, joints, parents, rest transforms) and its joint angles per frame | `SkeletonModel` + `[T]` coordinate values → segment world rotations and positions by the model's own forward kinematics | shape from bone lengths; pose by segment-rotation transfer with a calibration alignment |
| `joint_centres` | world positions of anatomical joint centres per frame | `[T, K, 3]` with K named centres, validity mask | shape from bone lengths; pose by position IK; root from the pelvis centre |
| `marker_trajectories` | labelled surface markers per frame | `[T, M, 3]` with labels, validity mask | marker set → joint centres → the `joint_centres` converter |

Skeleton models are themselves generic: an OpenSim-style model (custom joints, spline-coupled
coordinates) and a BVH-style one (offset hierarchy with Euler channels). FBX is not a skeleton
model here: an external Blender step exports BVH, which this package reads.

### 3.2 Formats (`smpl24.formats`)

Readers return **tables with the file's own names** and never interpret them: `npz` and the
whitelisting pickle reader (no code execution) return arrays by key; JSON returns objects; `osim`
returns a skeleton description; `mot`/`sto` return columns; `trc` and `c3d` return labelled
trajectories with rates and units; `b3d` returns the embedded `.osim`, the processing passes and
the subject fields as stored; `bvh` returns hierarchy plus channels; `mat` returns variables. A
format has no idea which dataset it serves, and it converts no units and no frames.

### 3.3 Profiles (`smpl24.profile`)

A profile is the only place a dataset appears. Schema `smpl24_profile_v1`, in outline:

```yaml
schema: smpl24_profile_v1
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
then as a shipped example, then under `$SHARED_DATASET_PATH/smpl24/profiles/`, resolves relative
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

### 3.5 Fit, repair, corpus, validate

- `fit.shape`: bone-length least squares with regularisation and optional stature and volume terms;
  a rest-skeleton rescale that records that it happened; skeleton pooling across trials.
- `fit.correspondence`: names → SMPL joints; the fill rule `weld` / `distribute` / `estimate` per
  missing joint; trunk body; lumbar split; aliases; demotion of joints nothing observed.
- `fit.pose_transfer` (segment rotations with a calibration alignment), `fit.pose_ik` (Kabsch
  initialisation, least squares with angle regularisation and temporal smoothing), `fit.root`.
- `repair.wrap` (unwrap before filtering, only the named coordinates), `repair.resample` (slerp for
  rotations, linear for translation), `repair.discontinuity` (flag, never edit).
- `corpus` writes and reads the format in `corpus-format.md`; every manifest carries the profile id
  and hash, the settings used, the source hashes, the converter version and commit, and the
  provenance codes.
- `validate` reports forward-kinematics reproduction against the source observations, bone-length
  residuals, joint-range coverage, and a round trip for `smpl_parameters` input.

### 3.6 Settings discipline

No numeric limit has a default in library code. Functions take a `settings` mapping resolved by the
profile and fail when a required key is missing; `configs/settings/` ships documented examples. A
threshold that lives in configuration is visible in the corpus manifest afterwards, and two runs
cannot disagree silently about it.

### 3.7 CLI

```
smpl24 extract-model --pkl <file> --gender <g> --out <dir>
smpl24 convert --profile <profile.yaml> --input <root> --models <dir> --out <corpus>
smpl24 convert --kind <kind> --format <fmt> [--correspondence ...] [--markerset ...] --settings ... \
               --input <file-or-root> --models <dir> --out <corpus>      # ad hoc, no profile
smpl24 profile validate <name-or-path>
smpl24 profile show <name-or-path>          # resolved bindings, referenced files and their hashes
smpl24 fbx2bvh --input <clip.fbx> --blender <exe> --out <clip.bvh>
smpl24 info <corpus>
smpl24 validate <corpus> [--against <source>]
```

A subcommand is added only together with the module it fronts, so no command exists without an
implementation.

---

## 4. Decisions

| Decision | Outcome |
|---|---|
| Name | `smpl24`: the skeleton's name, one word for the distribution, the import and the command |
| Licence | MIT (`LICENSE`), covering this code only. SMPL body models and every dataset keep their own licences |
| Repository | private while the work is in progress |
| `.c3d` reading | `ezc3d`, a regular dependency |
| FBX | a Blender bridge (`smpl24 fbx2bvh`); native FBX reading is out of scope |
| Profiles that cannot be published | kept outside this repository and found through `SHARED_DATASET_PATH` |

---

## 5. Roadmap

| Step | State |
|---|---|
| Model handling: clean-model extraction, loading with hashes, selection by gender with no default location | **done** |
| Skeleton: joint definition, rotations and Kabsch, forward kinematics, frame changes | **done** |
| Formats: osim, mot, trc, c3d, b3d, bvh, npz, json, mat, whitelisted pickle | **done** |
| Profiles: schema, loader, layout discovery, parameter binding, `profile validate` / `show` | **done** |
| Sources: skeleton FK models, joint-centre and marker kinds | next |
| Fit: shape, correspondence, pose transfer, position IK, root | next |
| Repair and corpus: wrap, resample, discontinuity; corpus write, read, summary | after fit |
| `convert` and the remaining CLI commands, one per kind | after corpus |
| Round-trip tests per format (synthesise SMPL-24 motion, export, convert back, compare) | with each new format |
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
