# smpl24 — founding plan

| | |
|---|---|
| Written | 2026-09-15, in the SOMA Synthetic IMU repository, lane `worktree-smpl24-primer`; revised the same day for the "generic source kinds, datasets as configuration" principle |
| Status | **Seed.** Phase 0 is in this change set; the later phases are proposals for the owner |
| Decisions already taken by the owner | separate repository mounted as a git submodule of the SOMA project; first release covers SMPL-family parameters, OpenSim skeletons (`.osim`+`.mot`, `.b3d`), marker trajectories (`.trc`/`.c3d`) and BVH/FBX; the existing code **moves** here and the SOMA project depends on this package (single source); **no code path is written for a particular dataset** — the package implements source kinds and file formats, and a dataset is a configuration (profile) that binds them |
| Language | English is the default for every document and message in this repository |
| Authority | none over the SOMA project. This plan releases no hold, changes no contract, and authorises no generation there |

---

## 0. Summary

The SOMA Synthetic IMU project already contains a working retarget engine
(`src/soma_synthetic_imu/addbio_retarget/`, ten modules, 172 unit tests), a marker-centre
retarget used for one cohort, and two copies of the SMPL model loader and forward kinematics.
The engine is generic in its mathematics but dataset-shaped in its surface: modules are named
after the dataset they were first written for, OpenSim body names and an AddBiomechanics landmark
table sit in code, the trunk body is a constant, the body-model path is hardcoded twice, and each
new dataset so far has meant a new driver script.

This plan carves that engine out into **`smpl24`**, an installable package built on one rule:

> **Code implements source kinds and file formats. A dataset is a profile.**

A *source kind* is what the source observes (SMPL parameters, an articulated skeleton's joint
angles, joint-centre trajectories, surface markers). A *format* is how it is stored (npz, pickle,
`.osim`/`.mot`, `.b3d`, `.trc`, `.c3d`, `.bvh`, `.mat`). A *profile* is a YAML file that says
which kind and format a dataset is, how its files are laid out, which field means what, its
units and frame, its joint correspondence, its repairs, and its settings. AddBiomechanics,
GAITEX, PRISM, AMASS and HKNU become five profiles over four kinds and seven formats; a sixth
dataset is a sixth profile, not a sixth driver.

The SOMA project keeps everything that is about *its* artifacts (attribution, holds, artifact
classes, unified8 bundles, sensor synthesis) and consumes `smpl24` as a pinned submodule. The
move happens in phases, each gated by parity: converting the same sample through the profile must
reproduce the corpus the project generates today.

---

## 1. What exists today (inventory, 2026-09-15)

### 1.1 The engine, classified by what it really is

| Today (SOMA project) | What it really is | Destination in `smpl24` |
|---|---|---|
| `addbio_retarget/osim_topology.py` | `.osim` XML reader (bodies, joints, parents, coordinates) | `formats/osim.py` |
| `addbio_retarget/osim_kinematics.py` | forward kinematics of an OpenSim-style skeleton (custom joints, spline-coupled coordinates) without OpenSim | `sources/skeleton/opensim_fk.py` — a *skeleton model* implementation, dataset-free |
| `addbio_retarget/b3d_frames.py` + vendored `SubjectOnDisk_pb2` | `.b3d` container reader (embedded `.osim` + passes + subject fields) | `formats/b3d.py` (+ `formats/_vendor/`) |
| `gaitex_retarget/gaitex_frames.py` | `.mot` reader, `.trc` reader, and a *frozen-root recovery* mechanism | `formats/mot.py`, `formats/trc.py`; the recovery becomes `sources/skeleton/root_recovery.py` switched on by profile |
| `addbio_retarget/smpl_correspondence.py` | SMPL-24 names/parents **plus** an OpenSim joint-name table and a root-translation coordinate list | skeleton definition → `skeleton/definition.py`; the naming table → **data**, `configs/correspondence/opensim_*.yaml` |
| `addbio_retarget/shape_fit.py` | betas from bone lengths, rest-joint rescale, stature/volume terms, model selection by gender, **and** `LANDMARK_OFFSETS_V1` (an AddBiomechanics-vs-SMPL landmark correction) | `fit/shape.py`, `model/select.py`; the landmark table → **data**, `configs/offsets/`, referenced by a profile |
| `addbio_retarget/pose_fit.py` | segment-rotation transfer `G_smpl[j] = R_world · G_source[body(j)] · A[j]`, root placement, **and** `_TRUNK_BODY = "torso"`, `_LUMBAR_SMPL = (3,6,9)` | `fit/pose_transfer.py`, `fit/root.py`; trunk body and lumbar split → **correspondence data** |
| `addbio_retarget/world_frame.py` | change of up axis | `skeleton/frames.py` (any pair of axes) |
| `addbio_retarget/wrap_repair.py` | unwrap-then-refilter named coordinates | `repair/wrap.py`; which coordinates and which filter → **profile** |
| `gaitex_retarget/shape.py` | pool one body from per-trial scaled skeletons | `fit/shape.py` (`pool_skeletons`), switched on by profile |
| `gaitex_retarget/provenance.py` | demote joints nothing observed | `fit/correspondence.py` (`restrict_to_driven`); the driven set → **profile** |
| `gaitex_retarget/joint_ranges.py` | fraction of a trial outside declared joint ranges | `validate/joint_ranges.py`; the ranges → **settings** |
| `kinematics/quaternion.py`, `kinematics/rigid_body.py` | rotation algebra, Kabsch with reflection correction | `skeleton/rotations.py` |
| `scripts/poc/anthro_smpl.py` (`rest_joints`, FK, batch FK, model path by gender) | SMPL rest skeleton and FK | `skeleton/kinematics.py`, `model/` |
| `scripts/poc/smpl_model.py`, `prepare_smpl_clean_npz.py` | two fail-closed pkl → clean-npz extractors | merged into `model/extract.py` |
| `scripts/diagnostics/hknu_retarget_probe.py` (`centres_from`, `rescale_rest_to_measured`, `root_placement_from`, `CORRESPONDENCE`, `TRUNK_ALIAS`) | the *joint-centres* source kind: centres from a Visual3D `.mat`, shape by measured bone lengths, root from the pelvis centre; plus a naming table and an alias | `formats/mat.py`, `sources/centres/`, `fit/shape.py`, `fit/root.py`; the table and alias → **data** |
| `scripts/poc/generate_amass_faithful.py` (`build_pose24`, `resample_trans`, per-joint slerp, per-sub-dataset fps fallback) | the *SMPL-parameters* source kind from npz; a fps fallback table | `sources/parameters/`, `repair/resample.py`; the fps table → **profile** |
| `scripts/poc/generate_prism_faithful.py` (parameter reading from PRISM pickles, gender from `subj_info`) | the *SMPL-parameters* source kind from a pickle container | `formats/pickle.py` (whitelisting unpickler), bindings → **profile** |
| `scripts/poc/generate_addbio_smpl24.py`, `generate_gaitex_smpl24.py` (six-step drivers, study/subject/trial discovery, skip rules) | orchestration + layout + skip policy | `cli.py convert --profile`; layout and skip rules → **profile** |
| tests: `tests/addbio_retarget/*` (172), the GAITEX reader/provenance/shape/joint-range/cluster-pose tests, `tests/poc/test_{anthro_smpl,smpl_model,prepare_smpl_clean_npz}.py` | | move with their code; dataset-specific expectations become profile-driven fixtures |

### 1.2 What stays in the SOMA project

- `addbio_retarget/attribution.py` (closed study table, licence fields), `artifact.py`
  (`spec_id = addbio_smpl24_raw`), artifact classes, holds, the experimental catalog: these
  describe the project's artifacts, not the conversion.
- Unified8 bundle emitters, sensor synthesis (`sensors/`, `gaitex_synthesis/`, `virtual_imu`),
  the anthro deliverable assembly (`compute_anthro`), the validator, the pipeline runner.
- `source_parsing/adapters/addbiomechanics.py` (the audit-lane B3D decoder coupled to governance).
- The project's own profiles for internal datasets (PRISM, HKNU) may live in the project's
  `configs/` and be passed by path; public-dataset profiles ship as examples in `smpl24`.

### 1.3 Facts that shape the design

- **One shared call.** All five SOMA generators reach the body model through
  `clean_model_path_for_gender`. `Model.for_gender` preserves it.
- **Model location is hardcoded twice** with different environment overrides. `smpl24` has one
  resolver (`--models` / `SMPL24_MODELS`) and no default path.
- **No thresholds hide in code.** The project forbids code-default tolerances. `smpl24` keeps the
  rule: numeric limits come from settings referenced by the profile, and are copied into every
  trial manifest.
- **No nimblephysics, no OpenSim dependency**: both are reimplemented in numpy/scipy. `smpl24`
  inherits that: numpy, scipy, protobuf, pyyaml.
- **No BVH/FBX code exists** (new work, phase 4).
- **`code_hash()` in the drivers hashes nine files by repository path** — replaced by the
  installed version plus the submodule commit, recorded by the project's run record.

---

## 2. Goals and non-goals

**Goals**

1. Generic source kinds and formats, each implemented once, tested on synthetic fixtures.
2. Datasets as profiles: AddBiomechanics, GAITEX, PRISM, AMASS and HKNU expressed entirely as
   YAML over the generic code, validated against a schema. Adding a dataset never adds a module.
3. One corpus format with per-joint provenance (`measured` / `derived` / `absent`), fit
   residuals, and the profile and settings that produced it.
4. Bit-for-bit reproduction of the corpora the SOMA project generates today, through profiles,
   before any behaviour changes.
5. The body model handled honestly: per subject by gender, located explicitly, hashed, never bundled.
6. English documentation with a quick start a new user can follow in ten minutes.

**Non-goals**

- Sensor synthesis, bundle formats, dataset attribution, governance holds (SOMA project).
- Mesh-level (marker-to-vertex) fitting in 0.1; markers go through joint centres first (§4.3).
- A public release; INTERNAL-ONLY until the owner decides otherwise (§7).

---

## 3. Repository shape

```
smpl24/
├── README.md  CHANGELOG.md  LICENSE (owner decision)  pyproject.toml
├── docs/
│   ├── plan.md               this document
│   ├── primer.md             SMPL-24 and how to reach it from SMPL or markers
│   ├── corpus-format.md      the on-disk corpus, versioned
│   ├── profile-schema.md     what a profile may say, field by field
│   └── kinds/                one page per source kind (parameters, skeleton, centres, markers)
├── configs/                  DATA, never code
│   ├── profiles/             addbiomechanics.yaml  gaitex.yaml  amass.yaml  prism.yaml  hknu.yaml  (examples)
│   ├── correspondence/       source joint/body names -> SMPL-24 joints, fill rules, trunk body, lumbar split
│   ├── markersets/           marker labels -> joint-centre rules
│   ├── offsets/              landmark offset tables (e.g. the AddBiomechanics-vs-SMPL table)
│   └── settings/             filter cut-offs, joint-rate limits, IK weights, joint ranges
├── src/smpl24/
│   ├── __init__.py  cli.py
│   ├── model/        extract.py  load.py  select.py
│   ├── skeleton/     definition.py  rotations.py  kinematics.py  frames.py
│   ├── formats/      npz.py  pickle.py  json.py  osim.py  mot.py  trc.py  c3d.py  b3d.py  bvh.py  mat.py  _vendor/
│   ├── sources/      base.py (the four kind dataclasses)
│   │   ├── parameters/   from_tables.py                       SMPL-family parameter streams
│   │   ├── skeleton/     model.py  opensim_fk.py  bvh_fk.py  root_recovery.py   articulated-skeleton motion
│   │   ├── centres/      from_tables.py                       joint-centre trajectories
│   │   └── markers/      conditioning.py  centres.py          surface markers -> joint centres
│   ├── profile/      schema.py  load.py  bind.py  layout.py   profile parsing, validation, field binding, file discovery
│   ├── fit/          shape.py  correspondence.py  pose_transfer.py  pose_ik.py  root.py
│   ├── repair/       wrap.py  resample.py  discontinuity.py
│   ├── corpus/       schema.py  write.py  read.py  summary.py
│   ├── validate/     fk_reproduction.py  bone_lengths.py  joint_ranges.py  roundtrip.py
│   └── convert.py    kind -> SMPL-24 pipelines, one per kind, driven by a bound profile
└── tests/            mirrors src/; profile fixtures are tiny YAML files over synthetic data
```

The dependency direction is fixed: `formats` know nothing about kinds; `sources` know nothing
about datasets; `fit`/`repair`/`corpus` know nothing about formats; `profile` binds them; only
`convert` and `cli` see the whole picture. No module imports a profile by name.

---

## 4. Design

### 4.1 Source kinds (`smpl24.sources`)

| Kind | Observes | Dataclass fields | Converter |
|---|---|---|---|
| `smpl_parameters` | SMPL-family `poses`, `betas`, `trans`, gender, fps | per trial: poses `[T, J, 3]` (J = 24/52/55), trans, fps, up axis; per subject: betas, gender | trim to 24 (hands identity), betas to model width, frame change, resample |
| `skeleton_motion` | an articulated skeleton (bodies, joints, parents, rest transforms) and its joint angles per frame | `SkeletonModel` + `[T]` coordinate values → segment world rotations/positions by the model's FK | shape from FK bone lengths; pose by segment-rotation transfer with a calibration alignment |
| `joint_centres` | world positions of anatomical joint centres per frame | `[T, K, 3]` with K named centres, validity mask | shape from bone lengths; pose by position IK; root from the pelvis centre |
| `marker_trajectories` | labelled surface markers per frame | `[T, M, 3]` with labels, validity mask | marker set → joint centres → the `joint_centres` converter |

Skeleton models are themselves generic: `opensim_fk` (custom joints, coupled coordinates, as
today's `osim_kinematics`) and `bvh_fk` (offset hierarchy with Euler channels). FBX is not a
skeleton model here: an external Blender step exports BVH (§4.4).

### 4.2 Formats (`smpl24.formats`)

Readers return **tables with the file's own names** and never interpret them: `npz` and
`pickle` (whitelisting unpickler, no code execution) return arrays by key; `json` returns
objects; `osim` returns a skeleton description; `mot`/`sto` return columns; `trc` and `c3d`
return labelled trajectories with rates and units; `b3d` returns the embedded `.osim`, the
passes, and the subject fields; `bvh` returns hierarchy plus channels; `mat` returns variables.
A format has no idea which dataset it serves.

### 4.3 Profiles (`smpl24.profile`)

A profile is the *only* place a dataset appears. Schema (`smpl24_profile_v1`), in outline:

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
correspondence: configs/correspondence/<file>.yaml     # names -> SMPL joints, fill rules, trunk body, lumbar split, aliases
markerset: configs/markersets/<file>.yaml              # marker_trajectories only
shape: {method: bone_lengths, landmark_offsets: <file>|none, rescale_to_measured: bool, pool_per_subject: bool}
root: {placement: pelvis_centre | source_translation | none, recover_frozen: bool}
provenance: {restrict_to_driven: [<source joints>]}
repairs:
  wrap: {coordinates: [...], filter: {order: ..., cutoff_hz: ...}}
  resample: {fps: ...}
  discontinuity: {settings_key: ...}
skip: {trials_without: <pass name>, subjects_with_unresolved_gender: bool}
settings: configs/settings/<file>.yaml
```

The loader validates the profile against the schema, refuses unknown keys, resolves relative
paths against the profile's directory, and hashes every referenced file; the hashes go into the
corpus manifests. `bind.py` turns a profile plus format tables into the kind's dataclass;
`layout.py` discovers subjects and trials on disk.

### 4.4 How the five current datasets read as profiles

| Dataset | kind | format | What today's code hardcodes that becomes profile data |
|---|---|---|---|
| AddBiomechanics | `skeleton_motion` | `b3d` | study/subject layout; `biological_sex` map with `unknown → neutral`; dynamics-pass-first with kinematics fallback; OpenSim naming table; `torso` as trunk body; lumbar split over spine1/2/3; landmark offsets v1; wrap repair on `pelvis_rotation`, `arm_rot_r`, `arm_flex_r` with the source's declared filter; skip trials without a dynamics pass |
| GAITEX | `skeleton_motion` | `osim_mot` (+ `trc`) | per-trial scaled skeletons pooled per subject; frozen-root recovery from `.trc`; gender declared neutral; provenance restricted to joints the nine worn sensors drive; joint-range settings |
| AMASS | `smpl_parameters` | `npz` | `poses[:, :66]` + hands identity; 16 betas to 10; `gender` field; `mocap_framerate` with a per-sub-dataset fallback table; Z-up |
| PRISM | `smpl_parameters` | `pickle` | pickle field bindings; gender from the subject-info table; frame and rate as the pickles state |
| HKNU | `joint_centres` | `mat` | Visual3D marker/segment-centre bindings; naming table with the `TA → trunk` alias; rescale rest joints to measured bone lengths; pelvis constant / lumbar zero policy; static-trial list for shape |

Nothing in this table needs a Python file. That is the test of the design: if a dataset needs
code, the code belongs to a kind or a format, and the profile only points at it.

### 4.5 Fit, repair, corpus, validate

- `fit.shape`: bone-length least squares with regularisation and optional stature/volume terms;
  `rescale_rest_joints` returning a `Shape` that records whether it was scaled; `pool_skeletons`.
- `fit.correspondence`: `Correspondence.from_yaml` (names → SMPL joints; fill rule `weld` /
  `distribute` / `estimate` per missing joint; trunk body; lumbar split; aliases);
  `restrict_to_driven`.
- `fit.pose_transfer` (segment rotations with calibration alignment), `fit.pose_ik` (Kabsch
  initialisation, least squares with angle regularisation and temporal smoothing), `fit.root`.
- `repair.wrap` (unwrap before filtering, only the named coordinates), `repair.resample` (slerp,
  linear), `repair.discontinuity` (flag, never edit).
- `corpus` writes/reads `docs/corpus-format.md`; every manifest carries profile id and hash,
  settings used, source hashes, converter version and commit, provenance codes.
- `validate` reports FK reproduction against the source observations, bone-length residuals,
  joint-range coverage, and round trips for `smpl_parameters` input.

### 4.6 Settings discipline

No numeric limit has a default in library code. Functions take a `settings` mapping resolved by
the profile and fail when a required key is missing. `configs/settings/` ships documented
examples. This mirrors the SOMA project's rule and keeps every threshold visible in the manifest.

### 4.7 CLI

```
smpl24 extract-model --pkl <file> --gender <g> --out <dir>
smpl24 convert --profile <profile.yaml> --input <root> --models <dir> --out <corpus>
smpl24 convert --kind <kind> --format <fmt> [--correspondence ...] [--markerset ...] --settings ... \
               --input <file-or-root> --models <dir> --out <corpus>      # ad hoc, no profile
smpl24 profile validate <profile.yaml>
smpl24 profile show <profile.yaml>          # resolved bindings, referenced files and their hashes
smpl24 fbx2bvh --input <clip.fbx> --blender <exe> --out <clip.bvh>
smpl24 info <corpus>
smpl24 validate <corpus> [--against <source>]
```

A subcommand is added only together with the module it fronts.

---

## 5. Migration phases

Each phase ends with the listed check green. Phases 2 and 3 carry the risk and are gated by
parity: the same sample converted through the dataset's profile must reproduce today's corpus.

| Phase | Content | Check |
|---|---|---|
| **0 Seed** (this change) | plan, README, pyproject, skeleton, primer, corpus-format draft, in-tree at `packages/smpl24/` | `pytest` in the seed; project's markdown-link test |
| **1 Split and mount** | owner creates the remote; `git init` from the seed, first commit, push; remove the seed directory from the project; `git submodule add <url> packages/smpl24`; dev install; project `pytest` `pythonpath` gains `packages/smpl24/src`; CI clones with `--recurse-submodules` | project tests unchanged; `smpl24 --version` from the project's environment |
| **2a Profile schema first** | `profile/` (schema, loader, binding, layout) with the five profiles of §4.4 written as data and validated; `formats/` for npz, pickle, osim, mot, trc, b3d, mat moved from today's readers | `smpl24 profile validate` passes for all five; format readers reproduce today's tables on fixtures |
| **2b Move and generalise the engine** | `model/`, `skeleton/`, `sources/{parameters,skeleton,centres}`, `fit/`, `repair/`, `corpus/` moved from §1.1 with dataset constants lifted into the configs; `convert.py` per kind; the project keeps one-release re-export shims with deprecation warnings | the 172 + moved tests green; **parity**: `convert --profile addbiomechanics.yaml` and `--profile gaitex.yaml` on a small fixed sample reproduce every npz array of today's drivers exactly; `--profile amass.yaml` reproduces today's `build_pose24` + resample arrays |
| **3 Project becomes a client** | the project's retarget drivers become thin calls to `smpl24 convert --profile` (or the API); `code_hash()` replaced by version + submodule commit; registry `library_imports` → `smpl24.*`; the pipeline runner records `smpl24` version, commit, profile hashes and body-model hashes into run provenance (closing the "body_model unavailable" item); shims removed; internal profiles (PRISM, HKNU) live in the project's `configs/` | **parity**: one study's corpus regenerated through the profile and compared with the corpus on disk (identical, or every difference listed and explained); ADR on `main` for the dependency change |
| **4 New kinds and formats** | `marker_trajectories` kind (conditioning + marker-set rules, generalising the project's marker stack), `c3d` format (extra), `bvh` format + `bvh_fk` skeleton model, `fbx2bvh` bridge; `docs/kinds/*.md` | round trips: synthesise SMPL-24 motion → export to the format → convert back → compare within float tolerance; FK reproduction on public sample files where licences allow |
| **5 Release discipline** | 0.1.0 tag; semantic versioning; CHANGELOG; the project pins the tag; licence decision applied | release checklist in `CHANGELOG.md` |

**Order inside 2b** (safest first): `model/` and `skeleton/` (pure, twice-implemented today) →
`sources/skeleton/opensim_fk` → `fit/` → `repair/` → `corpus/` → `convert`.

---

## 6. Integration with the SOMA project

- **Mount point** `packages/smpl24/` (the seed lives there, so the path does not change when the
  submodule replaces it; the project's plan link keeps resolving).
- **Profiles.** Public datasets' profiles ship in `smpl24/configs/profiles/` as examples; the
  project may override any of them and keeps internal ones (PRISM, HKNU) in its own `configs/`,
  passing paths explicitly. The project's registry (`source_pipelines_v1.yaml`) gains one field
  per source: `smpl24_profile`.
- **Pinning and provenance.** The project records the submodule commit, the profile id and hash,
  and `Model.sha256` per selected model in every run record; §10.2's "model/checkpoint hash" is
  satisfied.
- **Governance.** Moving code changes no field meaning, shape, order, unit, frame or rate, so
  phases 1–3 need no contract change; the dependency and the registry field need an ADR (numbers
  allocated on `main`). Holds are untouched: `smpl24` never generates inside the project; the
  runner gates every generation.
- **Data plane.** The package never reads `SOMA_DATA_ROOT` implicitly; the project passes model
  and corpus paths explicitly.
- **Windows and Dropbox.** The project's `.git` lives in Dropbox; submodule internals will too.
  Keep commits small and let Dropbox settle before switching branches.

---

## 7. Open decisions for the owner

| Decision | Options | Default in this plan |
|---|---|---|
| Remote for the new repository | private GitHub repository; a bare repository on the shared drive | private GitHub |
| Package and CLI name | `smpl24`; `smpl24-retarget`; `mocap2smpl24` | `smpl24` |
| Code licence | MIT / BSD-3 / Apache-2.0 / none (INTERNAL-ONLY) | INTERNAL-ONLY until decided |
| Where internal profiles live | in `smpl24/configs/profiles/` too; only in the project's `configs/` | project's `configs/` for PRISM and HKNU |
| `.c3d` reading | `ezc3d` as an extra; own reader | `ezc3d` extra |
| FBX route | Blender bridge (external); Autodesk FBX SDK; skip FBX in 0.1 | Blender bridge |
| Mount path in the project | `packages/smpl24`; `external/smpl24` | `packages/smpl24` |

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| A dataset name creeps back into code (`if profile.id == "gaitex"`) | review rule and a test that greps `src/` for profile ids; anything dataset-specific must be expressible in the schema, or the schema grows |
| The profile schema becomes a second programming language | keep it declarative: enumerated policies (`root.placement`, fill rules), no expressions; a policy that cannot be enumerated is a code feature of a kind |
| Two copies of the engine drift during phase 2 | one release with shims, then delete; parity on the same sample at both ends |
| A threshold sneaks in as a default | tests assert every settings key is required; review checklist |
| Model-path convenience returns (a hardcoded `D:` path) | `Model.for_gender` has no default root; a test asserts it fails without one |
| `filterwarnings = error` makes third-party deprecation warnings fatal | pin dependency majors; keep the setting |
| Submodule friction on Windows/Dropbox | small commits; documented `git submodule update --init`; CI uses `--recurse-submodules` |
| Reproduction differs after the move for correct reasons (a fixed bug) | phase 3 parity lists every difference; the project decides whether to regenerate |

---

## 9. Milestones

Working sessions, not dates.

1. Phase 0 (this change) and phase 1 (owner's remote + mount): one session.
2. Phase 2a (schema + five profiles + formats): one session.
3. Phase 2b (engine move with generalisation): two sessions plus the parity runs.
4. Phase 3: one session plus the corpus regeneration and its ADR.
5. Phase 4: one session per kind or format (markers, BVH, then FBX).
6. Phase 5: half a session.
