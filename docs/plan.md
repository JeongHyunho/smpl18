# smpl24 — founding plan

| | |
|---|---|
| Written | 2026-09-15, in the SOMA Synthetic IMU repository, lane `worktree-smpl24-primer` |
| Status | **Seed.** Phase 0 is in this change set; the later phases are proposals for the owner |
| Decisions already taken by the owner | separate repository mounted as a git submodule of the SOMA project; first release covers SMPL-family parameters, OpenSim skeletons (`.osim`+`.mot`, `.b3d`), marker trajectories (`.trc`/`.c3d`) and BVH/FBX; the existing code **moves** here and the SOMA project depends on this package (single source) |
| Language | English is the default for every document and message in this repository |
| Authority | none over the SOMA project. This plan releases no hold, changes no contract, and authorises no generation there |

---

## 0. Summary

The SOMA Synthetic IMU project already contains a working OpenSim-to-SMPL-24 retarget engine
(`src/soma_synthetic_imu/addbio_retarget/`, ten modules, 172 unit tests), a marker-based
retarget used for one cohort, and two copies of the SMPL model loader and forward kinematics
(`scripts/poc/anthro_smpl.py`, `scripts/poc/smpl_model.py`). Three sources reuse the engine
through it; none of it is installable, half of it is loaded by file path, and the body-model
location is hardcoded in two places.

This plan carves that engine out into **`smpl24`**, an installable package with one job: turn
motion capture into an SMPL-24 pose corpus with per-joint provenance. The SOMA project keeps
everything that is about *its* datasets (attribution, holds, artifact classes, the unified8
bundles, sensor synthesis) and consumes `smpl24` as a pinned submodule. The move happens in
phases, each with a parity check against the code as it stands today, so that the corpora the
project has already generated can be regenerated bit-for-bit from the new package before anything
new is added.

---

## 1. What exists today (inventory, 2026-09-15)

### 1.1 The engine that moves

| Today (SOMA project) | Purpose | Destination in `smpl24` |
|---|---|---|
| `addbio_retarget/osim_topology.py` | skeleton definition from an `.osim` (also embedded in `.b3d`), two index orders | `inputs/opensim/topology.py` |
| `addbio_retarget/osim_kinematics.py` | OpenSim forward kinematics without OpenSim (`parse_kinematics`, knee splines) | `inputs/opensim/kinematics.py` |
| `addbio_retarget/b3d_frames.py` + vendored `SubjectOnDisk_pb2` | decode `.b3d` frames without nimblephysics | `inputs/opensim/b3d.py` + `inputs/opensim/_vendor/` |
| `gaitex_retarget/gaitex_frames.py` | `.osim` + `.mot` + `.trc` reader; recovers a root translation a source's IK froze | `inputs/opensim/motfiles.py`, `inputs/markers/trc.py` |
| `addbio_retarget/smpl_correspondence.py` | SMPL-24 names/parents (`SMPL24_PARENTS`, `SMPL24_NAMES`) **and** an OpenSim joint-name table | split: skeleton definition → `skeleton/definition.py`; the naming table → `inputs/opensim/correspondence.py` (data, as YAML) |
| `addbio_retarget/shape_fit.py` | betas from measured bone lengths, rest-joint rescale, stature/volume, model selection by gender | `fit/shape.py`, `model/select.py` |
| `addbio_retarget/pose_fit.py` | `G_smpl[j] = R_world · G_source[body(j)] · A[j]`, root placement | `fit/pose_transfer.py`, `fit/root.py` |
| `addbio_retarget/world_frame.py` | Y-up source world → Z-up corpus world | `skeleton/frames.py` (generalised to any pair of up axes) |
| `addbio_retarget/wrap_repair.py` | undo a 2π staircase that a source filtered through | `repair/wrap.py` |
| `gaitex_retarget/shape.py` | pool one body per subject from per-trial scaled skeletons | `fit/shape.py` (`pool_skeletons`) |
| `gaitex_retarget/provenance.py` | demote joints nothing observed to non-measured | `fit/correspondence.py` (`restrict_to_driven`) |
| `gaitex_retarget/joint_ranges.py` | how much of a trial the source's IK failed to solve; bounds from config, no code default | `validate/joint_ranges.py` |
| `kinematics/quaternion.py`, `kinematics/rigid_body.py` (Kabsch with reflection correction) | rotation algebra, rigid pose from point sets | `skeleton/rotations.py` |
| `scripts/poc/anthro_smpl.py` (`rest_joints`, `smpl_fk_positions`, `fk_positions_batch`, `load_smpl_model`, `smpl_model_path_for_gender`) | SMPL rest skeleton and FK | `skeleton/kinematics.py`, `model/load.py` |
| `scripts/poc/smpl_model.py`, `scripts/poc/prepare_smpl_clean_npz.py` | two fail-closed pkl → clean-npz extractors with a whitelisting unpickler | merged into `model/extract.py` (tests from both) |
| `scripts/diagnostics/hknu_retarget_probe.py` (`centres_from`, `rescale_rest_to_measured`, `root_placement_from`) | marker/segment-centre retarget for a Visual3D cohort | generalised into `inputs/markers/centres.py`, `fit/shape.py`, `fit/root.py` |
| `scripts/poc/generate_amass_faithful.py` (`build_pose24`, `resample_trans`, per-joint slerp) | SMPL-H → 24 joints, resampling | `inputs/smplfamily.py`, `repair/resample.py` |
| `scripts/poc/generate_addbio_smpl24.py`, `generate_gaitex_smpl24.py` (the six-step drivers) | corpus-level orchestration | `cli.py convert b3d / opensim`, `corpus/write.py` |
| tests: `tests/addbio_retarget/*` (172), `tests/unit/test_gaitex_{frames_reader,provenance_and_shape,joint_ranges,cluster_pose}.py`, `tests/poc/test_{anthro_smpl,smpl_model,prepare_smpl_clean_npz}.py` | | move with their modules |

### 1.2 What stays in the SOMA project

- Everything named after a dataset: `addbio_retarget/attribution.py` (the closed study table and
  licence fields), `artifact.py` (`spec_id = addbio_smpl24_raw`), the `LANDMARK_OFFSETS_V1`
  table in `shape_fit.py` (an AddBiomechanics-vs-SMPL landmark correction; it becomes a settings
  input to `smpl24`, owned by the project), the `_TRUNK_BODY = "torso"` assumption in
  `pose_fit.py` (becomes a correspondence parameter).
- The unified8 bundle emitters, sensor synthesis (`sensors/`, `gaitex_synthesis/`,
  `virtual_imu`), the anthro deliverable assembly (`compute_anthro`), the validator, the pipeline
  runner, registries, holds, artifact classes.
- `source_parsing/adapters/addbiomechanics.py` (the audit-lane B3D decoder coupled to governance).

### 1.3 Facts that shape the design

- **One shared call.** All five SOMA generators reach the body model through
  `clean_model_path_for_gender`; the registry says so. That call is the first API `smpl24` must
  preserve (as `Model.for_gender`).
- **Model location is hardcoded twice** (`shape_fit.py:141`, `anthro_smpl.py:74`) with different
  environment overrides. `smpl24` has one resolver: `--models` / `SMPL24_MODELS`, no default path.
- **No thresholds hide in code.** The project forbids code-default tolerances; every tolerance in
  `rigid_body.py`, `markers.py`, `joint_ranges.py` is a required argument or config value.
  `smpl24` keeps that discipline so the project can consume it (§4.6).
- **No nimblephysics, no OpenSim dependency**: both are reimplemented in pure numpy/scipy and
  covered by tests. `smpl24` inherits that: numpy, scipy, protobuf, pyyaml.
- **No BVH/FBX code exists.** Those are new work (§5, phase 4).
- **`code_hash()` in the drivers hashes nine module files by repository path.** That breaks the
  moment the package moves; the replacement is the installed package version plus the submodule
  commit, recorded by the project's run record.

---

## 2. Goals and non-goals

**Goals**

1. One installable package that converts any of the four input kinds into one corpus format
   with per-joint provenance (`measured` / `derived` / `absent`) and fit residuals.
2. Bit-for-bit reproduction of the corpora the SOMA project generates today from the moved code
   before any behaviour changes.
3. The body model handled honestly: selected per subject by gender, located by explicit
   configuration, hashed and reported, never bundled.
4. Small, testable modules with pure-Python fixtures; no test needs licensed models or data.
5. English documentation with a quick start a new user can follow in ten minutes.

**Non-goals**

- Sensor synthesis, bundle formats, dataset attribution, governance holds: those stay in the
  SOMA project.
- Mesh-level fitting (marker-to-vertex MoSh-style optimisation) in 0.1. Marker input goes through
  joint-centre estimation first (§4.4); vertex-level fitting is a later extension.
- A public release. The repository is INTERNAL-ONLY until the owner decides otherwise (§7).

---

## 3. Repository shape

```
smpl24/
├── README.md                 quick start, interface, corpus format summary
├── CHANGELOG.md
├── LICENSE                   owner decision (§7); absent until taken
├── pyproject.toml            name smpl24, src layout, extras: c3d, dev
├── docs/
│   ├── plan.md               this document
│   ├── primer.md             SMPL-24 and how to reach it from SMPL or markers (English)
│   ├── corpus-format.md      the on-disk corpus, versioned
│   └── inputs/               one page per input kind (added with each phase)
├── maps/                     correspondence tables and joint maps (YAML, data not code)
├── settings/                 documented example settings files (every numeric limit lives here)
├── src/smpl24/
│   ├── __init__.py  cli.py
│   ├── model/      extract.py  load.py  select.py
│   ├── skeleton/   definition.py  rotations.py  kinematics.py  frames.py
│   ├── fit/        shape.py  correspondence.py  pose_transfer.py  pose_ik.py  root.py
│   ├── inputs/     smplfamily.py  opensim/{topology,kinematics,b3d,motfiles,correspondence}.py
│   │               markers/{trc,c3d,centres}.py  bvh.py  fbx.py
│   ├── repair/     wrap.py  resample.py  discontinuity.py
│   ├── corpus/     schema.py  write.py  read.py  summary.py
│   └── validate/   fk_reproduction.py  bone_lengths.py  joint_ranges.py  roundtrip.py
└── tests/          mirrors src/; fixtures build their own tiny skeletons and files
```

---

## 4. Design

### 4.1 Model (`smpl24.model`)

- `extract_clean(pkl, out, *, gender, num_betas, with_mesh)` — one extractor, merged from the two
  existing ones, whitelisting unpickler, exact shape asserts, writes `SMPL_<GENDER>_clean.npz`.
  Never executes pickle code; refuses unknown classes.
- `Model.for_gender(gender, root=None)` — resolves `root` from the argument, else
  `SMPL24_MODELS`, else fails. Genders: `male`, `female`, `neutral`; anything else raises
  `UnresolvedGender`. Exposes `path`, `sha256`, `rest_joints(betas)`, `num_betas`.
- The model directory is never defaulted to a machine path.

### 4.2 Skeleton (`smpl24.skeleton`)

- `definition.py`: the 24 names, parents, the 22-joint body subset, segment list. Frozen; a test
  pins it against the primer's table.
- `rotations.py`: axis-angle ↔ matrix ↔ quaternion (`(w,x,y,z)`), slerp, batched compose,
  Kabsch with reflection correction (one implementation for the whole package).
- `kinematics.py`: `rest_joints(model, betas)`, `fk(poses, trans, rest_joints)` → world joints
  and segment rotations, batched. The translation-versus-pelvis convention is documented on the
  function and tested (`p_0 = j_0 + t`).
- `frames.py`: change of up axis as one fixed rotation applied to root rotation and translation
  only; gravity vector carried along.

### 4.3 Fit (`smpl24.fit`)

- `shape.py`: `fit_shape(model, bone_lengths, settings)` — least squares on bone lengths with
  regularisation and optional stature/volume terms; `rescale_rest_joints` for skeletons the beta
  space cannot reach, returning a `Shape` that records whether it was scaled; `pool_skeletons`
  for sources that scale per trial. All weights come from `settings`.
- `correspondence.py`: `Correspondence.from_yaml(path)` — source joint/segment ↔ SMPL joint,
  with the fill rule per missing joint (`weld` / `distribute` / `estimate`) and the resulting
  provenance code. `restrict_to_driven` demotes joints nothing observed.
- `pose_transfer.py`: segment-rotation transfer with a calibration alignment `A[j]`; the trunk
  body name is a parameter of the correspondence, not a constant.
- `pose_ik.py`: position-based IK on joint centres (Kabsch initialisation, least squares with
  angle regularisation and temporal smoothing, warm start, missing observations dropped). Weights
  and iteration limits from `settings`.
- `root.py`: root rotation and translation from the pelvis observation; explicit "no root
  translation" state when the source has none.

### 4.4 Inputs (`smpl24.inputs`)

| Kind | Reader | Path to SMPL-24 |
|---|---|---|
| SMPL-family npz | `smplfamily.read` | trim to 24 (hands identity), betas to model width, frame change, resample |
| `.osim` + `.mot` (+ `.trc`) | `opensim.motfiles` | topology → FK → segment rotations → `pose_transfer`; shape from FK bone lengths |
| `.b3d` | `opensim.b3d` | as above, from the container's embedded model and passes |
| `.trc` / `.c3d` + marker set | `markers.trc`, `markers.c3d` (extra `ezc3d`), `markers.centres` | joint centres by marker-set rules → shape from a static trial → `pose_ik` |
| `.bvh` + joint map | `bvh.read` (own parser) | rest-pose alignment → `pose_transfer` |
| `.fbx` | `fbx.convert_with_blender` | Blender (external, path given) exports BVH → BVH path |

Each reader returns the same `SourceMotion` structure (frames, per-frame segment rotations
and/or point observations, fps, up axis, subject fields as the source states them). Nothing
downstream knows which reader ran.

### 4.5 Repair, corpus, validate

- `repair.wrap` unwraps before filtering and re-filters only the coordinates that wrapped;
  `repair.resample` slerps rotations and interpolates translations; `repair.discontinuity` flags
  frames whose joint rate exceeds a settings-declared limit and never edits them.
- `corpus` writes/reads the format in `docs/corpus-format.md`; every trial manifest carries the
  converter version and commit, the settings used, the source hashes, and the provenance codes.
- `validate` reports FK reproduction error against the source observations, bone-length
  residuals, and (for SMPL-family input) a round trip against the original parameters.

### 4.6 Settings discipline

No numeric limit has a default in library code. Functions take a `settings` mapping (loaded from
YAML by the CLI) and fail when a required key is missing. `settings/` ships documented examples.
This mirrors the SOMA project's rule and is what lets that project consume the package under its
own configuration hold; it also keeps every threshold visible in the trial manifest.

### 4.7 CLI

`smpl24 extract-model`, `smpl24 convert {smpl,opensim,b3d,markers,bvh,fbx}`, `smpl24 info`,
`smpl24 validate`. Every `convert` takes `--models`, `--out`, `--settings`; kind-specific inputs
as in the README. A subcommand is added only together with the module it fronts.

---

## 5. Migration phases

Each phase ends with the listed check green. Phases 2 and 3 are the ones that carry risk, and
both are gated by parity with the current code.

| Phase | Content | Check |
|---|---|---|
| **0 Seed** (this change) | plan, README, pyproject, skeleton, primer, corpus-format draft, in-tree at `packages/smpl24/` | `pytest` in the seed; project's markdown-link test |
| **1 Split and mount** | owner creates the remote; `git init` from the seed, first commit, push; remove the seed directory from the project; `git submodule add <url> packages/smpl24`; dev install `pip install -e packages/smpl24`; project `pytest` `pythonpath` gains `packages/smpl24/src`; CI clones with `--recurse-submodules` | project tests unchanged; `smpl24 --version` from the project's environment |
| **2 Move the engine** | modules of §1.1 move with their tests; dataset-specific pieces of §1.2 stay and become parameters; the project keeps one-release re-export shims (`soma_synthetic_imu.addbio_retarget.shape_fit` → `smpl24.fit.shape`) with deprecation warnings | the 172 + moved unit tests green in `smpl24`; **parity**: `generate_addbio_smpl24.py` and `generate_gaitex_smpl24.py` run on a small fixed sample before and after the move and every npz array is identical; project test suite green |
| **3 Project becomes a client** | the three retarget drivers call `smpl24` (API or `convert`); `code_hash()` replaced by `smpl24.__version__` + submodule commit; registry `library_imports` → `smpl24.*`; the pipeline runner records `smpl24` version, commit and the body-model hashes into the run provenance (closing the open "body_model unavailable" item); shims removed | **parity**: one study's corpus regenerated and compared with the corpus on disk (identical, or every difference listed and explained); ADR on `main` for the dependency change |
| **4 New inputs** | SMPL-family normaliser (from the AMASS builder's pure parts), general marker IK (from the Visual3D probe), BVH reader + joint maps, FBX bridge; `docs/inputs/*.md` per kind | round-trip tests: synthesise SMPL-24 motion → export to the input format → convert back → compare within float tolerance; FK reproduction on public sample files where licences allow |
| **5 Release discipline** | 0.1.0 tag; semantic versioning; CHANGELOG; the project pins the tag; licence decision applied | release checklist in `CHANGELOG.md` |

**Order inside phase 2** (safest first): `model/` and `skeleton/` (pure, twice-implemented today,
easiest to prove equal) → `inputs/opensim/` → `fit/` → `repair/` → `corpus/` → readers that
depend on all of them.

---

## 6. Integration with the SOMA project

- **Mount point** `packages/smpl24/` (the seed already lives there, so the path does not change
  when the submodule replaces it; the project's plan link keeps resolving).
- **Pinning.** The project records the submodule commit in every run record; bumping the pin is
  a reviewed change like any dependency bump. Provenance §10.2's "model/checkpoint hash" is
  satisfied by `Model.sha256` for the file actually selected per subject, plus the set hash.
- **Governance.** Moving code does not change any field meaning, shape, order, unit, frame or
  rate, so no contract change is needed for phases 1–3; the dependency itself needs an ADR
  (numbers are allocated on `main`). Holds are untouched: `smpl24` never generates anything on
  its own inside the project; the runner still gates every generation.
- **Data plane.** The package never reads `SOMA_DATA_ROOT` implicitly. The project passes model
  and corpus paths explicitly (its registry already names the corpora).
- **Windows and Dropbox.** The project's `.git` lives in Dropbox; submodule internals will too
  (`.git/modules/`). Keep commits small, avoid long-running rebases, and let Dropbox settle before
  switching branches. Nothing else changes.

---

## 7. Open decisions for the owner

| Decision | Options | Default in this plan |
|---|---|---|
| Remote for the new repository | private GitHub repository; a bare repository on the shared drive | private GitHub |
| Package and CLI name | `smpl24`; `smpl24-retarget`; `mocap2smpl24` | `smpl24` |
| Code licence | MIT / BSD-3 / Apache-2.0 / none (INTERNAL-ONLY) | INTERNAL-ONLY until decided |
| `.c3d` reading | `ezc3d` as an extra; own reader | `ezc3d` extra |
| FBX route | Blender bridge (external); Autodesk FBX SDK; skip FBX in 0.1 | Blender bridge |
| Mount path in the project | `packages/smpl24`; `external/smpl24` | `packages/smpl24` |

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Two copies of the engine drift during phase 2 | one release with shims, then delete; parity test on the same sample at both ends |
| A threshold sneaks in as a default | tests assert every settings key is required; review checklist |
| Model-path convenience returns (a hardcoded `D:` path) | `Model.for_gender` has no default root; test asserts it fails without one |
| `filterwarnings = error` makes third-party deprecation warnings fatal | pin dependency majors; keep the setting, it has caught real problems in the project |
| Submodule friction on Windows/Dropbox | small commits; documented `git submodule update --init`; CI uses `--recurse-submodules` |
| Reproduction differs after the move for reasons that are correct (e.g. a fixed bug) | phase 3 parity lists every difference; the project decides whether to regenerate |

---

## 9. Milestones

Working sessions, not dates.

1. Phase 0 (done with this change) and phase 1 (owner's remote + mount): one session.
2. Phase 2: two sessions (model/skeleton/opensim; then fit/repair/corpus) plus the parity run.
3. Phase 3: one session plus the corpus regeneration and its ADR.
4. Phase 4: one session per input kind (SMPL-family and markers first, BVH, then FBX).
5. Phase 5: half a session.
