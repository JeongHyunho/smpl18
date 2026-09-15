# smpl24

Convert motion capture into an **SMPL-24 pose corpus**: one body shape per subject, one
axis-angle pose sequence per trial on the 24-joint SMPL skeleton, with a record of where every
joint's motion came from.

The package is organised by **what a source observes** and **how it is stored**, never by which
dataset it is. A dataset is a *profile*: a YAML file that binds a source kind and a file format
to that dataset's layout, field names, units, joint correspondence, repairs and settings.

| Source kind | Observes | Formats in the first release |
|---|---|---|
| `smpl_parameters` | SMPL / SMPL-H / SMPL-X `poses`, `betas`, `trans` | `npz`, `pickle`, `json` |
| `skeleton_motion` | an articulated skeleton and its joint angles per frame | `osim`+`mot` (+`trc`), `b3d`, `bvh`; `fbx` through a Blender bridge |
| `joint_centres` | anatomical joint-centre trajectories | `mat`, `trc`, `c3d` |
| `marker_trajectories` | labelled surface markers | `trc`, `c3d`, with a marker-set description |

Example profiles ship for public datasets (AddBiomechanics, GAITEX, AMASS); a project keeps its
own profiles for internal ones. Adding a dataset means writing a profile, not a module.

**Status: planning seed.** This directory holds the founding plan, the target interface and a
package skeleton. The commands and API below are the v0.1 specification; the engine arrives in
the migration phases described in [`docs/plan.md`](docs/plan.md).

**INTERNAL-ONLY** until the owner decides otherwise. The repository contains no body-model files
and no motion data; see [Body models and licences](#body-models-and-licences).

---

## Quick start

### 1. Install

Python 3.12 or newer.

```bash
git clone https://github.com/JeongHyunho/smpl24.git
cd smpl24
python -m pip install -e ".[dev]"
smpl24 --version
```

`.c3d` reading uses `ezc3d`, installed as a regular dependency.

### 2. Prepare the body models (once per machine)

SMPL is licensed by the Max Planck Institute and is **not redistributed here**. Download the
`.pkl` models from the SMPL website under your own licence, then extract the few arrays this
package needs into a licence-free `.npz` per gender:

```bash
smpl24 extract-model --pkl path/to/basicmodel_m_lbs_10_207_0_v1.1.0.pkl --gender male   --out ~/smpl24-models
smpl24 extract-model --pkl path/to/basicmodel_f_lbs_10_207_0_v1.1.0.pkl --gender female --out ~/smpl24-models
smpl24 extract-model --pkl path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl --gender neutral --out ~/smpl24-models
```

The extractor never executes pickle code: it uses a whitelisting unpickler and writes only
`v_template`, `shapedirs`, `J_regressor`, `kintree_parents` (and `weights`, `posedirs`, `faces`
when a mesh is wanted). Point the tools at the directory with `--models` or `SMPL24_MODELS`.

### 3. Convert a dataset through its profile

```bash
# Look at what a profile binds, and check it against the schema
smpl24 profile show     configs/profiles/addbiomechanics.yaml
smpl24 profile validate configs/profiles/gaitex.yaml

# Convert: the profile says which kind and format the dataset is, how its files are laid out,
# which field means what, and which correspondence, repairs and settings apply
smpl24 convert --profile configs/profiles/addbiomechanics.yaml --input /data/addbiomechanics \
    --models ~/smpl24-models --out corpus/addbiomechanics

smpl24 convert --profile configs/profiles/gaitex.yaml --input /data/gaitex \
    --models ~/smpl24-models --out corpus/gaitex

smpl24 convert --profile configs/profiles/amass.yaml --input /data/amass \
    --models ~/smpl24-models --out corpus/amass

# An internal dataset: its profile is not in this repository. Point SHARED_DATASET_PATH at the
# shared drive where the project publishes internal profiles, then name the profile.
export SHARED_DATASET_PATH=/mnt/shared/SOMA_AI_SharedData      # or set it in the environment once
smpl24 convert --profile prism --input /data/prism --models ~/smpl24-models --out corpus/prism
```

A profile name resolves in this order: an existing path as given; `configs/profiles/<name>.yaml`
in this package; `$SHARED_DATASET_PATH/smpl24/profiles/<name>.yaml`. Values inside a profile may
use `${SHARED_DATASET_PATH}`, and relative paths resolve against the profile's own directory, so a
profile published on the shared drive finds its correspondence and settings files beside it.

Every converter writes the same corpus layout (see [Corpus format](#corpus-format)).

### 4. Convert a single file without a profile

For ad hoc use, name the kind and format on the command line; the same generic code runs.

```bash
# SMPL-H parameters in an npz: trim to 24 joints, fix the up axis, resample
smpl24 convert --kind smpl_parameters --format npz --up-axis z --fps 100 \
    --input motion.npz --models ~/smpl24-models --out corpus/adhoc

# An OpenSim skeleton with an inverse-kinematics result
smpl24 convert --kind skeleton_motion --format osim_mot --osim model.osim --mot ik.mot \
    --correspondence configs/correspondence/opensim_rajagopal.yaml --gender female \
    --settings configs/settings/default.yaml --models ~/smpl24-models --out corpus/adhoc

# Labelled markers: joint centres by a marker-set description, then position IK
smpl24 convert --kind marker_trajectories --format trc --input trial.trc --static static.trc \
    --markerset configs/markersets/plug_in_gait.yaml --gender male \
    --settings configs/settings/default.yaml --models ~/smpl24-models --out corpus/adhoc

# An animation skeleton with a joint map
smpl24 convert --kind skeleton_motion --format bvh --input clip.bvh \
    --correspondence configs/correspondence/bvh_mixamo.yaml \
    --settings configs/settings/default.yaml --models ~/smpl24-models --out corpus/adhoc

# FBX goes through Blender (external, not bundled): FBX -> BVH -> convert
smpl24 fbx2bvh --input clip.fbx --blender "C:/Program Files/Blender/blender.exe" --out clip.bvh
```

Numeric limits (filter cut-offs, joint-rate limits, IK weights, gap lengths) are never defaulted
inside the library. A profile points at a settings file; ad hoc runs pass `--settings`. The
values used are copied into every trial manifest.

### 5. Inspect and validate

```bash
smpl24 info corpus/gaitex                    # subjects, trials, frames, provenance summary
smpl24 validate corpus/gaitex                # forward-kinematics reproduction, bone-length residuals
smpl24 validate corpus/adhoc --against trial.trc   # compare FK joints with the source observations
```

### 6. Use from Python

```python
from smpl24 import Model, Skeleton
from smpl24.profile import Profile
from smpl24.convert import convert_subject
from smpl24.corpus import read_corpus

profile = Profile.load("configs/profiles/gaitex.yaml")          # validated; referenced files hashed
model_root = "~/smpl24-models"

for subject in profile.layout.subjects("/data/gaitex"):           # discovery from the profile's layout
    convert_subject(profile, subject, models=model_root, out="corpus/gaitex")

corpus = read_corpus("corpus/gaitex")
trial = corpus.subject("S03").trial("walk_01")
skel = Skeleton(Model.for_gender(trial.gender, root=model_root), trial.betas)
joints_world = skel.fk(trial.poses, trial.trans)                  # [T, 24, 3]
```

The generic layers are usable on their own:

```python
from smpl24.formats import osim, mot
from smpl24.sources.skeleton import OpenSimSkeleton
from smpl24.fit import fit_shape, Correspondence, transfer_segment_rotations

skeleton = OpenSimSkeleton(osim.read("model.osim"))
motion = skeleton.motion(mot.read("ik.mot"))                       # segment rotations per frame
corr = Correspondence.from_yaml("configs/correspondence/opensim_rajagopal.yaml")
shape = fit_shape(model, motion.bone_lengths(corr), settings=fit_settings)
pose = transfer_segment_rotations(model, shape, motion, corr, settings=pose_settings)
```

---

## Profiles

A profile is the only place a dataset is named. In outline:

```yaml
schema: smpl24_profile_v1
id: addbiomechanics
source_kind: skeleton_motion
format: b3d
layout: {subject: "{study}/{subject}.b3d", trial: within_container}
bindings:
  gender: {field: biological_sex, map: {female: female, f: female, male: male, unknown: neutral}}
  frames: {pass: dynamics, fallback: kinematics}
  root_translation: [pelvis_tx, pelvis_ty, pelvis_tz]
conventions: {up_axis: y, length_unit: m, angle_unit: rad}
correspondence: ../correspondence/opensim_rajagopal.yaml
shape: {method: bone_lengths, landmark_offsets: ../offsets/addbiomechanics_v1.yaml}
repairs:
  wrap: {coordinates: [pelvis_rotation, arm_rot_r, arm_flex_r], filter: {order: 2, cutoff_hz: 30}}
  resample: {fps: 100}
skip: {trials_without: dynamics}
settings: ../settings/default.yaml
```

The schema is documented field by field in [`docs/profile-schema.md`](docs/profile-schema.md).
The loader refuses unknown keys, resolves relative paths against the profile, and hashes every
referenced file into the corpus manifests.

Public datasets' profiles ship here as examples. Internal datasets' profiles are authored in the
consuming project and published to the shared drive under `smpl24/`, mirroring this package's
`configs/` layout (`profiles/`, `correspondence/`, `markersets/`, `offsets/`, `settings/`); the
`SHARED_DATASET_PATH` environment variable names that drive.

---

## Corpus format

```
corpus/
  SUMMARY.json                 counts, converter version, profile id + hash, settings hash
  <subject>/
    subject.json               gender, model file + sha256, betas, optional per-bone scale, fit residuals
    <trial>.npz                poses [T,24,3] axis-angle, trans [T,3], fps, up_axis,
                               joint_provenance [24] (measured | derived | absent), frame_valid [T]
    <trial>.manifest.json      source files + sha256, converter version and commit, profile id + hash,
                               correspondence id, repairs applied, discontinuity flags, settings used
```

Full definition: [`docs/corpus-format.md`](docs/corpus-format.md). Conventions of the skeleton
itself (joint table, rest pose, frames, forward kinematics): [`docs/primer.md`](docs/primer.md).

---

## Body models and licences

- SMPL body models are licensed by MPI and must be obtained by each user. This repository never
  contains them, and the extracted `.npz` files must not be committed or shared.
- Motion data is never stored in this repository either. Converters read from paths you give
  them and write only to `--out`.
- The code is under the MIT licence (`LICENSE`). The licence governs distribution once the owner
  publishes the repository; until then it stays INTERNAL-ONLY, as the parent project requires.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check src tests
```

Tests are pure-Python fixtures; nothing needs the body models or motion data. Integration checks
that do need them are skipped unless `SMPL24_MODELS` points at an extracted model set. A test
greps `src/` for profile ids so that no dataset name can creep back into code.

## Relationship to the SOMA Synthetic IMU project

This package was carved out of that project's retarget engine so that the conversion to SMPL-24
has one home with its own history and releases. The project consumes it as a git submodule and
pins a commit; it authors its internal profiles in its own `configs/smpl24/` and publishes them
to the shared drive that `SHARED_DATASET_PATH` names. The migration is staged in
[`docs/plan.md`](docs/plan.md).

**What FBX is, and why it goes through Blender.** FBX is Autodesk's binary interchange format for
3D scenes and skeletal animation, common in game and animation pipelines. Reading it needs the
proprietary Autodesk SDK or a reimplementation; Blender (free) imports FBX and exports BVH, which
this package reads natively. `smpl24 fbx2bvh` drives a Blender you already have installed.
