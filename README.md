# smpl24

Convert motion capture into an **SMPL-24 pose corpus**: one body shape per subject, one
axis-angle pose sequence per trial, on the 24-joint SMPL skeleton, with a record of where every
joint's motion came from.

Inputs the first release covers:

| Input | Files | Method |
|---|---|---|
| SMPL-family parameters | SMPL / SMPL-H / SMPL-X `.npz` (`poses`, `betas`, `trans`) | trim, re-frame, resample |
| OpenSim skeletons | `.osim` + `.mot` (+ `.trc`), or a `.b3d` container | shape fit from bone lengths, segment-rotation transfer |
| Marker trajectories | `.trc`, `.c3d` with a marker-set description | joint-centre estimation, position-based IK |
| Animation skeletons | `.bvh`; `.fbx` through a Blender bridge | joint map, segment-rotation transfer |

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
git clone <url> smpl24
cd smpl24
python -m pip install -e ".[dev]"
smpl24 --version
```

Optional extras: `.[c3d]` for `.c3d` reading.

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

### 3. Convert

Every converter writes the same corpus layout (see [Corpus format](#corpus-format)).

```bash
# SMPL / SMPL-H / SMPL-X parameters: trim to 24 joints, fix the up axis, resample
smpl24 convert smpl --input motion.npz --up-axis z --fps 100 \
    --models ~/smpl24-models --out corpus/

# OpenSim skeleton with an inverse-kinematics result
smpl24 convert opensim --osim model.osim --mot ik.mot --trc markers.trc \
    --correspondence maps/opensim_rajagopal.yaml --gender female \
    --models ~/smpl24-models --out corpus/

# One AddBiomechanics-style .b3d container (model, kinematics and subject fields inside)
smpl24 convert b3d --input subject.b3d --correspondence maps/opensim_rajagopal.yaml \
    --models ~/smpl24-models --out corpus/

# Labelled markers: joint centres from a marker-set description, then position IK
smpl24 convert markers --trc trial.trc --static static.trc --marker-set sets/plug_in_gait.yaml \
    --gender male --models ~/smpl24-models --out corpus/

# Animation skeleton with a joint map
smpl24 convert bvh --input clip.bvh --joint-map maps/bvh_mixamo.yaml \
    --models ~/smpl24-models --out corpus/

# FBX goes through Blender (external, not bundled): FBX -> BVH -> convert bvh
smpl24 convert fbx --input clip.fbx --blender "C:/Program Files/Blender/blender.exe" \
    --joint-map maps/bvh_mixamo.yaml --models ~/smpl24-models --out corpus/
```

Numeric limits (filter cut-offs, joint-rate limits, IK weights, gap lengths) are never defaulted
inside the library. They come from a settings file passed with `--settings settings.yaml`; the
repository ships documented examples under `settings/` and the values used are copied into every
trial manifest.

### 4. Inspect and validate

```bash
smpl24 info corpus/                  # subjects, trials, frames, provenance summary
smpl24 validate corpus/              # forward-kinematics reproduction, bone-length residuals
smpl24 validate corpus/ --against markers.trc   # compare FK joints with the source observations
```

### 5. Use from Python

```python
from smpl24 import Model, Skeleton
from smpl24.inputs import read_opensim
from smpl24.fit import fit_shape, Correspondence, transfer_segment_rotations
from smpl24.corpus import write_subject, write_trial

model = Model.for_gender("female", root="~/smpl24-models")     # exposes model.sha256
source = read_opensim("model.osim", "ik.mot")                    # bodies, joints, per-frame kinematics
corr = Correspondence.from_yaml("maps/opensim_rajagopal.yaml")   # source joint -> SMPL joint

shape = fit_shape(model, source.bone_lengths(corr), settings=fit_settings)   # once per subject
pose = transfer_segment_rotations(model, shape, source, corr, settings=pose_settings)  # per frame

write_subject("corpus/S01", shape, model)
write_trial("corpus/S01/walk_01", pose, source.provenance(corr), settings_used=...)

skel = Skeleton(model, shape)
joints_world = skel.fk(pose.poses, pose.trans)                   # [T, 24, 3]
```

---

## Corpus format

```
corpus/
  SUMMARY.json                 counts, converter version, settings hash
  <subject>/
    subject.json               gender, model file + sha256, betas, optional per-bone scale, fit residuals
    <trial>.npz                poses [T,24,3] axis-angle, trans [T,3], fps, up_axis,
                               joint_provenance [24] (measured | derived | absent), frame_valid [T]
    <trial>.manifest.json      source files + sha256, converter version and commit,
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
- The code licence is an open decision recorded in `docs/plan.md`. Until it is taken, treat the
  repository as INTERNAL-ONLY.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check src tests
```

Tests are pure-Python fixtures; nothing needs the body models or motion data. Integration checks
that do need them are skipped unless `SMPL24_MODELS` points at an extracted model set.

## Relationship to the SOMA Synthetic IMU project

This package was carved out of that project's retarget engine so that the conversion to SMPL-24
has one home with its own history and releases. The project consumes it as a git submodule and
pins a commit; the migration is staged in [`docs/plan.md`](docs/plan.md).
