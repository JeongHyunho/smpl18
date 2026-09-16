# Changelog

All notable changes are recorded here. The format follows Keep a Changelog, and semantic
versioning starts at 0.1.0, when a corpus can be produced end to end.

## [Unreleased]

### Added
- Design and roadmap (`docs/plan.md`), the interface (`README.md`), the corpus format draft
  (`docs/corpus-format.md`) and the SMPL-24 primer (`docs/primer.md`).
- The design rule: code implements source kinds (`smpl_parameters`, `skeleton_motion`,
  `joint_centres`, `marker_trajectories`) and file formats; a dataset is a profile (YAML) and
  never a module.
- Package skeleton with `smpl24 --version`.
- Decisions: name `smpl24`, MIT licence (`LICENSE`), `ezc3d` as a regular dependency, FBX through
  a Blender bridge, and profiles that cannot be published resolved from the drive named by
  `SHARED_DATASET_PATH`.
- `smpl24.profile` (schema `smpl24_profile_v1`, loader, layout discovery, parameter
  binding), `smpl24.sources.base` (the four kinds), `smpl24 profile validate` and
  `smpl24 profile show`; example profiles `addbiomechanics`, `gaitex`, `amass` with their
  correspondence, landmark-offset and settings files; `docs/profile-schema.md`, `docs/kinds/`.
- `smpl24.formats` readers for `.osim`, `.mot`, `.trc`, `.c3d`, `.b3d` (vendored
  SubjectOnDisk proto), `.bvh`, `.npz`, JSON, `.mat`, and a whitelisting pickle reader.
- `smpl24.model` (clean-model extraction, loading with hashes, selection by gender with no default
  location) and `smpl24.skeleton` (definition, rotations and Kabsch, forward kinematics, frame
  changes).

### Fixed
- The up-axis change keeps the body in place: `t' = C (j_0 + t) − j_0` (primer section 3.4 had
  `t' = C t`, which is right only with the pelvis at the origin).
