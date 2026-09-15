# Changelog

All notable changes to this package are recorded here. The format follows Keep a Changelog and
the versioning follows semantic versioning once the engine has moved in (0.1.0).

## [Unreleased]

### Added
- Founding plan (`docs/plan.md`), target interface (`README.md`), corpus format draft
  (`docs/corpus-format.md`) and the SMPL-24 primer (`docs/primer.md`).
- The design rule: code implements source kinds (`smpl_parameters`, `skeleton_motion`,
  `joint_centres`, `marker_trajectories`) and file formats; a dataset is a profile (YAML) and
  never a module.
- Package skeleton with `smpl24 --version`.
- Owner decisions of 2026-09-15: name `smpl24`, MIT licence (`LICENSE`), `ezc3d` as a regular
  dependency, FBX through a Blender bridge, internal profiles resolved from the shared drive named
  by `SHARED_DATASET_PATH`; the remote repository is deferred, so phases 2a/2b proceed in-tree.
- Phase 2a: `smpl24.profile` (schema `smpl24_profile_v1`, loader, layout discovery, parameter
  binding), `smpl24.sources.base` (the four kinds), `smpl24 profile validate` and
  `smpl24 profile show`; example profiles `addbiomechanics`, `gaitex`, `amass` with their
  correspondence, landmark-offset and settings files; `docs/profile-schema.md`, `docs/kinds/`.
- Phase 2a: `smpl24.formats` readers for `.osim`, `.mot`, `.trc`, `.c3d`, `.b3d` (vendored
  SubjectOnDisk proto), `.bvh`, `.npz`, JSON, `.mat`, and a whitelisting pickle reader.
- Phase 2b (first part): `smpl24.model` (clean-model extraction, loading with hashes, selection by
  gender with no default location) and `smpl24.skeleton` (definition, rotations and Kabsch,
  forward kinematics, frame changes), with parity tests against the code they came from.

### Fixed
- The up-axis change keeps the body in place: `t' = C (j_0 + t) − j_0` (primer section 3.4 had
  `t' = C t`, which is right only with the pelvis at the origin).
