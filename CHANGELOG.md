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
