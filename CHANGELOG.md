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
- Package skeleton with `smpl18 --version`.
- Decisions: name `smpl18`, MIT licence (`LICENSE`), `ezc3d` as a regular dependency, FBX through
  a Blender bridge, and profiles that cannot be published resolved from the drive named by
  `SHARED_DATASET_PATH`.
- `smpl18.profile` (schema `smpl18_profile_v1`, loader, layout discovery, parameter
  binding), `smpl18.sources.base` (the four kinds), `smpl18 profile validate` and
  `smpl18 profile show`; example profiles `addbiomechanics`, `gaitex`, `amass` with their
  correspondence, landmark-offset and settings files; `docs/profile-schema.md`, `docs/kinds/`.
- `smpl18.formats` readers for `.osim`, `.mot`, `.trc`, `.c3d`, `.b3d` (vendored
  SubjectOnDisk proto), `.bvh`, `.npz`, JSON, `.mat`, and a whitelisting pickle reader.
- `smpl18.model` (clean-model extraction, loading with hashes, selection by gender with no default
  location) and `smpl18.skeleton` (definition, rotations and Kabsch, forward kinematics, frame
  changes).

- File-by-file conversion for every source kind. Each takes its data straight to the 18-joint
  corpus:
  - `smpl18 convert markers` (`.trc`/`.c3d` with a marker set);
  - `smpl18 convert opensim` (`.osim` + `.mot`/`.sto`);
  - `smpl18 convert bvh`;
  - `smpl18 convert centres` (`.trc`/`.c3d`/`.npz`);
  - `smpl18 convert smpl` (SMPL / SMPL-H `.npz`; SMPL-X is refused with the way round it).
- The engine behind them:
  - `smpl18.fit` (targets, correspondence tables, the shape fit, the per-frame pose solve with
    calibrated segment orientations and priors on frozen and hinge joints);
  - `smpl18.sources.markers` (marker sets: `point`, `offset`, `chord`, `hinge` rules and segment
    frames);
  - `smpl18.sources.skeleton` (OpenSim and BVH forward kinematics);
  - `smpl18.sources.subject` (subject files);
  - `smpl18.convert`;
  - `smpl18.corpus` (write, read, summary, 24-joint rebuild).
- Commands `smpl18 info`, `smpl18 extract-model`, `smpl18 demo-models` (a stand-in body with the
  SMPL-24 tree, marked as such in every file and corpus) and `smpl18 fbx2bvh` (FBX to BVH through
  an installed Blender).
- Shipped tables:
  - `configs/markersets/conventional_full_body.yaml`;
  - `configs/correspondence/bvh_humanoid.yaml`;
  - `configs/correspondence/joint_centre_labels.yaml`.
- The `output`, `shape`, `pose`, `markers` and `checks` settings sections.
- Plausibility warnings for a wrong length unit, a wrong up axis or a stored pose far from the
  source, printed and recorded.
- `--replace`: a subject already in a corpus is refused otherwise, and a reader refuses a trial
  its subject record does not list.
- The OpenSim correspondence reads gait2392's knee names through aliases.
- The elbow `hinge` rule chooses between the two centres its geometry allows by where the
  upper-arm marker sits and by the centre holding still on the upper arm
  (`min_flexion_deg`, `drift_tolerance`); a trial it cannot decide loses the centre and is
  flagged, rather than getting the mirror image.
- `smpl18.formats.trc.write`.
- `examples/`: one runnable script per kind of input on synthetic captures, run by the tests;
  `smpl18.synthetic` generates those captures.
- Two ways back out of a corpus, for everything that reads the original 24-joint SMPL structure
  (`docs/blender.md`):
  - `smpl18 export-smpl` writes a trial as ordinary SMPL parameters (`poses [T, 72]`, `betas`,
    `trans`, `mocap_framerate`, `gender`), with the frozen joints at the subject's constants, the
    hands at identity, and `joint_provenance` widened to 24 joints (`constant`, `absent`).
    Converting the result again returns the same rotations.
  - `smpl18 blender` builds a Blender scene: the shaped surface skinned to 24 keyframed bones, a
    camera, a sun and a floor, saved as a `.blend` and rendered. `--frames` takes a slice,
    `--correctives` carries SMPL's 207 pose blend shapes in as shape keys.
- `smpl18.mesh`: pose blend shapes and linear blend skinning in numpy, and the per-joint
  rest-to-posed transforms a renderer's bones must carry.
- `smpl18.blender`: the scene plan (`smpl18_blender_plan_v1`), the render settings
  (`smpl18_render_v1`, `configs/render/default.yaml`) and the launcher; `smpl18.blender.scene` is
  the script Blender runs, which imports `bpy` and nothing from this package. The plan carries
  transforms rather than rotations, so Blender's own bone rest matrices cancel out of its skinning
  and no convention has to be agreed; the scene then checks its own deformation against vertices
  computed here and refuses to save when it differs.
- `smpl18 demo-models --with-mesh` and `smpl18.model.demo.demo_boxes`: the stand-in body can carry
  a blocky surface, so a scene can be built and rendered without a licensed model.

### Changed
- The README is organised by the data a user has, with the commands that work today; converting
  whole datasets through their profiles is marked as the next step.
- `SkeletonModel` declares `joint_child_bodies` and `rotational_coordinates`.
- The OpenSim correspondence uses only the pelvis body's rotation: its origin is not where SMPL
  puts the pelvis joint.
- The dataset-name guard reads the ids of the shipped profiles and settings and matches whole
  words; marker sets and correspondence tables are named after protocols, not datasets.
- Comments in the shipped configs describe the datasets, not particular conversion runs.

### Fixed
- The up-axis change keeps the body in place: `t' = C (j_0 + t) − j_0` (primer section 3.4 had
  `t' = C t`, which is right only with the pelvis at the origin).
