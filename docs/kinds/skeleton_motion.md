# Source kind: `skeleton_motion`

**What it observes.** An articulated skeleton (named bodies in a tree, joints between them,
rest transforms, a set of independent coordinates) and, per frame, the values of those
coordinates. The skeleton is a `smpl18.sources.SkeletonModel`. The trial is a
`smpl18.sources.SkeletonMotion`, whose `placed()` runs the model's forward kinematics and
returns world rotations and origins per body and joint centres per joint.

**Commands.**

- `smpl18 convert opensim --osim <model> --mot <mot/sto>... --correspondence <yaml>`
  ([README](../../README.md#kinematics-only-opensim),
  [example 2](../../examples/02_opensim_to_smpl18.py)).
- `smpl18 convert bvh --input <bvh>... --correspondence <yaml> --up-axis <axis> --length-unit
  <unit>` ([README](../../README.md#animation-skeleton-bvh-fbx),
  [example 3](../../examples/03_bvh_to_smpl18.py)). FBX goes through `smpl18 fbx2bvh` first.

**How it becomes SMPL-24.**

1. **Forward kinematics** (`smpl18.sources.skeleton`).
   - **OpenSim 4 models:** custom, pin, slider, universal, gimbal, ball, free, planar and weld
     joints; linear, polynomial, piecewise-linear and spline coordinate functions; coupled
     coordinates computed from their coupler. The coordinate file must hold every independent
     coordinate (a missing one is an error naming it). Degrees or radians follow the file's
     `inDegrees` header. The world frame comes from the model's gravity, or from `--up-axis`
     for a model that declares none.
   - **BVH:** each joint's channels in their written order (intrinsic rotations), offsets scaled
     by the declared length unit, end sites as extra points.
2. **Apply the correspondence** (`names: opensim_joints` or `bvh_joints`). An entry lists source
   joints:
   - the **first** gives the position target, its centre;
   - the body the **last** one moves gives the orientation target (the distal segment SMPL
     spans: the calcaneus past the subtalar joint, the radius past the radioulnar one).

   Further options:
   - `position: false` / `orientation: false` drop a target whose point or frame is not SMPL's
     (the OpenSim pelvis origin, a rig's spine joints).
   - A `lumbar` block turns a single trunk body into a position target for the lowest spine
     joint and an orientation target for `spine3`; the pose prior shares the trunk's turn over
     the three spine joints.
   - `fill: weld` holds joints the skeleton lacks (the neck of a Rajagopal model) at rest.
   - `prefixes` and `aliases` (`{name in the file: name in the table}`) absorb naming
     variants; the shipped table maps gait2392's `knee_l`/`knee_r` this way.
   - Joints the file lacks are reported in the manifest.
3. **Fit.** The shape, the per-frame pose and the reduction run as for every kind
   ([`plan.md`](../plan.md) §3.5–3.6). A segment's orientation is compared after a constant that
   a positions-only pass calibrates on all of the subject's trials, so the source's body frames
   need not match SMPL's.

**Then the reduction.** Whatever the kind, the 24-joint pose is reduced to the 18 joints the
corpus stores: four joints frozen to fitted per-subject constants, the two hands dropped,
orientations preserved exactly (`primer.md` section 5). A joint the source did not drive keeps
its provenance through the reduction, and a frozen joint's provenance is recorded with the
constants rather than per trial.

**Formats.** `osim_mot` (a model file and coordinate files), `bvh`; `b3d` (a container with the
model, several processing passes and subject fields) through profiles.

**Profile keys** (profile-driven conversion of whole datasets is on the roadmap):

| Key | Use |
|---|---|
| `layout` with `within_container` or `files` (`osim`, `mot`, `trc`, `metadata`) | where the model and coordinates are |
| `bindings.gender`, `stature_m`, `mass_kg` | subject fields, or a declared constant |
| `bindings.frames` (`pass`, `unfiltered_pass`) | which container pass supplies frames and which the wrap repair reads |
| `bindings.root_translation` | coordinates that are `trans`, not rotation |
| `bindings.markers.occlusion_sentinel` | how the marker file marks a lost marker |
| `conventions` (`angle_unit`, `up_axis_check: model_gravity`) | degrees or radians; verify the frame against the model's gravity |
| `correspondence` | source joint names to SMPL joints, fill rules, trunk body, lumbar split |
| `shape` (`bone_lengths`, `landmark_offsets`, `pool_per_subject`, `extra_spans`, `use_stature`, `use_mass`) | the fit |
| `root` (`pelvis_centre`, `child_offsets`, `recover_frozen`) | the root and a frozen-translation recovery from a marker cluster |
| `pose.method: segment_rotation_transfer` | the transfer |
| `provenance.restrict_to_driven` | demote joints nothing observed |
| `repairs.wrap` | unwrap-then-refilter coordinates a scalar filter corrupted, with the filter the source declares |
| `repairs.longest_solved_span`, `repairs.joint_ranges` | keep the time base uniform; measure declared ranges |
| `skip.trials_without_pass`, `skip.empty_trials`, `skip.trials_shorter_than` | refusal rules |
