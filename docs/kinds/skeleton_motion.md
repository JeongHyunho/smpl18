# Source kind: `skeleton_motion`

**What it observes.** An articulated skeleton (named bodies in a tree, joints between them,
rest transforms, a set of independent coordinates) and, per frame, the values of those
coordinates. The skeleton is a `smpl24.sources.SkeletonModel` (OpenSim-style custom joints
with coupled coordinates, or a BVH offset hierarchy); the trial is a
`smpl24.sources.SkeletonMotion`, whose `placed()` runs the model's forward kinematics and
returns world rotations and origins per body and joint centres per joint.

**How it becomes SMPL-24.** Shape: bone lengths are measured between the source's joint
centres (the median over frames) for every SMPL edge whose two ends the correspondence names,
plus any declared extra spans, and least squares finds the betas, with optional stature and
mass terms and optional landmark offsets. Pose: one constant rotation per bone carries the
source body's world rotation onto the SMPL joint it drives,
`G_smpl[j] = R_world · G_source[body(j)] · A[j]`; the root's constant and offset come from its
child offsets; the trunk's single rotation is distributed over the three spine joints; joints
the source lacks are welded (identity) and marked `absent`. Provenance is `measured` for a
joint whose source joint the model declares and, if `provenance.restrict_to_driven` is set,
whose body was actually observed.

**Formats.** `b3d` (a container with the model, several processing passes and subject
fields), `osim_mot` (a model file, a coordinate file, optionally a marker file and a metadata
file beside it), `bvh`.

**Profile keys that apply.**

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
