# Source kind: `marker_trajectories`

**What it observes.** Labelled surface markers per frame, `[T, M, 3]` in metres with a
`[T, M]` validity mask; an occluded sample is invalid, never a zero. The dataclass is
`smpl18.sources.MarkerTrajectories`.

**Command.** `smpl18 convert markers --input <trc/c3d>... --markerset <yaml> --up-axis <axis>`
with a subject file carrying the measurements the marker set declares
([README](../../README.md#markers-only), [example 1](../../examples/01_markers_to_smpl18.py)).

**How it becomes SMPL-24.**

1. **Read the file.** Length units come from the file. Samples the file marks lost are invalid,
   and so are exact zero triplets when `--occlusion-sentinel zero` is given. The positions are
   turned from the given up axis into the corpus frame.
2. **Bridge gaps.** Interior gaps up to `markers.max_gap_frames` are bridged linearly.
3. **Apply the marker-set rules** (`smpl18.sources.markers`, schema `smpl18_markerset_v1`). They
   give joint centres by `point`, `offset` (in a marker-built frame, with lengths and subject
   measurements), `chord` or `hinge`, and segment frames for the `segments` the set names. A
   centre is invalid on a frame where any marker it reads is. A `hinge` centre is also invalid
   where the limb is straighter than its `min_flexion_deg`, and where it does not hold still on
   the proximal segment within `drift_tolerance`. A trial whose hinge centre wanders beyond that
   loses the centre altogether, with a warning: its reference marker is probably placed where
   the rule cannot tell the joint centre from its mirror image.
4. **Build targets.** The centres become position targets and the frames orientation targets,
   with the set's weights. Provenance follows from which joints they reach.
5. **Fit.** The shape, the per-frame pose and the reduction run as for every kind
   ([`plan.md`](../plan.md) §3.5–3.6). A frame with fewer valid centres than
   `pose.min_position_targets` holds its neighbour and is marked invalid.

Mesh-level marker fitting (markers as points on the body surface) is a non-goal of 0.1. The
shipped [`conventional_full_body.yaml`](../../configs/markersets/conventional_full_body.yaml)
follows the conventional gait model's published hip, knee and ankle definitions and states its
approximations for the trunk, head, shoulders and elbows. Check its joint centres against your
lab's processing before trusting a corpus.

**Then the reduction.** Whatever the kind, the 24-joint pose is reduced to the 18 joints the
corpus stores: four joints frozen to fitted per-subject constants, the two hands dropped,
orientations preserved exactly (`primer.md` section 5). A joint the source did not drive keeps
its provenance through the reduction, and a frozen joint's provenance is recorded with the
constants rather than per trial.

**Formats.** `trc`, `c3d`.

**Profile keys** (profile-driven conversion of whole datasets is on the roadmap):

| Key | Use |
|---|---|
| `layout` | one file per trial, static trials beside them |
| `bindings.markers.occlusion_sentinel` | how the file marks a lost marker |
| `bindings.fps`, `bindings.static_trials`, `bindings.gender` | the rate, the calibration trials, the model |
| `markerset` | labels to joint-centre rules |
| `correspondence`, `shape`, `root`, `pose`, `skip` | as for `joint_centres` |
