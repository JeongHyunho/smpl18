# Source kind: `marker_trajectories`

**What it observes.** Labelled surface markers per frame, `[T, M, 3]` in metres with a
`[T, M]` validity mask; an occluded sample is invalid, never a zero. The dataclass is
`smpl18.sources.MarkerTrajectories`.

**How it becomes SMPL-24.** Markers do not fit SMPL directly in this release. A marker-set
description (`markerset`) says which labels form each rigid cluster and how each anatomical
joint centre is derived from them (midpoints, offsets along cluster axes, regression rules);
conditioning fills short gaps and blanks occlusions; the result is a `joint_centres` source and
the `joint_centres` converter runs from there, so every key that applies to that kind applies
here too. Mesh-level marker fitting is a non-goal of 0.1.

**Then the reduction.** Whatever the kind, the 24-joint pose is reduced to the 18 joints the
corpus stores: four joints frozen to fitted per-subject constants, the two hands dropped,
orientations preserved exactly (`primer.md` section 5). A joint the source did not drive keeps
its provenance through the reduction, and a frozen joint's provenance is recorded with the
constants rather than per trial.

**Formats.** `trc`, `c3d`.

**Profile keys that apply.**

| Key | Use |
|---|---|
| `layout` | one file per trial, static trials beside them |
| `bindings.markers.occlusion_sentinel` | how the file marks a lost marker |
| `bindings.fps`, `bindings.static_trials`, `bindings.gender` | the rate, the calibration trials, the model |
| `markerset` | labels to joint-centre rules |
| `correspondence`, `shape`, `root`, `pose`, `skip` | as for `joint_centres` |

No shipped profile uses this kind yet; it arrives with phase 4 of the plan.
