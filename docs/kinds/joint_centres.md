# Source kind: `joint_centres`

**What it observes.** World positions of named anatomical joint centres per frame,
`[T, K, 3]` in metres with a `[T, K]` validity mask: the output of a marker-based model
(Visual3D, Vicon Plug-in Gait) after its own processing. A source that also places its
segments may carry each body's world rotation `[T, 3, 3]`, which then steers each segment's
twist. The dataclass is `smpl18.sources.JointCentres`.

**Command.** `smpl18 convert centres --input <trc/c3d/npz>... --correspondence <yaml>
--up-axis <axis>` ([README](../../README.md#joint-centres),
[example 4](../../examples/04_joint_centres_to_smpl18.py)).

**How it becomes SMPL-24.**

1. **Read the file.** A `.trc` / `.c3d` is read with its own labels and units. An `.npz` holds
   `names`, `positions`, `fps` and `units`, and optionally `segment_names` with
   `segment_rotations`. Everything is turned into the corpus frame.
2. **Apply the correspondence** (`names: centres`). Each SMPL joint's entry names the centre that
   is its position target, and optionally the segment whose rotation is its orientation target.
   Entries may carry a weight (lower for centres placed differently from SMPL, such as trunk,
   neck and head) and joints without a source may carry a `fill` rule. Labels the file lacks are
   reported in the manifest.
3. **Fit.** The shape, the per-frame pose and the reduction run as for every kind
   ([`plan.md`](../plan.md) §3.5–3.6). Without segment rotations, the twist of a segment that
   nothing below it reveals (a head, a hand) stays at rest, and the knees' and elbows' hinge
   prior settles the thigh's and upper arm's twist.

**Then the reduction.** Whatever the kind, the 24-joint pose is reduced to the 18 joints the
corpus stores: four joints frozen to fitted per-subject constants, the two hands dropped,
orientations preserved exactly (`primer.md` section 5). A joint the source did not drive keeps
its provenance through the reduction, and a frozen joint's provenance is recorded with the
constants rather than per trial.

**Formats.** `trc`, `c3d`, `npz`; `mat` (variables and structs, addressed by key path) through
profiles.

**Profile keys** (profile-driven conversion of whole datasets is on the roadmap):

| Key | Use |
|---|---|
| `layout` (`subject`, `trial`, `exclude`, `subject_table`) | one file per trial; demographics in a table |
| `bindings.centres` (`all_fields_of`, `extra`) | which struct fields and paths are centres |
| `bindings.rotations` (`field` with `{body}`, `layout`, `aliases`) | segment rotations, if the source has them |
| `bindings.time`, `bindings.fps`, `bindings.static_trials` | the clock, the rate, the calibration trials |
| `bindings.gender`, `stature_m`, `mass_kg` | from the subject table |
| `correspondence` | segment and centre names to SMPL joints |
| `shape` (`bone_lengths`, `rescale_to_measured`, `from_trials`, `use_stature`, `use_mass`) | the fit and the rescale tables |
| `root` (`directions`, `anchor`, `tracked`, `from_trial`) | the pelvis policy |
| `pose` (`reference`, `place_unfitted_from_reference`, `lumbar_zero_from_reference`) | the neutral configuration |
| `skip.trials_with_nonfinite_input`, `skip.trials_with_fps_other_than` | refusal rules |
