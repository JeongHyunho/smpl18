# Source kind: `joint_centres`

**What it observes.** World positions of named anatomical joint centres per frame,
`[T, K, 3]` in metres with a `[T, K]` validity mask: the output of a marker-based model
(Visual3D, Vicon Plug-in Gait) after its own processing. A source that also places its
segments may carry each body's world rotation `[T, 3, 3]`; then pose can be transferred
rather than solved. The dataclass is `smpl24.sources.JointCentres`.

**How it becomes SMPL-24.** Shape: bone lengths between centres, measured on the trials
`shape.from_trials` names (a static pose, typically), then least squares for the betas; when
ten betas cannot reach the measured proportions, `shape.rescale_to_measured` scales the rest
skeleton's edges, chains and spans onto the measured lengths, directions untouched. Pose:
with segment rotations present, `pose.method: segment_rotation_transfer` as for a skeleton;
without them, `position_ik` (Kabsch initialisation, least squares with angle regularisation
and temporal smoothing). Root: `pelvis_centre`, with the constant from anatomical
`directions` read off a static trial when two hip offsets cannot fix it, an `anchor` pair the
translation must hit, and `tracked` joints the translation is solved against every frame. A
static `pose.reference` trial defines neutral: unfitted joints (a head with no child) take
their constant from it and the trunk's turn is measured against it.

**Formats.** `mat` (variables and structs, addressed by key path), `trc`, `c3d`.

**Profile keys that apply.**

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
