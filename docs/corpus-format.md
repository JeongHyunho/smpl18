# Reduced-model corpus format — draft for 1.0

| | |
|---|---|
| Format id | `smpl18_corpus` |
| Version | `1.0` (draft; frozen at the 0.1.0 release) |
| Shape | one directory per corpus, one file per trial, per-joint provenance beside the arrays; a reader needs no knowledge of the source |

A corpus is a directory. Every converter writes this layout; every reader consumes it. Nothing in
it is source-specific: the source is described, not assumed.

```
<corpus>/
  SUMMARY.json
  <subject>/
    subject.json
    <trial>.npz
    <trial>.manifest.json
```

## `SUMMARY.json`

| key | value |
|---|---|
| `format_id`, `format_version` | `smpl18_corpus`, `1.0` |
| `converter` | `{"package": "smpl18", "version": ..., "commit": ...}` |
| `profile` | `{"id": ..., "sha256": ..., "referenced_files": {relative path: sha256}}`, or `{"id": "adhoc", "command": ..., "arguments": {...}}` when no profile was used |
| `source_kind`, `format` | `smpl_parameters` / `skeleton_motion` / `joint_centres` / `marker_trajectories`; `npz` / `pickle` / `json` / `osim_mot` / `b3d` / `trc` / `c3d` / `bvh` / `mat` |
| `subjects`, `trials`, `frames` | counts |
| `settings_sha256` | hash of the merged settings, one value when every trial used the same |
| `skipped` | list of `{subject, trial, reason}` |
| `provenance_counts` | per joint, how many trials are `measured` / `derived` / `absent` |

## `<subject>/subject.json`

| key | value |
|---|---|
| `subject_id` | as the source names it |
| `gender` | `male` / `female` / `neutral`, and `gender_source` (`source_field` / `declared_neutral_unknown` / `subject_file` / `user_argument`: where the gender itself came from) |
| `model_file`, `model_sha256` | the extracted model actually used |
| `model_is_stand_in` | true when the model was the stand-in body of `smpl18 demo-models`, not SMPL |
| `subject_file` | where the subject's id, gender and measurements came from, and the measurements |
| `betas` | float list, model width |
| `bone_scale` | absent, or per-bone factors when the rest skeleton was rescaled beyond the beta space |
| `fit` | the shape fit: `bone_lengths` (per rigid pair: measured, model, difference), `bone_rms_m`, `position_rms_m` on the refinement sample, `frames_used`, `refinements`, `betas_fitted`, `prior_weight`; `null` for SMPL parameters, whose betas are taken as stored |
| `skeleton_pooling` | absent, or the spread across per-trial skeletons that were pooled |
| `reduced_model` | the freeze: `frozen_joints` (the four names), `constants` (`[4, 3]` axis-angle), `absorbed_by` (which kept joint took each frozen joint's turn), `frozen_joint_provenance`, and `fit` |
| `reduced_model.fit` | what the freeze cost and how it was chosen: `residual_rms_m` and `residual_max_m` over the affected joints, the same two for the mean-rotation `starting_guess`, `per_joint_rms_m`, `frames_used`, `solver`, `converged` |
| `betas_spread_across_trials` | for SMPL parameters, the largest difference between the trials' betas (the first trial's are used); `null` otherwise |
| `trials` | the trial ids converted with this record. A reader refuses a trial file the record does not list: it was fitted with another shape and other constants |
| `checks` | plausibility warnings for the subject and its trials (length unit, distance from the source, up axis); empty when none |

A reader that needs all 24 joints rebuilds them: the four frozen joints at `constants`, the 18
stored rotations as they are (`spine3` and the shoulders already carry what the freeze took from
the joints above them), the hands at identity (`CorpusTrial.local_rotations_24`). Segment world
orientations are then those of the fitted 24-joint pose exactly; joint positions are within
`reduced_model.fit.residual_rms_m` of it.

## `<subject>/<trial>.npz`

The corpus stores the **18-joint reduced model**, not the 24 joints of the SMPL skeleton: four
joints are frozen to per-subject constants (`spine1`, `spine2`, `left_collar`, `right_collar`) and
the two hand joints are dropped. `subject.json` carries the constants and what the freeze cost, so
the 24-joint pose can be rebuilt from a trial and its subject record.

| array | shape, dtype | meaning |
|---|---|---|
| `poses` | `[T, 18, 3]` float64 | axis-angle local rotations of the kept joints, in ascending SMPL index order; index 0 is the root's world rotation |
| `joint_names` | `[18]` U | the kept joints, naming what each row of `poses` is |
| `trans` | `[T, 3]` float64 | model-origin translation (**not** the pelvis position; `pelvis = j_0 + trans`) |
| `fps` | scalar | frames per second of this array (after resampling, if any) |
| `up_axis` | `U` | `y` or `z`: the world frame `poses[:, 0]` and `trans` are expressed in |
| `joint_provenance` | `[18]` `U` | `measured` / `derived` / `absent` per kept joint; a frozen joint's provenance is recorded in `subject.json`, not here |
| `frame_valid` | `[T]` bool | false where the source had no observation or a flagged discontinuity |

Unknown keys are not allowed; extensions go into the manifest.

A subject is written once, with all its trials: its betas and frozen constants are fitted on
them together. Writing a subject again replaces its record and trials (`--replace`); it never
adds trials to an existing record.

## `<subject>/<trial>.manifest.json`

| key | value |
|---|---|
| `trial_id`, `subject_id` | as the source names them |
| `source` | `{"kind": ..., "format": ..., "files": [{"path": file name, "sha256": ...}], "native_fps": ..., "native_up_axis": ..., "frame_change": {...}}`, plus what the kind knows (`native_units`, `model_gravity`; for SMPL parameters `model_family`, `joints_stored`, `betas_stored`) |
| `profile` | id and sha256 of the profile that bound this trial (as in `SUMMARY.json`) |
| `tables` | the marker set or correspondence used (id, sha256, file name) and what it found: position and orientation targets, names missing in the source, fill rules; for markers, the lengths measured and the subject measurements; `orientation_calibration`, the per-target constants shared by the subject's trials |
| `repairs` | e.g. `{"wrap": ["pelvis_rotation"], "resample": {"from": 120, "to": 100}}` — empty lists mean checked and untouched |
| `discontinuities` | frames flagged, and the settings key that flagged them |
| `settings` | the settings sections that affected this trial, copied verbatim; `settings_files` (names and sha256) and `settings_sha256` beside them |
| `frames` | `{"total": ..., "valid": ...}` |
| `checks` | this trial's plausibility warnings |
| `converter` | as in `SUMMARY.json` |
| `validation` | joint-centre error against the source's own targets, for the fitted 24-joint pose (`solved_24_joint`) and for the stored 18-joint pose (`stored_18_joint`): `rms_m`, `max_m`, `per_target_rms_m`; `orientation_rms_rad`; the solve's `passes`. Empty for SMPL parameters |

## Conventions

- Rotations are axis-angle in radians; positions in metres; time in seconds.
- The skeleton is the SMPL-24 of `primer.md` §2, reduced to the 18 kept joints as `primer.md` §5
  describes. A corpus never stores the hand joints: no source in scope drives them.
- `absent` means nothing observed the joint and it was welded to its parent; `derived` means a
  rule distributed or estimated it; `measured` means a source observation drove it.
