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
| `profile` | `{"id": ..., "sha256": ..., "referenced_files": {relative path: sha256}}`, or `{"id": "adhoc", ...}` with the command-line bindings when no profile was used |
| `source_kind`, `format` | `smpl_parameters` / `skeleton_motion` / `joint_centres` / `marker_trajectories`; `npz` / `pickle` / `json` / `osim_mot` / `b3d` / `trc` / `c3d` / `bvh` / `mat` |
| `subjects`, `trials`, `frames` | counts |
| `settings_sha256` | hash of the settings file used for the whole corpus |
| `skipped` | list of `{subject, trial, reason}` |
| `provenance_counts` | per joint, how many trials are `measured` / `derived` / `absent` |

## `<subject>/subject.json`

| key | value |
|---|---|
| `subject_id` | as the source names it |
| `gender` | `male` / `female` / `neutral`, and `gender_source` (`source_field` / `declared_neutral_unknown` / `user_argument`) |
| `model_file`, `model_sha256` | the extracted model actually used |
| `betas` | float list, model width |
| `bone_scale` | absent, or per-bone factors when the rest skeleton was rescaled beyond the beta space |
| `fit` | residuals: bone-length residual per bone, stature term if used, regularisation weight |
| `skeleton_pooling` | absent, or the spread across per-trial skeletons that were pooled |
| `reduced_model` | the freeze: `frozen_joints` (the four names), `constants` (`[4, 3]` axis-angle), `absorbed_by` (which kept joint took each frozen joint's turn), and `fit` |
| `reduced_model.fit` | what the freeze cost and how it was chosen: `residual_rms_m` and `residual_max_m` over the affected joints, the same two for the mean-rotation starting guess, `per_joint_rms_m`, `frames_used`, `solver`, `converged` |

A reader that needs all 24 joints rebuilds them: set the four frozen joints to `constants`, undo
the absorption on `spine3`, `left_shoulder` and `right_shoulder`, and set the hands to identity.
Segment world orientations come back exactly; joint positions come back to within
`reduced_model.fit.residual_rms_m`.

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

## `<subject>/<trial>.manifest.json`

| key | value |
|---|---|
| `trial_id` | as the source names it |
| `source` | `{"kind": ..., "format": ..., "files": [{"path": relative, "sha256": ...}], "native_fps": ..., "native_up_axis": ...}` |
| `profile` | id and sha256 of the profile that bound this trial (as in `SUMMARY.json`) |
| `correspondence` | id and sha256 of the map used, plus the fill rule per absent joint |
| `repairs` | e.g. `{"wrap": ["pelvis_rotation"], "resample": {"from": 120, "to": 100}}` — empty lists mean checked and untouched |
| `discontinuities` | frames flagged, and the settings key that flagged them |
| `settings` | the settings values that affected this trial, copied verbatim |
| `converter` | as in `SUMMARY.json` |
| `validation` | FK reproduction error against the source observations (median, max), when computed |

## Conventions

- Rotations are axis-angle in radians; positions in metres; time in seconds.
- The skeleton is the SMPL-24 of `primer.md` §2, reduced to the 18 kept joints as `primer.md` §5
  describes. A corpus never stores the hand joints: no source in scope drives them.
- `absent` means nothing observed the joint and it was welded to its parent; `derived` means a
  rule distributed or estimated it; `measured` means a source observation drove it.
