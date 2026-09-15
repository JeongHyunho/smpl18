# SMPL-24 corpus format — draft for 1.0

| | |
|---|---|
| Format id | `smpl24_corpus` |
| Version | `1.0` (draft; frozen at the 0.1.0 release) |
| Lineage | generalises the `addbio_smpl24_raw` / `gaitex_smpl24_raw` artifacts of the SOMA project (`retarget-v2`), whose fields it keeps |

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
| `format_id`, `format_version` | `smpl24_corpus`, `1.0` |
| `converter` | `{"package": "smpl24", "version": ..., "commit": ...}` |
| `input_kind` | `smpl` / `opensim` / `b3d` / `markers` / `bvh` / `fbx` |
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

## `<subject>/<trial>.npz`

| array | shape, dtype | meaning |
|---|---|---|
| `poses` | `[T, 24, 3]` float64 | axis-angle local rotations; index 0 is the root's world rotation |
| `trans` | `[T, 3]` float64 | model-origin translation (**not** the pelvis position; `pelvis = j_0 + trans`) |
| `fps` | scalar | frames per second of this array (after resampling, if any) |
| `up_axis` | `U` | `y` or `z`: the world frame `poses[:, 0]` and `trans` are expressed in |
| `joint_provenance` | `[24]` `U` | `measured` / `derived` / `absent` per joint |
| `frame_valid` | `[T]` bool | false where the source had no observation or a flagged discontinuity |

Unknown keys are not allowed; extensions go into the manifest.

## `<subject>/<trial>.manifest.json`

| key | value |
|---|---|
| `trial_id` | as the source names it |
| `source` | `{"kind": ..., "files": [{"path": relative, "sha256": ...}], "native_fps": ..., "native_up_axis": ...}` |
| `correspondence` | id and sha256 of the map used, plus the fill rule per absent joint |
| `repairs` | e.g. `{"wrap": ["pelvis_rotation"], "resample": {"from": 120, "to": 100}}` — empty lists mean checked and untouched |
| `discontinuities` | frames flagged, and the settings key that flagged them |
| `settings` | the settings values that affected this trial, copied verbatim |
| `converter` | as in `SUMMARY.json` |
| `validation` | FK reproduction error against the source observations (median, max), when computed |

## Conventions

- Rotations are axis-angle in radians; positions in metres; time in seconds.
- The skeleton is the SMPL-24 of `docs/primer.md` §2: hands (22, 23) carry identity unless the
  source drove them.
- `absent` means nothing observed the joint and it was welded to its parent; `derived` means a
  rule distributed or estimated it; `measured` means a source observation drove it.
