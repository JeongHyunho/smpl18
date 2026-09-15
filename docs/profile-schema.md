# Profile schema: `smpl24_profile_v1`

A profile is a YAML file that says everything `smpl24` needs to know about one dataset: which
source kind and file format it is, how its files are laid out, which field means what, its
units and frame, which correspondence and settings apply, which repairs run and which trials
are skipped. It is the only place a dataset is named. The engine implements source kinds and
formats; a profile binds them.

The schema is declarative. Every policy is an enumerated word (`root.placement`,
`shape.method`, a fill rule), never an expression. If a dataset needs something these words
cannot say, the schema or a source kind grows; a special case in code is a defect.

Validation (`smpl24 profile validate`, `Profile.load`) refuses any key the schema does not
name, anywhere in the file, and reports the dotted path of the offending key, for example
`bindings.gender.mapp: unknown key; allowed keys are [...]`. Keys marked *required* below must
be present; the others may be omitted, in which case the feature is off.

Field references (`field:`) are either a top-level key of the file's tables (`poses`) or a list
of keys walking into a nested container (`[smpl_params, poses]`, `[Segment, "{body}", SegRot]`).
File references resolve, in order, as an absolute path; relative to the profile's own directory;
relative to the package `configs/` directory; relative to `$SHARED_DATASET_PATH/smpl24/`. Any
string value may contain `${VAR}`, which is replaced from the environment; an unset variable is
an error naming the key.

---

## Top level

| Key | Required | Meaning |
|---|---|---|
| `schema` | yes | The constant `smpl24_profile_v1`. |
| `id` | yes | The dataset id, recorded in every corpus manifest. |
| `description` | no | One line for humans. |
| `source_kind` | yes | `smpl_parameters`, `skeleton_motion`, `joint_centres` or `marker_trajectories`; see `docs/kinds/`. |
| `format` | yes | `npz`, `pickle`, `json`, `osim_mot`, `b3d`, `trc`, `c3d`, `bvh` or `mat`. |
| `layout` | yes | How subjects and trials are found under `--input`. |
| `bindings` | yes | Which field means what, in the file's own names. |
| `conventions` | yes | Up axis, length unit, angle unit of the source. |
| `correspondence` | skeleton, centres, markers | Path to a `smpl24_correspondence_v1` file. Not allowed for `smpl_parameters`. |
| `markerset` | markers | Path to a marker-set description. Only for `marker_trajectories`. |
| `shape` | all but markers optional | How the subject's shape is decided. |
| `root` | skeleton, centres, markers | Where SMPL's root goes and how its constant is fixed. |
| `pose` | skeleton, centres, markers | How pose is produced and what counts as neutral. |
| `provenance` | no | Which joints may be called measured. |
| `repairs` | no | Wrap repair, resampling, discontinuity scan, solved-span trimming, joint ranges. |
| `skip` | no | Which trials and subjects are left out, and why. |
| `settings` | yes | One settings file or a list, merged in order (later keys win). Every numeric limit lives there. |

Per-kind requirements are checked after the structure: `smpl_parameters` needs `poses`,
`betas`, `trans`, `fps` and `gender` bindings and a `shape` block; `skeleton_motion` and
`joint_centres` need `correspondence`, `shape`, `root` and `pose`; `joint_centres` also needs
`centres` and `fps` bindings; `marker_trajectories` needs `markerset` as well.

---

## `layout`

Patterns are relative paths with `{placeholder}` names and the globs `*` (inside a segment)
and `**` (any number of segments). `{subject}` and `{trial}` are the identities; any other
placeholder (`{study}`, `{split}`, `{variant}`) is a grouping value kept on the subject and
trial for provenance and for keyed fallbacks (`bindings.fps.fallback_key`). A repeated
placeholder must match the same text. Discovery is sorted by relative path.

| Key | Required | Meaning |
|---|---|---|
| `subject` | yes | Pattern matching one entry per subject: a file (a container) or a directory. Must contain `{subject}`. |
| `trial` | yes | Pattern with `{trial}` matching one entry per trial, or the word `within_container` when trials live inside the subject's file and the format reader lists them. |
| `files` | no | Companion files per trial by role (`osim`, `mot`, `trc`, `metadata`), as patterns over the same placeholders. Not allowed with `within_container`. |
| `exclude` | no | File-name globs never treated as a subject or trial (`shape.npz`, `*_SubjectInfo.mat`). |
| `skip_dirs_starting_with` | no | Directory-name prefixes pruned from discovery (`_`, `.`). |
| `skip_empty_files` | no | Zero-byte files are not payload. |
| `subject_table` | no | A table of subject demographics under `--input`: `file`, `format` (`csv`, `json`, `xlsx`), optional `sheet`, the `id` column, and `columns` naming the `gender`, `mass_kg` and `stature_m` columns. Subject-level bindings then read that row. |

## `bindings`

| Key | Applies to | Meaning |
|---|---|---|
| `poses` | parameters | `field`; `layout` is `T,J,3` (default) or `T,J*3` (flat, reshaped). |
| `betas` | parameters | `field`; `frame: first` when betas are stored per frame and the first frame is the subject's shape. |
| `trans` | parameters | `field`. |
| `fps` | parameters, centres, markers | `field` with optional `aliases` (other field names tried in order), or `constant`. `fallback` is a table keyed by the value of the placeholder named in `fallback_key`, used when no field is present. |
| `gender` | all | Either `constant: female|male|neutral`, or `field` with `map` (source value, stripped and lower-cased, to `female`, `male` or `neutral`), an optional `default` for values not in the map, an optional `when_absent` for a missing field, and an optional `cross_check` field that must agree on the first initial. A value neither mapped nor defaulted is an error, so a profile has to say what it does with `unknown`. |
| `stature_m`, `mass_kg` | all | `field` of a subject-level measurement, used only when `shape.use_stature` / `use_mass` say so. |
| `root_translation` | skeleton | Coordinates that describe where the root is, not how it turns; they become `trans`. |
| `frames` | b3d | `pass` supplying the frames; `unfiltered_pass` the wrap repair reads its branch cuts from. |
| `centres` | centres | `all_fields_of`: a struct whose every field is a centre; `extra`: named centres from other paths. |
| `rotations` | centres | Segment world rotations: `field` with `{body}`, `layout` `T,3,3` or `3,3,T`, optional `bodies`, and `aliases` (alias to source body) so a correspondence may name a body the source does not. |
| `time` | centres | The frame clock field. |
| `markers` | trc, c3d | `occlusion_sentinel`: `zero` (an exact zero triplet is an occlusion) or `nan`. |
| `static_trials` | centres, markers | Trial ids that are static calibration poses. |

## `conventions`

| Key | Required | Values |
|---|---|---|
| `up_axis` | yes | `y` or `z`, of the source world. The corpus is Z-up. |
| `length_unit` | yes | `m` or `mm`. |
| `angle_unit` | yes | `rad` or `deg`, of stored joint angles. |
| `up_axis_check` | no | `model_gravity`: the frame is verified against the gravity the model declares; `none`. |

## `correspondence` and `markerset`

Paths. A correspondence file (`smpl24_correspondence_v1`) lists, per SMPL-24 joint, the source
joints or body that drive it and the centre it is fitted to, or a fill rule for a joint the
source lacks: `weld` (identity local rotation, the joint follows its parent), `distribute` (a
share of a measured parent rotation, as the three spine joints share the trunk's), `estimate`
(placed from a reference configuration). It also names the trunk body, the lumbar split, the
root translation coordinates and any aliases. A marker set maps labels to joint-centre rules.

## `shape`

| Key | Required | Meaning |
|---|---|---|
| `method` | yes | `bone_lengths` (least squares from measured segment lengths, with the settings' weights) or `parameters` (betas come from the source). |
| `landmark_offsets` | no | `none` or a path to an offsets table added to the measured lengths before the fit. |
| `rescale_to_measured` | no | `false`, or tables keyed by SMPL edges (`parent/child`) whose values are two source centres: `bones` scale one edge, `chains` scale every offset on a path together, `spans` move a pair about its midpoint. |
| `pool_per_subject` | no | Targets are the median across the subject's trials rather than one measurement. |
| `extra_spans` | no | Non-adjacent SMPL joint pairs added to the targets. |
| `from_trials` | no | `all`, or the trial ids the shape is measured on. |
| `use_stature`, `use_mass` | no | Whether the subject's stature and mass, when bound and present, enter the fit. |

## `root`

| Key | Required | Meaning |
|---|---|---|
| `placement` | yes | `pelvis_centre` (the root sits at the correspondence's root centre plus a fitted offset), `source_translation` (parameters' `trans`), `none`. |
| `alignment` | no | `child_offsets` (the root's constant from every child offset) or `directions` (from anatomical direction pairs). |
| `directions` | with `directions` | At least two `{smpl: [a, b], source: [A, B]}` pairs: the SMPL rest vector `a - b` against the source vector `A - B`. |
| `anchor` | no | `{smpl: [...], source: [...]}`: the SMPL joints whose centroid lands on the centroid of the named centres. |
| `tracked` | no | SMPL joints the translation is solved against every frame; empty keeps the constant offset. |
| `recover_frozen` | no | `false`, or `{markers, body, coordinates}`: recover a frozen root translation from a marker cluster rigid in one body and write it into the named coordinates. |
| `from_trial` | no | The static trial the directions and anchor are read from. |

## `pose`

| Key | Required | Meaning |
|---|---|---|
| `method` | yes | `segment_rotation_transfer` (one constant per bone carries the source's segment rotations), `position_ik` (solve from centres), `parameters` (as stored). |
| `reference` | no | `none`, or `{trial, aggregate: mean_rotation}`: a static trial whose mean segment rotations are the neutral configuration. |
| `place_unfitted_from_reference` | no | Joints with nothing to fit against take their constant from the reference. |
| `lumbar_zero_from_reference` | no | The trunk's turn is measured against the reference rather than against the raw frame difference. |

## `provenance`

`restrict_to_driven`: `none`; a list of source bodies that were observed; or
`{from: trial_metadata, file: <layout.files role>, field: <path>}` to read the set per trial
from a companion file. A joint whose driving body is not in the set becomes `absent`.

## `repairs`

| Key | Meaning |
|---|---|
| `wrap` | `none`, or `{coordinates: any | [names], filter: declared_by_source | from_settings, declared_pass, recompute_centres}`: unwrap-then-refilter coordinates that wrapped through a scalar filter. |
| `resample` | `none`, or `{fps, rotations: slerp, translation: linear}`. |
| `discontinuity` | `none`, or `{settings_key, action: flag}`: scan the result for steps and inverted trunks; flag, never edit. |
| `longest_solved_span` | Keep only the longest run of frames whose root was solved, so the time base stays uniform. |
| `joint_ranges` | `none`, or `{settings_key}`: measure the declared ranges and record; never drop a frame. |

## `skip`

| Key | Meaning |
|---|---|
| `trials_without_pass` | `none` or a pass name a container trial must carry. |
| `empty_trials` | Zero-length trials are skipped. |
| `subjects_with_unresolved_gender` | Skip a subject whose gender resolves to nothing instead of failing. |
| `trials_with_nonfinite_input` | A non-finite value in the input skips the trial rather than being filled. |
| `trials_with_fps_other_than` | `none` or a rate every trial must have. |
| `trials_shorter_than` | `none` or `{settings_key}` naming the minimum frame count in the settings. |

## `settings`

A path or a list of paths to `smpl24_settings_v1` files, deep-merged in order. Every numeric
limit, weight and sample size the engine uses is read from the merged mapping and copied into
every trial manifest; the library has no defaults for them. The package ships
`configs/settings/default.yaml` with every value cited to the code it came from.

---

## Example

```yaml
schema: smpl24_profile_v1
id: example
source_kind: skeleton_motion
format: b3d
layout: {subject: "{study}/{subject}/*.b3d", trial: within_container, skip_empty_files: true}
bindings:
  gender: {field: biological_sex, map: {female: female, male: male, unknown: neutral}}
  frames: {pass: dynamics, unfiltered_pass: kinematics}
  root_translation: [pelvis_tx, pelvis_ty, pelvis_tz]
conventions: {up_axis: y, length_unit: m, angle_unit: rad}
correspondence: ../correspondence/opensim_rajagopal.yaml
shape: {method: bone_lengths, landmark_offsets: none}
root: {placement: pelvis_centre, alignment: child_offsets}
pose: {method: segment_rotation_transfer}
repairs:
  wrap: {coordinates: any, filter: declared_by_source, declared_pass: lowPassFilter}
skip: {trials_without_pass: dynamics}
settings: ../settings/default.yaml
```

`smpl24 profile show example.yaml` prints the profile as loaded, every referenced file with
its absolute path and SHA-256, and the merged settings.
