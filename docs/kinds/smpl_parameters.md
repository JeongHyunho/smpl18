# Source kind: `smpl_parameters`

**What it observes.** SMPL-family parameters already fitted to a body: per trial, axis-angle
`poses [T, J, 3]` with `J` = 24 (SMPL), 52 (SMPL-H) or 55 (SMPL-X), a translation `trans
[T, 3]`, a frame rate; per subject (or per sequence), a `betas` vector and a gender. The
dataclass is `smpl24.sources.SmplParameters`.

**How it becomes SMPL-24.** No fit is needed. The converter keeps the first 24 joints (for
SMPL-H and SMPL-X the 22 body joints, with the two hands at identity), truncates or pads
`betas` to the model's width, changes the up axis if the source is not Z-up, and resamples to
the target rate with per-joint Slerp on rotations and linear interpolation on `trans`.
Every joint's provenance is `measured` except the two hands of an SMPL-H/X source, which are
`absent`.

**Formats.** `npz` (arrays by key), `pickle` (a whitelisting unpickler; nested containers
addressed by a key path), `json`.

**Profile keys that apply.**

| Key | Use |
|---|---|
| `layout.subject`, `layout.trial`, `layout.exclude` | one file per trial, or a container per subject |
| `bindings.poses` (`field`, `layout`) | the array and whether it is flat `[T, J*3]` |
| `bindings.betas` (`field`, `frame`) | the vector, or the first frame of a per-frame array |
| `bindings.trans` | the translation |
| `bindings.fps` (`field`, `aliases`, `fallback`, `fallback_key`) | the rate, with a table keyed by a layout placeholder when the file omits it |
| `bindings.gender` (`field`, `map`, `default`, `when_absent`, `cross_check`) | the model to select |
| `conventions.up_axis` | the source world's up axis |
| `shape.method: parameters` | betas are taken as stored |
| `root.placement: source_translation` | `trans` is the root |
| `repairs.resample` | the target rate |
| `skip.trials_with_nonfinite_input`, `skip.trials_with_fps_other_than` | refusal rules |

Binding is implemented in `smpl24.profile.bind.bind_parameters`.
