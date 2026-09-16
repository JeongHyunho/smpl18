# Source kind: `smpl_parameters`

**What it observes.** SMPL-family parameters already fitted to a body:
- per trial: axis-angle `poses [T, J, 3]` with `J` = 22 or 24 (SMPL) or 52 (SMPL-H), a
  translation `trans [T, 3]` and a frame rate;
- per subject (or per sequence): a `betas` vector and a gender.

The dataclass is `smpl18.sources.SmplParameters`.

**Command.** `smpl18 convert smpl --input <npz>... --up-axis <axis>`
([README](../../README.md#smpl-parameters),
[example 5](../../examples/05_smpl_parameters_to_smpl18.py)). The array names default to
`poses`, `trans` and `betas` (`--poses-key` etc. rename them). The rate is read from
`mocap_framerate`, `mocap_frame_rate`, `fps` or `frame_rate`, or given with `--fps`.

**How it becomes SMPL-24.** No fit is needed.
- **Joints:** the 22 body joints are kept, with the hands at identity.
- **Betas:** the first trial's `betas` (the first frame of a per-frame array) are cut or padded
  to the model's width.
- **Frame:** the up axis is changed with the rest pelvis of those betas,
  `t' = C (j_0 + t) − j_0` (primer §3.4).
- **Provenance:** every body joint is `measured`; the hands are `absent` and not stored.
- **Gaps:** a frame with a non-finite pose or translation is marked invalid and holds its
  nearest valid frame; the reduction's constants are fitted on the valid frames only.
- **Record:** the manifest keeps the model family (`smpl`, `smplh`), the joints and betas the
  file stored, and the subject record the largest difference between its trials' betas.

**SMPL-X is refused** (`J` = 55). Its betas describe the SMPL-X template and its rest pelvis
sits elsewhere, so neither carries onto SMPL. Compute the sequence's joint positions with the
SMPL-X model and convert them as `joint_centres`.

Resampling to another rate is part of profile-driven conversion (`repairs.resample`); the
command keeps the file's rate.

**Then the reduction.** Whatever the kind, the 24-joint pose is reduced to the 18 joints the
corpus stores: four joints frozen to fitted per-subject constants, the two hands dropped,
orientations preserved exactly (`primer.md` section 5). For this kind the reduction is the only
change to the joints' positions, and the subject record states its cost.

**Formats.** `npz` (arrays by key); `pickle` (a whitelisting unpickler; nested containers
addressed by a key path) and `json` through profiles.

**Profile keys** (binding is implemented in `smpl18.profile.bind.bind_parameters`;
profile-driven conversion of whole datasets is on the roadmap):

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
