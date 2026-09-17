# Examples

Each example starts from a situation — *I only have markers*, *I only have OpenSim
kinematics*, … — writes synthetic input files of that kind, converts them with the same
`smpl18` command you would type (it is printed before it runs), and reads the corpus back to
show how close the stored 18-joint poses are to the motion the files were made from.

| Script | You have | Command it shows |
|---|---|---|
| [`01_markers_to_smpl18.py`](01_markers_to_smpl18.py) | labelled markers (`.trc` / `.c3d`) | `smpl18 convert markers` |
| [`02_opensim_to_smpl18.py`](02_opensim_to_smpl18.py) | an OpenSim model and IK results (`.osim` + `.mot`) | `smpl18 convert opensim` |
| [`03_bvh_to_smpl18.py`](03_bvh_to_smpl18.py) | an animation skeleton (`.bvh`, or `.fbx` via Blender) | `smpl18 convert bvh` |
| [`04_joint_centres_to_smpl18.py`](04_joint_centres_to_smpl18.py) | joint-centre trajectories (`.trc` / `.c3d` / `.npz`) | `smpl18 convert centres` |
| [`05_smpl_parameters_to_smpl18.py`](05_smpl_parameters_to_smpl18.py) | SMPL / SMPL-H / SMPL-X parameters (`.npz`) | `smpl18 convert smpl` |
| [`06_read_the_corpus.py`](06_read_the_corpus.py) | a corpus from one of the above | reading it in Python |
| [`07_render_in_blender.py`](07_render_in_blender.py) | a corpus, and somewhere to look at it | `smpl18 export-smpl`, `smpl18 blender` |

## Running them

From the repository root, after `python -m pip install -e .` (or without installing — the
examples put `src/` on the path themselves):

```bash
python examples/01_markers_to_smpl18.py
python examples/06_read_the_corpus.py example-output/01_markers/corpus
```

Options every conversion example takes:

| Option | Meaning |
|---|---|
| `--work DIR` | where inputs and corpora go (default `./example-output`, one sub-directory per example) |
| `--models DIR` | your extracted SMPL models (see the main README); without it a **stand-in body** is written and used |
| `--frames N` | frames per synthetic trial (default 300) |

Example 7 takes two more: `--blender EXE` to build the scene with an installed Blender (without it
the plan is written and the command printed), and `--correctives` to carry SMPL's pose blend shapes
in as shape keys. Its `--models` directory must have been extracted with
`smpl18 extract-model --with-mesh`, since a render needs the skinning weights; the stand-in body it
falls back to is written with a surface of its own.

The stand-in body (`smpl18 demo-models`) has the SMPL-24 tree and ordinary adult proportions so
that everything runs without the licensed models. It is not a human shape model; a corpus made
with it says so (`model_is_stand_in` in `subject.json`, and `smpl18 info` prints it).

## What the numbers mean

The synthetic inputs are exact: markers obey the marker set's rules, joint centres and
skeletons are the motion's own. The remaining error — one to two millimetres at the median, a
few at the 95th percentile, and up to about fifteen for joints the source does not place (the
BVH rig's spine) — is what the method itself costs: the pose prior, the betas'
limited reach, and the 18-joint freeze, whose cost each subject record states. Real captures
add their own error on top (skin movement under markers, joint-centre rules that differ from the
lab's software, a skeleton whose proportions ten betas cannot reach); `smpl18 info` and each
trial's `.manifest.json` report how far the stored poses are from the source's own targets.
