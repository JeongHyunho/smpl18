# Rendering a corpus in Blender

A corpus stores 18 joints, which is what this package produces. Everything else that reads bodies
— renderers, viewers, the SMPL add-ons, other people's code — reads the original 24-joint SMPL
structure. There are two ways out, and both are exact:

| You want | Command | What you get |
|---|---|---|
| SMPL parameters, for any SMPL reader | `smpl18 export-smpl` | an `.npz` with `poses` `[T, 72]`, `betas`, `trans`, `mocap_framerate`, `gender` |
| A body moving in Blender | `smpl18 blender` | a scene plan, and (with `--blender`) a `.blend` and rendered frames |

Neither re-fits anything. The frozen joints come back at the subject's constants and the hands at
identity, which is exactly what the corpus means: every kept segment's world orientation is the
fitted one, and joint positions are within the residual `subject.json` states.

---

## Back to plain SMPL

```bash
smpl18 export-smpl --corpus corpus --out smpl/             # every subject, every trial
smpl18 export-smpl --corpus corpus --subject S01 --trial walk01 --out S01_walk01.npz
```

The file holds what an SMPL reader looks for, and a little more that would otherwise be lost:

| Key | What it is |
|---|---|
| `poses` | `[T, 72]` axis-angle, 24 joints (`--poses grouped` writes `[T, 24, 3]`) |
| `betas`, `trans`, `mocap_framerate`, `gender` | the subject's shape, the per-frame translation, the rate, the model's gender |
| `up_axis` | the corpus's vertical axis, which is what `trans` and `poses` are expressed in |
| `joint_names`, `joint_provenance` | 24 names, and per joint: `measured`, `derived`, `constant` (a frozen joint: one rotation for the whole subject) or `absent` (the hands) |
| `frame_valid` | which frames the capture really had, before gaps were held |
| `smpl18` | JSON: the subject, the trial, the frozen joints and what the reduction cost |

`trans` moves the model origin, not the pelvis: the pelvis sits at `J(betas)[0] + trans`
([`primer.md`](primer.md) §3.4). A reader that puts the pelvis at `trans` will float the body.

The round trip is checked in the test suite: converting an exported file with
`smpl18 convert smpl` returns the same rotations, translation and betas, and the corpus written
from it has nothing left for the reduction to lose.

---

## Into Blender

```bash
# 1. a body with a surface: the skinning weights and pose blend shapes live in the model file
smpl18 extract-model --pkl basicmodel_neutral_*.pkl --gender neutral --num-betas 10 \
    --out smpl18-models --with-mesh

# 2. build the scene (writes the plan, then runs Blender on it)
smpl18 blender --corpus corpus --subject S01 --trial walk01 \
    --models smpl18-models --out scene/ --blend --render \
    --blender "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
```

Without `--blender` the plan is written and the command to run is printed, which is what to do on a
machine that has no Blender (or to hand the plan to one that does). With no SMPL model at all:

```bash
smpl18 demo-models --out models --with-mesh       # a mannequin of blocks, not a body
```

### Options

| Option | Meaning |
|---|---|
| `--blend` / `--render` | save `<out>/scene.blend` / render PNGs into `<out>/frames` |
| `--frames START:STOP[:STEP]` | take a slice of the trial, to keep a long capture to a scene one can open |
| `--correctives` | carry SMPL's pose blend shapes in as 207 shape keys (see below) |
| `--render-settings FILE` | override the shipped numbers; repeat to merge, later files winning |
| `--models DIR` | where the extracted models are (else `$SMPL18_MODELS`) |

Every number a scene needs — engine, samples, resolution, the camera, the lights, the material,
the self-check — is in [`configs/render/default.yaml`](../configs/render/default.yaml) and nothing
is defaulted in code, as with conversion settings. Copy it, change what you want, and pass both:

```bash
smpl18 blender ... --render-settings configs/render/default.yaml --render-settings mine.yaml
```

### What is in the scene

* **`smpl18_body`** — the shaped rest surface as a mesh, one vertex group per SMPL joint from the
  model's skinning weights, with an armature modifier.
* **`smpl18_armature`** — 24 bones, one per SMPL joint, heads on the rest joint centres. Their
  tails only decide how the rig is *drawn* (see below).
* a camera looking at the body, a sun, a flat world light, and a floor at the lowest vertex.

The body is animated by keyframing each bone, so the scene is an ordinary Blender rig: retime it,
attach anything to a bone, look through the camera, render it however you like.

---

## Why the bones cannot be wrong

This is the part that usually goes wrong when moving SMPL into Blender, so it is worth stating.

The plan does not carry rotations for Blender to interpret. It carries, per frame and joint, the
**rest-to-posed transform** `A_j = [G_j | p_j − G_j j_j]` — the matrix that takes a rest vertex of
that joint to where it belongs. Blender deforms a vertex group by `pose.matrix @ bone.matrix_local⁻¹`,
so the script solves for the channel transform that makes `pose.matrix = A_j @ bone.matrix_local`:

```
matrix_basis = bone.matrix_local⁻¹ @ A_parent⁻¹ @ A_j @ bone.matrix_local
```

Both `matrix_local`s cancel out of the deformation. Nothing therefore depends on where a bone's
tail points, on its roll, or on any convention the two sides would have to agree about — which is
why the bones can be drawn along the body without a correction anywhere. The test suite proves the
cancellation over random bone rest matrices
([`tests/blender/test_scene_script.py`](../tests/blender/test_scene_script.py)).

And because a proof of the algebra is not a proof about Blender, the scene checks itself. The plan
carries a few frames of vertices computed by `smpl18.mesh`; the script evaluates the mesh Blender
actually deforms at those frames and compares:

```
self-check: 3 frames, worst vertex 0.0001 mm from this package's own skinning (tolerance 0.2000 mm)
```

Beyond the tolerance it prints what went wrong, saves nothing and exits non-zero. So a Blender
version that changes the rule, or a mistake here, fails the run instead of quietly bending the body.
`scene.check_frames` and `scene.check_tolerance_m` set how much is checked and how closely.

---

## Pose blend shapes

SMPL's surface is not skinning alone: 207 `posedirs` displace the template as a linear function of
the pose, which is what stops a bent elbow creasing. A static mesh in Blender cannot do that by
itself, so:

* by default they are **left out**. The command prints how far that moves the surface
  (`pose_blend_shapes_mm`) so you can judge whether it matters for your picture — a few millimetres
  around bent joints, nothing at all elsewhere.
* with `--correctives` they are carried in as **207 shape keys** whose values are keyframed per
  frame. The surface is then exact, at the price of a larger plan and file (`posedirs` is
  `[V, 3, 207]`) and slower keyframing.

The stand-in body has no pose blend shapes at all: it is a mannequin of rigid blocks, and says so.

---

## Rendering, in practice

* **`render.engine: cycles`** is the default because Cycles renders on the CPU and therefore works
  on a machine with no display, which is what `blender --background` usually is. EEVEE is far
  faster but needs a GPU and a display server; `--render-settings` can switch to it, and the script
  accepts whichever EEVEE identifier your Blender version uses.
* **`--factory-startup`** is always passed, so a teammate's add-ons and preferences cannot change
  the result.
* Frames land as `<out>/frames/frame_0001.png`; `render.frame_step` renders every *n*th one.
* A long capture is heavy in memory and in file size (a plan is
  `frames × 24 × 16 × 8` bytes of transforms plus the surface once). Use `--frames` to cut it.

## If something goes wrong

| It says | What to do |
|---|---|
| `carries no surface, only the skeleton` | extract the model with `--with-mesh`, or use `demo-models --with-mesh` |
| `Install Blender and pass its executable` | `--blender` did not name a runnable file; give the full path |
| `Blender's deformation is … beyond the … allowed` | the scene was not saved; report it with your Blender version — the plan and the tolerance are in the `.plan.npz` and the `.render.json` beside it |
| `render settings are missing …` | the settings file (or the pair of them) has no such value; nothing is defaulted |
| a body lying on its face | the corpus is Y-up and the object rotation stands it up; check `up_axis` in the plan's `about` if you built the plan by hand |

## Doing it from Python

```python
from smpl18.blender import build_plan, plan_for_trial, run_blender, load_render_settings
from smpl18.corpus import read_corpus
from smpl18.model import Model

trial = read_corpus("corpus").subject("S01").trial("walk01")
model = Model.for_gender(trial.subject.gender, root="smpl18-models")
settings, _ = load_render_settings(["configs/render/default.yaml"])
plan = plan_for_trial(trial, model, correctives=False, sample_frames=3, tolerance=2e-4,
                      leaf_reach=0.7)
plan.write("scene/S01_walk01.plan.npz")
```

`build_plan` takes a pose directly, so a caller with its own 24-joint motion (say, straight from
`smpl18 export-smpl`) need not go through a corpus at all. And `smpl18.mesh` skins a body without
Blender in the picture:

```python
from smpl18.mesh import posed_vertices

posed = posed_vertices(model, trial.subject.betas, trial.local_rotations_24(), trial.trans)
posed.vertices      # (T, V, 3)
posed.transforms    # (T, 24, 4, 4), the matrices a renderer's bones must carry
```
