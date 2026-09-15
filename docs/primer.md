# SMPL-24 primer — the skeleton, and how to reach it from SMPL or from markers

| | |
|---|---|
| Nature | explanatory. Defines no format, contract or limit, and states no numeric tolerance |
| Date | 2026-09-15 |
| Scope | the SMPL model and the SMPL-24 skeleton in general terms; the general procedure for obtaining an SMPL-24 pose sequence from SMPL-family parameters or from marker / joint-angle motion capture |
| Out of scope | any particular dataset's conversion record; this package's on-disk format (see `corpus-format.md`); acceptance criteria |

---

## 1. What SMPL is

SMPL (Skinned Multi-Person Linear model) is a **statistical model of the human body with a
small number of parameters**. One model is a triangle mesh of 6,890 vertices and a skeleton of
24 joints; three kinds of parameter fix the shape and the pose.

| Parameter | Symbol | Size | Meaning |
|---|---|---|---|
| shape | `β` | 10 (16 or 300 in some variants) | principal-component coefficients; bone lengths and flesh together |
| pose | `θ` | 24 × 3 = 72 | per joint, the local rotation relative to the parent as an axis-angle triple; the first three are the root's world orientation |
| translation | `t` | 3 | world position of the model origin |

It works in two steps.

1. **Shape → rest skeleton.** The rest-pose mesh vertices are a linear function of `β`, and the
   joint positions `J(β)` are those vertices multiplied by fixed weights (the joint regressor).
   **Bone lengths therefore depend on `β` alone, never on the pose.**
2. **Pose → deformation.** Joint rotations are accumulated along the skeleton tree (forward
   kinematics) and the mesh follows the joints through linear blend skinning (LBS).

If only joints are needed, no mesh has to be built: `J(β)` and forward kinematics give every
joint position and every segment rotation. The mesh matters only for attaching markers to surface
vertices or for estimating body mass from volume.

**Gender models.** Male, female and neutral models are distributed separately. Their rest meshes
and shape components differ, so **the same `β` gives a different skeleton on a different model.**
The gender choice is part of the parameters and must be recorded.

**Family.** SMPL (24 joints), SMPL-H (52 = body 22 + hands 30) and SMPL-X (55 = body 22 + jaw 1
+ eyes 2 + hands 30) **share the definition and order of the 22 body joints (indices 0–21).**
That is the common ground between them; SMPL-24 is those 22 joints plus the two hand joints
below the wrists (22, 23).

**Model files.** The originals are pickles distributed under licence. In practice one extracts the
needed arrays (rest vertices, shape directions, joint regressor, parent table, skinning weights)
into a separate file and reads only that.

---

## 2. The SMPL-24 skeleton

### 2.1 Joint table

| Index | Name | Parent | Segment moved by this joint's rotation |
|---|---|---|---|
| 0 | pelvis | — | pelvis (root); world orientation |
| 1 | left_hip | 0 | left thigh |
| 2 | right_hip | 0 | right thigh |
| 3 | spine1 | 0 | lower lumbar |
| 4 | left_knee | 1 | left shank |
| 5 | right_knee | 2 | right shank |
| 6 | spine2 | 3 | upper lumbar |
| 7 | left_ankle | 4 | left foot |
| 8 | right_ankle | 5 | right foot |
| 9 | spine3 | 6 | thorax |
| 10 | left_foot | 7 | left toes |
| 11 | right_foot | 8 | right toes |
| 12 | neck | 9 | neck |
| 13 | left_collar | 9 | left clavicle |
| 14 | right_collar | 9 | right clavicle |
| 15 | head | 12 | head |
| 16 | left_shoulder | 13 | left upper arm |
| 17 | right_shoulder | 14 | right upper arm |
| 18 | left_elbow | 16 | left forearm |
| 19 | right_elbow | 17 | right forearm |
| 20 | left_wrist | 18 | left hand |
| 21 | right_wrist | 19 | right hand |
| 22 | left_hand | 20 | left palm below the wrist (finger root) |
| 23 | right_hand | 21 | right palm below the wrist |

Joints 22 and 23 are the hand joints below the wrists. Body-only work leaves their rotation at
identity; they are not the place for finger poses (section 3.1).

### 2.2 Rest pose and local frames

- The **rest pose** is close to a T-pose with the arms out. With every `θ` at zero the skeleton
  stands in it.
- **In the rest pose every joint's local frame is aligned with the world axes.** SMPL has no
  per-joint rotation offset. A joint's local rotation `R_j` is therefore "how far from the rest
  direction", and the only rest offset between joints is the position difference
  `j_j − j_parent(j)`.
- The **model frame** is Y-up, face towards +Z, right-handed. Data recorded in another frame
  (Z-up, say) changes only the root rotation and the translation, never the local rotations
  (section 3.4).
- **A joint's rotation moves the child segment that starts at that joint.** The knee (4) rotates
  the shank; the thigh is rotated by the hip (1). When a *segment orientation* is needed, use the
  accumulated global rotation `G_j` of section 2.3, not the joint rotation.

### 2.3 Forward kinematics (FK)

With rest joint positions `j` (model frame, `J(β)`) and local rotations `R_j = exp(θ_j)`:

```
T_j = [ R_j | j_j − j_parent(j) ]          (local rigid transform of joint j: rotation + rest offset)
G_0 = [ R_0 | j_0 ]                         (root)
G_j = G_parent(j) · T_j                     (accumulated along the tree)
```

- world position of joint `j`: `p_j = G_j · [0,0,0,1]ᵀ + t`
- world rotation of the segment moved by joint `j`: the rotation part of `G_j`
- world position of the pelvis: `p_0 = j_0 + t`

The last line is the most common trap. **`t` is the translation of the model origin, not the
pelvis position**, and the pelvis does not sit at the origin of the rest skeleton. Moving to a
skeleton format rooted at the pelvis needs a correction by `j_0`.

### 2.4 Why the skeleton alone is enough

Inertial-sensor synthesis, joint-angle comparison and segment-orientation comparison need only
joint positions and segment rotations. The mesh is needed for (a) fitting markers to surface
vertices, (b) volume-based mass estimates, (c) visualisation. For skeleton-only work the model
file needs only the rest vertices, shape directions, joint regressor and parent table.

---

## 3. From SMPL-family parameters to SMPL-24

The input is already SMPL, SMPL-H or SMPL-X parameters (`poses`, `betas`, `trans`, gender, frame
rate). This is **trimming and frame matching**, not retargeting.

### 3.1 Trimming the pose vector

- In all three models the first 66 entries of `poses` are root 3 + body 21 joints × 3: the local
  rotations of joints 0–21.
- Joints 22 and 23 (hands) are filled with identity. The finger joints of SMPL-H and SMPL-X are
  defined differently from SMPL-24's hand joints and are not carried over.
- SMPL-X jaw and eye rotations are dropped.

### 3.2 Matching the shape width

- Use only as many `β` as the target model has (usually 10). Longer inputs are truncated,
  shorter ones zero-padded.
- **Use the same gender model.** The same `β` on a different gender model is a coefficient of a
  different shape space and gives a different skeleton. To change gender, refit `β`
  (section 4.1).

### 3.3 Translation

`t` is used as is. Get the pelvis position through forward kinematics when needed, and remember
`p_0 = j_0 + t` when moving to a pelvis-rooted format (section 2.3).

### 3.4 Frame (up-axis) change

When the data frame differs from the model frame (Z-up data, Y-up model), apply a fixed rotation
`C` to **the root rotation and the translation only**; the local rotations of joints 1–23 are
unchanged.

```
R_0' = C · R_0
t'   = C · (j_0 + t) − j_0   =   C · t + (C − I) · j_0
```

The `(C − I)·j_0` term is easy to drop. The root rotation turns the body about the **pelvis**
`j_0`, while `t` moves the **model origin** (section 2.3); with `t' = C·t` alone the whole body
lands `(C − I)·j_0` away from where it should be. On a real SMPL skeleton the pelvis is off the
origin, so a 90° up-axis change misplaces the body by tens of centimetres. `t' = C·t` is right
only when the pelvis sits at the origin. Check: world joint positions from forward kinematics
before and after the change must differ by the rigid rotation `C`.

`j_0` comes from the subject's rest skeleton (`J(β)`), so apply the change after the gender model
and `β` are fixed. Rotate every other frame-bound quantity (the gravity vector, for one) the same
way, and record which frame the output is stored in.

### 3.5 Resampling in time

- Interpolate local rotations with **unit-quaternion slerp** (or another SO(3) interpolation).
  Component-wise linear interpolation of axis-angle vectors is wrong near an angle of π and across
  wraps.
- Linear or spline interpolation is fine for the translation.
- Check that frame-rate metadata exists per file, and record where it came from when it does not.

### 3.6 Verification

Run forward kinematics on the converted parameters and compare the joint positions with those the
original model code computes from the original parameters. The difference should be at numerical
precision. This one check catches trimming, width, gender and frame mistakes together.

---

## 4. From marker or joint-angle capture to SMPL-24

Two kinds of input:

- **(A) labelled marker trajectories** (c3d, trc): surface point positions only.
- **(B) joint angles already solved on a skeleton** (OpenSim `.mot` + `.osim`, BVH, FBX): a source
  skeleton with its own definition, and a joint-angle time series on it.

Either way, "converting to SMPL-24" means **finding `(β, θ_t, t_t)` such that forward kinematics
on the SMPL skeleton reproduces the source observations** (marker positions or segment
orientations). Shape is fitted once, correspondence is fixed once, pose is solved per frame.

### 4.1 Shape `β`: once per subject

Bone lengths do not depend on pose, so `β` is fitted **once per subject**. Fitting per frame lets
the shape absorb pose errors.

1. **Measure bone lengths.** For (B), take joint-to-joint distances from the source skeleton's
   joint positions. For (A), estimate joint centres in a static calibration frame with a marker
   convention (the midpoint of a medial/lateral pair, or a fixed inward offset from the surface),
   then take the distances.
2. **Choose the corresponding SMPL bones** (the correspondence table of section 4.2).
3. **Solve a least-squares problem.**

   ```
   minimize_β  Σ_bones ( |J_child(β) − J_parent(β)| − L_measured )²  +  λ·|β|²  (+ optional stature term)
   ```

   `J(β)` is linear in `β`, so this converges well.
4. **Handle the undetermined directions.** Bone lengths say nothing about the girth directions of
   `β`. Keep those near zero with regularisation or pin them with auxiliary observations (stature,
   volume), and **record which was used.**
5. **Decide and record the gender.** If the source states it, use that model; if unknown, use
   neutral. Do not guess.
6. **Handle skeletons the model cannot reach.** Ten `β` may not reach some cohort's pelvis or
   shoulder width. Either (a) accept and record the residual, or (b) rescale `J(β)` bone by bone
   to the measurements. With (b) the output is no longer a pure SMPL skeleton, so **the rescaled
   skeleton itself must be stored in the output** and the pose fitted against it, or forward
   kinematics will not reproduce the source.

### 4.2 Joint correspondence

Fix the table from source joints/segments to SMPL joints before fitting poses. Common mismatches:

- the spine: one or two source links against SMPL's `spine1/2/3` plus `neck`;
- clavicles (`collar`): absent in most sources;
- head and hands: many source skeletons end at the neck and wrists;
- feet: toe joints (`foot`) are defined differently.

An SMPL joint the source lacks is handled in one of three ways, and **the choice is recorded per
joint.**

| Treatment | Meaning | Consequence |
|---|---|---|
| weld to parent | local rotation fixed at rest (identity) | the segment moves as one body with its parent; a sensor placed on it reports the parent's signal |
| distribute | one source rotation split across several SMPL joints (a lumbar rotation shared equally by three links) | the total rotation is right, the per-link split is an assumption |
| estimate | solved by IK from neighbouring markers or joints | only possible where observations exist |

The output labels every joint **measured / derived / absent**. Labelling a welded joint as
measured misleads any consumer that trusts that joint's data.

### 4.3 Pose `θ_t`: per frame

**Method I — segment-rotation transfer.** When the source gives segment rotations, as (B) does.

1. Run the source skeleton's forward kinematics to get each segment's world rotation `S_k(t)`.
2. In a calibration state where both skeletons are in their rest poses, take the **alignment
   matrix**

   ```
   A_k = S_k(rest)ᵀ · G_j(rest)       (source segment frame → SMPL segment frame, constant)
   ```

   If the source rest pose differs from SMPL's (an A-pose with the arms down, say), that
   difference is absorbed into `A_k`.
3. Per frame, `G_j(t) = S_k(t) · A_k`, local rotation `R_j = G_parent(j)ᵀ · G_j`, converted to
   axis-angle as `θ_j`.
4. Joints without a correspondence follow section 4.2.

**Method II — position-based IK.** For (A), or for (B) when joint positions are used.

1. **Initialise.** For each segment, solve Kabsch (with reflection correction) between three or
   more source points and their SMPL rest counterparts to get a starting rotation.
2. **Least squares.**

   ```
   minimize_{θ_t, t_t}  Σ_m w_m · | p_m^meas(t) − ( FK_{β,θ_t}(target_m) + t_t + offset_m ) |²
                      + joint-angle regularisation + temporal smoothing (difference to the previous frame)
   ```

   With marker input the targets are mesh vertices and `offset_m` is the marker thickness along
   the surface normal (a vertex correspondence is required). With joint centres only, the targets
   are joints and `offset_m = 0`.
3. Warm-start each frame from the previous solution; drop missing markers from that frame's sum.
4. Without a marker-to-vertex correspondence, first build joint centres from the marker convention
   and reduce to joint-position IK.

Joint range limits are optional in both methods; when used, their values are declared in settings
and copied into the output.

### 4.4 Root: `θ_0` and `t_t`

- The pelvis segment's world rotation is `θ_0`.
- With a measured pelvis world position `p_0(t)`, set `t_t = p_0(t) − R_0(t)·j_0` in the model
  frame (the pelvis is not at the origin of the rest skeleton, hence `j_0`). If the source has no
  root translation (an in-place animation), record that fact and store zeros.
- Up axis: with a Z-up source, either multiply a fixed `C` into every world rotation and position
  to reach the model frame, or keep the data frame Z-up and record it. Use one convention, never a
  mixture.

### 4.5 Post-processing

- **Angle wrapping.** If source joint angles (Euler, axis-angle) wrap at ±π, **unwrap before
  filtering.** A low-pass filter applied to a wrapped signal smears the ±π jump into neighbouring
  frames and produces apparent rotations beyond π. If a source has already done this, re-filter
  only the coordinates that wrapped.
- **Resampling.** Slerp for rotations, linear for positions (section 3.5).
- **Discontinuity scan.** Compute the joint rotation rate between adjacent frames and flag spans
  that exceed a physically possible value; the limit is declared in settings. Spans whose cause is
  not established are marked and excluded, never repaired.
- **Gaps.** Do not fill missing spans by interpolation; mark them.

### 4.6 Verification and record

A conversion is only useful together with its reproduction error.

| Check | Method |
|---|---|
| FK reproduction | distance between SMPL FK joint positions and source joint centres (or markers): median and maximum |
| bone-length residual | `J(β)` bone lengths against measured bone lengths |
| segment-rotation residual (method I) | angle between `G_j(t)` and `S_k(t)·A_k` |
| round trip | rebuild the source skeleton from the obtained `θ` and compare with the original joint angles |

Record: gender and the model file's hash, `β`, whether the skeleton was rescaled, the
correspondence table and each joint's measured/derived/absent label, frame, up axis and units,
where the resampling and filter settings came from, gap and discontinuity spans and the exclusion
list, and the code revision.

---

## 5. Common mistakes

| Mistake | Symptom | Remedy |
|---|---|---|
| using `t` as the pelvis position | the whole skeleton is offset by the rest pelvis offset | `p_0 = j_0 + t` (section 2.3) |
| changing the up axis with `t' = C·t` alone | the whole body is off by `(C − I)·j_0` | `t' = C·(j_0 + t) − j_0` (section 3.4) |
| mixing gender models | same `β`, different bone lengths | record gender as a parameter; refit `β` when it changes |
| component-wise axis-angle interpolation | rotations jump near π | quaternion slerp |
| ignoring rest-pose differences | constant offsets remain on arms and legs | alignment matrices `A_k` from a calibration frame |
| handedness / up-axis confusion | mirrored left and right, gravity in the wrong direction | one frame convention, recorded |
| degrees vs radians, mm vs m | rotations 57× too large, lengths 1000× off | convert where the value is read |
| reading a joint rotation as the parent segment's rotation | a sensor lands one segment too high | the "segment moved" column of section 2.1 |
| putting finger poses into hand joints 22/23 | twisted wrists | leave them at identity |
| unwrapping after filtering | overshoots beyond π remain | unwrap first, then filter |
| labelling a welded joint as measured | consumers trust motion that is not there | per-joint provenance labels |
| fitting `β` per frame | the shape wobbles and hides pose errors | once per subject |

---

## 6. Glossary

| Term | Meaning |
|---|---|
| rest pose (zero pose) | the pose with every joint rotation at zero; close to a T-pose in SMPL |
| local rotation | a joint's rotation seen from the parent joint's frame; `θ_j` |
| global rotation | a segment's orientation in the world frame; the rotation part of the accumulated `G_j` |
| forward kinematics (FK) | computing every joint's world position and orientation from local rotations and the rest skeleton |
| inverse kinematics (IK) | finding joint rotations that reproduce observed positions or orientations |
| joint regressor | the fixed weight matrix that produces joint positions from rest-mesh vertices |
| retarget | carrying motion from one skeleton to another; here, from a source skeleton to SMPL-24 |
| weld | fixing a joint's local rotation at rest so the child segment moves rigidly with the parent |
| wrap | the 2π jump an angle makes when crossing the ±π boundary; must be undone before filtering |
| Kabsch | the closed-form optimal rotation between two point sets; the reflection solution must be rejected |
| slerp | spherical linear interpolation between unit quaternions |
