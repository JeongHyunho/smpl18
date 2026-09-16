"""smpl18 -- convert motion capture into an SMPL-24 pose corpus.

One rule shapes the package: code implements source kinds and file formats; a dataset is a
profile (YAML) that binds them. No module knows a dataset by name.

The public surface planned for 0.1 (see docs/plan.md sections 3 and 4):

    smpl18.model     load and select the extracted SMPL body models, expose their hashes
    smpl18.skeleton  the 24-joint definition, rotations, forward kinematics, frame changes
    smpl18.formats   file readers that return tables in the file's own names (npz, pickle, json,
                     osim, mot, trc, c3d, b3d, bvh, mat)
    smpl18.sources   the four source kinds: smpl_parameters, skeleton_motion (with generic
                     skeleton models: OpenSim-style and BVH-style FK), joint_centres,
                     marker_trajectories
    smpl18.profile   profile schema, loading, field binding, file-layout discovery
    smpl18.fit       shape from bone lengths, joint correspondence, pose by rotation transfer or IK
    smpl18.reduce    the 18-joint reduction: four joints fitted to per-subject constants
    smpl18.repair    unwrap-before-filter, resampling, discontinuity scan
    smpl18.corpus    the on-disk corpus (write, read, summary)
    smpl18.validate  forward-kinematics reproduction, bone-length residuals, round trips
    smpl18.convert   one pipeline per source kind, driven by a bound profile

This seed release carries only the version; the modules arrive with the migration phases.
"""

__version__ = "0.0.0"

__all__ = ["__version__"]
