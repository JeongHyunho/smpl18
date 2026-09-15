"""smpl24 -- convert motion capture into an SMPL-24 pose corpus.

One rule shapes the package: code implements source kinds and file formats; a dataset is a
profile (YAML) that binds them. No module knows a dataset by name.

The public surface planned for 0.1 (see docs/plan.md sections 3 and 4):

    smpl24.model     load and select the extracted SMPL body models, expose their hashes
    smpl24.skeleton  the 24-joint definition, rotations, forward kinematics, frame changes
    smpl24.formats   file readers that return tables in the file's own names (npz, pickle, json,
                     osim, mot, trc, c3d, b3d, bvh, mat)
    smpl24.sources   the four source kinds: smpl_parameters, skeleton_motion (with generic
                     skeleton models: OpenSim-style and BVH-style FK), joint_centres,
                     marker_trajectories
    smpl24.profile   profile schema, loading, field binding, file-layout discovery
    smpl24.fit       shape from bone lengths, joint correspondence, pose by rotation transfer or IK
    smpl24.repair    unwrap-before-filter, resampling, discontinuity scan
    smpl24.corpus    the on-disk corpus (write, read, summary)
    smpl24.validate  forward-kinematics reproduction, bone-length residuals, round trips
    smpl24.convert   one pipeline per source kind, driven by a bound profile

This seed release carries only the version; the modules arrive with the migration phases.
"""

__version__ = "0.0.0"

__all__ = ["__version__"]
