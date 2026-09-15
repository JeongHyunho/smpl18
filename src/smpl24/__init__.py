"""smpl24 -- convert motion capture into an SMPL-24 pose corpus.

The public surface planned for 0.1 (see docs/plan.md section 4):

    smpl24.model     load and select the extracted SMPL body models, expose their hashes
    smpl24.skeleton  the 24-joint definition, rotations, forward kinematics, frame changes
    smpl24.fit       shape from bone lengths, joint correspondence, pose by rotation transfer or IK
    smpl24.inputs    readers: SMPL-family npz, OpenSim (.osim/.mot/.trc, .b3d), markers, BVH
    smpl24.repair    unwrap-before-filter, resampling, discontinuity scan
    smpl24.corpus    the on-disk corpus (write, read, summary)
    smpl24.validate  forward-kinematics reproduction, bone-length residuals, round trips

This seed release carries only the version; the modules arrive with the migration phases.
"""

__version__ = "0.0.0"

__all__ = ["__version__"]
