"""smpl18 -- convert motion capture into an 18-joint reduced SMPL pose corpus.

Labelled markers, OpenSim kinematics, joint-centre trajectories, BVH skeletons and SMPL-family
parameters all end in the same corpus: per-subject betas, per-trial poses of the 18 kept joints,
and the four frozen-joint constants that rebuild the 24-joint pose. One rule shapes the package:
code implements source kinds and file formats; a dataset is described by data (a profile, a
marker set, a correspondence table), never by a module.

    smpl18.formats    file readers that return tables in the file's own names (npz, pickle,
                      json, osim, mot, trc, c3d, b3d, bvh, mat)
    smpl18.sources    the four source kinds, the skeleton models (OpenSim, BVH), marker sets,
                      subject files
    smpl18.fit        targets, correspondence tables, the shape fit, the per-frame pose solve
    smpl18.reduce     the 18-joint reduction: four joints fitted to per-subject constants
    smpl18.convert    one reader per source kind, the subject fit, the corpus writer
    smpl18.corpus     the on-disk corpus: write, read, rebuild 24 joints
    smpl18.model      extracted SMPL models (and a stand-in body for trying the pipeline)
    smpl18.skeleton   the 24-joint definition, rotations, forward kinematics, frame changes
    smpl18.profile    dataset profiles: schema, loading, layout discovery, field binding
    smpl18.synthetic  synthetic motion and captures of it, for the examples and tests

The command line is ``smpl18`` (``smpl18.cli``).
"""

__version__ = "0.0.0"

__all__ = ["__version__"]
