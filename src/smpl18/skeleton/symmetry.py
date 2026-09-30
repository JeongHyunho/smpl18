"""Left/right symmetry of the rest skeleton: the mirror operator and the modes that apply it.

Why. The SMPL template is not left/right symmetric. At zero shape the left leg (thigh plus shank)
is 9.7 mm longer than the right in the male model and 7.2 mm and 8.2 mm shorter in the female and
neutral ones; the male right hip sits 8 mm below the left; and the template mesh is not its own
mirror image (nearest-vertex mismatch 3-4 mm at the median, 13-14 mm at most). The joint-level
shape directions carry an antisymmetric part too (5-7 % of their norm), which the large betas a
shape fit reaches can amplify: subjects whose measured left and right legs agree within 0.3 mm
were fitted with legs 7-34 mm apart, and a left/right equality term in the fit, on SMPL's own
skeleton, removed only about half of that. So the skeleton itself is made symmetric.

The operator. With ``Q`` the permutation that swaps every left joint with its right partner
(:data:`MIRROR_PARTNER`) and ``M = diag(-1, 1, 1)`` the reflection through the plane x = 0 (SMPL's
rest frame puts the subject's left towards +x)::

    S(J) = (J + Q M J) / 2        the symmetric part
    A(J) = (J - Q M J) / 2        the antisymmetric part, J = S(J) + A(J)

``S`` is idempotent and leaves a mirror-symmetric skeleton unchanged. It puts every midline joint
on x = 0; a mirror plane anywhere else would only translate the skeleton, which changes no bone.
A shape-direction array ``(24, 3, k)`` is treated direction by direction, so
``S(J0 + D beta) = S(J0) + S(D) beta``.

The modes, chosen by the settings key ``model.symmetry`` and carried by the model
(:attr:`smpl18.model.load.Model.symmetry`):

* ``none`` -- SMPL's own skeleton, ``J(beta) = J0 + D beta``, fitted as before the modes existed
  (the equality condition is refused here: no beta removes the template's own asymmetry);
* ``template`` -- the template is mirrored, the shape directions are not:
  ``J(beta) = S(J0) + D beta``; their antisymmetric part is left for the shape fit's left/right
  equality condition (``shape.lr_equality_weight``) to hold down;
* ``skeleton`` -- both, ``J(beta) = S(J0 + D beta)``: left and right bones are equal for every
  beta by construction. The default.

Only the joints are mirrored. The surface (:func:`smpl18.skeleton.kinematics.shaped_vertices`)
stays SMPL's, so in the two symmetric modes the joints a surface is skinned about are not where
the regressor reads them off that surface. The gap is the model's own asymmetry at those betas:
at most 4-9 mm for the three templates at zero shape, but it grows along the limbs with large
betas -- 36 mm at a foot for a male body whose bone-length fit reached betas of 13.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .definition import JOINT_NAMES, NUM_JOINTS

__all__ = [
    "DEFAULT_SYMMETRY",
    "MIRROR_PARTNER",
    "SETTINGS_SECTION",
    "SYMMETRY_MODES",
    "antisymmetric_part",
    "check_symmetry",
    "mirror",
    "mirror_pairs",
    "symmetric_part",
    "symmetry_setting",
]

SYMMETRY_MODES: tuple[str, ...] = ("none", "template", "skeleton")
#: What a model built or loaded without settings carries.
DEFAULT_SYMMETRY = "skeleton"
SETTINGS_SECTION = "model"


def _partner(name: str) -> int:
    for this, other in (("left_", "right_"), ("right_", "left_")):
        if name.startswith(this):
            return JOINT_NAMES.index(other + name[len(this):])
    return JOINT_NAMES.index(name)


#: Per joint, the joint it swaps with under the mirror: its other-side partner, or itself on the
#: midline (1-2, 4-5, 7-8, 10-11, 13-14, 16-17, 18-19, 20-21, 22-23).
MIRROR_PARTNER: tuple[int, ...] = tuple(_partner(name) for name in JOINT_NAMES)

_REFLECTION = np.array([-1.0, 1.0, 1.0])


def mirror(values) -> np.ndarray:
    """``Q M J``: the mirror image of ``(24, 3)`` joints or ``(24, 3, k)`` shape directions."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[:2] != (NUM_JOINTS, 3):
        raise ValueError(f"expected (24, 3) joints or (24, 3, k) directions, got {values.shape}")
    reflection = _REFLECTION if values.ndim == 2 else _REFLECTION[:, None]
    return values[list(MIRROR_PARTNER)] * reflection


def symmetric_part(values) -> np.ndarray:
    """``S(J) = (J + Q M J) / 2``."""
    values = np.asarray(values, dtype=np.float64)
    return 0.5 * (values + mirror(values))


def antisymmetric_part(values) -> np.ndarray:
    """``A(J) = (J - Q M J) / 2``, what :func:`symmetric_part` removes."""
    values = np.asarray(values, dtype=np.float64)
    return 0.5 * (values - mirror(values))


def mirror_pairs(pairs: Sequence[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """The joint pairs among ``pairs`` whose mirror image is among them too, as ``(left, right)``.

    A pair that is its own mirror image (the two hips) has no partner and is left out; each
    left/right couple is listed once, the pair holding a left joint first.
    """
    present = {frozenset(pair): tuple(pair) for pair in pairs}
    seen: set[frozenset] = set()
    out = []
    for pair in pairs:
        key = frozenset(pair)
        image = frozenset(MIRROR_PARTNER[joint] for joint in pair)
        if key in seen or image == key or image not in present:
            continue
        seen.update((key, image))
        if any(JOINT_NAMES[joint].startswith("left_") for joint in pair):
            out.append((tuple(pair), present[image]))
        else:
            out.append((present[image], tuple(pair)))
    return out


def check_symmetry(mode: str) -> str:
    """``mode`` when it is one of :data:`SYMMETRY_MODES`, else ``ValueError``."""
    if mode not in SYMMETRY_MODES:
        raise ValueError(f"symmetry must be one of {', '.join(SYMMETRY_MODES)}, got {mode!r}")
    return mode


def symmetry_setting(settings: Mapping[str, Any]) -> str:
    """``model.symmetry`` from a settings mapping, required and checked."""
    section = settings.get(SETTINGS_SECTION) if isinstance(settings, Mapping) else None
    if not isinstance(section, Mapping) or "symmetry" not in section:
        raise ValueError(f"settings is missing {SETTINGS_SECTION}.symmetry")
    return check_symmetry(str(section["symmetry"]))
