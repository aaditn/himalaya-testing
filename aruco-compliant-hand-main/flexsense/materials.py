"""Uniaxial stress-strain laws for the fibre-integrated beam.

TPU 95A is not a material with a Young's modulus. Quoting one is a statement
about the first few percent of strain, and this finger runs to more than twenty
percent in bending, where the tangent modulus has already fallen to a fraction
of its initial value. Everything here returns nominal (engineering) stress
against nominal strain, because that is what the beam kinematics feed it and
what a tensile pull on a printed strip measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Material:
    """A uniaxial law: nominal stress and tangent modulus from nominal strain."""

    def response(self, strain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @property
    def initial_modulus(self) -> float:
        stress, tangent = self.response(np.array([0.0]))
        return float(tangent[0])


@dataclass(frozen=True)
class LinearMaterial(Material):
    """Hookean. Correct for PLA and PETG at these strains, and the reference
    the fibre integration is checked against."""

    youngs_modulus: float

    def response(self, strain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        strain = np.asarray(strain, dtype=float)
        return (self.youngs_modulus * strain,
                np.full(strain.shape, self.youngs_modulus))


@dataclass(frozen=True)
class Yeoh(Material):
    """Incompressible Yeoh solid in uniaxial extension.

    W = c10*(I1-3) + c20*(I1-3)^2 + c30*(I1-3)^3, and with lambda = 1 + strain,
    I1 = lambda^2 + 2/lambda. Setting c20 = c30 = 0 gives neo-Hookean.

    The initial Young's modulus is 6*c10. c20 is negative for elastomers - it
    produces the early softening that a linear model misses entirely - and c30
    is positive so the curve turns back up instead of running away.
    """

    c10: float
    c20: float = 0.0
    c30: float = 0.0

    def response(self, strain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        strain = np.asarray(strain, dtype=float)
        # A fibre on the compressive side can be driven hard; clamp the stretch
        # away from zero so the 1/lambda terms stay finite during a Newton
        # iteration that has overshot.
        stretch = np.clip(1.0 + strain, 0.05, None)

        first_invariant = stretch ** 2 + 2.0 / stretch - 3.0
        slope = (self.c10 + 2.0 * self.c20 * first_invariant
                 + 3.0 * self.c30 * first_invariant ** 2)
        curvature = 2.0 * self.c20 + 6.0 * self.c30 * first_invariant

        kinematic = stretch - stretch ** -2.0
        stress = 2.0 * kinematic * slope

        d_kinematic = 1.0 + 2.0 * stretch ** -3.0
        d_invariant = 2.0 * stretch - 2.0 * stretch ** -2.0
        tangent = 2.0 * (d_kinematic * slope + kinematic * curvature * d_invariant)
        return stress, tangent


def neo_hookean(youngs_modulus: float) -> Yeoh:
    return Yeoh(c10=youngs_modulus / 6.0)


# Literature-typical printed TPU 95A, fitted over 0-50% nominal strain - the
# range this finger actually works in. Initial modulus 24.3 MPa, 2.16 MPa at
# 10% strain, 6.5 MPa at 50%. The tangent modulus stays positive out to 100%,
# so a Newton iteration that overshoots still lands somewhere physical.
#
# These are numbers for the hardness, NOT a measurement of your filament, your
# nozzle temperature or your layer orientation. The model is now more sensitive
# to this curve than to anything else in it. Replace it by pulling a printed
# strip and calling `fit_yeoh`.
TPU95A = Yeoh(c10=4.0462, c20=-1.93922, c30=1.271708)

# The samples TPU95A is fitted to, kept so the fit can be re-derived and checked.
TPU95A_REFERENCE = (
    (0.05, 1.30), (0.10, 2.20), (0.20, 3.60), (0.35, 5.20), (0.50, 6.50),
)


def fit_yeoh(strain, stress, c30_positive: bool = True) -> Yeoh:
    """Least-squares fit of the Yeoh coefficients to measured tensile data.

    Nominal stress is linear in (c10, c20, c30) once the stretch kinematics are
    factored out, so this is an ordinary linear least squares rather than an
    optimiser that needs babysitting.
    """
    strain = np.asarray(strain, dtype=float).ravel()
    stress = np.asarray(stress, dtype=float).ravel()
    if strain.size != stress.size or strain.size < 3:
        raise ValueError("need at least three matching strain/stress samples")

    stretch = 1.0 + strain
    if np.any(stretch <= 0.0):
        raise ValueError("strain must be greater than -1")

    invariant = stretch ** 2 + 2.0 / stretch - 3.0
    kinematic = 2.0 * (stretch - stretch ** -2.0)
    design = np.column_stack([
        kinematic,
        2.0 * kinematic * invariant,
        3.0 * kinematic * invariant ** 2,
    ])
    coefficients, *_ = np.linalg.lstsq(design, stress, rcond=None)
    c10, c20, c30 = (float(v) for v in coefficients)
    if c30_positive and c30 < 0.0:
        # Without a rising cubic term the fit can turn over and predict falling
        # stress at large stretch, which is not a material - it is an
        # extrapolation artefact. Refit with the term dropped instead.
        coefficients, *_ = np.linalg.lstsq(design[:, :2], stress, rcond=None)
        c10, c20 = (float(v) for v in coefficients)
        c30 = 0.0
    return Yeoh(c10=c10, c20=c20, c30=c30)
