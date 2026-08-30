"""Geometrically nonlinear 2D frame solver.

A Fin Ray finger only does its job at deflections where small-strain linear FEA
stops being true: the tip swings tens of degrees and the whole point of the
mechanism is that the load path rotates. So this is a co-rotational
Euler-Bernoulli beam formulation. Strains stay small (1.5 mm walls barely
stretch); rotations do not, and the co-rotational split handles exactly that
case at a fraction of the cost of a continuum model.

Everything is dense numpy. The finger meshes to a few hundred degrees of
freedom, so a dense factorisation per Newton iteration is far cheaper than
pulling in a sparse dependency.

Units are millimetre / newton / megapascal throughout, which is self-consistent:
E in MPa, lengths in mm, areas in mm^2, second moments in mm^4, forces in N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .materials import LinearMaterial, Material


@dataclass(frozen=True)
class Element:
    """One beam between two nodes, with a rectangular cross-section.

    Leave `material` unset for a Hookean element and the closed-form EA/EI
    relations are used directly. Give it a material and the section is
    integrated over fibres instead, which is what a hyperelastic law such as
    TPU needs: with 20% bending strain through the wall the stress is nowhere
    near proportional to the distance from the neutral axis, and the neutral
    axis itself shifts.
    """

    node_i: int
    node_j: int
    youngs_modulus: float
    area: float
    second_moment: float
    material: Material | None = None

    @property
    def thickness(self) -> float:
        """A rectangle has I = b t^3/12 and A = b t, so t = sqrt(12 I / A)."""
        return math.sqrt(12.0 * self.second_moment / self.area)

    @property
    def depth(self) -> float:
        return self.area / self.thickness


@lru_cache(maxsize=8)
def _gauss(count: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(count)
    return points, weights


# Stations along the element and fibres through the thickness. Three stations
# resolve the linearly varying curvature of a Hermite beam once the material
# makes the moment a nonlinear function of it; seven fibres resolve a neutral
# axis that no longer sits at mid-thickness.
LENGTH_STATIONS = 3
THICKNESS_FIBRES = 7


def _section_response(element: Element, axial_strain: float, curvature: float):
    """Integrate the material law over one cross-section.

    Returns the axial force, the moment, and the three section tangents
    S0 = int(Et dA), S1 = int(Et y dA), S2 = int(Et y^2 dA).
    """
    nodes, weights = _gauss(THICKNESS_FIBRES)
    half = element.thickness / 2.0
    y = half * nodes
    area_weights = element.depth * half * weights

    strain = axial_strain - y * curvature
    stress, tangent = element.material.response(strain)

    axial = float(stress @ area_weights)
    moment = float(-(stress * y) @ area_weights)
    s0 = float(tangent @ area_weights)
    s1 = float((tangent * y) @ area_weights)
    s2 = float((tangent * y * y) @ area_weights)
    return axial, moment, s0, s1, s2


def _fibre_local_state(element: Element, stretch: float, theta_i: float,
                       theta_j: float, rest_length: float):
    """Local forces and 3x3 tangent for (stretch, theta_i, theta_j).

    Uses the Hermite curvature of a chord-relative beam,
    kappa(xi) = [(6 xi - 4) theta_i + (6 xi - 2) theta_j] / L,
    which is exactly the shape function set the closed-form 4EI/L, 2EI/L
    stiffness comes from - so a linear material reproduces it identically.
    """
    nodes, weights = _gauss(LENGTH_STATIONS)
    stations = 0.5 * (nodes + 1.0)
    station_weights = 0.5 * weights

    axial_strain = stretch / rest_length
    q = np.zeros(3)
    local = np.zeros((3, 3))

    for xi, weight in zip(stations, station_weights):
        shape_i = 6.0 * xi - 4.0
        shape_j = 6.0 * xi - 2.0
        curvature = (shape_i * theta_i + shape_j * theta_j) / rest_length

        axial, moment, s0, s1, s2 = _section_response(element, axial_strain, curvature)

        q[0] += weight * axial
        q[1] += weight * moment * shape_i
        q[2] += weight * moment * shape_j

        d_axial_strain = np.array([1.0 / rest_length, 0.0, 0.0])
        d_curvature = np.array([0.0, shape_i / rest_length, shape_j / rest_length])
        d_axial = s0 * d_axial_strain - s1 * d_curvature
        d_moment = -s1 * d_axial_strain + s2 * d_curvature

        local[0] += weight * d_axial
        local[1] += weight * shape_i * d_moment
        local[2] += weight * shape_j * d_moment

    return q, local


@dataclass
class Frame:
    """Undeformed geometry plus connectivity. Nodes carry (x, y, theta)."""

    nodes: np.ndarray  # (n, 2)
    elements: list[Element]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes

    def rest_length(self, element: Element) -> float:
        delta = self.nodes[element.node_j] - self.nodes[element.node_i]
        return float(np.hypot(delta[0], delta[1]))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _element_dof(element: Element) -> list[int]:
    i, j = element.node_i, element.node_j
    return [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]


def _element_state(frame: Frame, element: Element, u: np.ndarray):
    """Co-rotational kinematics for one element.

    Returns the local force vector q = (axial, moment_i, moment_j), the 3x6
    strain-displacement matrix B, the current chord length and the vectors r
    and z used to assemble the geometric stiffness.
    """
    dof = _element_dof(element)
    ue = u[dof]

    p0i = frame.nodes[element.node_i]
    p0j = frame.nodes[element.node_j]
    d0 = p0j - p0i
    l0 = float(np.hypot(d0[0], d0[1]))
    beta0 = math.atan2(d0[1], d0[0])

    pi = p0i + ue[0:2]
    pj = p0j + ue[3:5]
    d = pj - pi
    ln = float(np.hypot(d[0], d[1]))
    if ln < 1e-12:
        raise FloatingPointError("element collapsed to zero length")
    beta = math.atan2(d[1], d[0])

    cos_b, sin_b = d[0] / ln, d[1] / ln
    rigid = _wrap(beta - beta0)

    # Deformational (strain-producing) quantities. Subtracting the rigid-body
    # rotation here is the whole trick: a rotated-but-unstrained element gives
    # exactly zero, no matter how large the rotation.
    stretch = ln - l0
    theta_i = _wrap(ue[2] - rigid)
    theta_j = _wrap(ue[5] - rigid)

    if element.material is None:
        axial = element.youngs_modulus * element.area / l0 * stretch
        bend = element.youngs_modulus * element.second_moment / l0
        q = np.array([axial,
                      bend * (4.0 * theta_i + 2.0 * theta_j),
                      bend * (2.0 * theta_i + 4.0 * theta_j)])
        local = np.array([
            [element.youngs_modulus * element.area / l0, 0.0, 0.0],
            [0.0, 4.0 * bend, 2.0 * bend],
            [0.0, 2.0 * bend, 4.0 * bend],
        ])
    else:
        q, local = _fibre_local_state(element, stretch, theta_i, theta_j, l0)

    r = np.array([-cos_b, -sin_b, 0.0, cos_b, sin_b, 0.0])
    z = np.array([sin_b, -cos_b, 0.0, -sin_b, cos_b, 0.0]) / ln

    b_matrix = np.empty((3, 6))
    b_matrix[0] = r
    b_matrix[1] = -z
    b_matrix[1, 2] += 1.0
    b_matrix[2] = -z
    b_matrix[2, 5] += 1.0

    return q, local, b_matrix, ln, l0, r, z


def internal_force(frame: Frame, u: np.ndarray) -> np.ndarray:
    force = np.zeros(frame.n_dof)
    for element in frame.elements:
        q, _local, b_matrix, *_ = _element_state(frame, element, u)
        force[_element_dof(element)] += b_matrix.T @ q
    return force


def tangent_stiffness(frame: Frame, u: np.ndarray) -> np.ndarray:
    stiffness = np.zeros((frame.n_dof, frame.n_dof))
    for element in frame.elements:
        q, local, b_matrix, ln, l0, r, z = _element_state(frame, element, u)
        material = b_matrix.T @ local @ b_matrix
        # Geometric stiffness (Crisfield): axial load acting through the
        # element's rotation, plus the end moments doing work on the chord.
        axial, moment_i, moment_j = q
        geometric = axial * ln * np.outer(z, z)
        geometric += (moment_i + moment_j) / ln * (np.outer(r, z) + np.outer(z, r))
        dof = _element_dof(element)
        stiffness[np.ix_(dof, dof)] += material + geometric
    return stiffness


@dataclass
class SolveResult:
    displacement: np.ndarray
    reaction: np.ndarray
    converged: bool
    iterations: int
    residual: float


def solve_static(
    frame: Frame,
    fixed_dof: np.ndarray,
    external_force: np.ndarray | None = None,
    extra_force=None,
    extra_stiffness=None,
    u0: np.ndarray | None = None,
    max_iterations: int = 60,
    tolerance: float = 1e-8,
    divergence_factor: float = 1e4,
) -> SolveResult:
    """Newton-Raphson with divergence-only backtracking.

    `extra_force(u)` and `extra_stiffness(u)` let a caller add configuration
    dependent loads - contact penalties, follower forces - without this module
    knowing anything about them.

    Full Newton steps are taken by default. A descent line search is wrong here:
    the first step of a bending problem leaves a large spurious axial residual
    (the nodes move sideways before they are allowed to draw in along the
    chord), so the residual norm legitimately spikes on the step that is about
    to converge quadratically. Backtracking only kicks in when the step is
    genuinely diverging or has gone non-finite.
    """
    n_dof = frame.n_dof
    u = np.zeros(n_dof) if u0 is None else u0.copy()
    external = np.zeros(n_dof) if external_force is None else external_force
    free = np.setdiff1d(np.arange(n_dof), np.asarray(fixed_dof, dtype=int))

    def parts(state: np.ndarray):
        internal = internal_force(frame, state)
        applied = external.copy()
        if extra_force is not None:
            applied = applied + extra_force(state)
        return internal, applied

    def residual_and_scale(state: np.ndarray):
        internal, applied = parts(state)
        out = internal - applied
        # Relative to whichever force system is actually present, so the same
        # tolerance means the same thing for a 0.01 N nudge and a 50 N squeeze.
        scale = max(float(np.linalg.norm(applied[free])),
                    float(np.linalg.norm(internal[free])), 1e-9)
        return out, scale

    residual, scale = residual_and_scale(u)
    for iteration in range(1, max_iterations + 1):
        norm = float(np.linalg.norm(residual[free]))
        if norm <= tolerance * scale:
            return SolveResult(u, residual, True, iteration - 1, norm / scale)

        stiffness = tangent_stiffness(frame, u)
        if extra_stiffness is not None:
            stiffness = stiffness + extra_stiffness(u)
        kff = stiffness[np.ix_(free, free)]
        try:
            step = np.linalg.solve(kff, -residual[free])
        except np.linalg.LinAlgError:
            # Near a limit point the tangent goes singular; a small diagonal
            # shift keeps the iteration moving instead of aborting the sweep.
            shift = 1e-9 * np.trace(kff) / len(free)
            step = np.linalg.solve(kff + shift * np.eye(len(free)), -residual[free])

        ceiling = divergence_factor * max(norm, scale)
        factor = 1.0
        for _ in range(25):
            trial = u.copy()
            trial[free] += factor * step
            trial_residual, trial_scale = residual_and_scale(trial)
            trial_norm = float(np.linalg.norm(trial_residual[free]))
            if np.isfinite(trial_norm) and trial_norm <= ceiling:
                break
            factor *= 0.5
        u, residual, scale = trial, trial_residual, trial_scale

    norm = float(np.linalg.norm(residual[free]))
    return SolveResult(u, residual, norm <= tolerance * scale * 100.0,
                       max_iterations, norm / scale)


def element_forces(frame: Frame, u: np.ndarray) -> np.ndarray:
    """Per-element (axial N, moment at i, moment at j) in the local frame."""
    return np.array([_element_state(frame, e, u)[0] for e in frame.elements])


def peak_fibre_strain(frame: Frame, u: np.ndarray) -> tuple[float, int]:
    """Largest |nominal strain| in any fibre, and which element carries it.

    For an elastomer this is the number that decides whether the constitutive
    law is being asked something it can answer, which matters more here than
    any stress limit - TPU will not break at these loads, but a Hookean fit to
    it stops meaning anything past a few percent.
    """
    worst, where = 0.0, -1
    for index, element in enumerate(frame.elements):
        q, _local, _b, ln, l0, *_ = _element_state(frame, element, u)
        axial_strain = (ln - l0) / l0
        if element.material is None:
            flexural = element.youngs_modulus * element.second_moment
            curvature = max(abs(q[1]), abs(q[2])) / flexural
        else:
            # q[1], q[2] are work-conjugate to the end rotations, so recover the
            # rotations from the displacement instead of inverting the law.
            dof = _element_dof(element)
            ue = u[dof]
            d = (frame.nodes[element.node_j] + ue[3:5]) - (frame.nodes[element.node_i] + ue[0:2])
            d0 = frame.nodes[element.node_j] - frame.nodes[element.node_i]
            rigid = _wrap(math.atan2(d[1], d[0]) - math.atan2(d0[1], d0[0]))
            theta_i = _wrap(ue[2] - rigid)
            theta_j = _wrap(ue[5] - rigid)
            curvature = max(abs(4.0 * theta_i + 2.0 * theta_j),
                            abs(2.0 * theta_i + 4.0 * theta_j)) / l0
        strain = abs(axial_strain) + curvature * element.thickness / 2.0
        if strain > worst:
            worst, where = strain, index
    return worst, where


def peak_fibre_stress(frame: Frame, u: np.ndarray,
                      thickness: np.ndarray) -> tuple[float, int]:
    """Largest |axial + bending| fibre stress, and which element carries it.

    Printed parts fail long before a beam model stops converging, so this is
    the number that decides whether a simulated deflection is one the real
    finger would survive.
    """
    forces = element_forces(frame, u)
    worst, where = 0.0, -1
    for index, (element, (axial, moment_i, moment_j)) in enumerate(
            zip(frame.elements, forces)):
        moment = max(abs(moment_i), abs(moment_j))
        section_modulus = element.second_moment / (thickness[index] / 2.0)
        stress = abs(axial) / element.area + moment / section_modulus
        if stress > worst:
            worst, where = stress, index
    return worst, where


def deformed_nodes(frame: Frame, u: np.ndarray) -> np.ndarray:
    return frame.nodes + u.reshape(-1, 3)[:, :2]


def rectangular_section(thickness: float, depth: float) -> tuple[float, float]:
    """Area and second moment for a rectangle bending about its thin axis."""
    return thickness * depth, depth * thickness ** 3 / 12.0
