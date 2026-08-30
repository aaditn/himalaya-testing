"""Press a rigid object into the finger and watch what the profile does.

The interesting output is not the tip deflection. It is whether the tip rotates
*towards* the object as the face is pushed in - the Fin Ray behaviour - and how
much of the contact face ends up touching. A plain cantilever does the opposite:
its tip rotates away and contact stays a single point.

Contact is a penalty formulation on the front member. Rather than pushing nodes
around by hand the obstacle is inflated by half the wall thickness, which makes
the centreline nodes an exact stand-in for the real outer surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fem2d import Frame, deformed_nodes, solve_static, tangent_stiffness
from .finray_geometry import FingerModel, FingerSpec, build_finger


class Obstacle:
    """A rigid body the finger presses against, positioned by one scalar.

    `gap` returns penetration depth, outward normal, and the local convex
    radius of the surface at each point. The radius lets the solver add the
    follower term that appears when a contacting node slides around curvature;
    flat stretches return infinity and contribute nothing.
    """

    def surface(self, advance: float) -> np.ndarray:
        raise NotImplementedError

    def gap(self, points: np.ndarray, advance: float):
        raise NotImplementedError


@dataclass
class FlatObstacle(Obstacle):
    """A rigid flat face of finite extent, with rounded edges.

    The extent matters. An unbounded plate eventually reaches the clamped root,
    where the face cannot move out of the way, and the penalty then reports
    hundreds of newtons of pure artefact.

    The edge radius matters too, and not only for realism: a square edge makes
    the constraint jump discontinuously as nodes slide past x = span, which is
    enough on its own to stall Newton at a perfectly stable configuration.
    Every real gripped object has some edge break.
    """

    offset: float = 0.0
    span: tuple[float, float] = (-1e9, 1e9)
    edge_radius: float = 1.0

    def _corners(self, advance: float):
        level = -self.offset + advance
        radius = max(self.edge_radius, 1e-6)
        lo = self.span[0] + radius
        hi = self.span[1] - radius
        return level, radius, lo, hi

    def gap(self, points: np.ndarray, advance: float):
        level, radius, lo, hi = self._corners(advance)
        x, y = points[:, 0], points[:, 1]
        depth = level - y
        normal = np.zeros_like(points)
        normal[:, 1] = 1.0
        curvature_radius = np.full(len(points), np.inf)

        for edge_x, outside in ((lo, x < lo), (hi, x > hi)):
            if not outside.any():
                continue
            centre = np.array([edge_x, level - radius])
            delta = points[outside] - centre
            distance = np.maximum(np.hypot(delta[:, 0], delta[:, 1]), 1e-9)
            depth[outside] = radius - distance
            normal[outside] = delta / distance[:, None]
            curvature_radius[outside] = radius
        return depth, normal, curvature_radius

    def surface(self, advance: float) -> np.ndarray:
        level, radius, lo, hi = self._corners(advance)
        angles = np.linspace(0.0, math.pi, 40)
        left = np.column_stack([lo - radius * np.sin(angles),
                                (level - radius) + radius * np.cos(angles)])
        right = np.column_stack([hi + radius * np.sin(angles[::-1]),
                                 (level - radius) + radius * np.cos(angles[::-1])])
        return np.vstack([left[::-1], right])


@dataclass
class CylinderObstacle(Obstacle):
    """A rigid circular object - the thing the gripper is actually holding."""

    radius: float
    station: float           # x of the centre
    clearance: float = 0.0   # gap below the face at advance 0

    def centre(self, advance: float) -> np.ndarray:
        return np.array([self.station, -(self.radius + self.clearance) + advance])

    def gap(self, points: np.ndarray, advance: float):
        delta = points - self.centre(advance)
        distance = np.maximum(np.hypot(delta[:, 0], delta[:, 1]), 1e-9)
        normal = delta / distance[:, None]
        return (self.radius - distance, normal,
                np.full(len(points), float(self.radius)))

    def surface(self, advance: float) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * math.pi, 121)
        return self.centre(advance) + self.radius * np.column_stack(
            [np.cos(angles), np.sin(angles)])


def _ramp(gap: np.ndarray, eps: float):
    """A C1 stand-in for max(0, gap), plus its derivative.

    The sharp version makes Newton chatter: a node sitting within rounding of
    the surface flips in and out of the active set between iterations and the
    solve stalls at a perfectly stable configuration. Blending across a band a
    few hundredths of a millimetre wide costs nothing physically and makes the
    residual differentiable, which is what Newton actually needs.
    """
    safe = np.maximum(gap, -2.0 * eps)
    blend = (safe + eps) ** 2 / (4.0 * eps)
    value = np.where(gap > eps, safe, np.where(gap < -eps, 0.0, blend))
    slope = np.where(gap > eps, 1.0, np.where(gap < -eps, 0.0, (safe + eps) / (2.0 * eps)))
    return value, slope


def _contact_stiffness(model: FingerModel, ratio: float = 300.0) -> float:
    """Penalty stiffness scaled to the finger's own normal compliance.

    Fixing a number here would be wrong: TPU and PLA are two orders of magnitude
    apart, and a penalty that is glassy for one is mush for the other.
    """
    frame = model.frame
    probe = model.front_nodes[len(model.front_nodes) // 2]
    force = np.zeros(frame.n_dof)
    force[3 * probe + 1] = 1.0
    stiffness = tangent_stiffness(frame, np.zeros(frame.n_dof))
    free = np.setdiff1d(np.arange(frame.n_dof), model.fixed_dof)
    deflection = np.linalg.solve(stiffness[np.ix_(free, free)], force[free])
    index = int(np.where(free == 3 * probe + 1)[0][0])
    return ratio / abs(float(deflection[index]))


@dataclass
class Step:
    advance: float
    displacement: np.ndarray
    contact_force: np.ndarray      # per contact node, global xy
    contact_nodes: np.ndarray      # node indices actually touching
    converged: bool

    @property
    def total_force(self) -> np.ndarray:
        return self.contact_force.sum(axis=0) if len(self.contact_force) else np.zeros(2)


@dataclass
class SimulationResult:
    model: FingerModel
    obstacle: Obstacle
    steps: list[Step] = field(default_factory=list)
    penalty: float = 0.0
    completed: bool = True

    @property
    def reached_mm(self) -> float:
        return self.steps[-1].advance if self.steps else 0.0

    def deformed(self, index: int) -> np.ndarray:
        return deformed_nodes(self.model.frame, self.steps[index].displacement)


def press(model: FingerModel, obstacle: Obstacle, max_advance: float,
          n_steps: int = 24, penalty: float | None = None,
          contact_tol: float = 0.01,
          max_substeps: int = 64,
          smoothing: float = 0.02,
          penalty_ratio: float = 300.0) -> SimulationResult:
    frame = model.frame
    spec = model.spec
    penalty = (penalty if penalty is not None
               else _contact_stiffness(model, penalty_ratio))
    # Only the free part of the contact face can touch anything. The base
    # block is buried in the gripper mount, and leaving its clamped nodes in
    # the contact set lets an obstacle "penetrate" a node that cannot move,
    # which manufactures unbounded penalty force out of nothing.
    clamped = set(np.asarray(model.fixed_dof, dtype=int) // 3)
    contact_dofs = np.asarray([n for n in model.front_nodes if n not in clamped],
                              dtype=int)
    inflate = spec.wall / 2.0

    def points_of(u: np.ndarray) -> np.ndarray:
        return frame.nodes[contact_dofs] + u.reshape(-1, 3)[contact_dofs, :2]

    def make_gap(advance: float):
        def evaluate(u: np.ndarray):
            points = points_of(u)
            depth, normal, curvature_radius = obstacle.gap(points, advance)
            # Inflating the obstacle by half a wall makes centreline nodes stand
            # in for the outer surface exactly, for any convex obstacle.
            return depth + inflate, normal, curvature_radius
        return evaluate

    def force_fn(advance: float):
        evaluate = make_gap(advance)

        def apply(u: np.ndarray) -> np.ndarray:
            depth, normal, _radius = evaluate(u)
            value, _slope = _ramp(depth, smoothing)
            out = np.zeros(frame.n_dof)
            active = np.where(value > 0.0)[0]
            for index in active:
                node = contact_dofs[index]
                out[3 * node:3 * node + 2] += penalty * value[index] * normal[index]
            return out
        return apply

    def stiffness_fn(advance: float):
        evaluate = make_gap(advance)

        def apply(u: np.ndarray) -> np.ndarray:
            depth, normal, curvature_radius = evaluate(u)
            value, slope = _ramp(depth, smoothing)
            out = np.zeros((frame.n_dof, frame.n_dof))
            for index in np.where(slope > 0.0)[0]:
                n = normal[index]
                block = penalty * slope[index] * np.outer(n, n)
                reach = curvature_radius[index]
                if np.isfinite(reach):
                    # A curved surface lets a contacting node slide around it;
                    # this follower term matters once the wrap gets deep.
                    block -= (penalty * value[index] / max(reach, 1e-9)
                              * (np.eye(2) - np.outer(n, n)))
                node = contact_dofs[index]
                slot = slice(3 * node, 3 * node + 2)
                out[slot, slot] += block
            return out
        return apply

    result = SimulationResult(model=model, obstacle=obstacle, penalty=penalty)
    u = np.zeros(frame.n_dof)

    def record(advance: float, state: np.ndarray, converged: bool) -> None:
        depth, normal, _radius = make_gap(advance)(state)
        value, _slope = _ramp(depth, smoothing)
        loaded = value > 0.0
        traction = (penalty * value[loaded][:, None] * normal[loaded]
                    if loaded.any() else np.zeros((0, 2)))
        touching = depth > -contact_tol
        result.steps.append(Step(float(advance), state.copy(), traction,
                                 contact_dofs[touching], converged))

    # Adaptive continuation. A soft finger at several millimetres of indentation
    # will not converge in one jump from the previous state; halving the
    # increment and walking in recovers it. Without this the sweep silently
    # returns a diverged state that still looks like a number.
    reached = 0.0
    increment = max_advance / n_steps
    while reached < max_advance - 1e-12:
        target = min(reached + increment, max_advance)
        solved = solve_static(frame, model.fixed_dof, None,
                              extra_force=force_fn(target),
                              extra_stiffness=stiffness_fn(target),
                              u0=u)
        if not solved.converged:
            if increment > max_advance / (n_steps * max_substeps):
                increment *= 0.5
                continue
            # Give back only states that actually solved. A diverged step still
            # carries a displacement vector and a contact force, and both are
            # nonsense; recording it would put a fake number in the summary
            # table. `reached_mm` tells the caller how far it really got.
            result.completed = False
            return result
        u = solved.displacement
        reached = target
        record(target, u, True)
        # Creep back towards the requested step size once it is behaving.
        increment = min(increment * 1.3, max_advance / n_steps)
    return result


def tip_pose(model: FingerModel, u: np.ndarray) -> tuple[np.ndarray, float]:
    """Tip position and rotation. Rotation is the number that matters:
    positive means the tip is curling towards the object."""
    node = model.tip_node
    position = model.frame.nodes[node] + u.reshape(-1, 3)[node, :2]
    return position, float(u[3 * node + 2])


def contact_patch(model: FingerModel, step: Step) -> tuple[float, int]:
    """Length of contact face touching the object, and how many nodes.

    A stiff penalty parks contacting nodes within microns of the surface, so a
    strict `penetration > 0` test undercounts the patch badly - it reports a
    single node for what is plainly a several-millimetre band. `Step` therefore
    stores nodes inside a small geometric tolerance instead.
    """
    if len(step.contact_nodes) == 0:
        return 0.0, 0
    stations = np.asarray(model.front_stations)
    order = {node: i for i, node in enumerate(model.front_nodes)}
    touched = sorted(stations[order[n]] for n in step.contact_nodes)
    return float(touched[-1] - touched[0]), len(touched)


def summarise(result: SimulationResult) -> list[dict]:
    rows = []
    for index, step in enumerate(result.steps):
        position, rotation = tip_pose(result.model, step.displacement)
        rest = result.model.frame.nodes[result.model.tip_node]
        patch, count = contact_patch(result.model, step)
        total = step.total_force
        rows.append({
            "advance_mm": step.advance,
            "normal_force_n": float(total[1]),
            "shear_force_n": float(total[0]),
            "tip_dx_mm": float(position[0] - rest[0]),
            "tip_dy_mm": float(position[1] - rest[1]),
            "tip_rotation_deg": math.degrees(rotation),
            "contact_patch_mm": patch,
            "contact_nodes": count,
            "converged": step.converged,
        })
    return rows


def default_model(material_mpa: float = 2000.0, depth: float = 15.0,
                  rib_thickness: float = 1.5) -> FingerModel:
    return build_finger(FingerSpec(youngs_modulus=material_mpa, depth=depth,
                                   rib_thickness=rib_thickness))
