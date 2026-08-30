"""The conforming finger's side profile, turned into a solvable frame.

Every number here is read off the CAD side view: two straight members meeting
at a sharp tip, 91 mm along the back and 86 mm along the front (the contact
face), 1.5 mm walls, and six ribs whose positions are given indirectly as the
clear inner height of the cavity at each rib.

The taper angle is not dimensioned on the drawing, so it is solved from the
constraint that the clear height at the base equals the dimensioned 26.4 mm.
That is a real cross-check rather than a fit: it also has to reproduce the two
member lengths and a base face square to the front member, and it does, which
is what `tests/test_finray_geometry.py` asserts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fem2d import Element, Frame, rectangular_section
from .materials import Material

# Clear inner heights dimensioned on the drawing, tip to base. Each one locates
# one rib; the last entry is the cavity height where the base block starts.
WINDOW_HEIGHTS_MM = (7.8, 10.9, 14.04, 17.16, 20.276, 24.3)
BASE_CLEAR_HEIGHT_MM = 26.4


@dataclass(frozen=True)
class FingerSpec:
    """Profile dimensions. Lengths are along the outer surface from the tip."""

    front_length: float = 86.0
    back_length: float = 91.0
    wall: float = 1.5
    rib_thickness: float = 1.5
    depth: float = 15.0            # out-of-plane width, NOT on the side view
    youngs_modulus: float = 2000.0  # MPa; the initial slope if `material` is set
    # Leave unset for a Hookean part (PLA, PETG). Set it for an elastomer,
    # where a single modulus stops meaning anything past a few percent strain.
    material: Material | None = None
    window_heights: tuple[float, ...] = WINDOW_HEIGHTS_MM
    base_clear_height: float = BASE_CLEAR_HEIGHT_MM
    elements_per_bay: int = 6
    elements_per_rib: int = 4

    @property
    def taper_angle(self) -> float:
        """Included angle between the two members, in radians.

        Solved from clear_height(front_length) == base_clear_height. The clear
        height perpendicular to the front member at station x is

            H(x) = x*tan(a) - wall/cos(a) - wall

        which is monotonic in a over the physical range, so bisection is safe.
        """
        target = self.base_clear_height

        def clear_at_base(angle: float) -> float:
            return (self.front_length * math.tan(angle)
                    - self.wall / math.cos(angle) - self.wall)

        low, high = 1e-4, math.radians(60.0)
        for _ in range(200):
            mid = 0.5 * (low + high)
            if clear_at_base(mid) < target:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    def clear_height(self, station: float) -> float:
        angle = self.taper_angle
        return station * math.tan(angle) - self.wall / math.cos(angle) - self.wall

    def station_of_height(self, height: float) -> float:
        """Inverse of `clear_height`: where the cavity is `height` tall."""
        angle = self.taper_angle
        return (height + self.wall + self.wall / math.cos(angle)) / math.tan(angle)

    @property
    def rib_stations(self) -> tuple[float, ...]:
        return tuple(self.station_of_height(h) for h in self.window_heights)

    @property
    def cavity_start(self) -> float:
        """Station where the two inner surfaces separate; solid tip before it."""
        return self.station_of_height(0.0)

    @property
    def centreline_apex(self) -> float:
        """Station where the two member centrelines cross - the tip node."""
        angle = self.taper_angle
        return (self.wall / 2.0) * (1.0 + 1.0 / math.cos(angle)) / math.tan(angle)

    def front_centreline(self, station: float) -> np.ndarray:
        return np.array([station, self.wall / 2.0])

    def back_centreline(self, station: float) -> np.ndarray:
        angle = self.taper_angle
        return np.array([station,
                         station * math.tan(angle) - (self.wall / 2.0) / math.cos(angle)])

    @property
    def base_outer_corner(self) -> np.ndarray:
        angle = self.taper_angle
        return np.array([self.back_length * math.cos(angle),
                         self.back_length * math.sin(angle)])


@dataclass
class FingerModel:
    """A meshed finger: the frame, its clamped degrees of freedom, and the
    bookkeeping needed to interpret results in engineering terms."""

    spec: FingerSpec
    frame: Frame
    fixed_dof: np.ndarray
    front_nodes: list[int]        # root-to-tip order along the contact face
    back_nodes: list[int]
    tip_node: int
    front_stations: list[float]   # undeformed x of each front node
    back_stations: list[float]
    rib_nodes: list[list[int]] = field(default_factory=list)

    def node_at_front_station(self, station: float) -> int:
        index = int(np.argmin(np.abs(np.asarray(self.front_stations) - station)))
        return self.front_nodes[index]

    def node_at_back_station(self, station: float) -> int:
        index = int(np.argmin(np.abs(np.asarray(self.back_stations) - station)))
        return self.back_nodes[index]


def _subdivide(start: float, stop: float, count: int) -> list[float]:
    """Interior stations only, so callers can chain segments without repeats."""
    return list(np.linspace(start, stop, count + 1)[1:-1])


def build_finger(spec: FingerSpec | None = None) -> FingerModel:
    spec = spec or FingerSpec()
    ribs = list(spec.rib_stations)
    apex = spec.centreline_apex
    clamp_station = ribs[-1]

    # Member stations: tip, every rib, and enough interior points per bay that
    # bending is resolved rather than piecewise-linearised.
    breaks = [apex] + ribs
    stations: list[float] = []
    for start, stop in zip(breaks[:-1], breaks[1:]):
        stations.append(start)
        stations.extend(_subdivide(start, stop, spec.elements_per_bay))
    stations.append(breaks[-1])
    # The solid base block between the last rib and the mounting face is
    # clamped anyway, but meshing it keeps the outline honest for plots.
    stations.extend(_subdivide(clamp_station, spec.front_length, 2))
    stations.append(spec.front_length)

    nodes: list[np.ndarray] = []
    elements: list[Element] = []

    wall_area, wall_inertia = rectangular_section(spec.wall, spec.depth)
    rib_area, rib_inertia = rectangular_section(spec.rib_thickness, spec.depth)

    tip_node = 0
    nodes.append(spec.front_centreline(apex))

    front_nodes = [tip_node]
    for station in stations[1:]:
        front_nodes.append(len(nodes))
        nodes.append(spec.front_centreline(station))

    back_nodes = [tip_node]
    for station in stations[1:]:
        back_nodes.append(len(nodes))
        nodes.append(spec.back_centreline(station))

    for chain in (front_nodes, back_nodes):
        for a, b in zip(chain[:-1], chain[1:]):
            elements.append(Element(a, b, spec.youngs_modulus, wall_area,
                                    wall_inertia, spec.material))

    rib_nodes: list[list[int]] = []
    for rib_station in ribs:
        index = int(np.argmin(np.abs(np.asarray(stations) - rib_station)))
        lower, upper = front_nodes[index], back_nodes[index]
        chain = [lower]
        p_low, p_high = nodes[lower], nodes[upper]
        for t in np.linspace(0.0, 1.0, spec.elements_per_rib + 1)[1:-1]:
            chain.append(len(nodes))
            nodes.append(p_low + t * (p_high - p_low))
        chain.append(upper)
        for a, b in zip(chain[:-1], chain[1:]):
            elements.append(Element(a, b, spec.youngs_modulus, rib_area,
                                    rib_inertia, spec.material))
        rib_nodes.append(chain)

    node_array = np.asarray(nodes, dtype=float)
    frame = Frame(node_array, elements)

    # Everything from the last rib to the mounting face is a solid block of
    # material 26 mm deep; treating it as rigid is closer to the truth than
    # meshing it as two thin walls.
    clamped = np.where(node_array[:, 0] >= clamp_station - 1e-9)[0]
    fixed_dof = np.concatenate([[3 * n, 3 * n + 1, 3 * n + 2] for n in clamped])

    return FingerModel(
        spec=spec,
        frame=frame,
        fixed_dof=np.sort(fixed_dof),
        front_nodes=front_nodes,
        back_nodes=back_nodes,
        tip_node=tip_node,
        front_stations=list(stations),
        back_stations=list(stations),
        rib_nodes=rib_nodes,
    )


def outline(spec: FingerSpec) -> np.ndarray:
    """Closed outer profile, for drawing the part rather than solving it."""
    angle = spec.taper_angle
    return np.array([
        [0.0, 0.0],
        [spec.front_length, 0.0],
        spec.base_outer_corner,
        [0.0, 0.0],
    ])


MATERIALS_MPA = {
    # Print-realistic tensile / flexural moduli for FDM parts, not datasheet
    # values for injection-moulded coupons.
    "pla": 3000.0,
    "petg": 1700.0,
    "abs": 1800.0,
    "pctg": 1500.0,
    "nylon": 1200.0,
    "tpu95a": 30.0,
    "tpu85a": 8.0,
}
