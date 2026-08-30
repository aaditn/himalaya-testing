"""Load a finger's CAD mesh and bend it along a measured spine.

The mesh arrives in whatever frame the CAD package happened to export, so it is
re-expressed in a canonical finger frame before anything else touches it:

    +y  along the finger, 0 at the root
    +z  out of the flat face the tags are stuck to
    +x  across the truss depth (y cross z)

That frame is derived from the geometry rather than declared, because the
alternative is a column of hand-measured offsets that silently rot the first
time the part is re-exported.

Bending is skinning: each vertex keeps its (x, z) offset and its distance along
the finger, and rides whatever frame the spine has at that distance. A flexure
really does deform this way to first order - cross-sections stay planar and
roughly rigid, which is the same Euler-Bernoulli assumption the spine itself
rests on.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# A finger is tens of millimetres. Anything whose longest dimension is under
# this is far more likely to be inches or centimetres than a real part.
MIN_PLAUSIBLE_MM = 20.0


def load_stl(path: str | Path) -> np.ndarray:
    """Return an (n, 3, 3) array of triangles. Handles binary and ASCII STL."""
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path}: too short to be an STL")

    count = struct.unpack("<I", raw[80:84])[0]
    if len(raw) == 84 + 50 * count:
        block = np.frombuffer(raw[84:84 + 50 * count], dtype=np.uint8).reshape(count, 50)
        # Each 50-byte record is normal(3f) + vertices(9f) + 2 spare bytes, so
        # the float view has to skip the normal and stop before the spare pair.
        return block[:, 12:48].copy().view(np.float32).reshape(count, 3, 3).astype(float)

    text = raw.decode("utf-8", errors="replace")
    if "facet" not in text:
        raise ValueError(f"{path}: not a recognisable STL")
    values = [
        [float(p) for p in line.split()[1:4]]
        for line in text.splitlines()
        if line.strip().startswith("vertex")
    ]
    if not values or len(values) % 3:
        raise ValueError(f"{path}: ASCII STL has {len(values)} vertices, not a multiple of 3")
    return np.asarray(values, dtype=float).reshape(-1, 3, 3)


def split_shells(triangles: np.ndarray) -> list[np.ndarray]:
    """Split a soup of triangles into connected components, largest first."""
    verts = np.round(triangles.reshape(-1, 3), 4)
    _unique, index = np.unique(verts, axis=0, return_inverse=True)
    faces = np.asarray(index).reshape(-1, 3)
    parent = list(range(faces.max() + 1))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b, c in faces:
        for x, y in ((a, b), (b, c)):
            ra, rb = find(int(x)), find(int(y))
            if ra != rb:
                parent[ra] = rb

    labels = np.array([find(int(f[0])) for f in faces])
    groups = [triangles[labels == label] for label in np.unique(labels)]
    return sorted(groups, key=len, reverse=True)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else vector


def _extrusion_axis(triangles: np.ndarray, tolerance: float = 0.05
                    ) -> np.ndarray | None:
    """Find the axis a prism was extruded along, or None if it is not one.

    A part swept from a 2D profile has every vertex on one of exactly two
    parallel planes, and the sweep direction is their shared normal. Testing
    for that beats picking the largest faces by area: on a truss the open
    profile has less area than its own side walls, so area would choose wrong.
    """
    vertices = np.unique(np.round(triangles.reshape(-1, 3), 4), axis=0)
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals[lengths > 1e-9] / lengths[lengths > 1e-9, None]

    seen: list[np.ndarray] = []
    for normal in normals:
        if any(abs(float(normal @ other)) > 0.999 for other in seen):
            continue
        seen.append(normal)
        heights = np.round(vertices @ normal, 2)
        levels, counts = np.unique(heights, return_counts=True)
        # Two planes is necessary but not sufficient - a tetrahedron also has
        # two. A real sweep puts the same profile on both, so demand matching
        # populations as well.
        if (len(levels) == 2 and abs(levels[1] - levels[0]) > tolerance
                and counts[0] == counts[1] and counts[0] >= 3):
            return normal
    return None


def _principal_axes(points: np.ndarray) -> np.ndarray:
    centred = points - points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    return np.asarray(vt)


@dataclass
class FingerMesh:
    """A finger in canonical local coordinates, ready to be bent.

    `vertices` are (m, 3) in the canonical frame; `faces` indexes them.
    `length_mm` is the extent along +y.
    """

    vertices: np.ndarray
    faces: np.ndarray
    length_mm: float
    width_mm: float
    depth_root_mm: float
    source: str = ""
    scale_note: str = ""

    @property
    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]

    @classmethod
    def from_stl(cls, path: str | Path, units: str = "auto",
                 face_sign: int = 1, roll_deg: float = 0.0) -> "FingerMesh":
        """Load one finger from an STL and canonicalise it.

        `units` is "mm", "in", "cm" or "auto". `face_sign` flips which of the
        two candidate faces is treated as the tagged one, for a mirrored print.
        `roll_deg` rotates the part about its own long axis, which is what sets
        which face the tags are actually stuck to - see `rolled`.
        """
        triangles = load_stl(path)
        shells = split_shells(triangles)
        # A "fingers" export often holds several copies. They are the same part,
        # so take the largest and ignore the rest rather than guessing which
        # copy is which finger.
        triangles = shells[0]
        triangles, note = _apply_units(triangles, units)
        mesh = cls._canonicalise(triangles, str(path), note, face_sign)
        return mesh.rolled(roll_deg) if roll_deg else mesh

    @classmethod
    def _canonicalise(cls, triangles: np.ndarray, source: str, note: str,
                      face_sign: int) -> "FingerMesh":
        points = triangles.reshape(-1, 3)
        centre = points.mean(axis=0)

        z = _extrusion_axis(triangles)
        if z is None:
            # Not a prism. Fall back to principal axes, taking the smallest as
            # the face normal; less exact, but it still gets a usable frame.
            axes = _principal_axes(points)
            spans = [np.ptp((points - centre) @ axes[i]) for i in range(3)]
            z = axes[int(np.argmin(spans))]

        # The finger's long axis lives in the plane of the flat face. Taking it
        # there rather than from a 3D fit matters: a tapered wedge tilts its
        # global principal axis out of that plane, which then leaks depth into
        # what should be the constant 20 mm thickness.
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(seed @ z)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        e1 = _unit(np.cross(z, seed))
        e2 = np.cross(z, e1)
        flat = np.column_stack([(points - centre) @ e1, (points - centre) @ e2])
        _u, _s, vt = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
        y = _unit(vt[0, 0] * e1 + vt[0, 1] * e2)

        # The root is the thick end: a cantilever tapers toward its free tip.
        along = (points - centre) @ y
        depth = (points - centre) @ np.cross(y, z)
        window = 0.25 * np.ptp(along)
        at_low = np.ptp(depth[along < along.min() + window]) if window > 0 else 0.0
        at_high = np.ptp(depth[along > along.max() - window]) if window > 0 else 0.0
        if at_high > at_low:
            y = -y

        z = z * (1 if face_sign >= 0 else -1)
        x = np.cross(y, z)

        basis = np.vstack([x, y, z])
        placed = (points - centre) @ basis.T
        placed[:, 1] -= placed[:, 1].min()      # root at y = 0
        placed[:, 0] -= placed[:, 0].mean()     # centre the truss depth
        placed[:, 2] -= placed[:, 2].mean()     # centre the flat faces

        vertices, index = np.unique(np.round(placed, 5), axis=0, return_inverse=True)
        faces = np.asarray(index).reshape(-1, 3)

        length = float(np.ptp(vertices[:, 1]))
        root = vertices[vertices[:, 1] < 0.15 * length]
        return cls(
            vertices=vertices, faces=faces,
            length_mm=length,
            width_mm=float(np.ptp(vertices[:, 2])),
            depth_root_mm=float(np.ptp(root[:, 0])) if len(root) else 0.0,
            source=source, scale_note=note,
        )

    def rolled(self, degrees: float) -> "FingerMesh":
        """Spin the finger about its own long axis.

        Canonicalisation can only guess which face carries the tags, and on a
        swept part it guesses the swept profile's face because that is the one
        the geometry makes special. On this hand the tags are on the narrow side
        wall instead, a quarter turn away, so the guess needs correcting once
        and declaring in the config rather than being re-derived every load.
        """
        angle = np.radians(float(degrees))
        cos, sin = np.cos(angle), np.sin(angle)
        spun = np.column_stack([
            self.vertices[:, 0] * cos + self.vertices[:, 2] * sin,
            self.vertices[:, 1],
            -self.vertices[:, 0] * sin + self.vertices[:, 2] * cos,
        ])
        return FingerMesh(
            vertices=spun, faces=self.faces, length_mm=self.length_mm,
            width_mm=float(np.ptp(spun[:, 2])),
            depth_root_mm=float(np.ptp(
                spun[spun[:, 1] < 0.15 * self.length_mm][:, 0])),
            source=self.source,
            scale_note=f"{self.scale_note}; rolled {degrees:g} deg about the long axis",
        )

    def face_line(self) -> tuple[float, float]:
        """Slope and intercept of the tagged surface in the y-z plane.

        Read off a line fitted to the upper envelope rather than the nearest
        vertices: the part is low-poly enough that a narrow slice can contain
        no vertices at all.
        """
        samples = []
        for y in np.linspace(0.0, self.length_mm, 24):
            band = self.vertices[np.abs(self.vertices[:, 1] - y) < 0.12 * self.length_mm]
            if len(band):
                samples.append((y, float(band[:, 2].max())))
        if len(samples) < 2:
            return 0.0, float(self.vertices[:, 2].max())
        data = np.asarray(samples)
        slope, intercept = np.polyfit(data[:, 0], data[:, 1], 1)
        return float(slope), float(intercept)

    def with_face_on_spine(self) -> "FingerMesh":
        """Lay the tagged surface onto the spine instead of straddling it.

        The tags sit on the outside of the finger and the spine is fitted
        through their centres, so the spine follows that surface - not the
        part's mid-plane, and not the axis a principal-component fit happens to
        pick. On this finger the tagged edge slants about five degrees away from
        that axis, which left the body drifting a centimetre off its own tags
        between root and tip. Rotating the surface flat first, then dropping it
        to z = 0, removes both errors at once.
        """
        slope, _intercept = self.face_line()
        angle = -np.arctan(slope)
        cos, sin = np.cos(angle), np.sin(angle)
        turned = np.column_stack([
            self.vertices[:, 0],
            self.vertices[:, 1] * cos - self.vertices[:, 2] * sin,
            self.vertices[:, 1] * sin + self.vertices[:, 2] * cos,
        ])
        turned[:, 1] -= turned[:, 1].min()

        flat = FingerMesh(
            vertices=turned, faces=self.faces,
            length_mm=float(np.ptp(turned[:, 1])),
            width_mm=self.width_mm, depth_root_mm=self.depth_root_mm,
            source=self.source, scale_note=self.scale_note)
        _slope, intercept = flat.face_line()
        flat.vertices[:, 2] -= intercept
        flat.scale_note = (f"{self.scale_note}; tagged face levelled "
                           f"({np.degrees(angle):+.1f} deg) and set on the spine")
        return flat

    def scaled_to_length(self, length_mm: float) -> "FingerMesh":
        """Uniformly rescale so the finger measures `length_mm` root to tip."""
        if self.length_mm <= 1e-6:
            raise ValueError("mesh has no length to scale")
        factor = float(length_mm) / self.length_mm
        return FingerMesh(
            vertices=self.vertices * factor, faces=self.faces,
            length_mm=length_mm, width_mm=self.width_mm * factor,
            depth_root_mm=self.depth_root_mm * factor,
            source=self.source,
            scale_note=f"{self.scale_note}; rescaled x{factor:.3f} to {length_mm:g} mm",
        )

    def deform(self, spine) -> np.ndarray:
        """Bend the mesh along `spine` and return world-space triangles.

        `spine` supplies `frames_at(y)` -> (origins, rotations) for an array of
        distances along the finger.
        """
        y = self.vertices[:, 1]
        origins, rotations = spine.frames_at(y)
        offsets = self.vertices.copy()
        offsets[:, 1] = 0.0
        # rotations is (m, 3, 3) with columns [x, y, z] of each local frame.
        moved = origins + np.einsum("mij,mj->mi", rotations, offsets)
        return moved[self.faces]


def _apply_units(triangles: np.ndarray, units: str) -> tuple[np.ndarray, str]:
    factors = {"mm": 1.0, "cm": 10.0, "in": 25.4, "inch": 25.4}
    if units in factors:
        return triangles * factors[units], f"units declared {units}"
    if units != "auto":
        raise ValueError(f"unknown units {units!r}")

    span = float(np.ptp(triangles.reshape(-1, 3), axis=0).max())
    if span >= MIN_PLAUSIBLE_MM:
        return triangles, f"units auto-detected mm (largest dimension {span:.1f})"
    # Inches and centimetres both leave a sub-20 number here. Inches is the
    # overwhelmingly common case for STL exported from imperial-default CAD,
    # and picking cm would give a finger a quarter of the plausible size.
    return triangles * 25.4, (
        f"units auto-detected inch (largest dimension {span:.3f} -> "
        f"{span * 25.4:.1f} mm); pass units= to override")
