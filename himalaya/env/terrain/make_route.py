"""Generate a climbing ROUTE: a walkable lane with walls either side.

The previous terrain was uniformly bumpy, which asks the robot to invent a
path and to pull itself up with its arms. Both are wrong for this machine.

The G1's arms are not a climbing mechanism. Shoulders cap at 25 Nm against
139 Nm at the knee, and the wrists cap at 5 Nm -- they fold under load. The
legs do the climbing; the hands are a BALANCING aid. So the terrain should
not require crafting a way up by hand. It should offer an easiest path, with
something either side to steady against while the legs work.

Hence a corridor:

    lane    low and gentle, ~1.1 m wide, walkable, winding up the slope
    walls   steep, rising ~0.7 m -- shoulder height, within a 0.34 m reach
    beyond  broken ground, so straying off the route is worse than staying on

The lane wanders laterally as it climbs, so the route has to be followed
rather than memorised as a straight line, and the walls come and go.
"""

import numpy as np


def _smooth_path(res, rng, wiggle=2.2, octaves=3):
  """Lateral centre of the lane at each uphill position, in grid columns."""
  y = np.zeros(res)
  amp, freq = 1.0, 1.0
  for _ in range(octaves):
    ph = rng.random() * 2 * np.pi
    y += amp * np.sin(2 * np.pi * freq * np.arange(res) / res + ph)
    amp *= 0.5
    freq *= 2.0
  y /= np.abs(y).max() + 1e-9
  return res / 2.0 + y * wiggle * res / 12.0


def route(res=256, extent=12.0, lane_w=1.1, wall_h=0.55, wall_w=0.9,
          rough=0.10, lane_rough=0.055, seed=0):
  """Return (heights_in_metres, stats).

  lane_w   walkable width, metres
  wall_h   how far the walls rise above the lane
  wall_w   width of the wall band either side of the lane
  rough       amplitude of bumps on the walls and outer ground
  lane_rough  amplitude of bumps in the lane itself -- the footing the legs
              have to adapt to
  """
  rng = np.random.default_rng(seed)
  cell = extent / res

  # Uphill runs along axis 0 (world -x); lateral is axis 1.
  centre = _smooth_path(res, rng)
  lat = np.arange(res)[None, :]
  dist = np.abs(lat - centre[:, None]) * cell      # metres from lane centre

  # Everything below varies ALONG the climb. A constant-width lane between two
  # equal walls reads as built rather than found, so width, each wall's height
  # independently, and the wall profile all wander as you ascend.
  def _wander(scale, octaves=4):
    """Slow random variation along the uphill axis, mean 1.0."""
    v = np.zeros(res)
    amp, freq = 1.0, 1.0
    for _ in range(octaves):
      ph = rng.random() * 2 * np.pi
      v += amp * np.sin(2 * np.pi * freq * np.arange(res) / res + ph)
      amp *= 0.55
      freq *= 2.1
    v /= np.abs(v).max() + 1e-9
    return 1.0 + scale * v

  # Lane pinches and opens: 0.6x to 1.4x the nominal width.
  half_lane = (lane_w / 2.0) * _wander(0.4)[:, None]

  # LEFT and RIGHT walls are independent -- the main source of the symmetry.
  # One side can fall away to nothing while the other rises.
  side = np.sign(lat - centre[:, None])
  h_left = wall_h * _wander(0.55)[:, None]
  h_right = wall_h * _wander(0.55, octaves=5)[:, None]
  wall_here = np.where(side < 0, h_left, h_right)

  # Wall steepness varies too, so some sections are scrambles and others walls.
  steep = (0.45 * _wander(0.35))[:, None]
  band = wall_w * _wander(0.3)[:, None]

  t = np.clip((dist - half_lane) / np.maximum(1e-6, band), 0.0, 1.0)
  h = wall_here * (t ** steep)

  # Texture: none in the lane, growing outward, so the route reads as the
  # easy line and everything else looks worse.
  n = rng.random((res // 4 + 2, res // 4 + 2))
  yi = np.linspace(0, n.shape[0] - 2, res)
  xi = np.linspace(0, n.shape[1] - 2, res)
  y0, x0 = yi.astype(int)[:, None], xi.astype(int)[None, :]
  fy = (1 - np.cos((yi - y0.ravel()) * np.pi))[:, None] * 0.5
  fx = (1 - np.cos((xi - x0.ravel()) * np.pi))[None, :] * 0.5
  grain = (n[y0, x0] * (1 - fy) * (1 - fx) + n[y0 + 1, x0] * fy * (1 - fx)
           + n[y0, x0 + 1] * (1 - fy) * fx + n[y0 + 1, x0 + 1] * fy * fx)
  h += rough * grain * np.clip((dist - half_lane) / 0.5, 0.0, 1.5)

  # The lane itself is NOT flat. A dead-level floor is the other thing that
  # looks manufactured, and a policy that only ever walks on flat ground has
  # learned nothing about adapting its footing. Small bumps here are what the
  # legs have to cope with while the hands steady against the walls.
  lane_grain = (n[y0, x0] * (1 - fy) * (1 - fx) + n[y0 + 1, x0 + 1] * fy * fx)
  h += lane_rough * (lane_grain - 0.5) * 2.0

  h -= h.min()

  lane_mask = dist < half_lane
  gy, gx = np.gradient(h, cell)
  face = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
  stats = {
      "lane_frac": lane_mask.mean() * 100,
      "lane_face": np.percentile(face[lane_mask], 50),
      "wall_face": np.percentile(face[~lane_mask], 90),
      "max_h": h.max(),
      "mean_h": h.mean(),
  }
  return h, stats


def write_png(path, z_scale, **kw):
  from PIL import Image
  h, stats = route(**kw)
  Image.fromarray((np.clip(h / z_scale, 0, 1) * 255).astype(np.uint8),
                  mode="L").save(path)
  return h, stats


if __name__ == "__main__":
  import sys
  out = sys.argv[1] if len(sys.argv) > 1 else "himalaya/env/xmls/assets/mountain.png"
  h, s = write_png(out, z_scale=0.9)
  print("wrote", out, {k: round(v, 3) for k, v in s.items()})
