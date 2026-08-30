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
    walls   raised banks rising ~0.55 m -- shoulder height, within a 0.34 m
            reach, and ROUNDED: something to lean a palm on, not a blade
    beyond  broken ground, so straying off the route is worse than staying on

Everything here is bumps, not spikes. An earlier version raised the walls with
t**0.45 -- an exponent below 1, so the ground went near-vertical the moment a
foot left the lane -- and added unsmoothed grain on top. Measured, the median
face angle INSIDE the lane was 22 degrees, 23% of the surface exceeded 45
degrees and 5.6% exceeded 60, peaking at 80. On a 35 degree base tilt a 22
degree local face is a 57 degree effective slope underfoot. Nothing balances
on that, on two limbs or four, so the terrain was rejecting every policy
before the policy was the problem.

The lane wanders laterally as it climbs, so the route has to be followed
rather than memorised as a straight line, and the walls come and go.
"""

import numpy as np

# Total relief, metres. Must not exceed the hfield z_scale in the scene XML
# (size="6 6 <z_scale> 1.0"), or the PNG clips and the peaks flatten to a mesa.
_PEAK_M = 1.05

# Wall profile exponent. ABOVE 1 eases out of the lane and steepens as it goes,
# which is a bank; BELOW 1 leaves the lane edge with infinite slope, which is a
# blade. scripts/map.py overrides these so the map can be tuned without editing
# this file.
_WALL_STEEP = 2.1
# Final smoothing radius in cells. 1 removes single-cell spikes; 2 also flattens
# the corridor walls until the route stops reading as a route.
_BLUR_K = 1
# Lane bump size. "coarse" samples a res/16 lattice -- features ~1 m across,
# wider than the 0.18 m foot, so a foot lands on a slope. "fine" samples the
# same res/4 lattice the wall texture uses, which is the original behaviour and
# puts bumps smaller than a foot in the walking line.
_LANE_GRAIN = "coarse"
# How many median passes to run. Each pass removes spikes up to one cell wider
# than the last; 2-3 clears the field without touching the corridor.
_DESPIKE_PASSES = 2


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


def route(res=256, extent=12.0, lane_w=1.4, wall_h=0.85, wall_w=1.6,
          rough=0.16, lane_rough=0.14, n_routes=4, seed=0):
  """Return (heights_in_metres, stats).

  lane_w   walkable width, metres
  wall_h   how far the walls rise above the lane
  wall_w   width of the wall band either side of the lane. WIDE on purpose:
           the same 0.55 m rise spread over 1.6 m is a bank you can lean on,
           over 0.9 m it is a wall you fall off.
  rough       amplitude of bumps on the walls and outer ground
  lane_rough  amplitude of bumps in the lane itself -- the footing the legs
              have to adapt to. Small: this is texture underfoot, not an
              obstacle course.
  """
  rng = np.random.default_rng(seed)
  cell = extent / res

  # SEVERAL corridors side by side in one heightfield, rather than one.
  #
  # hfield_data is a single array shared by every environment -- it has no
  # per-world dimension in the MJX schema, so 8192 envs cannot each get their
  # own terrain the way they get their own friction. Building N routes into
  # one field and spawning at a random one gets the same effect: the policy
  # meets a different corridor each episode and cannot memorise a path,
  # because a policy that learned one route fails on the next.
  lat = np.arange(res)[None, :]
  band = res / n_routes
  centres, dists = [], []
  for r in range(n_routes):
    # Each route wanders within its own band, with its own everything.
    c = _smooth_path(res, rng, wiggle=1.1) - res / 2.0 + band * (r + 0.5)
    centres.append(c)
    dists.append(np.abs(lat - c[:, None]) * cell)
  # A point belongs to whichever route it is nearest.
  dist = np.min(np.stack(dists), axis=0)
  centre = centres[0]

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

  # Wall steepness varies, so some sections are gentle banks and others are
  # proper scrambles -- but ALL of them ease in.
  #
  # This exponent used to be ~0.45. Below 1.0 the curve leaves the lane with
  # infinite slope: the ground goes vertical at the lane edge and the "wall"
  # is a blade. At 1.6-2.6 the rise starts flat and steepens, which is the
  # shape of a bank. Combined with the wider wall_w the peak face angle drops
  # from 80 degrees to something a foot can stand on.
  steep = (_WALL_STEEP * _wander(0.25))[:, None]
  band = wall_w * _wander(0.3)[:, None]

  t = np.clip((dist - half_lane) / np.maximum(1e-6, band), 0.0, 1.0)
  # smoothstep: zero SLOPE at both ends, so the lane floor meets the bank
  # without a crease and the bank tops out without an edge.
  t = t * t * (3.0 - 2.0 * t)
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
  # Round the grain off before laying it down. The interpolation above is
  # cosine on a res/4 lattice, which still leaves a crease at every cell
  # boundary; a couple of box-blur passes turn those creases into domes.
  # Bumps are for adapting footing, so they should be wider than a foot is
  # long (0.18 m) rather than sharp enough to stand a foot on edge.
  def _despike(a, k=1, passes=1):
    """Median filter: kill isolated spikes, keep edges.

    A blur AVERAGES, so it drags wall tops down and lane floors up -- run hard
    enough to remove spikes it also erases the corridor, which is exactly what
    happened here (cross-section flattened to 0.06-0.20 m and the route stopped
    reading as a route). A median replaces a cell with the MIDDLE value of its
    neighbourhood: a lone peak is outvoted and disappears, while a real wall
    edge survives untouched because half its neighbours are high too.
    """
    out = a.astype(float)
    for _ in range(passes):
      pad = np.pad(out, k, mode="edge")
      stack = np.stack([
          pad[k + dy:k + dy + out.shape[0], k + dx:k + dx + out.shape[1]]
          for dy in range(-k, k + 1) for dx in range(-k, k + 1)])
      out = np.median(stack, axis=0)
    return out

  def _blur(a, k=3, passes=2):
    out = a.astype(float)
    for _ in range(passes):
      pad = np.pad(out, k, mode="edge")
      acc = np.zeros_like(out)
      for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
          acc += pad[k + dy:k + dy + out.shape[0], k + dx:k + dx + out.shape[1]]
      out = acc / ((2 * k + 1) ** 2)
    return out

  if _BLUR_K > 0:
    grain = _blur(grain, k=2, passes=1)
  h += rough * grain * np.clip((dist - half_lane) / 0.5, 0.0, 1.5)

  # The lane itself is NOT flat. A dead-level floor is the other thing that
  # looks manufactured, and a policy that only ever walks on flat ground has
  # learned nothing about adapting its footing. Small bumps here are what the
  # legs have to cope with while the hands steady against the walls.
  # The lane's own undulation, on ITS OWN length scale rather than the wall
  # texture's. Blurring the fine grain enough to be safe underfoot also
  # flattens it to nothing -- measured, the lane came out at 0.6 degrees, a
  # dead-level floor, which teaches the policy nothing about footing. So
  # sample a COARSER lattice instead of blurring a fine one harder: features
  # roughly 1 m across, comfortably wider than the 0.18 m foot, so a foot
  # lands on a slope rather than astride a peak.
  _lattice = 4 if _LANE_GRAIN == "fine" else 16
  ln = rng.random((res // _lattice + 2, res // _lattice + 2))
  lyi = np.linspace(0, ln.shape[0] - 2, res)
  lxi = np.linspace(0, ln.shape[1] - 2, res)
  ly0, lx0 = lyi.astype(int)[:, None], lxi.astype(int)[None, :]
  lfy = (1 - np.cos((lyi - ly0.ravel()) * np.pi))[:, None] * 0.5
  lfx = (1 - np.cos((lxi - lx0.ravel()) * np.pi))[None, :] * 0.5
  lane_grain = (ln[ly0, lx0] * (1 - lfy) * (1 - lfx)
                + ln[ly0 + 1, lx0] * lfy * (1 - lfx)
                + ln[ly0, lx0 + 1] * (1 - lfy) * lfx
                + ln[ly0 + 1, lx0 + 1] * lfy * lfx)
  if _BLUR_K > 0:
    lane_grain = _blur(lane_grain, k=2, passes=1)
  h += lane_rough * (lane_grain - lane_grain.mean()) * 2.0

  # One last light pass over the WHOLE field. Cheap insurance: whatever the
  # wander parameters do, no cell ends up as a spike its neighbours do not
  # support.
  #
  # k=1, not k=2. A 5x5 kernel at this resolution averages over 23 cm and
  # flattened the corridor walls along with the spikes: the cross-section came
  # out wobbling between 0.06 and 0.20 m where it should show troughs between
  # raised banks, so the route stopped reading as a route. A 3x3 removes the
  # single-cell spikes, which is all it was ever needed for.
  if _BLUR_K > 0:
    # Median, not blur: removes the spikes, leaves the walls standing.
    h = _despike(h, k=_BLUR_K, passes=_DESPIKE_PASSES)

  h -= h.min()
  # Normalise to the scene's z_scale budget. The XML declares the hfield as
  # size="6 6 1.10 1.0", so a PNG is written as h/z_scale clipped to [0,1] --
  # anything above 1.10 m does not become a taller wall, it becomes a FLAT
  # PLATEAU where the clip bites, silently replacing the terrain's top with a
  # mesa. Rescaling here keeps the shape and lets the XML own the height.
  peak = float(h.max())
  if peak > 0 and _PEAK_M > 0:
    h *= _PEAK_M / peak

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
  """Write the heightfield PNG, and the lane centreline beside it.

  The centreline is saved because the reward needs it: progress has to be
  measured ALONG the route, not as world-space height. The lane wanders 4.4 m
  laterally at a median 38 degrees off the uphill axis, so a plain uphill
  projection pays only cos(38) ~ 0.79 for following the corridor and would
  make charging straight up the wall the better-paid option.
  """
  from PIL import Image
  h, stats = route(**kw)
  Image.fromarray((np.clip(h / z_scale, 0, 1) * 255).astype(np.uint8),
                  mode="L").save(path)

  # Save every route's mouth, so reset() can start at a random one. Only the
  # entry point is needed -- the reward is height gained, which says nothing
  # about where the lane goes, so the policy still has to find the corridor
  # for itself rather than being steered along it.
  res = kw.get("res", 256)
  extent = kw.get("extent", 12.0)
  n_routes = kw.get("n_routes", 4)
  rng = np.random.default_rng(kw.get("seed", 0))
  cell = extent / res
  band = res / n_routes
  mouths = []
  for r in range(n_routes):
    c = _smooth_path(res, rng, wiggle=1.1) - res / 2.0 + band * (r + 0.5)
    mouths.append((c[0] - res / 2.0) * cell)   # lateral offset at the bottom
  np.save(str(path).replace(".png", "_centre.npy"), np.array(mouths))
  return h, stats


if __name__ == "__main__":
  import sys
  out = sys.argv[1] if len(sys.argv) > 1 else "himalaya/env/xmls/assets/mountain.png"
  h, s = write_png(out, z_scale=0.9)
  print("wrote", out, {k: round(v, 3) for k, v in s.items()})
