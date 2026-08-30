"""Generate a bumpy incline: mostly flat rock, scattered with sharp bumps.

The terrain this replaces was globally terraced -- the whole height field
quantised into bands -- which produced a regular, stepped landscape rather
than a slope you scramble up. It also sat too high on average, so the surface
read as a plateau.

What is actually wanted is the stock rough terrain with teeth: a low, flat
base with individual bumps scattered over it, each one steep-sided enough
that a hand can catch on it. The base stays near zero so the mean height is
low; the bumps are sparse, so between them there is ordinary ground to walk
on.

Two ingredients:

  base    fine, low-amplitude noise -- surface grain, a few cm
  bumps   scattered blobs with a sharp profile, tall enough to hook
          (a hand capsule is 0.05 m in radius, so a bump needs ~0.10 m of
          rise over a short run to be catchable)

The bump profile is the part that matters. A smooth blob is a ramp a hand
slides off; raising the blob to a power keeps its peak but steepens the
flanks, which is what turns it into a feature with an edge.
"""

import numpy as np


def _value_noise(shape, cells, rng):
  """Smooth noise: random values on a coarse lattice, cosine-interpolated."""
  h, w = shape
  lattice = rng.random((cells + 1, cells + 1))
  yi = np.linspace(0, cells, h, endpoint=False)
  xi = np.linspace(0, cells, w, endpoint=False)
  y0 = yi.astype(int)[:, None]
  x0 = xi.astype(int)[None, :]
  fy = (1 - np.cos((yi - y0.ravel()) * np.pi)) [:, None] * 0.5
  fx = (1 - np.cos((xi - x0.ravel()) * np.pi)) [None, :] * 0.5
  return (lattice[y0, x0] * (1 - fy) * (1 - fx)
          + lattice[y0 + 1, x0] * fy * (1 - fx)
          + lattice[y0, x0 + 1] * (1 - fy) * fx
          + lattice[y0 + 1, x0 + 1] * fy * fx)


def bumpy_incline(res=256, extent=12.0, n_bumps=350, bump_r=(0.10, 0.45),
                  bump_h=(0.15, 0.35), grain=0.03, sharpness=10.0, seed=0):
  """Return (heights_in_metres, stats).

  Density is a balance, not a maximum. At 650 bumps half the ground was a
  steep flank -- a spike field rather than an incline with features on it, and
  the robot would be walking on spikes. At 350 the mix is roughly a quarter
  flat, a third moderate, a third steep, which leaves ordinary ground between
  the holds while keeping a hold within arm reach from most positions.

  n_bumps   how many discrete features over the patch
  bump_r    bump radius range, metres -- around the 0.05 m hand capsule
  bump_h    bump height range, metres -- >= 0.10 m to be hookable
  grain     amplitude of the background surface noise, metres
  sharpness exponent on the bump profile; higher = steeper flanks, more edge
  """
  rng = np.random.default_rng(seed)
  cell = extent / res

  # Low background grain, so the mean height stays near zero.
  h = _value_noise((res, res), 48, rng) * grain

  yy, xx = np.mgrid[0:res, 0:res].astype(float)
  for _ in range(n_bumps):
    cy, cx = rng.random(2) * res
    r_m = rng.uniform(*bump_r)
    amp = rng.uniform(*bump_h)
    r_cells = max(1.5, r_m / cell)

    # Elongate and rotate each bump, so they are shards at random angles
    # rather than one dome stamped repeatedly. Aspect up to 3:1.
    th = rng.uniform(0, np.pi)
    agg = rng.uniform(1.0, 3.0)
    dy, dx = yy - cy, xx - cx
    u = (dx * np.cos(th) + dy * np.sin(th)) / (r_cells * agg)
    v = (-dx * np.sin(th) + dy * np.cos(th)) / r_cells
    d = np.sqrt(u ** 2 + v ** 2)
    inside = d < 1.0
    if not inside.any():
      continue

    # Profile shape is randomised per bump, because one repeated shape is
    # what made the last terrain look stamped:
    #   flat-topped  a mesa with a near-vertical rim -- a real ledge to catch
    #   wedge        linear ramp up to a sharp crest
    #   dome         the old rounded blob, kept as a minority for variety
    kind = rng.random()
    profile = np.zeros_like(d)
    if kind < 0.45:
      # Plateau out to `flat`, then drop steeply. The rim is the hold.
      flat = rng.uniform(0.35, 0.7)
      w = np.clip((1.0 - d[inside]) / max(1e-6, 1.0 - flat), 0.0, 1.0)
      profile[inside] = w ** 0.35          # low exponent = square shoulder
    elif kind < 0.8:
      profile[inside] = (1.0 - d[inside]) ** 0.6   # wedge, sharp crest
    else:
      profile[inside] = (1.0 - d[inside] ** 2) ** (1.0 / sharpness)

    # Break the outline so bumps are not smooth ellipses.
    jag = 1.0 + 0.25 * np.sin(rng.uniform(0, 6.28) + 5.0 * np.arctan2(v, u + 1e-9))
    profile[inside] *= np.clip(jag[inside], 0.5, 1.4)

    h = np.maximum(h, profile * amp)   # max, not sum: bumps overlap as rock

  gy, gx = np.gradient(h, cell)
  face = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
  steps = np.abs(np.diff(h, axis=0))
  stats = {
      "mean_h": h.mean(), "max_h": h.max(),
      "median_face": np.percentile(face, 50),
      "p90_face": np.percentile(face, 90),
      "max_step": steps.max(),
      "hookable_pct": (steps >= 0.10).mean() * 100,
  }
  return h, stats


def write_png(path, z_scale, **kw):
  """Write the PNG. z_scale must match the scene's hfield size z."""
  from PIL import Image
  h, stats = bumpy_incline(**kw)
  img = np.clip(h / z_scale, 0, 1)
  Image.fromarray((img * 255).astype(np.uint8), mode="L").save(path)
  return h, stats


if __name__ == "__main__":
  import sys
  out = sys.argv[1] if len(sys.argv) > 1 else "himalaya/env/xmls/assets/mountain.png"
  h, s = write_png(out, z_scale=0.35)
  print("wrote", out, {k: round(v, 3) for k, v in s.items()})
