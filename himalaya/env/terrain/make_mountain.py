"""Generate a mountainside heightfield: rock at several scales at once.

The stock Playground hfield is 5 cm of relief over 7.8 cm cells -- gentle
undulation. A hand capsule is 5 cm in radius, so it rests ON that surface and
has nothing to hook. Pressing friction is all it can contribute.

A real mountainside is structured across scales:

    boulders / slabs   0.3-1.0 m   features to route over or around
    ledges / edges     0.1-0.3 m   what a hand actually grips
    grain              0.02-0.05 m surface roughness underfoot

So the terrain is built as summed octaves of value noise (each half the
wavelength and roughly half the amplitude of the last), then TERRACED -- the
heights are quantised so the surface breaks into flats separated by steps.
Terracing is what turns smooth noise into something with edges, and edges are
what distinguish a mountainside from a bumpy ramp.

Writes a PNG that MuJoCo loads as an hfield.
"""

import numpy as np


def _value_noise(shape, cells, rng):
  """Smooth noise: random values on a coarse lattice, cosine-interpolated."""
  h, w = shape
  gh, gw = cells + 1, cells + 1
  lattice = rng.random((gh, gw))

  yi = np.linspace(0, cells, h, endpoint=False)
  xi = np.linspace(0, cells, w, endpoint=False)
  y0 = yi.astype(int)[:, None]
  x0 = xi.astype(int)[None, :]
  fy = (yi - y0.ravel())[:, None]
  fx = (xi - x0.ravel())[None, :]
  # Cosine easing gives C1 continuity, so the surface has no lattice creases.
  fy = (1 - np.cos(fy * np.pi)) * 0.5
  fx = (1 - np.cos(fx * np.pi)) * 0.5

  v00 = lattice[y0, x0]
  v10 = lattice[y0 + 1, x0]
  v01 = lattice[y0, x0 + 1]
  v11 = lattice[y0 + 1, x0 + 1]
  return (v00 * (1 - fy) * (1 - fx) + v10 * fy * (1 - fx)
          + v01 * (1 - fy) * fx + v11 * fy * fx)


def mountain_height(res=256, octaves=4, terrace_steps=6, seed=0):
  """Return a float array in [0,1]: fractal rock, terraced into ledges."""
  rng = np.random.default_rng(seed)
  h = np.zeros((res, res))
  amp, cells, norm = 1.0, 4, 0.0
  for _ in range(octaves):
    h += amp * _value_noise((res, res), cells, rng)
    norm += amp
    amp *= 0.55      # each octave a bit over half the last: rocky, not cloudy
    cells *= 2       # and half the wavelength
  h /= norm

  # Terracing: quantise to flats separated by steps. Few, tall terraces --
  # a hand needs a step at least twice its 0.05 m capsule radius to hook, so
  # the risers have to stay sharp. A softening exponent much above ~6 rounds
  # them back into a ramp, which is the defect this terrain exists to fix.
  q = np.floor(h * terrace_steps) / terrace_steps
  frac = h * terrace_steps - np.floor(h * terrace_steps)
  h = q + (frac ** 6) / terrace_steps

  h -= h.min()
  h /= h.max()
  return h


def write_png(path, res=256, z_scale=0.6, **kw):
  """Write the heightfield PNG and report the geometry it will produce."""
  from PIL import Image
  h = mountain_height(res=res, **kw)
  Image.fromarray((h * 255).astype(np.uint8), mode="L").save(path)
  return h


if __name__ == "__main__":
  import sys
  out = sys.argv[1] if len(sys.argv) > 1 else "himalaya/env/xmls/assets/mountain.png"
  h = write_png(out)
  print("wrote", out, h.shape)
