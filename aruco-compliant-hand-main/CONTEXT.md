# FlexSense — Project Context

Working notes for continuing this work in a new session. Written 2026-08-30.

## What this is

A compliant, passive, 3D-printed hand for a humanoid that supports itself on
inclines. The fingers are flexures that deform against whatever they press on.
ArUco tags on the fingers let a wrist camera read that deformation **fully
passively** — no strain gauges, no wiring — giving the robot a read on the
surface it is touching (slope, hardness, slip).

Demo platform is a LeRobot SO-101 arm (`~/lerobot-MakerMods`). This repo is the
sensing half and runs standalone.

## Hardware facts established

| Thing | Value |
|---|---|
| Camera | USB webcam, 1280×720 |
| Calibrated intrinsics | fx 729.9, fy 725.7, cx 640, cy 360 |
| Distortion | `[-0.265, 0.161, -0.0024, -0.0008, -0.075]` |
| Calibration quality | rms 0.431 px, 64 views, HFOV 82.5° |
| Finger tags | ArUco `DICT_4X4_50`, **18 mm** |
| Calibration board | ArUco `DICT_5X5_100`, 30 mm squares nominal |
| Hand layout | **BUILT**: 3 fingers × 2 tags + reference tag 11 on the blue cuff |
| Tag IDs | tips 0/2/4 (upper), bases 1/3/5 (lower), reference 11. Left→right in camera POV |
| Finger CAD | `assets/finger_v2.stl` — 104.8 mm long, 20 mm wide, tapering from ~29 mm deep at the root to a point |
| Tag mounting | On the **narrow 20 mm side wall**, not the wide swept face. `mesh.roll_deg: -90` in `hand.yaml` is what says so |

**The printer scales to ~93.3%.** The ChArUco board printed at 28 mm squares
instead of 30. Measure anything printed before trusting a millimetre from it.
This is harmless for camera intrinsics (uniform scale cancels) but directly
scales every deflection reading.

## Files

### The hand declaration — start here
- **`config/hand.yaml`** — the single source of truth. Tag IDs per finger per
  station, finger geometry, reference tag IDs, camera position. The simulator,
  the estimator and the live tracker all read this file.
- `flexsense/handconfig.py` — schema, YAML load/dump, validation, and the
  adapters (`to_sim_rig`, `to_estimator_config`, `tag_manifest`).
- `flexsense/handtools.py` — `check` / `preview` / `rehearse`.

### The 3D render
- `flexsense/fingermesh.py` — loads the finger STL and re-expresses it in a
  canonical frame (+y along the finger from the root, +z out of the tagged
  face). Bends it by skinning: every vertex rides the spine frame at its own
  distance along the finger.
- `flexsense/spine.py` — the curve between the tags. Cubic Hermite through
  root/base/tip, with a rotation-minimising frame carried along it.
- `flexsense/render3d.py` — pinhole and orthographic projectors, painter's
  algorithm mesh drawing.
- `flexsense/hud.py` — theme, real-typeface text batching, gauges, traces.
- `flexsense/gripview.py` — composes the live display.
- `flexsense/smoothing.py` — pose ambiguity resolution and one-euro filtering.

### Measurement
- `flexsense/pose3d.py` — pose primitives: `Pose`, single-marker PnP,
  rigid-group PnP, `fit_plane` (for the incline readout).
- `flexsense/estimator3d.py` — `SpatialDeformationEstimator`. Full 6-DoF, **no
  planar assumption**. Handles in-plane and out-of-plane bending identically.
  Outputs per-finger 3D deflection, signed normal/shear/along, tip rotation,
  contact-plane normal, incline angle.
- `flexsense/estimator.py` — the older 2D homography estimator. Still what
  `track` uses. Superseded by `estimator3d` for the incline work.
- **`flexsense/grip.py`** — grip classification from finger curvature, plus the
  live `grip` command. Bend is a *signed* rotation about each finger's bend
  axis; the unsigned angle between two normals is symmetric and cannot separate
  wrapping from back-bending.
- `flexsense/vision.py` — `MarkerDetector` and `build_detector`. Lens correction
  is applied here, at the single choke point, so the tracker and the force
  calibration cannot disagree about whether distortion was removed.

### Calibration
- `flexsense/camera_calib.py` — ChArUco calibration, `robust_calibrate`,
  plausibility checks, `refit_from_frames`.
- `flexsense/screen_target.py` — full-screen animated target (what actually
  produced the good calibration).
- `calibration/camera_intrinsics.json` — the current calibration.
- `calibration/views_screen/` — 64 saved frames. Refit from these any time
  without recapturing.

### Simulation / viewing
- `flexsense/simrig.py` — synthetic hand renderer. Cantilever bending, renders
  through the real intrinsics.
- `flexsense/watch.py` — live marker viewer with 3D pose boxes.

## Commands

```bash
# validate the hand declaration and predict its optics
python -m flexsense hand-check

# see what a camera position would actually see
python -m flexsense hand-preview --deflect "left=8,right=12"

# run the whole pipeline in simulation against ground truth
python -m flexsense hand-rehearse --deflect "left=8,middle=3,right=12"

# live grip classification with the 3D finger render
python -m flexsense grip --source 0
python -m flexsense grip --source 0 --no-mesh    # plain readout, no CAD

# live marker view with pose boxes
python -m flexsense watch --source 0

# camera calibration (screen target is the one that worked)
python -m flexsense screen-calibrate --source 0
python -m flexsense camera-refit --frames-dir calibration/views_screen

# printable sheets
python -m flexsense markers --output markers.svg
python -m flexsense camera-board            # ChArUco board, SVG not PNG
```

Use `./.venv/bin/python`. Tests: `python -m unittest discover -s tests` (123 pass).

## Measured findings worth not rediscovering

**Pixel size dominates everything.** Corner error for a 20 mm tag:
~0.5 mm at 30 px, ~0.12 mm at 50 px, ~0.044 mm at 90 px. Below 50 px is
unusable. This swamps every other variable including marker family.

**Camera at ~140 mm** puts tags at 103–116 px and gives ~0.1 mm deflection
error. At 175 mm they drop to 82 px and error rises toward 1 mm.

**Stay with ArUco `DICT_4X4_50`.** Tested against AprilTag 36h11 and others:
corner accuracy differs by ~5% between families while pixel size differs by
1200%. Zero ID confusions in 1600 harsh trials, because the estimator ignores
IDs outside its configured set. 4×4 actually detected *better* at small sizes
(6×6 cells stay larger than 8×8 at the same physical size).

**Out-of-plane bending measures better than in-plane** — the opposite of the
obvious guess. Out-of-plane tilts the tag, and tilt is what makes single-square
PnP well-conditioned; in-plane keeps it face-on, the degenerate case.

**Low reprojection error does not mean good calibration.** Flat-on-only views
produced rms 0.21 px with fx=146 when truth was 900 — a degenerate fit that
reprojects perfectly. Always check implied field of view and fx/fy match.

**Calibration needs an explicit focal-length seed.** OpenCV's internal
initialisation fell into a basin giving fx=3838 (19° FOV) on real data. Every
explicit seed converged to the right answer. `robust_calibrate` multi-starts
across 40–120° FOV and filters for physical plausibility.

**Fix the principal point at image centre.** Letting it float gave fx =
913 ± 275 (30% instability); pinned it gave 732 ± 37 (5%). It is only weakly
observable and drags focal length with it.

**Flat targets matter.** A ~5 mm bow in the printed board produced rms 2.75 and
fx off by 20%. Same applies to tags on the hand.

**Reference tag spacing must exceed tag width** (≥1.3×). Three 20 mm tags at
16 mm pitch overlapped and the outer two never decoded.

**Magnitude is biased, projections are not.** `|deflection|` reads ~0.4 mm at
rest because the norm of a noisy vector never reads zero. Use the signed
projection onto the derived axis as the primary output.

**The cubic between the tags is exact, not an approximation.** A cantilever's
deflection curve is a cubic and two poses fix a unique cubic, so interpolating
between the tags reproduces a constant-curvature arc to under 0.05 mm anywhere
inside the measured span. All the error lives outside it.

**The unmeasured tip is where the error is.** With tags at 24 and 58 mm on the
104.8 mm finger, 47 mm is extrapolated. At 40 degrees of bend that costs 7.3 mm
at the tip; moving the tags to 30 and 85 mm cuts it to 2.1 mm. Past the last tag
the spine extends straight, which is right rather than lazy - bending moment
vanishes at a free end, so curvature really does decay there.

**Thresholds are chosen in millimetres of tip travel, not degrees.**
`grip.bend_deg_for_tip_travel` converts between them. It is pure cantilever
geometry - the shape a tip-loaded beam takes, differenced between the two tag
stations - so it does not depend on how stiff the flexure is and transfers
straight to the printed part. For tags at 24/58 mm on a 104.8 mm finger:

| bend | tip travel |
|---|---|
| 3.0 deg (`neutral_deg`) | 9.3 mm |
| 4.0 deg (`backbend_deg`) | 12.5 mm |
| 4.5 deg (`wrap_deg`) | 14.1 mm |
| 6.0 deg (the old value) | 19.0 mm |

Fitting this from simulated frames instead gives the wrong answer - PnP scatter
of a few tenths of a degree swamps a fit over a 10 mm span, and produced 0.23
deg/mm against a true 0.32. Derive it, do not measure it.

**The forward curve turns over.** Both slope angles saturate toward 90 degrees,
so their difference peaks near 120% of the finger length and falls after.
`tip_travel_for_bend_deg` bounds its search at one length so the bisection
cannot converge onto the far branch.

**A single square marker flips.** IPPE gives two poses that reproject almost
identically and separate least when the tag is near face-on, which is most of
the time here. `MarkerPoseTracker` asks `solvePnPGeneric` for both and keeps the
one closest to last frame. This was invisible while the output was a number and
obvious the moment a mesh was drawn on the finger.

**STL carries no units.** This export is in inches; `fingermesh` sniffs that
from the overall size. The extrusion axis is found by looking for the direction
along which every vertex falls onto exactly two equally populated planes -
picking the largest faces by area chooses wrong, because an open truss profile
has less area than its own side walls.

**Geometry cannot tell you which face carries the tags.** Canonicalisation
picks the swept profile's own face, because that is the one the geometry makes
special. On this hand the tags are a quarter turn away on the narrow side wall,
which is why `mesh.roll_deg` exists. Same category of fact as `grip.wrap_sign`:
declared once from looking at the part, not derived.

**The tags sit on the outside, so the mesh hangs behind the spine.** The spine
is fitted through tag centres, so the tagged *surface* is what it follows - not
the part's mid-plane, and not the axis a principal-component fit picks. On this
finger the tagged edge slants about 6 degrees off that axis, which left the body
drifting a centimetre away from its own tags between root and tip.
`with_face_on_spine` levels the surface and drops it to z = 0.

**Grip sign cannot be derived from geometry.** Whether positive bend means
wrapping or back-bending depends on which face of the finger the tags are stuck
to. Calibrated on the built hand 2026-08-30: a genuine wrapping grip reads
*negative*, so `grip.wrap_sign: -1.0`. Recalibrate if the tags are ever moved to
the other face of a finger.

## CAD requirements

- Reference tags on a **flat pad**, not the domed hub. A curved reference tag
  corrupts all fingers, not just one.
- **Two or three reference tags**, not one. The estimator learns their relative
  geometry during zeroing, so any visible subset fixes the frame. One tag means
  a single occlusion invalidates every finger that frame.
- Camera mount at **~140 mm** from the fingertips. Check the webcam's minimum
  focus distance.
- Keep the vertical span tight — extra span forces the camera back, costing
  pixels on every tag.

## Open items

1. **`stations_mm` (24, 58) are still guesses.** `length_mm` now comes from the
   CAD (104.8). The station numbers scale the rendered shape, the extrapolated
   tip, and every threshold expressed in millimetres of travel, so calipers on
   the built hand are worth it. (`base_mm` ±26 is consistent now that the
   fingers are known to be 20 mm wide across, not 30.)
2. **`track3d` CLI is not wired.** `estimator3d` is library-only, driven by
   tests. This is the last piece before live 3D tracking.
3. **Two-tag joint pose** — the biggest accuracy lever left. Each finger has
   tags ~30 mm apart; that baseline is far better conditioned than one 20 mm
   square. No hardware change needed.
4. **Bend direction still unknown.** The estimator handles either, but knowing
   it sets accuracy expectations. One press on a flat surface answers it.
5. **Only one reference tag (11)** on the built hand. Any occlusion or glare on
   it invalidates every finger that frame. Two more on the cuff would give
   graceful degradation; the estimator learns their geometry automatically.
6. **Occlusion is untested.** `simrig` renders tags only, not finger bodies, so
   it cannot tell you whether fingers hide each other. A paper mockup can.
7. **Confirm the 18 mm tag size with calipers.** The printer scales ~93.3%, and
   tag size scales every millimetre and every distance the system reports.
8. **`config/so101.yaml`** still describes the old 5-tag 2-finger layout used by
   the 2D `track` path. `config/hand.yaml` is the current one.


## Suggested next physical step

The hand is built and the ids are in `config/hand.yaml`. Next:

1. `flexsense watch --source 0` — confirm all 7 tags detect and read ≥90 px.
2. `flexsense grip --source 0` — zero with fingers unloaded, then grip. The
   sign is already calibrated (`wrap_sign: -1.0`).
3. Tune `wrap_deg` (4.5° = 14 mm of tip travel) and `backbend_deg` (4.0° =
   12.5 mm) against real grips. The gauges draw both thresholds as ticks, so
   watch where the needle sits during a good hold, a marginal one and a slip.
   Set the target in millimetres and convert with
   `grip.tip_travel_for_bend_deg`.
4. Measure the finger and put the real `stations_mm` in `config/hand.yaml`.
