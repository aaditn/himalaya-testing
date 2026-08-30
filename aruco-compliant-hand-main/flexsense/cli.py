from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .calibrate import collect_calibration
from .camera_calib import (
    BoardSpec,
    generate_board,
    refit_from_frames,
    run_camera_calibration,
)
from .screen_target import run_screen_calibration
from .handconfig import load_hand, to_grip_thresholds
from .grip import run_grip_live
from .hud import append_control_bar
from .handtools import check as hand_check, preview as hand_preview, rehearse as hand_rehearse
from .watch import run_watch
from .config import AppConfig, load_config
from .estimator import FrameMeasurement, PlanarDeformationEstimator
from .markers import generate_hand_marker_sheet, generate_marker_sheet
from .models import ForceModelSet
from .outputs import CsvMeasurementWriter, MeasurementPublisher
from .vision import DEFAULT_INTRINSICS, build_detector, open_capture, parse_source, require_cv2


def _format_number(value: float | None, suffix: str) -> str:
    return "--" if value is None else f"{value:+.3f}{suffix}"


def _draw_overlay(frame, corners, ids, measurement: FrameMeasurement) -> None:
    cv2 = require_cv2()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    status_color = (40, 220, 40) if measurement.status == "tracking" else (30, 210, 240)
    if measurement.status == "zeroing":
        status = f"ZEROING {measurement.zero_progress}/{measurement.zero_target} | keep unloaded"
    else:
        status = measurement.status.upper()
    cv2.putText(frame, status, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, status_color, 2)

    y = 62
    for name, sensor in measurement.sensors.items():
        validity = "OK" if sensor.valid else ("STALE" if sensor.stale else "LOST")
        force = (
            f" Fn={_format_number(sensor.force_normal_n, 'N')}"
            if sensor.force_normal_n is not None
            else ""
        )
        line = (
            f"{name} {validity}  normal={_format_number(sensor.normal_deflection_mm, 'mm')} "
            f"shear={_format_number(sensor.shear_deflection_mm, 'mm')}{force}"
        )
        color = (60, 230, 60) if sensor.valid else (40, 80, 240)
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, color, 2)
        y += 25
    cv2.putText(
        frame,
        "q: quit   z: re-zero unloaded fingers",
        (18, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )


def run_tracker(config: AppConfig, source: int | str, display: bool,
                intrinsics: str | None = DEFAULT_INTRINSICS) -> int:
    cv2 = require_cv2()
    detector, note = build_detector(
        config.markers.dictionary, intrinsics,
        (config.camera.width, config.camera.height),
    )
    print(note, file=sys.stderr)
    force_models = ForceModelSet.load(config.output.calibration_file)
    estimator = PlanarDeformationEstimator(config, force_models)
    capture = open_capture(source, config.camera.width, config.camera.height)
    publisher = MeasurementPublisher(
        json_stdout=config.output.json_stdout,
        stdout_hz=config.output.stdout_hz,
        udp_host=config.output.udp_host,
        udp_port=config.output.udp_port,
    )
    try:
        with CsvMeasurementWriter(config.output.csv_path) as writer:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                by_id, corners, ids = detector.detect(frame)
                measurement = estimator.observe(by_id)
                writer.write(measurement)
                publisher.publish(measurement)
                if display:
                    _draw_overlay(frame, corners, ids, measurement)
                    frame = append_control_bar(
                        frame, (("Q", "quit"), ("Z", "re-zero")))
                    cv2.imshow("FlexSense SO-101", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord("z"):
                        estimator.reset_zero()
        return 0
    finally:
        publisher.close()
        capture.release()
        if display:
            cv2.destroyAllWindows()


def _load_args_config(args: argparse.Namespace) -> AppConfig:
    return load_config(args.config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flexsense",
        description="Track printed flexure deformation with an end-effector camera.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    markers = subparsers.add_parser("markers", help="Generate a print-at-100%% marker sheet")
    markers.add_argument("--hand", default="config/hand.yaml",
                         help="Hand declaration to print the tags for")
    markers.add_argument("--config", default=None,
                         help="Use the older so101.yaml planar layout instead")
    markers.add_argument("--output", default="markers.svg")

    track = subparsers.add_parser("track", help="Run live deformation tracking")
    track.add_argument("--config", default="config/so101.yaml")
    track.add_argument("--source", help="Camera index or video file; overrides config")
    track.add_argument("--no-display", action="store_true")
    track.add_argument("--intrinsics", default=DEFAULT_INTRINSICS,
                       help="Camera calibration used to remove lens distortion")
    track.add_argument("--no-undistort", action="store_true",
                       help="Skip lens correction; millimetre readings will be biased")

    board = subparsers.add_parser(
        "camera-board", help="Generate a print-at-100%% ChArUco board for camera calibration"
    )
    board.add_argument("--config", default="config/so101.yaml")
    board.add_argument("--output", default="charuco_board.svg")
    board.add_argument("--dpi", type=int, default=300)

    camcal = subparsers.add_parser(
        "camera-calibrate", help="Estimate camera intrinsics and lens distortion"
    )
    camcal.add_argument("--config", default="config/so101.yaml")
    camcal.add_argument("--source", help="Camera index or video file; overrides config")
    camcal.add_argument("--output", default="calibration/camera_intrinsics.json")
    camcal.add_argument("--views", type=int, default=20)
    camcal.add_argument(
        "--free-principal-point",
        action="store_true",
        help="Let the principal point float. Off by default: on typical webcam "
             "data it is weakly observable and destabilises the focal length.",
    )
    camcal.add_argument(
        "--min-coverage",
        type=float,
        default=0.55,
        help="Keep capturing until board corners have touched this fraction of "
             "the frame. Low coverage leaves the principal point extrapolated.",
    )
    camcal.add_argument(
        "--frames-dir",
        default="calibration/views",
        help="Where to save the accepted frames, so a bad run can be diagnosed "
             "without recapturing. Pass an empty string to disable.",
    )
    camcal.add_argument(
        "--square-mm",
        type=float,
        default=None,
        help="Measured printed size of one black square. Default assumes the "
             "nominal 30 mm; set this to what your calipers actually read.",
    )
    camcal.add_argument("--no-display", action="store_true")

    screen = subparsers.add_parser(
        "screen-calibrate",
        help="Calibrate against a full-screen animated target instead of a print",
    )
    screen.add_argument("--config", default="config/so101.yaml")
    screen.add_argument("--source", help="Camera index or video file; overrides config")
    screen.add_argument("--output", default="calibration/camera_intrinsics.json")
    screen.add_argument("--views", type=int, default=24)
    screen.add_argument("--min-coverage", type=float, default=0.55)
    screen.add_argument("--free-principal-point", action="store_true")
    screen.add_argument("--frames-dir", default="calibration/views_screen")

    refit = subparsers.add_parser(
        "camera-refit", help="Recalibrate from frames saved by an earlier capture"
    )
    refit.add_argument("--config", default="config/so101.yaml")
    refit.add_argument("--frames-dir", default="calibration/views_screen")
    refit.add_argument("--output", default="calibration/camera_intrinsics.json")
    refit.add_argument("--square-mm", type=float, default=None)
    refit.add_argument("--free-principal-point", action="store_true")

    watch = subparsers.add_parser("watch", help="Just show the camera with detected markers")
    watch.add_argument("--config", default="config/so101.yaml")
    watch.add_argument("--source", help="Camera index or video file; overrides config")
    watch.add_argument("--intrinsics", default="calibration/camera_intrinsics.json")
    watch.add_argument("--marker-mm", type=float, default=None,
                       help="Physical size of the finger markers; defaults to config")
    watch.add_argument("--board-square-mm", type=float, default=30.0,
                       help="Measured square size of the printed ChArUco board")
    watch.add_argument(
        "--dictionaries",
        help="Comma-separated ArUco families to look for. Defaults to the finger "
             "dictionary plus the calibration board's.",
    )

    finray = subparsers.add_parser(
        "finray", help="Simulate how the conforming finger deforms under an object")
    finray.add_argument("--material", default="petg",
                        help="pla|petg|abs|pctg|nylon|tpu95a|tpu85a, or a modulus in MPa")
    finray.add_argument("--depth", type=float, default=15.0,
                        help="out-of-plane finger width in mm (not on the side view)")
    finray.add_argument("--rib", type=float, default=1.5, help="rib thickness in mm")
    finray.add_argument("--object", dest="shape", default="cylinder",
                        choices=("cylinder", "flat"))
    finray.add_argument("--radius", type=float, default=15.0)
    finray.add_argument("--station", type=float, default=55.0,
                        help="where the object touches, measured from the tip")
    finray.add_argument("--span", type=float, nargs=2, default=(8.0, 58.0),
                        help="extent of a flat object, in stations from the tip")
    finray.add_argument("--advance", type=float, default=8.0,
                        help="how far to press the object in, in mm")
    finray.add_argument("--steps", type=int, default=20)
    finray.add_argument("--refine", type=int, default=8, help="elements per rib bay")
    finray.add_argument("--hookean", action="store_true",
                        help="force a single-modulus law even for an elastomer")
    finray.add_argument("--svg", default=None, help="write the deformed shapes here")
    finray.add_argument("--json", dest="json_path", default=None)

    hcheck = subparsers.add_parser(
        "hand-check", help="Validate a hand declaration and predict how it will image")
    hcheck.add_argument("--hand", default="config/hand.yaml")

    hprev = subparsers.add_parser(
        "hand-preview", help="Render what the declared camera position would see")
    hprev.add_argument("--hand", default="config/hand.yaml")
    hprev.add_argument("--output", default="preview.png")
    hprev.add_argument("--deflect", default="",
                       help="Per-finger tip deflection, e.g. left=6,right=10")

    hreh = subparsers.add_parser(
        "hand-rehearse", help="Run the full pipeline in simulation against ground truth")
    hreh.add_argument("--hand", default="config/hand.yaml")
    hreh.add_argument("--deflect", default="")

    gripcmd = subparsers.add_parser(
        "grip", help="Live grip classification from finger curvature")
    gripcmd.add_argument("--hand", default="config/hand.yaml")
    gripcmd.add_argument("--source", default="0")
    gripcmd.add_argument("--intrinsics", default=None)
    gripcmd.add_argument("--zero-frames", type=int, default=30)
    gripcmd.add_argument("--no-display", action="store_true")
    gripcmd.add_argument("--no-mesh", action="store_true",
                         help="skip the 3D finger render and use the plain readout")

    labelcmd = subparsers.add_parser(
        "label", help="Capture human GOOD/BAD labels for grip-model training")
    labelcmd.add_argument("--hand", default="config/hand.yaml")
    labelcmd.add_argument("--source", default="0")
    labelcmd.add_argument("--intrinsics", default=None)
    labelcmd.add_argument("--zero-frames", type=int, default=30)
    labelcmd.add_argument("--dataset", default="data/grip_labels",
                          help="local dataset directory (frames plus labels.jsonl)")
    labelcmd.add_argument("--no-mesh", action="store_true",
                          help="skip the 3D finger render and use the plain readout")

    calibrate = subparsers.add_parser("calibrate", help="Fit deflection-to-force calibration")
    calibrate.add_argument("--config", default="config/so101.yaml")
    calibrate.add_argument("--source", help="Camera index or video file; overrides config")
    calibrate.add_argument("--sensor", required=True)
    calibrate.add_argument("--axis", choices=["normal", "shear"], required=True)
    calibrate.add_argument(
        "--loads-g",
        required=True,
        help="Comma-separated applied masses in grams, preferably up then down",
    )
    calibrate.add_argument("--samples", type=int, default=40)
    calibrate.add_argument("--degree", type=int, choices=[1, 2], default=1)
    calibrate.add_argument("--no-display", action="store_true")
    calibrate.add_argument("--intrinsics", default=DEFAULT_INTRINSICS)
    calibrate.add_argument("--no-undistort", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "finray":
            from .finray_run import run as run_finray, write_report
            outcome = run_finray(
                material=args.material, depth=args.depth, rib_thickness=args.rib,
                shape=args.shape, radius=args.radius, station=args.station,
                span=tuple(args.span), advance=args.advance, steps=args.steps,
                refine=args.refine, hookean=args.hookean)
            report, result = outcome["report"], outcome["result"]
            geometry = report["geometry"]
            print(f"taper {geometry['taper_angle_deg']:.3f} deg, back member "
                  f"{geometry['back_length_mm_derived']:.2f} mm (drawing says 91), "
                  f"base cavity {geometry['base_clear_height_mm']:.2f} mm")
            print(f"{report['solver']['nodes']} nodes, "
                  f"{report['solver']['elements']} elements, "
                  f"{geometry['constitutive_law']} material, "
                  f"initial modulus {geometry['initial_modulus_mpa']:.1f} MPa")
            header = (f"{'press mm':>9}{'force N':>10}{'tip dx':>9}{'tip dy':>9}"
                      f"{'tip rot':>9}{'patch mm':>10}{'peak MPa':>10}{'strain %':>10}")
            print(header)
            for row in report["steps"]:
                print(f"{row['advance_mm']:9.2f}{row['normal_force_n']:10.3f}"
                      f"{row['tip_dx_mm']:9.3f}{row['tip_dy_mm']:9.3f}"
                      f"{row['tip_rotation_deg']:9.2f}{row['contact_patch_mm']:10.2f}"
                      f"{row['peak_stress_mpa']:10.2f}{row['peak_strain_pct']:10.1f}")
            if not result.completed:
                print(f"stopped converging at {result.reached_mm:.2f} mm "
                      f"of the {args.advance:.2f} mm requested")
            if args.svg:
                from .finray_svg import render
                print(f"wrote {render(result, args.svg)}")
            if args.json_path:
                print(f"wrote {write_report(report, args.json_path)}")
            return

        if args.command in ("grip", "label"):
            hand = load_hand(args.hand)
            raw_source = args.source
            src = int(raw_source) if str(raw_source).isdigit() else raw_source
            raise SystemExit(run_grip_live(
                hand, src, args.intrinsics,
                display=(not args.no_display if args.command == "grip" else True),
                thresholds=to_grip_thresholds(hand), zero_frames=args.zero_frames,
                use_mesh=not args.no_mesh,
                dataset_dir=args.dataset if args.command == "label" else None))
        if args.command in ("hand-check", "hand-preview", "hand-rehearse"):
            hand = load_hand(args.hand)
            if args.command == "hand-check":
                result = hand_check(hand)
                print(json.dumps({k: v for k, v in result.items() if k != "tags"}, indent=2))
                print(f"\n{'id':>4} {'part':<10} {'station':<9} {'px':>7}  visible")
                for row in result["tags"]:
                    px = "--" if row["pixels"] is None else f"{row['pixels']:.0f}"
                    print(f"{row['id']:4d} {row['part']:<10} {row['station']:<9} {px:>7}  "
                          f"{'yes' if row['visible'] else 'NO'}")
                for problem in result["problems"]:
                    print(f"PROBLEM: {problem}", file=sys.stderr)
                for warning in result["warnings"]:
                    print(f"WARNING: {warning}", file=sys.stderr)
                raise SystemExit(1 if result["problems"] else 0)
            deflections = {}
            for item in (args.deflect or "").split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    deflections[key.strip()] = float(value)
            if args.command == "hand-preview":
                hand_preview(hand, deflections, args.output)
                print(Path(args.output).resolve())
                return
            print(json.dumps(hand_rehearse(hand, deflections), indent=2))
            return
        if args.command == "markers":
            if args.config:
                output = generate_marker_sheet(load_config(args.config), args.output)
            else:
                output = generate_hand_marker_sheet(load_hand(args.hand), args.output)
            print(output.resolve())
            return
        config = _load_args_config(args)
        if args.command == "camera-board":
            output = generate_board(BoardSpec(), args.output, dpi=args.dpi)
            spec = BoardSpec()
            print(output.resolve())
            print(
                f"Print at 100% / actual size. After printing, measure one black square "
                f"with calipers: it must be {spec.square_mm:.1f} mm."
            )
            return
        if args.command == "camera-refit":
            result = refit_from_frames(
                frames_dir=args.frames_dir,
                output=args.output,
                measured_square_mm=args.square_mm,
                fix_principal_point=not args.free_principal_point,
            )
            print(json.dumps(result, indent=2))
            for warning in result["warnings"]:
                print(f"WARNING: {warning}", file=sys.stderr)
            if not result["warnings"]:
                print("Sanity checks passed.", file=sys.stderr)
            return
        source = parse_source(args.source, config.camera.source)
        if args.command == "screen-calibrate":
            result = run_screen_calibration(
                source=source,
                width=config.camera.width,
                height=config.camera.height,
                output=args.output,
                target_views=args.views,
                min_coverage=args.min_coverage,
                fix_principal_point=not args.free_principal_point,
                frames_dir=args.frames_dir or None,
            )
            print(json.dumps(result, indent=2))
            rms = result["rms_reprojection_error_px"]
            verdict = "good" if rms < 0.5 else ("usable" if rms < 1.0 else "POOR - recapture")
            print(f"\nRMS reprojection error {rms:.3f} px -> {verdict}", file=sys.stderr)
            print(f"Tilt spread: {result['tilt_histogram']}", file=sys.stderr)
            for warning in result["warnings"]:
                print(f"WARNING: {warning}", file=sys.stderr)
            if not result["warnings"] and rms < 1.0:
                print("Sanity checks passed.", file=sys.stderr)
            return
        if args.command == "camera-calibrate":
            result = run_camera_calibration(
                source=source,
                width=config.camera.width,
                height=config.camera.height,
                output=args.output,
                target_views=args.views,
                measured_square_mm=args.square_mm,
                frames_dir=args.frames_dir or None,
                min_coverage=args.min_coverage,
                fix_principal_point=not args.free_principal_point,
                display=not args.no_display,
            )
            print(json.dumps(result, indent=2))
            rms = result["rms_reprojection_error_px"]
            verdict = "good" if rms < 0.5 else ("usable" if rms < 1.0 else "POOR - recapture")
            print(f"\nRMS reprojection error {rms:.3f} px -> {verdict}", file=sys.stderr)
            if result["views_dropped_as_outliers"]:
                print(f"Dropped {result['views_dropped_as_outliers']} outlier view(s) and refit.",
                      file=sys.stderr)
            print(f"Tilt spread: {result['tilt_histogram']}", file=sys.stderr)
            for warning in result["warnings"]:
                print(f"WARNING: {warning}", file=sys.stderr)
            if not result["warnings"] and rms < 1.0:
                print("Sanity checks passed.", file=sys.stderr)
            return
        if args.command == "watch":
            families = ([d.strip() for d in args.dictionaries.split(",") if d.strip()]
                        if args.dictionaries else None)
            raise SystemExit(run_watch(config, source, args.intrinsics, families,
                                       args.marker_mm, args.board_square_mm))
        if args.command == "track":
            raise SystemExit(run_tracker(
                config, source, display=not args.no_display,
                intrinsics=None if args.no_undistort else args.intrinsics))
        if args.command == "calibrate":
            loads = [float(item.strip()) for item in args.loads_g.split(",") if item.strip()]
            points, coefficients, r_squared = collect_calibration(
                config=config,
                source=source,
                sensor_name=args.sensor,
                axis_name=args.axis,
                loads_g=loads,
                samples_per_load=args.samples,
                degree=args.degree,
                display=not args.no_display,
                intrinsics=None if args.no_undistort else args.intrinsics,
            )
            print(
                json.dumps(
                    {
                        "saved": str(config.output.calibration_file),
                        "coefficients": coefficients,
                        "r_squared": r_squared,
                        "points": [point.__dict__ for point in points],
                    },
                    indent=2,
                )
            )
            return
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        raise SystemExit(130)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
