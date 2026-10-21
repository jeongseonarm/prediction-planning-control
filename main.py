"""
Prediction -> planning -> control pipeline on the inD dataset.

    python main.py                     # run and show the animation
    python main.py --no-viz            # headless
    python main.py --save result.gif   # write the animation instead of showing it
    python main.py --controller nmpc   # SLSQP reference instead of LTV-MPC
"""

import argparse
import dataclasses

from config import ScenarioConfig
from simulation.runner import run_simulation
from simulation.world import build_world


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--controller", choices=("ltv", "nmpc"))
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--no-viz", action="store_true", help="skip the animation")
    parser.add_argument("--save", metavar="PATH", help="save the animation (e.g. .gif)")
    parser.add_argument("--fps", type=float, help="playback speed (default: real time)")
    parser.add_argument("--loop", action="store_true", help="make the saved GIF repeat forever")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    overrides = {
        "controller": args.controller,
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
    }
    cfg = dataclasses.replace(
        ScenarioConfig(), **{k: v for k, v in overrides.items() if v is not None}
    )

    visualize = not args.no_viz
    world = build_world(cfg, with_background=visualize)

    # imported here so that --help and the tests do not need torch
    from trajectory_prediction.predictor import TrajectoryPredictor

    predictor = TrajectoryPredictor(
        data_path=cfg.data_path,
        log_dir=cfg.log_dir,
        input_frame=cfg.input_frame,
        output_frame=cfg.output_frame,
        frame_dt=cfg.frame_dt,
    )

    result = run_simulation(cfg, world, predictor, keep_tracks=visualize)

    if visualize:
        from simulation.visualization import animate

        animate(cfg, world, result, fps=args.fps, show=args.save is None,
                save_path=args.save, loop=args.loop)

    return result


if __name__ == "__main__":
    main()
