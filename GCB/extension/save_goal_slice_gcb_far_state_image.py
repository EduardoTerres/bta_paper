"""Save a single state image with the state placed in a corner of the maze.

Reuses the same environment construction as the other goal-slice GCB figure
scripts. The state is the valid, non-goal position closest to the requested
grid corner (an exact corner cell is typically a wall, so this snaps inward).
"""
import argparse
import logging
from pathlib import Path

import torch

from plot_goal_slice_gcb_decoder_samples import image_for_plot, load_checkpoint
from save_goal_slice_gcb_decoder_images import save_image


LOG = logging.getLogger("goal_slice_far_state_image")

CORNERS = {
    "top-left": lambda position, n, m: position[0] + position[1],
    "top-right": lambda position, n, m: position[0] + (m - 1 - position[1]),
    "bottom-left": lambda position, n, m: (n - 1 - position[0]) + position[1],
    "bottom-right": lambda position, n, m: (n - 1 - position[0]) + (m - 1 - position[1]),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Checkpoint to read the environment configuration from")
    parser.add_argument("--corner", choices=sorted(CORNERS), default="top-left")
    parser.add_argument("--output", type=Path,
                        default=Path("figures/goal_slice_gcb_state_corner.png"))
    parser.add_argument("--image-size", type=int, default=256,
                        help="Side length, in pixels, of the saved PNG; 0 keeps native resolution")
    return parser.parse_args()


def corner_position(corner, goals, valid_positions, n, m):
    """Valid, non-goal position closest (Manhattan) to the requested grid corner."""
    key = CORNERS[corner]
    candidates = [position for position in valid_positions if position not in goals] or valid_positions
    return min(candidates, key=lambda position: key(position, n, m))


def main():
    args = parse_args()
    if args.image_size < 0:
        raise ValueError("--image-size must not be negative")

    model_args, _step, env, _model = load_checkpoint(args.checkpoint, torch.device("cpu"))
    goals = {tuple(goal) for goal in env.goal_positions}
    valid_positions = [tuple(position) for position in env.env.possiblePositions]

    state = corner_position(args.corner, goals, valid_positions, env.env.n, env.env.m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(image_for_plot(env, [state]), args.output, args.image_size)
    LOG.info("Saved state %s (%s corner) to %s", state, args.corner, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
