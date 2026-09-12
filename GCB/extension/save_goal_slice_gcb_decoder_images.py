"""Save decoder input images (state/query/goal-set/next-state) as separate PNGs.

Same sample generation as plot_goal_slice_gcb_decoder_samples.py, but instead of
laying every input and prediction out in one comparison figure, each of the four
per-sample images is written to its own file so it can be inspected pixel-for-pixel.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from plot_goal_slice_gcb_decoder_samples import (
    image_for_plot,
    load_checkpoint,
    make_samples,
)


LOG = logging.getLogger("goal_slice_decoder_images")

COLUMNS = ("state", "query", "goal_set", "next_state")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Checkpoint to read the environment configuration from")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("figures/goal_slice_gcb_decoder_images"))
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of transition/goal-set combinations to render")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-set-split", choices=("auto", "train", "held-out", "random"), default="auto",
                        help="auto uses held-out sets when the checkpoint was trained with them")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Side length, in pixels, of each saved PNG; 0 keeps native resolution")
    return parser.parse_args()


def save_image(array, path, size):
    image = Image.fromarray(array.astype(np.uint8))
    if size:
        image = image.resize((size, size), Image.NEAREST)
    image.save(path)


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.image_size < 0:
        raise ValueError("--image-size must not be negative")

    model_args, _step, env, _model = load_checkpoint(args.checkpoint, torch.device("cpu"))
    rng = np.random.RandomState(args.seed)
    samples, selected_split = make_samples(env, model_args, args.num_samples, args.goal_set_split, rng)
    print("goal_set_split=%s" % selected_split)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples, start=1):
        inputs = dict(state=[sample["state"]], query=[sample["query"]],
                      goal_set=sample["goal_set"], next_state=[sample["next_state"]])
        for column in COLUMNS:
            path = args.output_dir / ("sample%02d_%s.png" % (index, column))
            save_image(image_for_plot(env, inputs[column]), path, args.image_size)
        print("row %d: %s; state=%s action=%d next_state=%s query=%s goal_set=%s target_reward=%.3f" %
              (index, sample["description"], sample["state"], sample["action"], sample["next_state"],
               sample["query"], list(sample["goal_set"]), sample["reward"]))

    LOG.info("Saved %d x %d images to %s", len(samples), len(COLUMNS), args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
