"""Smoke test: `offline_gcab_four_rooms.py --num-rooms 4 --device cpu` initializes
training from vector (x, y) state-action-goal transitions, with no convolutions
anywhere in the trained modules.

Run with: python GCB/extension/scripts/smoke_test_state_four_rooms.py
"""
import argparse
import os
import sys
import tempfile

import torch.nn as nn

EXTENSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GCB_ROOT = os.path.dirname(EXTENSION_DIR)
for path in (GCB_ROOT, EXTENSION_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from offline_gcab_four_rooms import build_variant, collect_four_rooms_replay  # noqa: E402
from state_gcab.trainer import train_state_representation_offline  # noqa: E402


def assert_no_convolutions(module):
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    convs = [name for name, m in module.named_modules() if isinstance(m, conv_types)]
    assert not convs, f"Found convolutional layers, expected a pure MLP/state pipeline: {convs}"


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = argparse.Namespace(
            num_rooms=4,
            dataset_loc=os.path.join(tmpdir, "replay_four_rooms_state_4_smoke.pt"),
            dataset_steps=300,
            number_training_points=200,
            training_iterations=3,
            eval_freq=3,
            log_freq=3,
            max_episode_steps=20,
            device="cpu",
            use_wandb=False,
            no_collect=False,
            eval_video_save_dir=os.path.join(tmpdir, "eval_videos"),
            eval_video_record_number=2,
            no_video=False,
        )

        variant = build_variant(args)
        collect_four_rooms_replay(variant, variant["dataset_loc"], args.dataset_steps)

        agent, _ = train_state_representation_offline(variant)

        obs_shape = agent.actor.encoder.obs_shape
        assert obs_shape == (2,), f"Expected vector obs_shape (2,), got {obs_shape}"
        assert not hasattr(agent.actor.encoder, "convs"), "Encoder should not have conv layers"

        assert_no_convolutions(agent.actor)
        assert_no_convolutions(agent.critic)
        assert_no_convolutions(agent.critic_representation)

        gifs = [f for f in os.listdir(variant["eval_video_save_dir"]) if f.endswith(".gif")]
        assert gifs, "Expected rollout/goal GIFs to be saved during eval"

    print("OK: state-based four-rooms training initialized with vector observations, "
          "no convolutions, and saved rollout images.")


if __name__ == "__main__":
    main()
