"""Standalone one-hot state-vector goal-slice GCB experiment for four rooms.

The learned task representation is phi(x_s, x_g, x_G_plus).  The network gets
only those three vectors: query membership is used to compute a scalar reward
and is not kept as a feature, ID, coordinate, or input tensor.
"""
import argparse
import itertools
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from state_gcab.environment import VectorFourRoomsGoalEnv

log = logging.getLogger("goal_slice_gcb_state")


def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def umap_snapshot_steps(training_iterations, num_images):
    """Return exactly num_images evenly spaced update indices, including the final one."""
    if num_images <= 0 or training_iterations <= 0:
        return set()
    return set(np.linspace(0, training_iterations - 1, min(num_images, training_iterations), dtype=int).tolist())


def transition(position, action, walls, valid_positions):
    deltas = ((-1, 0), (0, 1), (1, 0), (0, -1), (0, 0))
    dx, dy = deltas[action]
    candidate = position[0] + dx, position[1] + dy
    return candidate if candidate not in walls and candidate in valid_positions else position


def sample_set(goals, query, contains_query, rng):
    """Sample a nonempty goal set; contains_query is never stored in replay."""
    other_goals = [g for g in goals if g != query]
    result = [query] if contains_query else []
    result.extend(g for g in other_goals if rng.rand() < .5)
    if not result:
        result.append(other_goals[rng.randint(len(other_goals))])
    return tuple(sorted(result))


def extended_reward(next_position, query, goal_set, r_universal, r_empty, r_min):
    if next_position == query:
        return r_universal if query in goal_set else r_empty
    if next_position in goal_set:
        return r_min
    return 0.0


def goal_set_vector(env, positions):
    """One-hot for singleton inputs and multi-hot for the desired goal set."""
    vector = np.zeros(env.observation_space.shape, dtype=np.float32)
    for position in positions:
        vector[env._position_to_index[tuple(position)]] = 1.0
    return vector


class GoalSliceReplayBuffer:
    """Offline GCB-style replay with separate state, query, and goal-set vectors."""
    VERSION = 2

    def __init__(self, capacity, vector_shape, action_shape):
        self.capacity = capacity
        self.vector_shape, self.action_shape = tuple(vector_shape), tuple(action_shape)
        shape = (capacity, *vector_shape)
        self.states = np.empty(shape, dtype=np.float32)
        self.next_states = np.empty(shape, dtype=np.float32)
        self.queries = np.empty(shape, dtype=np.float32)
        self.goal_sets = np.empty(shape, dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.query_hits = np.empty(capacity, dtype=bool)
        self.idx = 0

    def add(self, state, action, reward, next_state, query, goal_set, query_hit):
        i = self.idx
        self.states[i], self.next_states[i] = state, next_state
        self.queries[i], self.goal_sets[i] = query, goal_set
        self.actions[i], self.rewards[i, 0] = action, reward
        self.query_hits[i] = query_hit
        self.idx += 1

    def sample(self, size, device, query_hit_oversample=0., pool_indices=None):
        if not 0. <= query_hit_oversample <= 1.:
            raise ValueError("query_hit_oversample must be in [0, 1]")
        pool = np.arange(self.idx) if pool_indices is None else np.asarray(pool_indices)
        if not len(pool):
            raise ValueError("cannot sample from an empty replay subset")
        if query_hit_oversample == 0.:
            indices = np.random.choice(pool, size=size, replace=True)
        else:
            hit_indices = pool[self.query_hits[pool]]
            miss_indices = pool[~self.query_hits[pool]]
            if not len(hit_indices) or not len(miss_indices):
                raise ValueError("query-hit oversampling requires both query-hit and non-hit transitions")
            natural_rate = len(hit_indices) / len(pool)
            target_rate = natural_rate + query_hit_oversample * (.5 - natural_rate)
            num_hits = int(round(size * target_rate))
            indices = np.concatenate((
                np.random.choice(hit_indices, size=num_hits, replace=True),
                np.random.choice(miss_indices, size=size - num_hits, replace=True),
            ))
            np.random.shuffle(indices)
        vector = lambda x: torch.as_tensor(x[indices], device=device).float()
        return dict(state=vector(self.states), next_state=vector(self.next_states),
                    query=vector(self.queries), goal_set=vector(self.goal_sets),
                    action=torch.as_tensor(self.actions[indices], device=device),
                    reward=torch.as_tensor(self.rewards[indices], device=device))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(version=self.VERSION, vector_shape=self.vector_shape,
                        action_shape=self.action_shape, idx=self.idx,
                        states=self.states[:self.idx], next_states=self.next_states[:self.idx],
                        queries=self.queries[:self.idx], goal_sets=self.goal_sets[:self.idx],
                        actions=self.actions[:self.idx], rewards=self.rewards[:self.idx], query_hits=self.query_hits[:self.idx]),
                   path, pickle_protocol=4)

    @classmethod
    def load(cls, path, limit=None):
        payload = torch.load(path, map_location="cpu")
        if payload.get("version") != cls.VERSION:
            raise ValueError("Not a state goal-slice replay dataset: " + str(path))
        count = payload["idx"] if limit is None else min(payload["idx"], limit)
        result = cls(count, payload["vector_shape"], payload["action_shape"])
        for name in ("states", "next_states", "queries", "goal_sets", "actions", "rewards"):
            getattr(result, name)[:count] = payload[name][:count]
        if "query_hits" in payload:
            result.query_hits[:count] = payload["query_hits"][:count]
        else:
            # Legacy replays did not store this flag; singleton renderings/vectors
            # are identical exactly when the transition entered the query cell.
            axes = tuple(range(1, result.next_states.ndim))
            result.query_hits[:count] = np.all(result.next_states[:count] == result.queries[:count], axis=axes)
        result.idx = count
        return result


def collect(args, path):
    env = VectorFourRoomsGoalEnv(num_rooms=args.num_rooms, max_episode_steps=args.max_episode_steps, slip_prob=0.0)
    replay = GoalSliceReplayBuffer(args.dataset_steps, env.observation_space.shape, env.action_space.shape)
    rng = np.random.RandomState(args.seed)
    positions = list(env.env.possiblePositions)
    goals = [tuple(goal) for goal in env.goal_positions]
    train_goal_sets = None
    if args.evaluate_on_held_out:
        train_goal_sets, held_out_goal_sets = write_goal_set_split(path, goals)
        log.info("Held-out split: %d train and %d eval goal sets", len(train_goal_sets), len(held_out_goal_sets))
    walls, valid = set(env.env.walls), set(positions)
    for i in range(args.dataset_steps):
        state_position = positions[rng.randint(len(positions))]
        action_id = int(rng.randint(env.action_space.shape[0]))
        next_position = transition(state_position, action_id, walls, valid)
        query = goals[rng.randint(len(goals))]
        goal_set = train_goal_sets[rng.randint(len(train_goal_sets))] if train_goal_sets is not None else sample_set(goals, query, bool(rng.randint(2)), rng)
        reward = extended_reward(next_position, query, goal_set,
                                 args.r_universal, args.r_empty, args.r_min)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[action_id] = 1
        # All three encoder inputs are position-indexed vectors; the goal-set vector is multi-hot.
        replay.add(goal_set_vector(env, [state_position]), action, reward,
                   goal_set_vector(env, [next_position]), goal_set_vector(env, [query]),
                   goal_set_vector(env, goal_set), next_position == query)
        if (i + 1) % max(1, args.dataset_steps // 10) == 0:
            log.info("Collected %d/%d transitions", i + 1, args.dataset_steps)
    replay.save(path)
    log.info("Saved goal-slice replay: %s", path)


class GoalSliceGCB(nn.Module):
    """GCB next-observation bisimulation with a three-vector MLP encoder."""
    def __init__(self, vector_shape, args, device):
        super().__init__()
        input_dim = 3 * int(np.prod(vector_shape))
        hidden_dim = args.num_filters
        layers = []
        for _ in range(args.num_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, args.feature_dim))
        self.encoder = nn.Sequential(*layers).to(device)
        self.decoder = nn.Sequential(nn.Linear(2 * args.feature_dim, 512), nn.LayerNorm(512),
                                     nn.ReLU(), nn.Linear(512, 1)).to(device)
        self.discount, self.action_weight, self.metric_loss = args.discount, args.action_weight, args.metric_loss
        self.optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                           lr=args.lr, weight_decay=args.weight_decay)

    def encode(self, state, query, goal_set):
        return self.encoder(torch.cat((state, query, goal_set), dim=1))

    def _loss_terms(self, batch):
        z = self.encode(batch["state"], batch["query"], batch["goal_set"])
        z_next = self.encode(batch["next_state"], batch["query"], batch["goal_set"])
        permutation = torch.randperm(z.shape[0], device=z.device)
        p = 1 if self.metric_loss == "l1" else 2
        z_distance = torch.norm(z - z[permutation], p=p, dim=1)
        reward_distance = F.smooth_l1_loss(batch["reward"], batch["reward"][permutation], reduction="none").squeeze(1)
        next_distance = torch.norm(z_next - z_next[permutation], p=2, dim=1)
        # This is PairedStateGoal.encoder_loss with transition_model_type=next_observation.
        bisim_target = (self.action_weight * reward_distance + self.discount * next_distance).detach()
        bisim_loss = (z_distance - bisim_target).square().mean()
        reward_loss = F.mse_loss(self.decoder(torch.cat((z, z_next), dim=1)), batch["reward"])
        return bisim_loss + reward_loss, bisim_loss, reward_loss, z

    
    @staticmethod
    def _metrics(prefix, total, bisim_loss, reward_loss, z):
        return {prefix + "/phi_loss": total.item(),
                prefix + "/phi_bisimulation_loss": bisim_loss.item(),
                prefix + "/phi_reward_decoder_loss": reward_loss.item(),
                prefix + "/phi_mean_norm": torch.norm(z.detach(), p=1, dim=1).mean().item()}

    def update(self, batch):
        self.train()
        total, bisim_loss, reward_loss, z = self._loss_terms(batch)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        return self._metrics("train", total, bisim_loss, reward_loss, z)

    @torch.no_grad()
    def validation_metrics(self, batch):
        self.eval()
        total, bisim_loss, reward_loss, z = self._loss_terms(batch)
        return self._metrics("stats_eval", total, bisim_loss, reward_loss, z)


@torch.no_grad()
def evaluate_collapse(model, env, args, device):
    """Compare unseen same-status and different-status task representations."""
    model.eval()
    rng = np.random.RandomState(args.seed + 1009)
    positions, goals = list(env.env.possiblePositions), [tuple(goal) for goal in env.goal_positions]
    within, between = [], []
    for state_position in positions[:min(len(positions), args.collapse_eval_states)]:
        state = torch.as_tensor(goal_set_vector(env, [state_position]), device=device).unsqueeze(0)
        for query_position in goals:
            query = torch.as_tensor(goal_set_vector(env, [query_position]), device=device).unsqueeze(0)
            status_embeddings = []
            for member in (False, True):
                sets = evaluation_goal_sets(goals, query_position, member, args, rng, args.collapse_sets_per_status)
                goal_vectors = torch.as_tensor(np.stack([goal_set_vector(env, x) for x in sets]), device=device)
                z = model.encode(state.expand_as(goal_vectors), query.expand_as(goal_vectors), goal_vectors)
                status_embeddings.append(z)
                # Evaluate in the same L1 geometry as the default bisimulation loss.
                distances = torch.cdist(z, z, p=1)
                within.extend(distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)].cpu().tolist())
            between.extend(torch.cdist(status_embeddings[0], status_embeddings[1], p=1).flatten().cpu().tolist())
    in_class, out_class = float(np.mean(within)), float(np.mean(between))
    return {"collapse/within_status_distance": in_class,
            "collapse/between_status_distance": out_class,
            "collapse/separation_ratio": out_class / max(in_class, 1e-8),
            "collapse/within_pairs": len(within), "collapse/between_pairs": len(between)}


def start_wandb(args, name):
    if not args.use_wandb:
        return None
    import wandb
    return wandb.init(project=args.project_name,
                      group=args.group or "goal-slice-gcb-four-rooms-state-" + str(args.num_rooms),
                      name=name, config=vars(args))


def train(args, dataset_path):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    replay = GoalSliceReplayBuffer.load(dataset_path, args.number_training_points)
    env = VectorFourRoomsGoalEnv(num_rooms=args.num_rooms, max_episode_steps=args.max_episode_steps, slip_prob=0.0)
    model = GoalSliceGCB(replay.vector_shape, args, device)
    if not 0. < args.validation_fraction < 1.:
        raise ValueError("validation_fraction must be in (0, 1)")
    if args.validation_freq <= 0:
        raise ValueError("validation_freq must be positive")
    if not 0. < args.lr_factor < 1.:
        raise ValueError("lr_factor must be in (0, 1)")
    split_rng = np.random.RandomState(args.seed + 2718)
    split_indices = split_rng.permutation(replay.idx)
    validation_size = max(1, int(round(replay.idx * args.validation_fraction)))
    if validation_size >= replay.idx:
        raise ValueError("validation_fraction leaves no training transitions")
    validation_indices, train_indices = split_indices[:validation_size], split_indices[validation_size:]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    name = "goal-slice-gcb-state-rooms%d-fdim%d-iters%d-bs%d-qhit%.3g-seed%d" % (
        args.num_rooms, args.feature_dim, args.training_iterations, args.batch_size,
        args.query_hit_oversample, args.seed)
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = start_wandb(args, name)
    started = time.time()
    umap_steps = umap_snapshot_steps(args.training_iterations, args.num_umap_images) if wandb_run and args.log_umap else set()
    log.info("Training on %d transitions; outputs: %s", replay.idx, output_dir)
    with (output_dir / "metrics.jsonl").open("w") as output:
        for step in range(args.training_iterations):
            metrics = model.update(replay.sample(args.batch_size, device, args.query_hit_oversample, train_indices))
            if step % args.log_freq == 0:
                metrics.update(step=step, **{"train/steps_per_second": (step + 1) / max(time.time() - started, 1e-8)})
                log.info("step=%d phi=%.6f bisim=%.6f decoder=%.6f", step, metrics["train/phi_loss"],
                         metrics["train/phi_bisimulation_loss"], metrics["train/phi_reward_decoder_loss"])
                output.write(json.dumps(metrics) + "\n"); output.flush()
                if wandb_run: wandb_run.log(metrics, step=step)
            if step % args.validation_freq == 0 or step + 1 == args.training_iterations:
                validation = model.validation_metrics(replay.sample(args.batch_size, device, pool_indices=validation_indices))
                scheduler.step(validation["stats_eval/phi_loss"])
                validation.update(step=step, **{"stats_eval/lr": model.optimizer.param_groups[0]["lr"]})
                log.info("step=%d val_phi=%.6f lr=%.2e", step, validation["stats_eval/phi_loss"], validation["stats_eval/lr"])
                output.write(json.dumps(validation) + "\n"); output.flush()
                if wandb_run: wandb_run.log(validation, step=step)
            if step % args.eval_freq == 0 or step + 1 == args.training_iterations:
                metrics = evaluate_collapse(model, env, args, device)
                metrics.update({"stats_eval/" + key: value for key, value in metrics.items() if key.startswith("collapse/")})
                metrics["step"] = step
                log.info("step=%d within=%.6f between=%.6f ratio=%.3f", step,
                         metrics["collapse/within_status_distance"], metrics["collapse/between_status_distance"],
                         metrics["collapse/separation_ratio"])
                output.write(json.dumps(metrics) + "\n"); output.flush()
                if wandb_run: wandb_run.log(metrics, step=step)
            if wandb_run and args.log_umap and step in umap_steps:
                log_training_umap(model, env, args, device, wandb_run, output_dir, step, split="train")
                if args.evaluate_on_held_out:
                    log_training_umap(model, env, args, device, wandb_run, output_dir, step, split="eval")
    torch.save({"model": model.state_dict(), "args": vars(args)}, output_dir / "goal_slice_gcb_state.pt")
    if wandb_run: wandb_run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-rooms", type=int, default=4, choices=(4, 8, 16))
    parser.add_argument("--max-episode-steps", type=int, default=75)
    parser.add_argument("--dataset-loc", default="")
    parser.add_argument("--dataset-steps", type=int, default=60000)
    parser.add_argument("--number-training-points", type=int, default=50000)
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument("--training-iterations", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--query-hit-oversample", type=float, default=0.,
                        help="0 keeps natural replay sampling; 1 makes batches 50% query-entering transitions")
    parser.add_argument("--validation-fraction", type=float, default=.1,
                        help="Fraction of the loaded replay held out for validation-loss scheduling")
    parser.add_argument("--validation-freq", type=int, default=500,
                        help="Training updates between validation-loss scheduler checks")
    parser.add_argument("--lr-patience", type=int, default=5,
                        help="Validation checks without improvement before reducing the learning rate")
    parser.add_argument("--lr-factor", type=float, default=.5,
                        help="Factor applied by ReduceLROnPlateau when validation loss plateaus")
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--log-freq", type=int, default=100)
    # Match the existing one-hot-state GCB MLP defaults where applicable.
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--metric-loss", choices=("l1", "l2"), default="l1")
    parser.add_argument("--discount", type=float, default=.99)
    parser.add_argument("--action-weight", type=float, default=25.)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--r-universal", type=float, default=1.)
    parser.add_argument("--r-empty", type=float, default=-1.)
    parser.add_argument("--r-min", type=float, default=-.25)
    parser.add_argument("--evaluate-on-held-out", action="store_true", help="Train on odd-cardinality goal sets and evaluate only on the disjoint even-cardinality sets")
    parser.add_argument("--collapse-eval-states", type=int, default=32)
    parser.add_argument("--collapse-sets-per-status", type=int, default=16)
    parser.add_argument("--log-umap", action="store_true", help="Log global and individual state-goal UMAPs to this training W&B run")
    parser.add_argument("--umap-log-freq", type=int, default=5000, help="Deprecated; use --num-umap-images to control snapshot count")
    parser.add_argument("--num-umap-images", type=int, default=50, help="Total UMAP snapshots to save and log")
    parser.add_argument("--global-umap-states", type=int, default=32, help="Number of fixed states represented in the global UMAP")
    parser.add_argument("--individual-umap-contexts", type=int, default=4,
                        help="Number of fixed state-goal contexts to log individually (0 disables them)")
    parser.add_argument("--umap-sets-per-class", type=int, default=64)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=.1)
    parser.add_argument("--umap-usetex", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="goal_slice_gcb_state_runs")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--project-name", default="gcb-four-rooms-state")
    parser.add_argument("--group", default="")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    if args.number_training_points > args.dataset_steps and not args.no_collect:
        raise ValueError("number-training-points cannot exceed dataset-steps")
    if args.num_umap_images <= 0:
        raise ValueError("--num-umap-images must be positive")
    if args.r_universal == args.r_empty:
        log.warning("r_universal equals r_empty: query status has no direct reward signal")
    seed_everywhere(args.seed)
    # A separate name prevents reusing a replay collected under the legacy 50/50 split.
    dataset_suffix = "_heldout80_20" if args.evaluate_on_held_out else ""
    dataset = args.dataset_loc or "replay_goal_slice_four_rooms_state_%d%s.pt" % (args.num_rooms, dataset_suffix)
    if not args.no_collect and not os.path.exists(dataset):
        collect(args, dataset)
    elif not os.path.exists(dataset):
        raise FileNotFoundError("Dataset missing: %s (omit --no-collect to build it)" % dataset)
    else:
        if args.evaluate_on_held_out:
            manifest_path = Path(dataset).with_suffix(".goal_set_split.json")
            if not manifest_path.exists():
                raise FileNotFoundError("Held-out evaluation requires the matching .goal_set_split.json manifest")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("scheme") != "goal_set_complement_pairs_80_20":
                raise ValueError("Held-out replay uses a different goal-set split; collect a new 80/20 replay")
        log.info("Using existing goal-slice dataset: %s", dataset)
    train(args, dataset)






def _umap_goal_sets(goals, query, contains_query, limit, rng):
    """Distinct vector task sets; the empty set is retained for evaluation."""
    others = [goal for goal in goals if goal != query]
    goal_sets = []
    for mask in itertools.product((False, True), repeat=len(others)):
        selected = ([query] if contains_query else []) + [goal for goal, keep in zip(others, mask) if keep]
        goal_sets.append(tuple(sorted(selected)))
    if limit <= 0 or limit >= len(goal_sets):
        return goal_sets
    return [goal_sets[index] for index in rng.choice(len(goal_sets), limit, replace=False)]


@torch.no_grad()
def _umap_embeddings(model, env, state_position, query_position, limit, args, rng, device, split="train"):
    goals = [tuple(goal) for goal in env.goal_positions]
    state = torch.as_tensor(goal_set_vector(env, [state_position]), device=device).unsqueeze(0)
    query = torch.as_tensor(goal_set_vector(env, [query_position]), device=device).unsqueeze(0)
    features, labels = [], []
    for member in (True, False):
        sets = evaluation_goal_sets(goals, query_position, member, args, rng, limit, split=split)
        vectors = torch.as_tensor(np.stack([goal_set_vector(env, goal_set) for goal_set in sets]), device=device)
        features.append(model.encode(state.expand_as(vectors), query.expand_as(vectors), vectors).cpu().numpy())
        labels.extend([int(member)] * len(sets))
    return np.concatenate(features), np.asarray(labels, dtype=int)


def _umap_diagnostics(vectors, labels):
    """Raw-latent collapse, rank, scatter, and spectral diagnostics."""
    positive, negative = vectors[labels == 1], vectors[labels == 0]
    def within_l1(x):
        distances = np.abs(x[:, None] - x[None, :]).sum(axis=2)
        return float(distances[np.triu_indices(len(x), 1)].mean())
    d_plus, d_minus = within_l1(positive), within_l1(negative)
    d_between = float(np.abs(positive[:, None] - negative[None, :]).sum(axis=2).mean())
    within = (d_plus + d_minus) / 2
    singular = np.linalg.svd(vectors, compute_uv=False)
    centered = np.linalg.svd(vectors - vectors.mean(axis=0, keepdims=True), compute_uv=False)
    positive_mean, negative_mean = positive.mean(axis=0), negative.mean(axis=0)
    positive_variance = float(np.square(positive - positive_mean).sum(axis=1).mean())
    negative_variance = float(np.square(negative - negative_mean).sum(axis=1).mean())
    centroid_distance = float(np.linalg.norm(positive_mean - negative_mean))
    euclidean = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=2)
    tau = max(float(np.median(euclidean[np.triu_indices(len(vectors), 1)])), 1e-8)
    affinity = np.exp(-np.square(euclidean) / (2 * tau * tau)); np.fill_diagonal(affinity, 0)
    degree = affinity.sum(axis=1)
    laplacian = np.eye(len(vectors)) - affinity / np.sqrt(np.outer(degree, degree) + 1e-8)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    max_k = min(6, len(vectors) - 1)

    fitted, bics, silhouettes = {}, {}, {}
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0).fit(vectors)
        prediction = gmm.predict(vectors)
        fitted[k], bics[k] = prediction, float(gmm.bic(vectors))
        silhouettes[k] = float(silhouette_score(vectors, prediction)) if 1 < len(np.unique(prediction)) < len(vectors) else float("nan")
    selected_k = min(bics, key=bics.get)
    selected_labels = fitted[selected_k]
    kmeans_labels = KMeans(n_clusters=2, random_state=0, n_init=20).fit_predict(vectors)
    kmeans_accuracy = max(float(np.mean(labels == kmeans_labels)), float(np.mean(labels == (1 - kmeans_labels))) )

    return {
        "D_plus": d_plus, "D_minus": d_minus, "D_between": d_between,
        "collapse_ratio": d_between / (within + 1e-8),
        "collapse_index": 1 - within / (d_between + 1e-8),
        "rank2_explained_variance": float(np.square(singular[:2]).sum() / max(np.square(singular).sum(), 1e-8)),
        "centered_rank1_explained_variance": float(centered[0] ** 2 / max(np.square(centered).sum(), 1e-8)),
        "within_variance_positive": positive_variance, "within_variance_negative": negative_variance,
        "centroid_distance": centroid_distance,
        "selected_k": int(selected_k), "silhouette_k2": silhouettes.get(2, float("nan")),
        "ARI": float(adjusted_rand_score(labels, selected_labels)),
        "NMI": float(normalized_mutual_info_score(labels, selected_labels)),
        "kmeans2_accuracy": kmeans_accuracy,
        "nc_collapse": (positive_variance + negative_variance) / (centroid_distance ** 2 + 1e-8),
        "laplacian_lambda1": float(eigenvalues[0]), "laplacian_lambda2": float(eigenvalues[1]),
        "laplacian_lambda3": float(eigenvalues[2]), "laplacian_eigengap": float(eigenvalues[2] - eigenvalues[1]),
        "singular_values": singular, "centered_singular_values": centered,
        "laplacian_eigenvalues": eigenvalues,
    }


def _global_umap_goal_sets(goals, query, args, rng, split="train"):
    """Distinct goal sets for one global-UMAP context, split by query membership."""
    if args.evaluate_on_held_out:
        train_sets, held_out_sets = goal_set_split(goals)
        all_sets = train_sets if split == "train" else held_out_sets
    else:
        all_sets = [goal_set for goal_set in _umap_goal_sets(goals, query, False, 0, rng) if goal_set]
        all_sets += _umap_goal_sets(goals, query, True, 0, rng)
    member_sets = [goal_set for goal_set in all_sets if query in goal_set]
    nonmember_sets = [goal_set for goal_set in all_sets if query not in goal_set]
    limit = args.umap_sets_per_class
    if 0 < limit < len(member_sets):
        member_sets = [member_sets[index] for index in rng.choice(len(member_sets), limit, replace=False)]
    if 0 < limit < len(nonmember_sets):
        nonmember_sets = [nonmember_sets[index] for index in rng.choice(len(nonmember_sets), limit, replace=False)]
    return member_sets, nonmember_sets


@torch.no_grad()
def _global_centered_umap_embeddings(model, env, args, device, split="train"):
    """Pool contexts only after removing each context shared state/query offset."""
    rng = np.random.RandomState(args.seed + 2027)
    positions = list(env.env.possiblePositions)[:min(len(env.env.possiblePositions), args.global_umap_states)]
    goals = [tuple(goal) for goal in env.goal_positions]
    features, labels = [], []
    for state_position in positions:
        state = torch.as_tensor(goal_set_vector(env, [state_position]), device=device).unsqueeze(0)
        for query_position in goals:
            query = torch.as_tensor(goal_set_vector(env, [query_position]), device=device).unsqueeze(0)
            member_sets, nonmember_sets = _global_umap_goal_sets(goals, query_position, args, rng, split=split)
            member_vectors = torch.as_tensor(np.stack([goal_set_vector(env, goal_set) for goal_set in member_sets]), device=device)
            nonmember_vectors = torch.as_tensor(np.stack([goal_set_vector(env, goal_set) for goal_set in nonmember_sets]), device=device)
            member_z = model.encode(state.expand_as(member_vectors), query.expand_as(member_vectors), member_vectors)
            nonmember_z = model.encode(state.expand_as(nonmember_vectors), query.expand_as(nonmember_vectors), nonmember_vectors)
            context_z = torch.cat((member_z, nonmember_z), dim=0)
            features.append((context_z - context_z.mean(dim=0, keepdim=True)).cpu().numpy())
            labels.extend([1] * len(member_z) + [0] * len(nonmember_z))
    return np.concatenate(features), np.asarray(labels, dtype=int), len(positions) * len(goals)


def log_training_umap(model, env, args, device, wandb_run, output_dir, step, split="train"):
    """Save and log one global UMAP snapshot of per-context centered task embeddings."""
    try:
        import umap
    except ImportError:
        log.warning("Skipping global UMAP: install umap-learn in gcb")
        return
    import matplotlib
    matplotlib.use("Agg")
    if args.umap_usetex:
        matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif"})
    import matplotlib.pyplot as plt
    import wandb

    namespace = "stats_train" if split == "train" else "stats_eval"

    with torch.no_grad():
        vectors, labels, num_contexts = _global_centered_umap_embeddings(model, env, args, device, split=split)
    coordinates = umap.UMAP(n_neighbors=min(args.umap_n_neighbors, len(vectors) - 1),
                            min_dist=args.umap_min_dist, metric="manhattan",
                            random_state=args.seed).fit_transform(vectors)
    global_dir = "global_centered_umap" if split == "train" else "eval_global_centered_umap"
    snapshot_dir = output_dir / global_dir / ("step_%07d" % step)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    png = snapshot_dir / "global_centered_umap.png"
    figure, axis = plt.subplots(figsize=(6, 5))
    for member, color, label in ((1, "#238b45", "query included"),
                                 (0, "#d95f0e", "query excluded")):
        mask = labels == member
        axis.scatter(coordinates[mask, 0], coordinates[mask, 1], c=color, s=15, alpha=.65, label=label)
    axis.set_title("Global centered UMAP (%d fixed (s, g) contexts)" % num_contexts)
    axis.set_xlabel("UMAP 1"); axis.set_ylabel("UMAP 2"); axis.legend(frameon=False)
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    wandb_run.log({namespace + "/global_centered_umap": wandb.Image(str(png), caption="per-(s,g)-centered embeddings")}, step=step)
    # Keep fixed raw (state, query) contexts alongside the aggregated plot.
    # Pairing states and queries retains the historical four Four-Rooms contexts.
    positions = list(env.env.possiblePositions)
    goals = [tuple(goal) for goal in env.goal_positions]
    num_contexts = max(0, min(args.individual_umap_contexts, len(positions), len(goals)))
    individual_dir = output_dir / ("training_umap" if split == "train" else "eval_umap") / ("step_%07d" % step)
    individual_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed + 4049)
    for context_index, (state_position, query_position) in enumerate(zip(positions, goals)):
        if context_index >= num_contexts:
            break
        vectors, labels = _umap_embeddings(model, env, state_position, query_position,
                                           args.umap_sets_per_class, args, rng, device, split=split)
        coordinates = umap.UMAP(n_neighbors=min(args.umap_n_neighbors, len(vectors) - 1),
                                min_dist=args.umap_min_dist, metric="manhattan",
                                random_state=args.seed + context_index).fit_transform(vectors)
        png = individual_dir / ("context_%02d_umap.png" % context_index)
        figure, axis = plt.subplots(figsize=(6, 5))
        for member, color, label in ((1, "#238b45", "query included"),
                                     (0, "#d95f0e", "query excluded")):
            mask = labels == member
            axis.scatter(coordinates[mask, 0], coordinates[mask, 1], c=color, s=15,
                         alpha=.65, label=label)
        axis.set_title("UMAP: state=%s, query=%s" % (state_position, query_position))
        axis.set_xlabel("UMAP 1"); axis.set_ylabel("UMAP 2"); axis.legend(frameon=False)
        figure.savefig(png, dpi=180, bbox_inches="tight")
        figure.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        wandb_run.log({namespace + "/state_goal_umap/context_%02d" % context_index:
                       wandb.Image(str(png), caption="state=%s, query=%s" %
                                   (state_position, query_position))}, step=step)
    log.info("Saved and logged %s global centered UMAP and %d individual state-goal UMAPs at step %d",
             split, num_contexts, step)


def goal_set_split(goals):
    """Deterministic, approximately 80/20 split of vector goal sets.

    Whole complement pairs are assigned together. Each selected pair contains one
    set that includes any fixed query and one that excludes it, so both train and
    evaluation retain exactly balanced query-membership labels. The ratio is as
    close to 80/20 as an integral number of pairs permits.
    """
    all_sets = [tuple(goal for goal, take in zip(goals, mask) if take)
                for mask in itertools.product((False, True), repeat=len(goals))]
    pair_count = len(all_sets) // 2
    pair_indices = np.random.RandomState(0).permutation(pair_count)
    held_out_pair_count = min(max(1, int(round(.2 * pair_count))), pair_count - 1)
    held_out_indices = set()
    for index in pair_indices[:held_out_pair_count]:
        held_out_indices.update((int(index), len(all_sets) - 1 - int(index)))
    return ([goal_set for index, goal_set in enumerate(all_sets) if index not in held_out_indices],
            [goal_set for index, goal_set in enumerate(all_sets) if index in held_out_indices])


def write_goal_set_split(path, goals):
    train_sets, held_out_sets = goal_set_split(goals)
    manifest = {
        "scheme": "goal_set_complement_pairs_80_20",
        "target_train_fraction": .8,
        "actual_train_fraction": len(train_sets) / (len(train_sets) + len(held_out_sets)),
        "goals": [list(goal) for goal in goals],
        "train_goal_sets": [[list(goal) for goal in goal_set] for goal_set in train_sets],
        "held_out_goal_sets": [[list(goal) for goal in goal_set] for goal_set in held_out_sets],
    }
    Path(path).with_suffix(".goal_set_split.json").write_text(json.dumps(manifest, indent=2))
    return train_sets, held_out_sets


def evaluation_goal_sets(goals, query, member, args, rng, limit, split="eval"):
    """Return sets from the requested train or held-out evaluation split."""
    if args.evaluate_on_held_out:
        train_sets, held_out_sets = goal_set_split(goals)
        source_sets = train_sets if split == "train" else held_out_sets
        return [goal_set for goal_set in source_sets if (query in goal_set) == member]
    return [sample_set(goals, query, member, rng) for _ in range(limit)]


if __name__ == "__main__":
    main()
