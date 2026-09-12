# Goal-slice GCB experiment

`goal_slice_gcb_four_rooms.py` is separate from the existing GCB code and
four-rooms scripts. It reuses GCB's 64px `PixelEncoder`, paired reward decoder,
AdamW settings, and `next_observation` bisimulation target.

The input is `[I_s, I_g, I_G+]` concatenated in channels (nine RGB channels).
No membership bit, set ID, or symbolic positions reach the network. The requested
reward values are configurable with `--r-universal`, `--r-empty`, and `--r-min`.

Run locally from `GCB/extension` with `PYTHONPATH` set to `GCB`, or submit:

```bash
sbatch scripts/goal_slice_gcb_four_rooms.sh
```

Each run writes JSONL metrics and a checkpoint under `goal_slice_gcb_runs/`.
Evidence for the proposed collapse is low `collapse/within_status_distance` and
larger `collapse/between_status_distance` (a separation ratio above one).

A nonzero `r_min` makes a non-query member behaviorally relevant when an action
reaches it. Thus this experiment tests the proposed collapse rather than
supervising it. Use `--r-min 0` as the direct irrelevance ablation.
