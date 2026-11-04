# JSON output (`schema_version: 1`)

`robot-data-diff diff A B --format json` prints, and `--json OUT` writes, one
JSON object. From Python, `DiffResult.to_dict()` returns the same structure.

**Stability:** `schema_version` is bumped only when an existing key is
renamed or removed, or its meaning changes. New keys may be added in any
release, so ignore keys you don't recognise. Values that are not finite
numbers (NaN, inf) are written as `null`.

```jsonc
{
  "schema_version": 1,
  "tool": {"name": "robot-data-diff", "version": "0.2.0"},
  "checkpoint": "ckpt-42",          // --checkpoint, or null
  "match_by": "index",              // "index" | "content"

  "a": {                             // same shape for "b"
    "name": "lerobot/pusht",         // the ref as given on the command line
    "codebase_version": "v3.0",
    "fps": 10,
    "robot_type": "unknown",
    "episodes": 206,                 // after --episodes / --task filters
    "frames": 25650,                 // sum of episode lengths
    "mean_idle_ratio": 0.0309,       // null with --no-quality
    "total_spikes": 25,              // null with --no-quality
    "stats_verified": false,         // true when --verify-stats ran
    "stats_mismatches": [            // only with --verify-stats
      {"episode": 3, "feature": "action", "stat": "max",
       "metadata": [1.0, 2.0], "data": [1.5, 2.0]},
      {"episode": 7, "feature": null, "stat": "length",
       "metadata": 120, "data": 118}
    ]
  },
  "b": { ... },

  "summary": {"added": 1, "removed": 6, "changed": 5, "moved": 0, "unchanged": 195},

  "episodes": {
    "added":   [206],                // episode indices in B
    "removed": [200, 201],           // episode indices in A
    "moved":   [{"from": 12, "to": 40}],   // only with match_by "content"
    "changed": [
      {
        "index": 0,                  // index in B
        "old_index": 0,              // index in A (differs only when moved)
        "old_length": 161, "new_length": 161, "d_length": 0,
        "tasks_changed": false,
        "old_tasks": ["Push the T-shaped block onto the T-shaped target."],
        "new_tasks": ["Push the T-shaped block onto the T-shaped target."],
        // feature -> stat -> per-dimension (B - A). Only stats that changed
        // are present. null = can't subtract (shape changed / missing).
        "feature_deltas": {
          "action": {"min": [46.5, 35.5], "max": [187.5, 224.5],
                     "mean": [104.5031, 144.1553], "std": [20.1, 31.7]}
        },
        "d_idle_ratio": 0.0,         // null with --no-quality
        "d_spike_count": 0
      }
    ]
  },

  "tasks": [{"task": "Push the T-shaped block ...", "a": 206, "b": 201}],

  "dataset_changes": [{"field": "fps", "a": 10, "b": 30}],
                                     // fields: fps, robot_type, codebase_version

  "schema": {
    "added":   ["observation.images.wrist"],
    "removed": [],
    "changed": [{"feature": "observation.state",
                 "a": {"dtype": "float32", "shape": [2]},
                 "b": {"dtype": "float32", "shape": [7]}}]
  },

  "fail_on": {                       // only when --fail-on is given
    "rules": ["schema", "changed>5%"],
    "failed": ["changed>5%: 12 changed = 6.0% of 200 episodes in A"]
  }
}
```
