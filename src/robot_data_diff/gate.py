"""--fail-on: turn a DiffResult into a CI pass/fail decision.

A spec is a comma-separated list of rules. Each rule is a name, optionally
followed by `>N`, `>=N` or (for episode counts) `>N%` of A's episodes:

    schema, fps, robot_type, codebase_version, dataset, tasks, stale
        fail if that kind of change is present (no threshold)
    added, removed, changed, moved
        fail if the count is > 0, or exceeds the threshold
    idle-increase, spikes-increase
        fail if the mean idle ratio / total spike count rose by more than
        the threshold (default 0)
"""
from __future__ import annotations

import operator
import re
from dataclasses import dataclass

from .differ import DiffResult

FLAG_RULES = ("schema", "fps", "robot_type", "codebase_version", "dataset", "tasks",
              "stale")
COUNT_RULES = ("added", "removed", "changed", "moved")
QUALITY_RULES = ("idle-increase", "spikes-increase")

_RULE = re.compile(r"^([a-z_-]+)\s*(?:(>=|>)\s*(\d+(?:\.\d+)?|\.\d+)\s*(%)?)?$")
_OPS = {">": operator.gt, ">=": operator.ge}


@dataclass(frozen=True)
class Rule:
    name: str
    op: str = ">"
    threshold: float = 0.0
    percent: bool = False
    text: str = ""


def parse_fail_on(spec: str) -> list[Rule]:
    """Parse 'schema,fps,changed>5%' into rules; ValueError on a bad spec."""
    rules = []
    for raw in spec.split(","):
        text = raw.strip()
        m = _RULE.match(text)
        if not m:
            raise ValueError(f"bad --fail-on rule {text!r} "
                             "(expected e.g. 'schema', 'removed', 'changed>5%')")
        name, op, num, pct = m.groups()
        if name not in FLAG_RULES + COUNT_RULES + QUALITY_RULES:
            known = ", ".join(FLAG_RULES + COUNT_RULES + QUALITY_RULES)
            raise ValueError(f"unknown --fail-on rule {name!r} (known: {known})")
        if name in FLAG_RULES and op:
            raise ValueError(f"--fail-on rule {name!r} takes no threshold")
        if pct and name not in COUNT_RULES:
            raise ValueError(f"--fail-on rule {name!r} can't use a percentage")
        rules.append(Rule(name, op or ">", float(num) if num else 0.0, bool(pct), text))
    return rules


def _flag(r: DiffResult, name: str) -> str | None:
    changes = {n: (va, vb) for n, va, vb in r.dataset_changes}
    if name == "schema":
        n = len(r.features_added) + len(r.features_removed) + len(r.features_changed)
        return f"{n} feature(s) added/removed/changed" if n else None
    if name in ("fps", "robot_type", "codebase_version"):
        if name in changes:
            va, vb = changes[name]
            return f"{va} -> {vb}"
        return None
    if name == "dataset":
        return ", ".join(f"{n} {va} -> {vb}" for n, (va, vb) in changes.items()) or None
    if name == "tasks":
        diff = [t for t, (ca, cb) in r.task_counts.items() if ca != cb]
        return f"episode counts changed for {len(diff)} task(s)" if diff else None
    if name == "stale":
        if not (r.a_stats_verified or r.b_stats_verified):
            raise ValueError("--fail-on stale needs --verify-stats")
        n = len(r.a_stats_mismatches) + len(r.b_stats_mismatches)
        return f"{n} metadata stat(s) disagree with data/" if n else None
    raise AssertionError(name)


def evaluate(rules: list[Rule], r: DiffResult) -> list[str]:
    """Messages for the rules that fail; empty list means the gate passes."""
    hits = []
    for rule in rules:
        if rule.name in FLAG_RULES:
            msg = _flag(r, rule.name)
            if msg:
                hits.append(f"{rule.text}: {msg}")
            continue
        if rule.name in COUNT_RULES:
            count = len(getattr(r, rule.name))
            if rule.percent:
                value = 100.0 * count / r.a_episodes if r.a_episodes else 0.0
                detail = (f"{count} {rule.name} = {value:.1f}% of "
                          f"{r.a_episodes} episodes in A")
            else:
                value, detail = count, f"{count} {rule.name}"
        else:
            if r.mean_idle_a is None or r.mean_idle_b is None:
                raise ValueError(f"--fail-on {rule.name} needs quality stats "
                                 "(drop --no-quality)")
            if rule.name == "idle-increase":
                value = r.mean_idle_b - r.mean_idle_a
                detail = (f"mean idle ratio {r.mean_idle_a:.4f} -> "
                          f"{r.mean_idle_b:.4f} ({value:+.4f})")
            else:
                value = r.spikes_b - r.spikes_a
                detail = f"total spikes {r.spikes_a} -> {r.spikes_b} ({value:+d})"
        if _OPS[rule.op](value, rule.threshold):
            hits.append(f"{rule.text}: {detail}")
    return hits
