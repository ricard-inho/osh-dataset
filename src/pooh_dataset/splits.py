"""Train / val / test splits at trajectory granularity.

The source of truth is ``splits.json`` at the dataset root::

    {"version": 3,
     "train": ["trajectory_010", "trajectory_hera_000001", ...],
     "val":   [],
     "test":  ["trajectory_014", "trajectory_hera_000007", ...],
     "summary": {"real/antena": {"train": 24, "test": 6}, "sim_3d/approach": {"train": 400, "test": 100}, ...}}

The real part is written by :func:`update_real_splits` (``pooh splits update``): the real
trajectories with a target model (a mockup of ``models/real/mockups.json`` or an object of
``models/objects.yaml``) are grouped by lighting *setting* (target, light colour temperature
and intensity), and whole settings go to ``test`` so that no test lighting is seen in
training, about ``eval_fraction`` of each target's trajectories, spread over the colour
temperatures. Also append-only. Real trajectories without a target model are in no split.

Before version 3, real trajectories that ``splits.json`` did not list were assigned by a stable
hash of their name; :func:`load_splits` still does that for such files. Simulated
trajectories it does not list are in no split until :func:`update_sim_splits` adds them.

The simulated part is written by :func:`update_sim_splits` (``pooh splits update``, also run
by the sim publisher): append-only, so a trajectory never changes split. Its unit is the
recording *run* (one simulator process, a few episodes of the same scenario with the same
seed): all episodes of a run share a split, so no near-duplicate of an eval episode is
trained on. Runs are balanced per scenario config (``zerog_cubesat_arm_wall_docking``,
``orbital_iss_earth_orbit``, ...), so every group, task, target and background gets eval
runs: a new run goes to ``test`` when that keeps the config's share of test runs closest
to ``eval_fraction``.

Frozen benchmarks (``benchmarks/<version>.json`` in this package, e.g. ``benchmark_v1``) pin a
Hub revision and its train / test trajectories for papers: :func:`load_benchmark`,
``ds.split(..., benchmark="v1")``, ``download(benchmark="v1")``. They never change; the Hub's
``splits.json`` carries a copy of each (``benchmark_v1``, ...), written by
:func:`update_sim_splits`.

This module only uses the standard library: the sim publisher loads it by path.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Iterable
from pathlib import Path

__all__ = ["BENCHMARKS", "EVAL_FRACTION", "SPLIT_NAMES", "hash_split", "light_setting", "load_benchmark",
           "load_splits", "real_targets", "sim_run", "update_real_splits", "update_sim_splits"]

SPLIT_NAMES = ("train", "val", "test")
#: share of the recording runs of each simulated scenario config that go to ``test``
EVAL_FRACTION = 0.2
VERSION = 3

BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmarks"
#: frozen benchmark versions shipped with the package
BENCHMARKS = tuple(sorted(p.stem for p in BENCHMARK_DIR.glob("v*.json")))

# <config>_<scenario index>_<launch time>_<process index>, e.g.
# zerog_cubesat_arm_wall_docking_00012_20261017T101530_002
_RUN_TAG = re.compile(r"^(?P<config>.+)_\d{5}_(?P<time>\d{8}T\d{6})_\d{3}$")


def hash_split(name: str, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> str:
    h = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < ratios[0]:
        return "train"
    return "val" if h < ratios[0] + ratios[1] else "test"


def sim_run(source_dir: str) -> dict:
    """Recording run of a simulated trajectory, from its raw directory name
    (``<run>_env<k>_ep<j>``): ``run``, ``config`` (scenario config without the index),
    ``task`` (its last word: approach / orbit in sim_3d, docking / inspection in sim_zerog)
    and ``time`` (launch time, sorts runs). Unknown names are a run of their own, task None."""
    run = source_dir.rsplit("_env", 1)[0]
    m = _RUN_TAG.match(run)
    if m is None:
        return {"run": run, "config": run, "task": None, "time": ""}
    config = m["config"]
    return {"run": run, "config": config, "task": config.rsplit("_", 1)[-1], "time": m["time"]}


def load_benchmark(name: str) -> dict:
    """Frozen benchmark ``name`` ("v1" or "benchmark_v1"): ``repo_id``, ``revision``, ``train`` /
    ``val`` / ``test`` trajectory lists, ``summary``."""
    version = name.removeprefix("benchmark_")
    path = BENCHMARK_DIR / f"{version}.json"
    if not path.exists():
        raise ValueError(f"unknown benchmark {name!r}; have {list(BENCHMARKS)}")
    return json.loads(path.read_text())


def load_splits(root: Path, trajectories: list[str], unlisted: Iterable[str] = ()) -> dict[str, list[str]]:
    """Splits of ``trajectories`` from ``splits.json``; before version 3 (or without the file)
    those of ``unlisted`` (the real trajectories) that it does not list get :func:`hash_split`."""
    path = Path(root) / "splits.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    if data.get("version", 0) >= 3:
        unlisted = ()
    known = set(trajectories)
    splits = {k: [t for t in data.get(k, []) if t in known] for k in SPLIT_NAMES}
    splits.update({k: v for k, v in data.items() if k not in SPLIT_NAMES and isinstance(v, list)})
    listed = {t for v in splits.values() for t in v}
    hashed = sorted(t for t in unlisted if t in known and t not in listed)
    if hashed:
        warnings.warn(
            f"{len(hashed)} real trajectories have no official split in splits.json; using a hash-based "
            "split for them. Results are only comparable across people with the same dataset revision.",
            stacklevel=3,
        )
    for t in hashed:
        splits[hash_split(t)].append(t)
    return splits


RULE = ("simulated: per recording run, the eval fraction of the runs of each scenario config in test; real: per "
        "lighting setting (target, colour temperature, intensity), the eval fraction of each target's trajectories "
        "in test; real trajectories without a target model are in no split. Append-only (pooh_dataset.splits)")


def light_setting(light: dict | None) -> tuple:
    """(colour temperature K, intensity) of a real trajectory's ``light`` entry in manifest.json."""
    light = light or {}
    return light.get("color_temperature_k"), light.get("intensity")


def real_targets(objects_yaml: dict | None, mockups: dict | None) -> dict[str, str]:
    """Real trajectory -> its target model, from ``models/objects.yaml`` (objects that list their
    ``trajectories``) and ``models/real/mockups.json``."""
    out = {}
    for name, d in (objects_yaml or {}).items():
        for t in (d or {}).get("trajectories") or []:
            out[str(t)] = name
    out.update({t: e["model"] for t, e in (mockups or {}).items()})
    return out


def update_real_splits(splits: dict | None, targets: dict[str, str], light: dict[str, dict],
                       eval_fraction: float = EVAL_FRACTION) -> dict:
    """``splits.json`` content with the real trajectories of ``targets`` (trajectory -> target
    model, see :func:`real_targets`) that it does not list yet added, per lighting setting
    (``light``: manifest.json's trajectory -> light entry). Listed trajectories and those of the
    frozen benchmarks never move; a setting that already has a split keeps it. Returns a new dict."""
    old = dict(splits or {})
    assigned = {t: k for k in SPLIT_NAMES for t in old.get(k, [])}
    for version in BENCHMARKS:
        b = load_benchmark(version)
        for k in SPLIT_NAMES:
            for t in b[k]:
                if t in targets:
                    assigned.setdefault(t, k)
    groups: dict[tuple, list[str]] = {}
    for t in sorted(targets):
        groups.setdefault((targets[t],) + light_setting(light.get(t)), []).append(t)

    lists = {k: set(old.get(k, [])) for k in SPLIT_NAMES}
    count: dict[str, dict[str, int]] = {}
    new = []
    for key, ts in groups.items():
        cnt = count.setdefault(key[0], {"train": 0, "test": 0})
        have = [assigned[t] for t in ts if t in assigned]
        for t in ts:
            if t in assigned:
                lists[assigned[t]].add(t)
                cnt[assigned[t]] = cnt.get(assigned[t], 0) + 1
        if len(have) < len(ts):
            new.append((key, [t for t in ts if t not in assigned], have[0] if have else None))
    # New settings in order of colour temperature, then intensity: the test settings spread over
    # the range. A setting goes to test when that keeps its target's test share closest to
    # eval_fraction.
    for key, ts, fixed in sorted(new, key=lambda x: (x[0][0], x[0][1] is None, x[0][1] or 0, x[0][2] is None,
                                                     x[0][2] or 0, x[1][0])):
        cnt = count[key[0]]
        if fixed is None:
            n = cnt["train"] + cnt["test"] + len(ts)
            to_test = abs(cnt["test"] + len(ts) - eval_fraction * n) < abs(cnt["test"] - eval_fraction * n)
            fixed = "test" if to_test else "train"
        for t in ts:
            lists[fixed].add(t)
        cnt[fixed] = cnt.get(fixed, 0) + len(ts)

    out = dict(old)
    for k in SPLIT_NAMES:
        out[k] = sorted(lists[k])
    summary = {k: v for k, v in old.get("summary", {}).items() if not k.startswith("real/")}
    for t, target in targets.items():
        for k in SPLIT_NAMES:
            if t in lists[k]:
                cell = summary.setdefault(f"real/{target}", {s: 0 for s in SPLIT_NAMES})
                cell[k] += 1
    out.update({"version": VERSION, "rule": RULE, "eval_fraction": eval_fraction,
                "summary": dict(sorted(summary.items()))})
    return out


def update_sim_splits(splits: dict | None, sim_manifest: dict, eval_fraction: float = EVAL_FRACTION) -> dict:
    """``splits.json`` content with the simulated trajectories of ``sim_manifest`` that it
    does not list yet added (see the module docstring). Listed trajectories, real or
    simulated, never move, and neither do the simulated trajectories of the frozen benchmarks.
    Returns a new dict; ``splits`` may be None (no file yet)."""
    old = dict(splits or {})
    assigned = {t: k for k in SPLIT_NAMES for t in old.get(k, [])}
    entries = sim_manifest.get("trajectories", {})
    for version in BENCHMARKS:  # frozen benchmarks fix the split of their simulated trajectories
        b = load_benchmark(version)
        for k in SPLIT_NAMES:
            for t in b[k]:
                if t in entries:
                    assigned.setdefault(t, k)
    runs: dict[str, dict] = {}
    for t, e in entries.items():
        r = sim_run(e.get("source_dir", t))
        run = runs.setdefault(r["run"], {**r, "group": e.get("group"), "trajectories": []})
        run["trajectories"].append(t)

    # Splits of the runs already assigned (a run is never split; the first listed episode
    # decides for a run whose episodes are new), counted per scenario config.
    run_split: dict[str, str] = {}
    per_config: dict[tuple, dict[str, int]] = {}
    new_runs = []
    for name, run in runs.items():
        key = (run["group"], run["config"])
        cnt = per_config.setdefault(key, {"train": 0, "test": 0})
        have = [assigned[t] for t in sorted(run["trajectories"]) if t in assigned]
        if have:
            run_split[name] = have[0]
            if have[0] in cnt:
                cnt[have[0]] += 1
        else:
            new_runs.append(name)

    for name in sorted(new_runs, key=lambda n: (runs[n]["time"], n)):
        cnt = per_config[(runs[name]["group"], runs[name]["config"])]
        n = cnt["train"] + cnt["test"] + 1
        split = "test" if int(eval_fraction * n + 0.5) > cnt["test"] else "train"
        cnt[split] += 1
        run_split[name] = split

    out = {k: v for k, v in old.items() if k not in SPLIT_NAMES and k not in ("version", "summary")}
    lists = {k: set(old.get(k, [])) for k in SPLIT_NAMES}
    for name, run in runs.items():
        for t in run["trajectories"]:
            if t in assigned:
                lists[assigned[t]].add(t)  # listed, or fixed by a benchmark
            else:
                lists[run_split[name]].add(t)
    summary: dict[str, dict[str, int]] = {}
    for k in SPLIT_NAMES:
        for t in lists[k]:
            if t in entries:
                run = runs[sim_run(entries[t].get("source_dir", t))["run"]]
                cell = summary.setdefault(f"{run['group']}/{run['task']}", {s: 0 for s in SPLIT_NAMES})
                cell[k] += 1
    summary.update({k: v for k, v in old.get("summary", {}).items() if k.startswith("real/")})
    out.update({
        "version": VERSION,
        "rule": RULE,
        "eval_fraction": eval_fraction,
        **{k: sorted(lists[k]) for k in SPLIT_NAMES},
        "summary": dict(sorted(summary.items())),
    })
    for version in BENCHMARKS:  # frozen copies, never edited here
        b = load_benchmark(version)
        out[b["name"]] = b
    return out
