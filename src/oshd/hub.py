"""Selective download of the dataset from the Hugging Face Hub into a local folder.

Hub layout (one repo for the real and the simulated recordings)::

    <modality>/<trajectory>/*.parquet               real: events, firefly, realsense_color, ...
    sim_<group>_<modality>/<trajectory>/*.parquet   simulated, group sim_3d or sim_zerog
    manifest.json, sim_manifest.json                trajectories of each domain
    calibration/<trajectory>.yaml, calibration/rig.yaml
    metadata/<trajectory>.yaml                      simulated only (scenario, rates, ...)
    models/                                         object models (real)
    models/sim/targets.json, models/sim/<model>/    simulated targets: trajectory -> CAD + scale
    <variant>/                                      a derived copy of the real recordings in the
                                                    layout above (manifest.json, calibration/,
                                                    models/, <modality>/<trajectory>/) plus isolate/
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "DEFAULT_REPO_ID",
    "DOMAINS",
    "META_PATTERNS",
    "REAL_MODALITIES",
    "SIM_GROUPS",
    "SIM_MODALITIES",
    "VARIANTS",
    "default_root",
    "download",
    "is_twin",
    "remote_manifest",
    "remote_sim_manifest",
]

DEFAULT_REPO_ID = os.environ.get("OSHD_REPO_ID", "OSHDataset/oshd")

REAL_MODALITIES = ("events", "firefly", "realsense_color", "realsense_depth", "lidar", "imu", "mocap",
                   "robot_messages")
#: simulated recordings: Hub folder ``<group>_<modality>``
SIM_GROUPS = ("sim_3d", "sim_zerog")
SIM_MODALITIES = ("rgb", "depth", "semantic", "events", "lidar", "mocap", "imu", "joint_states", "steps")
#: values of ``domain``: a domain, both, one simulation group, or ``twin``: the zero-G lab twins of
#: the real mockup recordings (sim_zerog trajectories whose target is a real mockup, ``real_<mockup>``)
DOMAINS = ("real", "synthetic", "all") + SIM_GROUPS + ("twin",)

#: derived copies of the real recordings, each in its own Hub folder: without the lab (``isolated``),
#: the original / isolated recordings re-expressed in one reference view (``coregistered`` /
#: ``isolated_coregistered``, modality ``coregistered``), all from oshd-isolate, and the simulator's
#: replay of the real recordings along their tracked trajectories (``twin_replay``)
VARIANTS = ("isolated", "coregistered", "isolated_coregistered", "twin_replay")
#: a variant's own metadata: corrections, configuration, statistics (``isolate/``), run records (``twin/``)
VARIANT_META = ["isolate/*", "twin/*"]

MANIFEST = "manifest.json"
SIM_MANIFEST = "sim_manifest.json"
#: small files always fetched: manifests, dataset card, splits, object models
META_PATTERNS = ["README.md", MANIFEST, SIM_MANIFEST, "splits.json", "models/*"]
#: the simulated targets' CAD is fetched per trajectory (download_sim_targets), not with the metadata
META_IGNORE = ["models/sim/*"]
SIM_TARGETS = "models/sim/targets.json"
#: never fetched with the recordings: Hub patterns match across folders (``*/<trajectory>/*``)
IGNORE = META_IGNORE + [f"{v}/*" for v in VARIANTS]


def default_root(repo_id: str = DEFAULT_REPO_ID) -> Path:
    """``$OSHD_DATA_DIR/<repo>`` or ``~/.cache/oshd/<repo>``."""
    base = os.environ.get("OSHD_DATA_DIR")
    base_path = Path(base) if base else Path.home() / ".cache" / "oshd"
    return base_path / repo_id.replace("/", "--")


def _as_list(x: str | Iterable[str] | None) -> list[str] | None:
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def domain_parts(domain: str) -> tuple[bool, list[str]]:
    """``domain`` -> (includes the real data, simulation groups it includes)."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; use one of {DOMAINS}")
    real = domain in ("real", "all")
    if domain == "twin":
        return False, ["sim_zerog"]
    groups = list(SIM_GROUPS) if domain in ("synthetic", "all") else [g for g in SIM_GROUPS if g == domain]
    return real, groups


def is_twin(info: dict) -> bool:
    """Whether a simulated trajectory (its manifest entry) is the lab twin of a real mockup
    recording: scenario ``zerog/real_<mockup>/static/<task>``."""
    parts = str(info.get("scenario") or "").split("/")
    return info.get("group") == "sim_zerog" and len(parts) > 1 and parts[1].startswith("real_")


def remote_manifest(repo_id: str = DEFAULT_REPO_ID, revision: str | None = None) -> dict:
    """``manifest.json`` of the real recordings."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, MANIFEST, repo_type="dataset", revision=revision)
    return json.loads(Path(path).read_text())


def remote_sim_manifest(repo_id: str = DEFAULT_REPO_ID, revision: str | None = None) -> dict:
    """``sim_manifest.json``: simulated trajectory -> group, scenario, counts, goal errors."""
    from huggingface_hub import hf_hub_download

    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # huggingface_hub < 0.24
        from huggingface_hub.utils import EntryNotFoundError
    try:
        path = hf_hub_download(repo_id, SIM_MANIFEST, repo_type="dataset", revision=revision)
    except EntryNotFoundError:
        return {"trajectories": {}}
    return json.loads(Path(path).read_text())


def download(
    repo_id: str = DEFAULT_REPO_ID,
    root: str | Path | None = None,
    trajectories: str | Iterable[str] | None = None,
    modalities: str | Iterable[str] | None = None,
    revision: str | None = None,
    max_workers: int = 8,
    domain: str = "real",
    split: str | None = None,
    task: str | None = None,
    benchmark: str | None = None,
    variant: str | None = None,
) -> Path:
    """Download (or resume / update) part of the dataset and return the local root.

    ``trajectories`` / ``modalities`` = None means all of them. Without ``trajectories``,
    ``domain`` picks the recordings: ``"real"`` (default), ``"synthetic"`` (all simulated),
    ``"sim_3d"`` / ``"sim_zerog"`` (one simulation group) or ``"all"``. Explicit
    ``trajectories`` are fetched whatever their domain. Modality names are the folder names
    of the domain: ``realsense_color``, ... for real data, ``rgb``, ``depth``, ``semantic``, ...
    for simulated data (see :data:`REAL_MODALITIES`, :data:`SIM_MODALITIES`).

    ``split`` ("train", "val", "test", from ``splits.json``) and ``task`` ("approach",
    "orbit", "docking", "inspection") narrow the trajectories of ``domain`` further, e.g.
    ``download(domain="sim_zerog", task="docking", split="test")``. ``benchmark`` ("v1") takes
    the split from that frozen benchmark and downloads at its Hub revision (``revision`` must
    then be None or the same).

    Already-downloaded files are skipped, so calling this again after new trajectories were
    added only fetches the new ones. Pin ``revision`` (a commit sha or tag) for reproducible
    experiments.

    ``variant`` (one of :data:`VARIANTS`) downloads that derived copy of the real recordings
    instead, with the same ``trajectories``, ``modalities``, ``split`` and ``revision``, into
    ``<root>/<variant>``, and returns that folder. Without it, the derived copies are never
    fetched.
    """
    from huggingface_hub import snapshot_download

    root = Path(root) if root is not None else default_root(repo_id)
    trajs, mods = _as_list(trajectories), _as_list(modalities)
    if variant is not None:
        return _download_variant(repo_id, root, variant, trajs, mods, revision, max_workers, domain, split, task,
                                 benchmark)
    real, groups = domain_parts(domain)
    revision, models_revision = _pin(benchmark, revision)
    if trajs is None and _selects(domain, split, task, benchmark):
        # Pick the trajectories from the manifests and splits.json, then fetch them by name.
        snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=root,
                          allow_patterns=META_PATTERNS, ignore_patterns=IGNORE, max_workers=max_workers)
        trajs = _select(root, repo_id, domain, split, task, benchmark)
    if trajs is not None:
        names, data, sim = trajs, _named_patterns(trajs, mods), True
    else:
        names, data, sim = _all_patterns(repo_id, root, revision, mods, real, groups)
    side = _side_patterns(names, every_sim=sim and trajs is None)
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=root,
        allow_patterns=META_PATTERNS + side + data,
        ignore_patterns=IGNORE,
        max_workers=max_workers,
    )
    if sim:
        download_sim_targets(repo_id, root, _sim_names(root, groups) if trajs is None else trajs, models_revision,
                             max_workers)
    return root


def _selects(domain: str, split: str | None, task: str | None, benchmark: str | None) -> bool:
    """Whether the trajectories come from the manifests and splits.json rather than whole domains."""
    return split is not None or task is not None or benchmark is not None or domain == "twin"


def _side_patterns(names: list[str], every_sim: bool) -> list[str]:
    """Calibration and metadata of ``names`` (``every_sim``: of every simulated trajectory too)."""
    side = ["calibration/rig.yaml", "calibration/clicks_*"]
    side += [f"{d}/{t}.yaml" for t in names for d in ("calibration", "metadata")]
    return side + (["calibration/*", "metadata/*"] if every_sim else [])


def _pin(benchmark: str | None, revision: str | None) -> tuple[str | None, str | None]:
    """Revisions of the data and of the simulated targets' models: a frozen benchmark pins both (its
    data revision predates the models)."""
    if benchmark is None:
        return revision, revision
    from .splits import load_benchmark

    b = load_benchmark(benchmark)
    if revision not in (None, b["revision"]):
        raise ValueError(f"{b['name']} is pinned to revision {b['revision']}, not {revision}")
    return b["revision"], b.get("models_revision", b["revision"])


def _select(root: Path, repo_id: str, domain: str, split: str | None, task: str | None,
            benchmark: str | None = None) -> list[str]:
    """The trajectories of a split / task, from the manifests and splits.json under ``root``."""
    from .dataset import OshDataset

    trajs = OshDataset(root, repo_id).split(split or "all", domain, task=task, benchmark=benchmark)
    if not trajs:
        raise ValueError(f"no trajectory in {root} with domain={domain!r} split={split!r} task={task!r} "
                         f"benchmark={benchmark!r}")
    return trajs


def _named_patterns(trajs: list[str], mods: list[str] | None) -> list[str]:
    """Data of named trajectories. Trajectory names are unique across domains: match both layouts."""
    if mods is None:
        return [f"*/{t}/*.parquet" for t in trajs]
    return [p for m in mods for t in trajs for p in (f"{m}/{t}/*.parquet", f"sim_*_{m}/{t}/*.parquet")]


def _all_patterns(repo_id: str, root: Path, revision: str | None, mods: list[str] | None, real: bool,
                  groups: list[str]) -> tuple[list[str], list[str], bool]:
    """(real trajectory names, data patterns, whether simulated data is included) of whole domains."""
    data = [f"{g}_{m}/*/*.parquet" for g in groups for m in (mods or ["*"])]
    if not real:
        return [], data, bool(groups)
    from huggingface_hub import hf_hub_download

    # real calibration by name: the sim calibration files are many
    manifest = hf_hub_download(repo_id, MANIFEST, repo_type="dataset", revision=revision, local_dir=root)
    names = list(json.loads(Path(manifest).read_text()).get("trajectories", {}))
    return names, [f"{m}/*/*.parquet" for m in (mods or REAL_MODALITIES)] + data, bool(groups)


def _sim_names(root: Path, groups: list[str]) -> list[str]:
    """Simulated trajectories of ``groups`` in the local ``sim_manifest.json``."""
    path = root / SIM_MANIFEST
    sim_trajs = json.loads(path.read_text()).get("trajectories", {}) if path.exists() else {}
    return [t for t, e in sim_trajs.items() if e.get("group") in groups]


def _check_variant(variant: str, domain: str, benchmark: str | None) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; use one of {VARIANTS}")
    if domain != "real":
        raise ValueError(f"variant {variant!r} holds real recordings only, not domain {domain!r}")
    if benchmark is not None:
        raise ValueError(f"variant {variant!r} is not part of a frozen benchmark; select with split=")


def _download_variant(repo_id: str, root: Path, variant: str, trajs: list[str] | None, mods: list[str] | None,
                      revision: str | None, max_workers: int, domain: str, split: str | None, task: str | None,
                      benchmark: str | None) -> Path:
    """:func:`download` of the derived copy ``variant``: its metadata, then the data asked for of the
    trajectories its manifest lists."""
    from huggingface_hub import snapshot_download

    _check_variant(variant, domain, benchmark)

    def fetch(patterns):
        snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=root,
                          allow_patterns=[f"{variant}/{p}" for p in patterns], max_workers=max_workers)

    sub = root / variant
    fetch(META_PATTERNS + VARIANT_META)
    if trajs is None and (split is not None or task is not None):
        trajs = _select(sub, repo_id, "real", split, task)
    if trajs is None:
        trajs = list(json.loads((sub / MANIFEST).read_text()).get("trajectories", {}))
    fetch(["calibration/rig.yaml", "calibration/clicks_*"]
          + [p for t in trajs for p in (f"calibration/{t}.yaml", *(f"{m}/{t}/*.parquet" for m in mods or ["*"]))])
    return sub


def download_sim_targets(repo_id: str, root: Path, trajectories: list[str], revision: str | None = None,
                         max_workers: int = 8) -> list[str]:
    """CAD of the simulated targets of ``trajectories`` (``models/sim/<model>/``) and
    ``models/sim/targets.json``; returns the models fetched. Real trajectories are ignored."""
    from huggingface_hub import hf_hub_download, snapshot_download

    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # huggingface_hub < 0.24
        from huggingface_hub.utils import EntryNotFoundError
    try:
        path = hf_hub_download(repo_id, SIM_TARGETS, repo_type="dataset", revision=revision, local_dir=root)
    except EntryNotFoundError:
        warnings.warn(f"{repo_id}@{revision or 'main'} has no {SIM_TARGETS}: no target CAD", stacklevel=2)
        return []
    targets = json.loads(Path(path).read_text())
    models = sorted({targets[t]["model"] for t in trajectories if t in targets})
    if models:
        snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=root,
                          allow_patterns=["models/sim/README.md"] + [f"models/sim/{m}/*" for m in models],
                          max_workers=max_workers)
    return models
