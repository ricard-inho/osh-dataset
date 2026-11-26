"""Derived copies of the real recordings on the Hub (``<variant>/`` folders with the root layout)."""

import json
from contextlib import nullcontext

import pytest
from huggingface_hub.utils import filter_repo_objects

from oshd import OshDataset, hub

T = "trajectory_007"
MANIFEST = json.dumps({"trajectories": {T: {}}})
ORIGINAL = ["README.md", "manifest.json", "splits.json", "models/objects.yaml", "calibration/rig.yaml",
            f"calibration/{T}.yaml", f"firefly/{T}/a.parquet", f"events/{T}/a.parquet"]
ISOLATED = ["isolated/manifest.json", "isolated/splits.json", "isolated/models/objects.yaml",
            "isolated/calibration/rig.yaml", f"isolated/calibration/{T}.yaml", "isolated/isolate/config.json",
            "isolated/isolate/stats/a.json", f"isolated/firefly/{T}/a.parquet",
            f"isolated/firefly_semantic/{T}/a.parquet", f"isolated/events/{T}/a.parquet"]
COREGISTERED = ["coregistered/manifest.json", f"coregistered/calibration/{T}.yaml",
                f"coregistered/coregistered/{T}/a.parquet", "coregistered/isolate/coregistered/manifest.json"]
BOTH = [f"isolated_coregistered/coregistered/{T}/a.parquet", "isolated_coregistered/manifest.json"]
#: the simulator's replay of the real recordings, its run records under ``twin/``
TWIN = ["twin_replay/manifest.json", "twin_replay/calibration/rig.yaml", f"twin_replay/calibration/{T}.yaml",
        f"twin_replay/events/{T}/a.parquet", f"twin_replay/firefly/{T}/a.parquet", "twin_replay/twin/counts/a.json"]
#: a trajectory folder the variants' manifests do not list: never fetched
UNLISTED = ["isolated/firefly/trajectory_008/a.parquet", "isolated/calibration/trajectory_008.yaml",
            "coregistered/coregistered/trajectory_008/a.parquet"]
REPO = ORIGINAL + ISOLATED + COREGISTERED + BOTH + TWIN + UNLISTED


class Fetched(set):
    revisions: set


@pytest.fixture
def fetched(monkeypatch, tmp_path):
    """Files a download would fetch from a repo holding REPO (huggingface_hub's own matching)."""
    import huggingface_hub

    got, revisions = Fetched(), set()

    def fetch(path):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(MANIFEST if path.endswith("manifest.json") else "{}")
        got.add(path)

    def fake_snapshot(repo_id, allow_patterns=None, ignore_patterns=None, revision=None, **kw):
        revisions.add(revision)
        for path in filter_repo_objects(REPO, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns):
            fetch(path)

    def fake_file(repo_id, filename, **kw):
        if filename not in REPO:
            raise huggingface_hub.errors.EntryNotFoundError(filename)
        fetch(filename)
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_file)
    got.revisions = revisions
    return got


@pytest.mark.parametrize("kw", [{}, {"trajectories": [T]}, {"modalities": ["firefly"]},
                                {"trajectories": [T], "modalities": ["firefly"]}])
def test_the_original_recordings_download_without_the_derived_copies(fetched, tmp_path, kw):
    assert hub.download(root=tmp_path, **kw) == tmp_path
    assert fetched and fetched <= set(ORIGINAL)


def test_a_variant_downloads_its_own_folder(fetched, tmp_path):
    root = hub.download(root=tmp_path, variant="isolated", trajectories=[T], modalities=["firefly"])
    assert root == tmp_path / "isolated"
    assert fetched == set(ISOLATED) - {f"isolated/firefly_semantic/{T}/a.parquet", f"isolated/events/{T}/a.parquet"}


def test_a_variant_without_filters_downloads_all_of_it(fetched, tmp_path):
    hub.download(root=tmp_path, variant="coregistered")
    assert fetched == set(COREGISTERED)


def test_the_twin_replay_downloads_with_its_run_records(fetched, tmp_path):
    root = hub.download(root=tmp_path, variant="twin_replay")
    assert root == tmp_path / "twin_replay" and fetched == set(TWIN)
    fetched.clear()
    hub.download(root=tmp_path, variant="twin_replay", modalities=["events"])
    assert fetched == set(TWIN) - {f"twin_replay/firefly/{T}/a.parquet"}


def test_a_variant_split_selects_from_its_own_manifest(fetched, tmp_path, monkeypatch):
    monkeypatch.setattr(OshDataset, "split", lambda self, *a, **k: [T] if self.root == tmp_path / "isolated" else [])
    hub.download(root=tmp_path, variant="isolated", split="test", modalities=["events"])
    assert f"isolated/events/{T}/a.parquet" in fetched and fetched <= set(ISOLATED)


def test_the_dataset_opens_the_variant(fetched, tmp_path):
    ds = OshDataset.download(root=tmp_path, variant="isolated_coregistered")
    assert ds.root == tmp_path / "isolated_coregistered" and ds.trajectory_names == [T]


@pytest.mark.parametrize("kw, match", [({"variant": "filtered"}, "unknown variant"),
                                       ({"variant": "isolated", "domain": "sim_3d"}, "real"),
                                       ({"variant": "isolated", "benchmark": "v1"}, "benchmark")])
def test_bad_variant_requests_fail_before_any_download(fetched, tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        hub.download(root=tmp_path, **kw)
    assert not fetched


def test_the_cli_downloads_a_variant(fetched, tmp_path, capsys):
    from oshd.cli import main

    main(["--root", str(tmp_path), "download", "--variant", "isolated", "-t", T, "-m", "events"])
    assert f"isolated/events/{T}/a.parquet" in fetched and fetched <= set(ISOLATED)
    assert f"dataset at {tmp_path / 'isolated'}" in capsys.readouterr().out


@pytest.mark.parametrize("variant", [None, "isolated"])
def test_an_empty_selection_fails_before_the_data_download(fetched, tmp_path, monkeypatch, variant):
    monkeypatch.setattr(OshDataset, "split", lambda self, *a, **k: [])
    with pytest.raises(ValueError, match="no trajectory"):
        hub.download(root=tmp_path, variant=variant, split="test")
    assert not any(p.endswith(".parquet") for p in fetched)


@pytest.mark.parametrize("kw", [{"domain": "sim_zerog", "task": "docking"}, {"benchmark": "v1"}, {"domain": "twin"},
                                {"split": "test"}])
def test_a_split_task_benchmark_or_twin_request_selects_the_trajectories(fetched, tmp_path, monkeypatch, kw):
    asked = []
    monkeypatch.setattr(OshDataset, "split", lambda self, *a, **k: asked.append((a, k)) or [T])
    with pytest.warns(UserWarning, match="no target CAD"):
        hub.download(root=tmp_path, modalities=["events"], **kw)
    (args, _), = asked
    assert args[0] == kw.get("split", "all") and f"events/{T}/a.parquet" in fetched


@pytest.mark.parametrize("variant", [None, "isolated"])
def test_explicit_trajectories_win_over_a_split(fetched, tmp_path, monkeypatch, variant):
    monkeypatch.setattr(OshDataset, "split", lambda self, *a, **k: ["trajectory_008"])
    with pytest.warns(UserWarning) if variant is None else nullcontext():
        hub.download(root=tmp_path, variant=variant, trajectories=[T], split="test", modalities=["firefly"])
    assert f"{variant + '/' if variant else ''}firefly/{T}/a.parquet" in fetched
    assert not any("trajectory_008" in p for p in fetched)


def test_a_benchmark_holding_the_variant_pins_its_revision_and_split(fetched, tmp_path, monkeypatch):
    from oshd.splits import load_benchmark

    asked = []
    monkeypatch.setattr(OshDataset, "split", lambda self, *a, **k: asked.append((a, k)) or [T])
    root = hub.download(root=tmp_path, variant="isolated", benchmark="v4", split="test", modalities=["events"])
    assert root == tmp_path / "isolated" and f"isolated/events/{T}/a.parquet" in fetched
    assert fetched.revisions == {load_benchmark("v4")["revision"]}
    (args, kw), = asked
    assert args[0] == "test" and kw["benchmark"] == "v4"
    with pytest.raises(ValueError, match="pinned"):
        hub.download(root=tmp_path, variant="isolated", benchmark="v4", revision="main")
