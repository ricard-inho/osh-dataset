"""Real mockups (models/real/), rig calibration relative to the RealSense colour camera, and the
real lighting-setting splits."""

import json

import numpy as np
import pytest
import yaml
from conftest import T0

from oshd.calibration import Calibration
from oshd.geometry import make_pose
from oshd.labels import object_pose
from oshd.objects import MOCKUP_BODY, ObjectModel, load_object_models
from oshd.splits import load_benchmark, load_splits, real_targets, update_real_splits

INTR = {"width": 64, "height": 48, "K": [50.0, 0, 32, 0, 50.0, 24, 0, 0, 1], "D": [0.0] * 5}


def test_extrinsics_from_realsense_relative_calibration():
    T_body_rs = make_pose(np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    T_rs_ff = {"translation_m": [0.0, 0.2, 0.0], "rotation_xyzw": [0.0, 0.0, 1.0, 0.0]}
    cal = Calibration.from_dict("t", {
        "extrinsics": {"reference_body": "SensorStack",
                       "cameras": {"realsense_color": {"T_body_cam": T_body_rs.ravel().tolist()},
                                   "events": {"T_body_cam": np.eye(4).ravel().tolist()}}},
        "realsense_color": {**INTR, "T_realsense_color_cam": {"translation_m": [0, 0, 0]}},
        "firefly": {**INTR, "T_realsense_color_cam": T_rs_ff},
        "events": {**INTR, "T_realsense_color_cam": T_rs_ff},
    })
    ff = cal.T_body_cam("firefly")
    assert np.allclose(ff[:3, 3], [0.1, 0.2, 0.0]) and np.allclose(ff[:3, :3], np.diag([-1, -1, 1]))
    assert np.allclose(cal.T_body_cam("realsense_color"), T_body_rs)
    assert np.allclose(cal.T_body_cam("events"), np.eye(4))  # an explicit T_body_cam wins


CUBE_OBJ = "\n".join(
    [f"v {x} {y} {z}" for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)]
    + ["f 1 2 4", "f 1 4 3", "f 5 7 8", "f 5 8 6", "f 1 5 6", "f 1 6 2", "f 3 4 8", "f 3 8 7",
       "f 1 3 7", "f 1 7 5", "f 2 6 8", "f 2 8 4"]) + "\n"


def write_models(models, mockups, objects=None):
    (models / "real").mkdir(parents=True)
    for name in {e["model"] for e in mockups.values()}:
        (models / "real" / name).mkdir()
        if name != "no_cad":
            (models / "real" / name / "model.obj").write_text(CUBE_OBJ)
    (models / "real" / "mockups.json").write_text(json.dumps(mockups))
    if objects is not None:
        (models / "objects.yaml").write_text(yaml.safe_dump(objects))


def test_mockups_are_objects(tmp_path):
    pytest.importorskip("trimesh")
    pose = {"frame_id": "world", "position": [1.0, 2.0, 0.5], "orientation_xyzw": [0, 0, 0, 1]}
    write_models(tmp_path, {"trajectory_010": {"model": "antena", "static_pose": pose},
                            "trajectory_011": {"model": "antena"},
                            "trajectory_040": {"model": "antena_1_solar"},
                            "trajectory_041": {"model": "no_cad"}},
                 {"cubesat": {"box": [0.1, 0.1, 0.1], "trajectories": ["trajectory_007"]}})
    with pytest.warns(UserWarning, match="no_cad"):
        objs = load_object_models(tmp_path)
    assert sorted(objs) == ["antena", "antena_1_solar", "cubesat"]
    a = objs["antena"]
    assert (a.class_id, a.mocap_body, a.trajectories) == (2, MOCKUP_BODY, ["trajectory_010", "trajectory_011"])
    assert objs["antena_1_solar"].class_id == 3
    assert np.allclose(a.static_poses["trajectory_010"][:3, 3], [1.0, 2.0, 0.5])
    assert a.present_in("trajectory_011") and not a.present_in("trajectory_040")


def test_static_pose_fallback(ds):
    traj = ds["trajectory_a"]  # mocap: SensorStack and cubesat at (0, 0, 2), no "target"
    static = make_pose(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    untracked = ObjectModel.from_box("m", [0.1] * 3, mocap_body=MOCKUP_BODY)
    untracked.static_poses[traj.name] = static
    T, valid = object_pose(traj, untracked, np.array([T0, T0 + 1]))
    assert valid.all() and np.allclose(T[:, 2, 3], 3.0)
    tracked = ObjectModel.from_box("c", [0.1] * 3, mocap_body="cubesat")
    tracked.static_poses[traj.name] = static
    T, valid = object_pose(traj, tracked, np.array([T0, T0 + 10**12]))  # the second: no mocap
    assert valid.all() and np.isclose(T[0, 2, 3], 2.0) and np.isclose(T[1, 2, 3], 3.0)
    T, valid = object_pose(traj, ObjectModel.from_box("c", [0.1] * 3, mocap_body="cubesat"), T0 + 10**12)
    assert not valid


def light(temp, intensity):
    return {"color_temperature_k": temp, "intensity": intensity, "intensity_unit": "%"}


def real_fixture():
    """antena: 10 settings x 3 trajectories (names in no frozen benchmark); cubesat 007-009
    (benchmark_v1) without light."""
    targets, lights, n = {}, {}, 10
    for k in range(10):
        for _ in range(3):
            t = f"trajectory_fixture_{n:03d}"
            targets[t], lights[t] = "antena", light(3000 + 300 * k, 20 + k)
            n += 1
    targets.update({f"trajectory_00{i}": "cubesat" for i in (7, 8, 9)})
    return targets, lights


def test_real_splits_hold_out_whole_lighting_settings():
    targets, lights = real_fixture()
    s = update_real_splits(None, targets, lights)
    split_of = {t: k for k in ("train", "test") for t in s[k]}
    assert set(split_of) == set(targets)
    by_setting: dict[tuple, set] = {}
    for t in targets:
        if targets[t] == "antena":
            by_setting.setdefault(tuple(lights[t].values()), set()).add(split_of[t])
    assert all(len(v) == 1 for v in by_setting.values())  # a setting is never split
    assert s["summary"]["real/antena"] == {"train": 24, "val": 0, "test": 6}  # 2 of 10 settings
    test_temps = sorted({lights[t]["color_temperature_k"] for t in s["test"] if t in lights})
    assert test_temps[0] > 3000 and test_temps[-1] < 5700  # spread, not the extremes
    b = load_benchmark("v1")  # the frozen benchmark's real trajectories keep their split
    assert {t for t in b["test"] if t in targets} <= set(s["test"])
    assert {t for t in b["train"] if t in targets} <= set(s["train"])
    assert update_real_splits(s, targets, lights) == s  # idempotent


def test_real_splits_append_only_and_v3_has_no_hash_fallback(tmp_path):
    targets, lights = real_fixture()
    first = {t: v for t, v in targets.items() if t < "trajectory_fixture_025"}
    s1 = update_real_splits(None, first, lights)
    s2 = update_real_splits(s1, targets, lights)
    for k in ("train", "test"):
        assert set(s1[k]) <= set(s2[k])
    assert s2["version"] == 3
    (tmp_path / "splits.json").write_text(json.dumps(s2))
    s = load_splits(tmp_path, [*targets, "trajectory_001"], unlisted=[*targets, "trajectory_001"])
    assert all("trajectory_001" not in v for v in s.values())  # no target model: no split


def test_real_targets():
    objects = {"cubesat": {"trajectories": ["trajectory_007"]}, "other": {"box": [1, 1, 1]}}
    assert real_targets(objects, {"trajectory_010": {"model": "antena"}}) == {
        "trajectory_007": "cubesat", "trajectory_010": "antena"}


def test_mockup_override_and_simple_mesh(tmp_path):
    pytest.importorskip("trimesh")
    write_models(tmp_path, {"trajectory_010": {"model": "antena"}, "trajectory_040": {"model": "antena_1_solar"}},
                 {"cubesat": {"class_id": 1, "box": [0.1] * 3, "trajectories": ["trajectory_007"]},
                  "antena": {"class_id": 2, "mocap_body": MOCKUP_BODY,
                             "T_body_model": {"translation": [0, 0, 0], "rotation_xyzw": [0, 0, 1, 0]}}})
    simple = tmp_path / "real" / "antena" / "model_simple.obj"
    simple.write_text("\n".join(CUBE_OBJ.splitlines()[:8]) + "\nf 1 2 4\n")
    objs = load_object_models(tmp_path)
    a = objs["antena"]
    assert a.class_id == 2 and np.allclose(a.T_body_model[:3, :3], np.diag([-1, -1, 1]))
    assert len(a.faces) == 1 and a.mesh_path.name == "model.obj"  # labels: simple mesh; source: full
    assert a.trajectories == ["trajectory_010"]
    assert objs["antena_1_solar"].class_id == 3 and len(objs["antena_1_solar"].faces) == 12
