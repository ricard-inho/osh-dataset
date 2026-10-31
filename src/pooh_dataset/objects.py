"""Tracked object models (CAD mesh or box) and the mocap-marker -> model transform.

Read from ``models/objects.yaml`` in the dataset and from the real mockups of
``models/real/mockups.json`` (trajectory -> mockup, written by the real pipeline): each mockup
is ``models/real/<mockup>/model.obj`` in the frame of the mocap body ``target`` (metres), with a
``static_pose`` for trajectories where the mockup stood still and its tracking dropped out.
See ``DATA_REQUIREMENTS.md``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .calibration import parse_transform
from .geometry import make_pose

__all__ = ["MOCKUP_BODY", "ObjectModel", "load_object_models"]

#: mocap rigid body of the real mockups (models/real/)
MOCKUP_BODY = "target"


@dataclass(eq=False)
class ObjectModel:
    name: str
    #: 1-based class / object id (BOP obj_id, COCO category_id; YOLO uses class_id - 1)
    class_id: int
    #: mocap rigid-body name tracking this object
    mocap_body: str
    #: vertices (N, 3) in metres, model frame
    vertices: np.ndarray
    #: triangle indices (M, 3)
    faces: np.ndarray
    #: pose of the model frame in the mocap rigid-body (marker) frame
    T_body_model: np.ndarray = field(default_factory=lambda: np.eye(4))
    mesh_path: Path | None = None
    #: BOP-style symmetries, passed through to exports untouched
    symmetries: dict | None = None
    #: trajectories the object was physically present in (None = all)
    trajectories: list[str] | None = None
    #: (K, 3) model-frame points a person can click in images (e.g. cube corners); used by
    #: `pooh calibrate`. Defaults to the corners of the mesh's bounding box.
    keypoints: np.ndarray | None = None
    #: Motive rigid-body export T_body_model was measured from (``pooh align-markers``), if any
    motive_file: str | None = None
    #: trajectory -> T_world_body of a mockup that stood still, used where its mocap track is
    #: missing (``models/real/mockups.json``)
    static_poses: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def click_points(self) -> np.ndarray:
        return self.corners if self.keypoints is None else self.keypoints

    def present_in(self, trajectory: str) -> bool:
        return self.trajectories is None or trajectory in self.trajectories

    @classmethod
    def from_box(cls, name: str, size_xyz, class_id: int = 1, mocap_body: str | None = None,
                 T_body_model=None) -> ObjectModel:
        """Axis-aligned box centred on the model origin; ``size_xyz`` in metres."""
        sx, sy, sz = (float(s) / 2 for s in size_xyz)
        v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
        f = np.array([
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],  # -x, +x
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],  # -y, +y
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],  # -z, +z
        ])  # fmt: skip
        return cls(name, class_id, mocap_body or name, v, f,
                   np.eye(4) if T_body_model is None else np.asarray(T_body_model, float))

    @classmethod
    def from_mesh_file(cls, name: str, path: Path, scale: float = 1.0, **kw) -> ObjectModel:
        try:
            import trimesh
        except ImportError as e:  # pragma: no cover
            raise ImportError("Loading CAD meshes needs `pip install pooh-dataset[mesh]`") from e
        mesh = trimesh.load(path, force="mesh", process=False)
        return cls(name=name, vertices=np.asarray(mesh.vertices, float) * scale,
                   faces=np.asarray(mesh.faces, np.int64), mesh_path=Path(path), **kw)

    @property
    def corners(self) -> np.ndarray:
        """(8, 3) corners of the model's axis-aligned bounding box."""
        lo, hi = self.vertices.min(0), self.vertices.max(0)
        return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    @property
    def diameter(self) -> float:
        """Max distance between two vertices (approximated on the bbox for big meshes)."""
        pts = self.vertices if len(self.vertices) <= 2000 else self.corners
        d = pts[:, None, :] - pts[None, :, :]
        return float(np.sqrt((d**2).sum(-1)).max())


def load_object_models(models_dir: Path) -> dict[str, ObjectModel]:
    """Parse ``models/objects.yaml`` and add the mockups of ``models/real/mockups.json``.
    Returns {} when neither is there."""
    models_dir = Path(models_dir)
    cfg_path = models_dir / "objects.yaml"
    cfg = (yaml.safe_load(cfg_path.read_text()) or {}) if cfg_path.exists() else {}
    out: dict[str, ObjectModel] = {}
    for i, (name, d) in enumerate(sorted(cfg.items()), start=1):
        common = dict(
            class_id=int(d.get("class_id", i)),
            mocap_body=d.get("mocap_body", name),
            T_body_model=parse_transform(d["T_body_model"]) if "T_body_model" in d else np.eye(4),
        )
        if "mesh" in d:
            m = ObjectModel.from_mesh_file(name, models_dir.parent / d["mesh"],
                                           scale=float(d.get("mesh_scale", 1.0)), **common)
        elif "box" in d:
            m = ObjectModel.from_box(name, d["box"], **common)
        elif (models_dir / "real" / name).is_dir():  # a mockup: its CAD is in models/real/<name>/
            m = _mockup_model(models_dir, name, **common)
            if m is None:
                continue
        else:
            raise ValueError(f"models/objects.yaml: {name!r} needs either `mesh` or `box`")
        m.symmetries = d.get("symmetries")
        m.motive_file = d.get("motive_file")
        if d.get("keypoints") is not None:
            m.keypoints = np.asarray(d["keypoints"], dtype=np.float64).reshape(-1, 3)
        if d.get("trajectories") is not None:
            m.trajectories = [str(t) for t in d["trajectories"]]
        out[name] = m
    _add_mockups(out, models_dir)
    return out


def _add_mockups(out: dict[str, ObjectModel], models_dir: Path) -> None:
    """The real mockups (``models/real/``) as objects on the ``target`` mocap body. An entry of
    the same name in objects.yaml (without ``mesh``) wins, e.g. to set ``class_id`` and
    ``T_body_model``; its trajectories and static poses are filled in from mockups.json."""
    path = models_dir / "real" / "mockups.json"
    if not path.exists():
        return
    by_mockup: dict[str, dict[str, dict]] = {}
    for traj, entry in json.loads(path.read_text()).items():
        by_mockup.setdefault(entry["model"], {})[traj] = entry
    next_id = max((m.class_id for m in out.values()), default=0) + 1
    for name, entries in sorted(by_mockup.items()):
        m = out.get(name)
        if m is None:
            m = _mockup_model(models_dir, name, class_id=next_id, mocap_body=MOCKUP_BODY)
            if m is None:
                continue
            next_id += 1
            out[name] = m
        if m.trajectories is None:
            m.trajectories = sorted(entries)
        for traj, e in entries.items():
            p = e.get("static_pose")
            if p is not None and traj not in m.static_poses:
                m.static_poses[traj] = make_pose(np.asarray(p["position"], float),
                                                 np.asarray(p["orientation_xyzw"], float))


def _mockup_model(models_dir: Path, name: str, **kw) -> ObjectModel | None:
    """Mockup ``name`` from ``models/real/<name>/``: the labels use ``model_simple.obj`` (a
    watertight ~10k-face version with the same silhouette) when it is there, ``mesh_path`` is
    the full ``model.obj``."""
    folder = models_dir / "real" / name
    full, simple = folder / "model.obj", folder / "model_simple.obj"
    if not full.exists():
        warnings.warn(f"models/real/{name}: no model.obj; mockup {name!r} has no object model", stacklevel=4)
        return None
    m = ObjectModel.from_mesh_file(name, simple if simple.exists() else full, **kw)
    m.mesh_path = full
    return m
