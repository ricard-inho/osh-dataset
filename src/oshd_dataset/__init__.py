"""oshd-dataset: download and process the OSHD multimodal robotics dataset.

Quick start::

    from oshd_dataset import OshdDataset
    ds = OshdDataset.download(modalities=["mocap", "realsense_color"])
    traj = ds["trajectory_007"]
    frame = traj.frame("realsense_color", 0)

    sim = OshdDataset.download(domain="sim_3d", modalities=["rgb", "mocap"])
"""

__version__ = "1.0.0"

from .calibration import Calibration, CameraIntrinsics, MissingCalibrationError
from .dataset import OshdDataset
from .events import Events
from .hub import DEFAULT_REPO_ID, download
from .objects import ObjectModel
from .trajectory import Trajectory

__all__ = [
    "Calibration",
    "CameraIntrinsics",
    "DEFAULT_REPO_ID",
    "Events",
    "MissingCalibrationError",
    "ObjectModel",
    "OshdDataset",
    "Trajectory",
    "download",
]
