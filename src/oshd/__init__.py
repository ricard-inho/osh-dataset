"""oshd: download and process the OSHD multimodal robotics dataset.

Quick start::

    from oshd import OshDataset
    ds = OshDataset.download(modalities=["mocap", "realsense_color"])
    traj = ds["trajectory_007"]
    frame = traj.frame("realsense_color", 0)

    sim = OshDataset.download(domain="sim_3d", modalities=["rgb", "mocap"])
"""

__version__ = "1.0.0"

from .calibration import Calibration, CameraIntrinsics, MissingCalibrationError
from .dataset import OshDataset
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
    "OshDataset",
    "Trajectory",
    "download",
]
