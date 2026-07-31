# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.2.31"

import os

# Set ENV Variables (place before imports)
os.environ["OMP_NUM_THREADS"] = "1"  # reduce CPU utilization during training

# High-level APIs are loaded lazily so detector construction does not require
# optional medical-volume dependencies such as SimpleITK.
def __getattr__(name):
    if name == "Explorer":
        from .data.explorer.explorer import Explorer
        return Explorer
    if name == "BaseModel":
        from .models import YOLO
        return YOLO
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))

from .utils import ASSETS, SETTINGS
from .utils.checks import check_yolo as checks
from .utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "BaseModel",
    #"YOLO",
    # "YOLOWorld",
    # "NAS",
    # "SAM",
    # "FastSAM",
    # "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
