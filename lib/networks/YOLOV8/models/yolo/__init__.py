# Ultralytics YOLO 🚀, AGPL-3.0 license

# from . import classify, detect, obb, pose, segment, world

# from .model import YOLO, YOLOWorld

# __all__ = "classify", "segment", "detect", "pose", "obb", "world", "YOLO", "YOLOWorld"

###duan
from . import detect, pose

from .model import YOLO

__all__ = "detect", "pose",  "YOLO"