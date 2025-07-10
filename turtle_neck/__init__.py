"""
거북목 감지 시스템 패키지
"""

from .model_manager import ModelManager
from .pose_decoder import OpenPoseDecoder
from .image_processor import ImageProcessor
from .turtle_neck_detector import TurtleNeckDetector
from .pose_visualizer import PoseVisualizer
from .system_runner import TurtleNeckSystem, run_turtle_neck_detection

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "ModelManager",
    "OpenPoseDecoder", 
    "ImageProcessor",
    "TurtleNeckDetector",
    "PoseVisualizer",
    "TurtleNeckSystem",
    "run_turtle_neck_detection"
]