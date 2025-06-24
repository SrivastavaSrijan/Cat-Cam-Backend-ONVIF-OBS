"""
Detection and motion processing modules.
"""

from .motion_processor import detect_objects, save_detection_frame, detect_motion, process_stream, init

__all__ = ['detect_objects', 'save_detection_frame', 'detect_motion', 'process_stream', 'init']
