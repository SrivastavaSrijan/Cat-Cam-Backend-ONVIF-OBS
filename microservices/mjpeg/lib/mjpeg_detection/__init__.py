"""
Detection and motion processing modules.
"""

from .motion_processor import (
    init_detection,
    detect_cats,
    detect_motion,
    process_frame_for_detection,
    log_detection_event,
    toggle_detection,
    get_detection_status
)