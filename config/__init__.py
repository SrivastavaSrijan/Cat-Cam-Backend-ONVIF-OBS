"""
Configuration package for the camera control application.
"""

from .shared_config import (
    STREAMS,
    CAMERA_CONFIGS,
    OBS_PASSWORD,
    OBS_URL,
    CANVAS_HEIGHT,
    CANVAS_WIDTH
)

from .config import Config

__all__ = [
    'STREAMS',
    'CAMERA_CONFIGS', 
    'OBS_PASSWORD',
    'OBS_URL',
    'CANVAS_HEIGHT',
    'CANVAS_WIDTH',
    'Config'
]
