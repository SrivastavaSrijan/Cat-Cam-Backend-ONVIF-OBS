"""
Configuration module for the camera control application.
"""
import os
from .shared_config import CAMERA_CONFIGS, OBS_PASSWORD, OBS_URL, STREAMS, CANVAS_HEIGHT, CANVAS_WIDTH

class Config:
    """Application configuration"""
    # Flask settings
    HOST = "0.0.0.0"
    PORT = 5001
    DEBUG = True
    
    # CORS settings
    CORS_ORIGINS = "*"
    
    # Camera settings
    CAMERA_CONFIGS = CAMERA_CONFIGS
    
    # OBS settings
    OBS_URL = OBS_URL
    OBS_PASSWORD = OBS_PASSWORD
    OBS_CANVAS_WIDTH = CANVAS_WIDTH
    OBS_CANVAS_HEIGHT = CANVAS_HEIGHT
    
    # Streams
    STREAMS = STREAMS
    
    # Motion detection
    MOTION_DETECTION_LOG = "motion_detection.log"
