"""
Enhanced ONVIF Camera Control System - Modular Flask Application
"""
import logging
from flask import Flask
from flask_cors import CORS

from config import Config
from services.camera_service import CameraService
from services.obs_service import OBSService
from services.detection_service import DetectionService
from routes.ptz_routes import init_ptz_routes
from routes.obs_routes import init_obs_routes
from routes.detection_routes import init_detection_routes


def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, resources={r"/*": {"origins": Config.CORS_ORIGINS}})
    
    # Initialize services
    print("🚀 Starting Enhanced ONVIF Camera Control System...")
    
    # Initialize OBS service
    obs_service = OBSService(Config.OBS_URL, Config.OBS_PASSWORD)
    
    # Initialize camera service
    camera_service = CameraService(Config.CAMERA_CONFIGS)
    
    # Initialize detection service
    detection_service = DetectionService(Config.MOTION_DETECTION_LOG)
    
    # Register blueprints with service dependencies
    ptz_bp = init_ptz_routes(camera_service)
    obs_bp = init_obs_routes(obs_service)
    detection_bp = init_detection_routes(detection_service)
    
    app.register_blueprint(ptz_bp)
    app.register_blueprint(obs_bp)
    app.register_blueprint(detection_bp)
    
    print("\n🌐 Starting Flask web server...")
    print("   API will be available at: http://localhost:5001")
    print("   Use Ctrl+C to stop the server")
    print("-" * 50)
    
    return app


# Create the Flask application
app = create_app()


if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
