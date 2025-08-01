"""
Camera service wrapper for managing ONVIF camera operations.
"""
import logging
from threading import Lock
from lib.camera.onvif_manager import ONVIFCameraManager, ONVIFCameraInstance
from utils.response import success_response, error_response, data_response


class CameraService:
    """Service for managing camera operations"""
    
    def __init__(self, camera_configs):
        self.camera_manager = ONVIFCameraManager()
        self.lock = Lock()
        self._initialize_cameras(camera_configs)
    
    def _initialize_cameras(self, camera_configs):
        """Initialize all cameras"""
        print("\n📹 Initializing cameras...")
        self.camera_manager.initialize_cameras(camera_configs)
        
        # Check how many cameras were successfully initialized
        initialized_count = len(self.camera_manager.cameras)
        total_count = len(camera_configs)
        print(f"\n📊 Camera initialization complete: {initialized_count}/{total_count} cameras online")

        if initialized_count == 0:
            print("⚠️  WARNING: No cameras were successfully initialized!")
            print("   - Check camera network connectivity")
            print("   - Verify ONVIF credentials in shared_config.py")
            print("   - Ensure cameras support ONVIF protocol")
        else:
            print("✅ Camera system ready!")
            for nickname in self.camera_manager.cameras.keys():
                print(f"   - {nickname}: Online")
    
    def perform_operation(self, nickname, operation):
        """Perform a thread-safe operation on a specific camera"""
        with self.lock:
            return self.camera_manager.perform_operation(nickname, operation)
    
    def get_camera_list(self):
        """Get list of all cameras with their status"""
        cameras = []
        for nickname, camera in self.camera_manager.cameras.items():
            if camera:
                safe_status = camera.get_status_safe()
                camera_info = {
                    "nickname": nickname,
                    "host": safe_status.get("host", "unknown"),
                    "port": getattr(camera, '_port', 554),  # Default ONVIF port
                    "status": "online" if safe_status.get("online", False) else "offline",
                    "online": safe_status.get("online", False)
                }
                if not safe_status.get("online", False):
                    camera_info["error"] = safe_status.get("error")
                cameras.append(camera_info)
            else:
                cameras.append({
                    "nickname": nickname,
                    "host": "unknown",
                    "port": 554,
                    "status": "offline",
                    "online": False,
                    "error": "Camera not initialized"
                })
        return cameras
