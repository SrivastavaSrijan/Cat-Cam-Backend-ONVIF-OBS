"""
Camera service wrapper for managing ONVIF camera operations.
"""
import logging
from threading import Lock
from lib.camera.onvif_manager import ONVIFCameraManager, ONVIFCameraInstance


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
            status = camera.get_status_safe()
            cameras.append({
                "nickname": nickname,
                "host": status.get("host"),
                "online": status.get("online", False),
                "ptz_available": status.get("ptz_available", False),
                "presets_available": status.get("presets_available", False)
            })
        return cameras
