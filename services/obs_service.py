"""
OBS service wrapper for managing OBS WebSocket operations.
"""
import logging
from lib.obs.websocket_client import OBSWebSocketClient


class OBSService:
    """Service for managing OBS operations"""
    
    def __init__(self, obs_url, obs_password):
        self.obs_client = OBSWebSocketClient(obs_url, obs_password)
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize OBS WebSocket connection"""
        if not self.obs_client.connect():
            logging.error("Could not connect to OBS WebSocket")
            print("⚠️  OBS WebSocket connection failed - OBS features will not be available")
            return False
        else:
            print("✅ OBS WebSocket connected successfully")
            return True
    
    def get_scenes(self):
        """Get all OBS scenes"""
        try:
            self.obs_client.retrieve_scene_sources("Mosaic")
            return {"success": "Scene sources retrieved"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_current_scene(self):
        """Get current OBS scene"""
        try:
            # Note: This method may need to be implemented in OBSWebSocketClient
            return {"current_scene": "Not implemented"}
        except Exception as e:
            return {"error": str(e)}
    
    def switch_scene(self, scene_name):
        """Switch OBS scene"""
        try:
            self.obs_client.switch_scene(scene_name)
            return {"success": f"Switched to scene: {scene_name}"}
        except Exception as e:
            return {"error": str(e)}
    
    def apply_transformation(self, transformation_type, active_source=None):
        """Apply transformation to OBS scene"""
        try:
            self.obs_client.retrieve_scene_sources("Mosaic")
            
            if transformation_type == "grid":
                self.obs_client.update_obs_layout("Mosaic")
            elif transformation_type == "highlight":
                self.obs_client.update_obs_layout("Mosaic", active_source)
            
            return {"success": f"Applied {transformation_type} transformation"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_current_transformation(self):
        """Get current highlighted source"""
        try:
            if hasattr(self.obs_client, 'current_highlighted_source') and self.obs_client.current_highlighted_source:
                return {"highlighted_source": self.obs_client.current_highlighted_source}
            else:
                return {"highlighted_source": None}
        except Exception as e:
            return {"error": str(e)}
    
    def reconnect(self):
        """Reconnect to OBS"""
        try:
            self.obs_client.close()
            import time
            time.sleep(5)
            self.obs_client.connect()
            return {"success": "Reconnected to OBS"}
        except Exception as e:
            return {"error": str(e)}
    
    def start_fullscreen_projector(self, scene_name, monitor_index=0):
        """Start fullscreen projector"""
        return self.obs_client.start_fullscreen_projector(scene_name, monitor_index)
    
    def close_projector(self, projector_type="source"):
        """Close projector"""
        return self.obs_client.close_projector(projector_type)
    
    def start_virtual_camera(self):
        """Start virtual camera"""
        return self.obs_client.start_virtual_camera()
    
    def stop_virtual_camera(self):
        """Stop virtual camera"""
        return self.obs_client.stop_virtual_camera()
    
    def get_virtual_camera_status(self):
        """Get virtual camera status"""
        return self.obs_client.get_virtual_camera_status()
