"""
OBS service wrapper for managing OBS WebSocket operations.
"""
import logging
from lib.obs.websocket_client import OBSWebSocketClient
from utils.response import success_response, error_response, data_response


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
            from config.shared_config import STREAMS
            self.obs_client.retrieve_scene_sources("Mosaic")
            # STREAMS is a list of dicts, convert to expected format
            scenes = [{"name": stream.get("name", ""), "obs_scene": stream.get("name", "")} for stream in STREAMS]
            return data_response({"scenes": scenes})
        except Exception as e:
            return error_response(str(e))
    
    def get_current_scene(self):
        """Get current OBS scene"""
        try:
            # TODO: Implement get_current_scene in OBSWebSocketClient if needed
            return data_response({"name": "Mosaic", "current_scene": "Mosaic"})
        except Exception as e:
            return error_response(str(e))
    
    def switch_scene(self, scene_name):
        """Switch OBS scene"""
        try:
            self.obs_client.switch_scene(scene_name)
            return success_response(message=f"Switched to scene: {scene_name}")
        except Exception as e:
            return error_response(str(e))
    
    def apply_transformation(self, transformation_type, active_source=None):
        """Apply transformation to OBS scene"""
        try:
            self.obs_client.retrieve_scene_sources("Mosaic")
            
            if transformation_type == "grid":
                self.obs_client.update_obs_layout("Mosaic")
            elif transformation_type == "highlight":
                self.obs_client.update_obs_layout("Mosaic", active_source)
            
            return success_response(message=f"Applied {transformation_type} transformation")
        except Exception as e:
            return error_response(str(e))
    
    def get_current_transformation(self, scene_name="Mosaic"):
        """Get current highlighted source"""
        try:
            print(self.obs_client.current_highlighted_source)
            # Check if OBS client has a current highlighted source stored
            if hasattr(self.obs_client, 'current_highlighted_source') and self.obs_client.current_highlighted_source:
                transformation_state = {
                    "layout_mode": "highlight",
                    "highlighted_source": self.obs_client.current_highlighted_source
                }
            else:
                # No highlighted source, assume grid layout
                transformation_state = {
                    "layout_mode": "grid",
                    "highlighted_source": ""
                }
            return data_response(transformation_state)
        except Exception as e:
            return error_response(str(e))
    
    def reconnect(self):
        """Reconnect to OBS"""
        try:
            self.obs_client.close()
            import time
            time.sleep(5)
            self.obs_client.connect()
            return success_response(message="Reconnected to OBS")
        except Exception as e:
            return error_response(str(e))
    
    def start_fullscreen_projector(self, scene_name, monitor_index=0):
        """Start fullscreen projector"""
        result = self.obs_client.start_fullscreen_projector(scene_name, monitor_index)
        if "error" in result:
            return error_response(result["error"])
        return success_response(message=result.get("success", "Projector started"))
    
    def close_projector(self, projector_type="source"):
        """Close projector"""
        result = self.obs_client.close_projector(projector_type)
        if "error" in result:
            return error_response(result["error"])
        return success_response(message=result.get("success", "Projector closed"))
    
    def start_virtual_camera(self):
        """Start virtual camera"""
        result = self.obs_client.start_virtual_camera()
        if "error" in result:
            return error_response(result["error"])
        return success_response(message=result.get("success", "Virtual camera started"))
    
    def stop_virtual_camera(self):
        """Stop virtual camera"""
        result = self.obs_client.stop_virtual_camera()
        if "error" in result:
            return error_response(result["error"])
        return success_response(message=result.get("success", "Virtual camera stopped"))
    
    def get_virtual_camera_status(self):
        """Get virtual camera status"""
        result = self.obs_client.get_virtual_camera_status()
        if "error" in result:
            return error_response(result["error"])
        # For now, return a mock status since the OBS client doesn't return actual status
        virtual_camera_status = {
            "active": False,
            "status": "unknown"
        }
        return data_response(virtual_camera_status)
