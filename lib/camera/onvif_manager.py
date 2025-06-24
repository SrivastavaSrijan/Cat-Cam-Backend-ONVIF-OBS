import logging

# Suppress specific Zeep logger
logging.getLogger("zeep.xsd.types.simple").setLevel(logging.CRITICAL)

from onvif import ONVIFCamera, ONVIFError
from threading import Lock
import time
from .ptz_controller import FixedPtzCam  # Import our fixed PTZ implementation

class ONVIFCameraManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        # Camera storage
        self.cameras = {}

    def initialize_cameras(self, camera_configs):
        """
        Initialize all cameras using the provided configurations.
        """
        for cam in camera_configs:
            nickname = cam['nickname']
            try:
                camera_instance = ONVIFCameraInstance()
                initialized_camera = camera_instance.setup_move(cam['host'], cam['port'], cam['username'], cam['password'])
                self.cameras[nickname] = initialized_camera
            except Exception as e:
                print(f"❌ Failed to initialize camera {nickname}: {str(e)}")
                logging.error(f"Camera initialization failed for {nickname}: {e}")
                continue

    def get_camera(self, nickname):
        """
        Get an initialized camera by nickname.
        """
        return self.cameras.get(nickname)

    def perform_operation(self, nickname, operation, *args, **kwargs):
        """
        Perform a thread-safe operation on a specific camera.
        """
        camera = self.get_camera(nickname)
        if not camera:
            raise ValueError(f"Camera with nickname {nickname} not found.")

        with self._lock:
            try:
                return operation(camera, *args, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Error performing operation: {str(e)}")

class ONVIFCameraInstance:
    def __init__(self):
        self._ptz_cam = None  # Our main PTZ implementation
        self._onvif_ptz = None  # Backup ONVIF for presets only
        self._onvif_media_profile = None
        self._onvif_camera = None  # Add this missing reference
        self.host = None
        self._port = None
        self._username = None
        self._password = None
        self.is_online = False

    def setup_move(self, IP, PORT, USER, PASS):
        self.host = IP
        self._port = PORT
        self._username = USER
        self._password = PASS
        
        # Try to initialize FixedPtzCam first (this is our primary interface)
        try:
            self._ptz_cam = FixedPtzCam(ip_address=IP, port=PORT, user=USER, pword=PASS)
            self.is_online = True
        except Exception as e:
            logging.error(f"FixedPtzCam failed for {IP}: {e}")
            self._ptz_cam = None
        
        # Try to setup basic ONVIF for presets (as backup)
        try:
            self._onvif_camera = ONVIFCamera(IP, PORT, USER, PASS)
            self._onvif_ptz = self._onvif_camera.create_ptz_service()
            profiles = self._onvif_camera.create_media_service().GetProfiles()
            if profiles and len(profiles) > 0:
                self._onvif_media_profile = profiles[0]
                if not self.is_online:  # Only mark online if FixedPtzCam failed
                    self.is_online = True
        except Exception as e:
            logging.warning(f"ONVIF backup failed for {IP}: {e}")
            self._onvif_ptz = None
            self._onvif_media_profile = None
            self._onvif_camera = None
        
        if not self.is_online:
            print(f"⚠️  Camera {IP} failed to initialize completely")
        
        return self

    def move_direction(self, direction, velocity_factor=1.0):
        """Quick incremental movement for taps - moves briefly then stops"""
        if not self._ptz_cam:
            raise RuntimeError("Camera not available for movement")
        
        # Use high speed for responsive incremental movement
        move_speed = velocity_factor
        
        direction_map = {
            "up": (0.0, move_speed),
            "down": (0.0, -move_speed),
            "left": (-move_speed, 0.0),
            "right": (move_speed, 0.0),
        }
        
        if direction not in direction_map:
            raise ValueError(f"Invalid direction: {direction}")
        
        x_velocity, y_velocity = direction_map[direction]
        
        # Start movement with high speed, let it run briefly, then stop
        self._ptz_cam.move(x_velocity, y_velocity)
        time.sleep(0.5)  # Brief movement duration
        self._ptz_cam.stop()
        
        return {"success": f"Incremental move {direction}"}

    def continuous_move(self, pan_speed, tilt_speed):
        """Fast continuous movement for holds"""
        if not self._ptz_cam:
            raise RuntimeError("Camera not available for movement")
        
        self._ptz_cam.move(pan_speed, tilt_speed)
        return {"success": "Continuous movement started"}

    def stop_movement(self):
        """Immediately stop all movement"""
        if not self._ptz_cam:
            raise RuntimeError("Camera not available for movement")
        
        self._ptz_cam.stop()
        return {"success": "Movement stopped"}

    def get_status(self):
        """Get current camera position using PtzCam"""
        if not self._ptz_cam:
            raise RuntimeError("PTZ camera not initialized")
        
        try:
            postion = self._ptz_cam.get_position()
            x = postion['PanTilt']['x']
            y = postion['PanTilt']['y']
            return {
                "PTZPosition": {
                    "PanTilt": {
                        "x": x,
                        "y": y,
                    },
                }
            }

        except Exception as e:
            # Return safe defaults if position can't be retrieved
            logging.warning(f"Could not get position: {e}")
            return {
                "PTZPosition": {
                    "PanTilt": {
                        "x": 0.0,
                        "y": 0.0,
                    },
                    "Zoom": {
                        "x": 0.0
                    }
                }
            }
        
    def get_presets(self):
        """Get presets using ONVIF backup"""
        if not self._onvif_ptz or not self._onvif_media_profile:
            return []  # Return empty list instead of raising error

        try:
            presets = self._onvif_ptz.GetPresets({"ProfileToken": self._onvif_media_profile.token})
            if not presets:
                return []
                
            presets_list = []
            for preset in presets:
                preset_data = {
                    "Name": getattr(preset, 'Name', "Unknown"),
                    "Token": getattr(preset, 'token', None),
                    "PTZPosition": {}
                }
                
                # Simple preset data extraction
                if hasattr(preset, 'PTZPosition') and preset.PTZPosition:
                    if hasattr(preset.PTZPosition, 'PanTilt'):
                        preset_data["PTZPosition"]["PanTilt"] = {
                            "x": getattr(preset.PTZPosition.PanTilt, 'x', 0.0),
                            "y": getattr(preset.PTZPosition.PanTilt, 'y', 0.0),
                        }
                    if hasattr(preset.PTZPosition, 'Zoom'):
                        preset_data["PTZPosition"]["Zoom"] = {
                            "x": getattr(preset.PTZPosition.Zoom, 'x', 0.0),
                        }
                
                presets_list.append(preset_data)
            return presets_list
        except Exception as e:
            logging.warning(f"Failed to get presets: {e}")
            return []
    
    def goto_preset(self, preset_token):
        """Go to preset using ONVIF backup"""
        if not self._onvif_ptz or not self._onvif_media_profile:
            raise RuntimeError("Presets not available on this camera")
        
        try:
            request = self._onvif_ptz.create_type('GotoPreset')
            request.ProfileToken = self._onvif_media_profile.token
            request.PresetToken = preset_token
            self._onvif_ptz.GotoPreset(request)
            return {"success": f"Moved to preset"}
        except Exception as e:
            raise RuntimeError(f"Failed to go to preset: {str(e)}")

    def is_initialized(self):
        """Check if camera is online"""
        return self.is_online

    def get_status_safe(self):
        """Get safe camera status"""
        return {
            "online": self.is_online,
            "host": self.host,
            "ptz_available": self._ptz_cam is not None,
            "presets_available": self._onvif_ptz is not None
        }

    # Simplified placeholder methods for API compatibility
    def set_movement_speed(self, pan_tilt_speed=0.2, zoom_speed=0.1):
        return {"success": "Speed setting noted"}

    def get_movement_status(self):
        return {"is_moving": False, "direction": None, "progress": 0}

    def get_imaging_settings(self):
        """Get imaging settings AND relay information using ONVIF backup"""
        if not self._onvif_camera or not self._onvif_media_profile:
            return {"error": "Imaging settings not available on this camera"}

        try:
            result = {}
            
            # Get imaging settings
            imaging_service = self._onvif_camera.create_imaging_service()
            
            # Get video source token
            video_source_token = None
            if (hasattr(self._onvif_media_profile, 'VideoSourceConfiguration') and 
                self._onvif_media_profile.VideoSourceConfiguration and
                hasattr(self._onvif_media_profile.VideoSourceConfiguration, 'SourceToken')):
                video_source_token = self._onvif_media_profile.VideoSourceConfiguration.SourceToken
            else:
                video_sources = imaging_service.GetVideoSources()
                if video_sources and len(video_sources) > 0:
                    video_source_token = video_sources[0].token
                else:
                    return {"error": "No video source token found"}
            
            settings = imaging_service.GetImagingSettings({"VideoSourceToken": video_source_token})
            
            # Extract imaging settings
            result["imaging"] = {
                "brightness": getattr(settings, 'Brightness', None),
                "contrast": getattr(settings, 'Contrast', None),
                "saturation": getattr(settings, 'ColorSaturation', None),
                "sharpness": getattr(settings, 'Sharpness', None),
                "ir_cut_filter": getattr(settings, 'IrCutFilter', None),
                "video_source_token": video_source_token
            }
            
            # Get relay information
            try:
                device_service = self._onvif_camera.create_devicemgmt_service()
                relays = device_service.GetRelayOutputs()
                
                if relays:
                    result["relays"] = []
                    for relay in relays:
                        relay_info = {
                            "token": getattr(relay, 'token', 'unknown'),
                            "properties": {
                                "mode": getattr(relay.Properties, 'Mode', None) if hasattr(relay, 'Properties') else None,
                                "delay_time": getattr(relay.Properties, 'DelayTime', None) if hasattr(relay, 'Properties') else None,
                                "idle_state": getattr(relay.Properties, 'IdleState', None) if hasattr(relay, 'Properties') else None,
                            }
                        }
                        result["relays"].append(relay_info)
                        print(f"Found relay: {relay_info}")
                else:
                    result["relays"] = []
                    print("No relays found")
                    
            except Exception as e:
                result["relays"] = f"Error getting relays: {str(e)}"
                print(f"Relay discovery failed: {e}")
            
            result["night_mode_strategy"] = "brightness=0, saturation=0, relay=ON"
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to get settings: {str(e)}"}

    def set_night_mode(self, enable=True):
        """Night mode: brightness=0, saturation=0, relay=ON"""
        if not self._onvif_camera or not self._onvif_media_profile:
            return {"error": "Night mode not available on this camera"}

        try:
            results = {"steps": [], "success": False}
            
            # Step 1: Set imaging settings
            try:
                imaging_service = self._onvif_camera.create_imaging_service()
                
                # Get video source token
                video_source_token = None
                if (hasattr(self._onvif_media_profile, 'VideoSourceConfiguration') and 
                    self._onvif_media_profile.VideoSourceConfiguration and
                    hasattr(self._onvif_media_profile.VideoSourceConfiguration, 'SourceToken')):
                    video_source_token = self._onvif_media_profile.VideoSourceConfiguration.SourceToken
                else:
                    video_sources = imaging_service.GetVideoSources()
                    if video_sources and len(video_sources) > 0:
                        video_source_token = video_sources[0].token
                
                if video_source_token:
                    settings = imaging_service.GetImagingSettings({"VideoSourceToken": video_source_token})
                    
                    if enable:
                        # Night mode settings
                        if hasattr(settings, 'Brightness'):
                            settings.Brightness = 0.0  # Dark
                            results["steps"].append("Set brightness to 0")
                        
                        if hasattr(settings, 'ColorSaturation'):
                            settings.ColorSaturation = 0.0  # No color (grayscale)
                            results["steps"].append("Set saturation to 0")
                    else:
                        # Day mode settings
                        if hasattr(settings, 'Brightness'):
                            settings.Brightness = 50.0  # Normal
                            results["steps"].append("Restored brightness to 50")
                        
                        if hasattr(settings, 'ColorSaturation'):
                            settings.ColorSaturation = 50.0  # Normal color
                            results["steps"].append("Restored saturation to 50")
                    
                    # Apply imaging settings
                    imaging_service.SetImagingSettings({
                        "VideoSourceToken": video_source_token,
                        "ImagingSettings": settings,
                        "ForcePersistence": True
                    })
                    results["steps"].append("Applied imaging settings")
                    
            except Exception as e:
                results["steps"].append(f"Imaging settings failed: {str(e)}")
            
            # Step 2: Control relay (IR illuminator)
            try:
                device_service = self._onvif_camera.create_devicemgmt_service()
                relays = device_service.GetRelayOutputs()
                
                if relays:
                    for relay in relays:
                        try:
                            relay_state = "active" if enable else "inactive"
                            device_service.SetRelayOutputState({
                                "RelayOutputToken": relay.token,
                                "LogicalState": relay_state
                            })
                            results["steps"].append(f"Set relay {relay.token} to {relay_state}")
                            results["ir_illuminator"] = f"{'ON' if enable else 'OFF'}"
                            break  # Use first relay found
                        except Exception as e:
                            results["steps"].append(f"Relay {relay.token} control failed: {str(e)}")
                else:
                    results["steps"].append("No relays found for IR control")
                    
            except Exception as e:
                results["steps"].append(f"Relay control failed: {str(e)}")
            
            # Determine success
            imaging_success = any("Applied imaging settings" in step for step in results["steps"])
            relay_success = any("Set relay" in step and "failed" not in step for step in results["steps"])
            
            if imaging_success or relay_success:
                results["success"] = True
                results["message"] = f"Night mode {'enabled' if enable else 'disabled'}"
                if relay_success:
                    results["message"] += " - IR illuminator controlled"
            else:
                results["message"] = "Night mode control failed"
            
            return results
                
        except Exception as e:
            return {"error": f"Failed to set night mode: {str(e)}"}
