"""Fixed PtzCam class that handles cameras without full imaging capabilities"""

import logging
from onvif import ONVIFCamera

log = logging.getLogger(__name__)

def _check_zeroness(number):
    """Almost-zero check"""
    e = .001
    if -e < number < e:
        return 0
    else:
        return number

class FixedPtzCam:
    """Simplified PTZ camera class that works with cameras that have limited imaging settings"""
    
    def __init__(self, ip_address=None, port='80', user=None, pword=None):
        """Initialize PTZ camera with error handling for imaging settings"""
        
        mycam = ONVIFCamera(ip_address, port, user, pword)
        media_service = mycam.create_media_service()
        self.ptz_service = mycam.create_ptz_service()
        
        # Try to create imaging service, but don't fail if it doesn't work fully
        try:
            self.imaging_service = mycam.create_imaging_service()
        except Exception as e:
            log.warning(f"Imaging service not available: {e}")
            self.imaging_service = None

        # Get media profile
        profiles = media_service.GetProfiles()
        if not profiles:
            raise RuntimeError("No media profiles available")
        self.media_profile = profiles[0]

        # Get video source
        try:
            self.video_source = media_service.GetVideoSources()[0]
        except Exception as e:
            log.warning(f"Video source not available: {e}")
            self.video_source = None

        # Set default bounds
        self.pan_bounds = [-1.0, 1.0]
        self.tilt_bounds = [-1.0, 1.0]
        self.zoom_bounds = [0.0, 1.0]

        # Try to get imaging settings, but handle gracefully if they fail
        self.imaging_settings = None
        if self.imaging_service and self.video_source:
            try:
                vst = {'VideoSourceToken': self.video_source.token}
                self.imaging_settings = self.imaging_service.GetImagingSettings(vst)
                
                # Safely get exposure bounds
                if (hasattr(self.imaging_settings, 'Exposure') and 
                    self.imaging_settings.Exposure):
                    
                    exposure = self.imaging_settings.Exposure
                    
                    # Safe access to iris bounds
                    try:
                        min_iris = getattr(exposure, 'MinIris', None)
                        max_iris = getattr(exposure, 'MaxIris', None)
                        if min_iris is not None and max_iris is not None:
                            self.iris_bounds = [min_iris, max_iris]
                        else:
                            self.iris_bounds = [0.0, 1.0]  # Default bounds
                    except Exception:
                        self.iris_bounds = [0.0, 1.0]
                    
                    # Safe access to exposure time bounds
                    try:
                        min_exp = getattr(exposure, 'MinExposureTime', None)
                        max_exp = getattr(exposure, 'MaxExposureTime', None)
                        if min_exp is not None and max_exp is not None:
                            self.exposure_time_bounds = [min_exp, max_exp]
                        else:
                            self.exposure_time_bounds = [0.0, 1.0]  # Default bounds
                    except Exception:
                        self.exposure_time_bounds = [0.0, 1.0]
                else:
                    # No exposure settings available
                    self.iris_bounds = [0.0, 1.0]
                    self.exposure_time_bounds = [0.0, 1.0]
                    
            except Exception as e:
                log.warning(f"Could not get imaging settings: {e}")
                self.iris_bounds = [0.0, 1.0]
                self.exposure_time_bounds = [0.0, 1.0]
        else:
            # No imaging service or video source
            self.iris_bounds = [0.0, 1.0]
            self.exposure_time_bounds = [0.0, 1.0]

        log.info(f"FixedPtzCam initialized successfully for {ip_address}")

    def __del__(self):
        log.debug('FixedPtzCam object deletion.')

    def move(self, x_velocity, y_velocity):
        """Move camera with continuous movement"""
        move_request = self.ptz_service.create_type('ContinuousMove')
        move_request.ProfileToken = self.media_profile.token
        move_request.Velocity = {'PanTilt': {'x': x_velocity, 'y': y_velocity},
                                 'Zoom': {'x': 0.0}}
        self.ptz_service.ContinuousMove(move_request)

    def move_w_zoom(self, x_velocity, y_velocity, zoom_command):
        """Move camera with zoom"""
        x_velocity = float(_check_zeroness(x_velocity))
        y_velocity = float(_check_zeroness(y_velocity))
        zoom_command = _check_zeroness(zoom_command)

        move_request = self.ptz_service.create_type('ContinuousMove')
        move_request.ProfileToken = self.media_profile.token
        move_request.Velocity = {'PanTilt': {'x': x_velocity, 'y': y_velocity},
                                 'Zoom': {'x': zoom_command}}
        self.ptz_service.ContinuousMove(move_request)

    def get_position(self):
        """Get current camera position"""
        status = self.ptz_service.GetStatus({'ProfileToken': self.media_profile.token})
        position = status.Position
        
        # x = pan, y = tilt
        return position

    def absmove(self, x_pos, y_pos):
        """Absolute move to specific position"""
        move_request = self.ptz_service.create_type('AbsoluteMove')
        move_request.ProfileToken = self.media_profile.token
        
        # Get current position for template
        current_status = self.ptz_service.GetStatus({'ProfileToken': self.media_profile.token})
        if current_status and hasattr(current_status, 'Position'):
            move_request.Position = current_status.Position
            move_request.Position.PanTilt.x = x_pos
            move_request.Position.PanTilt.y = y_pos
        else:
            # Create position manually
            move_request.Position = self.ptz_service.create_type('PTZVector')
            move_request.Position.PanTilt = self.ptz_service.create_type('Vector2D')
            move_request.Position.PanTilt.x = x_pos
            move_request.Position.PanTilt.y = y_pos
            move_request.Position.Zoom = self.ptz_service.create_type('Vector1D')
            move_request.Position.Zoom.x = 0.0
        
        self.ptz_service.AbsoluteMove(move_request)

    def absmove_w_zoom(self, pan_pos, tilt_pos, zoom_pos):
        """Absolute move with zoom"""
        zoom_pos = _check_zeroness(zoom_pos)
        pan_pos = _check_zeroness(pan_pos)
        tilt_pos = _check_zeroness(tilt_pos)

        move_request = self.ptz_service.create_type('AbsoluteMove')
        move_request.ProfileToken = self.media_profile.token
        
        # Get current position for template
        current_status = self.ptz_service.GetStatus({'ProfileToken': self.media_profile.token})
        if current_status and hasattr(current_status, 'Position'):
            move_request.Position = current_status.Position
            move_request.Position.PanTilt.x = pan_pos
            move_request.Position.PanTilt.y = tilt_pos
            move_request.Position.Zoom.x = zoom_pos
        else:
            # Create position manually
            move_request.Position = self.ptz_service.create_type('PTZVector')
            move_request.Position.PanTilt = self.ptz_service.create_type('Vector2D')
            move_request.Position.PanTilt.x = pan_pos
            move_request.Position.PanTilt.y = tilt_pos
            move_request.Position.Zoom = self.ptz_service.create_type('Vector1D')
            move_request.Position.Zoom.x = zoom_pos
        
        self.ptz_service.AbsoluteMove(move_request)

    def stop(self):
        """Stop all movement"""
        stop_request = self.ptz_service.create_type('Stop')
        stop_request.ProfileToken = self.media_profile.token
        stop_request.PanTilt = True
        stop_request.Zoom = True
        self.ptz_service.Stop(stop_request)

    def zoom(self, zoom_command):
        """Set zoom level"""
        zoom_command = _check_zeroness(zoom_command)
        
        move_request = self.ptz_service.create_type('AbsoluteMove')
        move_request.ProfileToken = self.media_profile.token
        
        # Get current position
        current_status = self.ptz_service.GetStatus({'ProfileToken': self.media_profile.token})
        if current_status and hasattr(current_status, 'Position'):
            move_request.Position = current_status.Position
            move_request.Position.Zoom.x = zoom_command
        else:
            # Create position manually
            move_request.Position = self.ptz_service.create_type('PTZVector')
            move_request.Position.PanTilt = self.ptz_service.create_type('Vector2D')
            move_request.Position.PanTilt.x = 0.0
            move_request.Position.PanTilt.y = 0.0
            move_request.Position.Zoom = self.ptz_service.create_type('Vector1D')
            move_request.Position.Zoom.x = zoom_command
        
        self.ptz_service.AbsoluteMove(move_request)
