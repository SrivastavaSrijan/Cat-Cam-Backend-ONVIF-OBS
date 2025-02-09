import logging

# Suppress specific Zeep logger
logging.getLogger("zeep.xsd.types.simple").setLevel(logging.CRITICAL)

import sys
from onvif import ONVIFCamera, ONVIFError
from threading import Lock

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
                self.cameras[nickname] = ONVIFCameraInstance().setup_move(cam['host'], cam['port'], cam['username'], cam['password'])
                print(f"Initialized camera: {nickname}, {self.cameras[nickname].host}")
            except Exception as e:
                print(f"Failed to initialize camera {nickname}: {str(e)}")

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
                # Pass the camera instance to the operation
                return operation(camera, *args, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Error performing operation: {str(e)}")

class ONVIFCameraInstance:

    def __init__(self):
        self._XMAX = 1
        self._XMIN = -1
        self._XNOW = 0.5
        self._YMAX = 1
        self._YMIN = -1
        self._YNOW = 0.5
        self._Move = 0.1
        self._Velocity = 1
        self._Zoom = 0
        self._positionrequest = None
        self._ptz = None
        self._active = False
        self._ptz_configuration_options = None
        self._media_profile = None
        self.host = None

    def get_status(self):
        if not self._ptz or not self._media_profile or not self._ptz_configuration_options:
            raise RuntimeError("PTZ service or media profile not set up.")

        self._XMAX = self._ptz_configuration_options.Spaces.AbsolutePanTiltPositionSpace[0].XRange.Max
        self._XMIN = self._ptz_configuration_options.Spaces.AbsolutePanTiltPositionSpace[0].XRange.Min
        self._YMAX = self._ptz_configuration_options.Spaces.AbsolutePanTiltPositionSpace[0].YRange.Max
        self._YMIN = self._ptz_configuration_options.Spaces.AbsolutePanTiltPositionSpace[0].YRange.Min
        status = self._ptz.GetStatus({'ProfileToken': self._media_profile.token})
        self._XNOW = status.Position.PanTilt.x
        self._YNOW = status.Position.PanTilt.y
        self._Velocity = self._ptz_configuration_options.Spaces.PanTiltSpeedSpace[0].XRange.Max
        self._Zoom = status.Position.Zoom.x
        return {
            "PTZPosition" : {
                "PanTilt": {
                    "x": status.Position.PanTilt.x,
                    "y": status.Position.PanTilt.y,
                }
            }
        };


    def setup_move(self, IP, PORT, USER, PASS):
        mycam = ONVIFCamera(IP, PORT, USER, PASS)
        self.host = IP
        media = mycam.create_media_service()

        self._ptz = mycam.create_ptz_service()
        self._media_profile = media.GetProfiles()[0]

        request = self._ptz.create_type('GetConfigurationOptions')
        request.ConfigurationToken = self._media_profile.PTZConfiguration.token
        self._ptz_configuration_options = self._ptz.GetConfigurationOptions(request)

        request_configuration = self._ptz.create_type('GetConfiguration')
        request_configuration.PTZConfigurationToken = self._media_profile.PTZConfiguration.token
        ptz_configuration = self._ptz.GetConfiguration(request_configuration)

        request_setconfiguration = self._ptz.create_type('SetConfiguration')
        request_setconfiguration.PTZConfiguration = ptz_configuration

        self._positionrequest = self._ptz.create_type('AbsoluteMove')
        self._positionrequest.ProfileToken = self._media_profile.token

        if self._positionrequest.Position is None:
            self._positionrequest.Position = self._ptz.GetStatus({'ProfileToken': self._media_profile.token}).Position
            self._positionrequest.Position.PanTilt.space = (
                self._ptz_configuration_options.Spaces.AbsolutePanTiltPositionSpace[0].URI
            )

        if self._positionrequest.Speed is None:
            self._positionrequest.Speed = self._ptz.GetStatus({'ProfileToken': self._media_profile.token}).Position
            self._positionrequest.Speed.PanTilt.space = (
                self._ptz_configuration_options.Spaces.PanTiltSpeedSpace[0].URI
            )
        return  self

    def do_move(self):
        if self._active:
            self._ptz.Stop({'ProfileToken': self._media_profile.token})
        self._active = True
        self._ptz.AbsoluteMove(self._positionrequest)

    def move_up(self):
        if self._YNOW - self._Move <= -1:
            self._positionrequest.Position.PanTilt.y = self._YNOW
        else:
            self._positionrequest.Position.PanTilt.y = self._YNOW - self._Move
        self.do_move()

    def move_down(self):
        if self._YNOW + self._Move >= 1:
            self._positionrequest.Position.PanTilt.y = self._YNOW
        else:
            self._positionrequest.Position.PanTilt.y = self._YNOW + self._Move
        self.do_move()

    def move_right(self):
        if self._XNOW - self._Move >= -0.99:
            self._positionrequest.Position.PanTilt.x = self._XNOW - self._Move
        elif abs(self._XNOW + self._Move) >= 0.0:
            self._positionrequest.Position.PanTilt.x = abs(self._XNOW) - self._Move
        elif abs(self._XNOW) <= 0.01:
            self._positionrequest.Position.PanTilt.x = self._XNOW
        self._positionrequest.Position.PanTilt.y = self._YNOW
        self.do_move()

    def move_left(self):
        if self._XNOW + self._Move <= 1.0:
            self._positionrequest.Position.PanTilt.x = self._XNOW + self._Move
        elif self._XNOW <= 1.0 and self._XNOW > 0.99:
            self._positionrequest.Position.PanTilt.x = -self._XNOW
        elif self._XNOW < 0:
            self._positionrequest.Position.PanTilt.x = self._XNOW + self._Move
        elif self._XNOW <= -0.105556 and self._XNOW > -0.11:
            self._positionrequest.Position.PanTilt.x = self._XNOW
        self._positionrequest.Position.PanTilt.y = self._YNOW
        self.do_move()

    def move_upleft(self):
        if self._YNOW == -1:
            self._positionrequest.Position.PanTilt.y = self._YNOW
        else:
            self._positionrequest.Position.PanTilt.y = self._YNOW - self._Move
        if self._XNOW + self._Move <= 1.0:
            self._positionrequest.Position.PanTilt.x = self._XNOW + self._Move
        elif self._XNOW <= 1.0 and self._XNOW > 0.99:
            self._positionrequest.Position.PanTilt.x = -self._XNOW
        elif self._XNOW < 0:
            self._positionrequest.Position.PanTilt.x = self._XNOW + self._Move
        elif self._XNOW <= -0.105556 and self._XNOW > -0.11:
            self._positionrequest.Position.PanTilt.x = self._XNOW
        self.do_move()

    def move_upright(self):
        if self._YNOW == -1:
            self._positionrequest.Position.PanTilt.y = self._YNOW
        else:
            self._positionrequest.Position.PanTilt.y = self._YNOW - self._Move
        if self._XNOW - self._Move >= -0.99:
            self._positionrequest.Position.PanTilt.x = self._XNOW - self._Move
        elif abs(self._XNOW + self._Move) >= 0.0:
            self._positionrequest.Position.PanTilt.x = abs(self._XNOW) - self._Move
        elif abs(self._XNOW) <= 0.01:
            self._positionrequest.Position.PanTilt.x = self._XNOW
        self.do_move()

    def move_downleft(self):
        if self._YNOW - self._Move == 1:
            self._positionrequest.Position.PanTilt.y = self._YNOW
        else:
            self._positionrequest.Position.PanTilt.y = self._YNOW - self._Move
        if self._XNOW + self._Move <= 1.0:
            self._positionrequest.Position.PanTilt.x = self._XNOW + self._Move
        elif self._XNOW <= 1.0 and self._XNOW > 0.99:
            self._positionrequest.Position.PanTilt.x = -self._XNOW
        elif self._XNOW < 0:
            self._positionrequest.Position.PanTilt.x = self._XNOW + self._Move
        elif self._XNOW <= -0.105556 and self._XNOW > -0.11:
            self._positionrequest.Position.PanTilt.x = self._XNOW
        self.do_move()

    def move_downright(self):
        if self._YNOW == -1:
            self._positionrequest.Position.PanTilt.y = self._YNOW
        else:
            self._positionrequest.Position.PanTilt.y = self._YNOW - self._Move
        if self._XNOW - self._Move >= -0.99:
            self._positionrequest.Position.PanTilt.x = self._XNOW - self._Move
        elif abs(self._XNOW + self._Move) >= 0.0:
            self._positionrequest.Position.PanTilt.x = abs(self._XNOW) - self._Move
        elif abs(self._XNOW) <= 0.01:
            self._positionrequest.Position.PanTilt.x = self._XNOW
        self.do_move()

    def Zoom_in(self):
        if self._Zoom + self._Move >= 1.0:
            self._positionrequest.Position.Zoom = 1.0
        else:
            self._positionrequest.Position.Zoom = self._Zoom + self._Move
        self.do_move()

    def Zoom_out(self):
        if self._Zoom - self._Move <= 0.0:
            self._positionrequest.Position.Zoom = 0.0
        else:
            self._positionrequest.Position.Zoom = self._Zoom - self._Move
        self.do_move()


    def get_presets(self):
        if not self._ptz or not self._media_profile:
            raise RuntimeError("PTZ service or media profile is not initialized.")

        try:
            presets = self._ptz.GetPresets({"ProfileToken": self._media_profile.token})
            presets_list = [
                {
                    "Name": preset.Name,
                    "Token": preset.token,
                    "PTZPosition": {
                        "PanTilt": {
                            "x": preset.PTZPosition.PanTilt.x if preset.PTZPosition.PanTilt else None,
                            "y": preset.PTZPosition.PanTilt.y if preset.PTZPosition.PanTilt else None,
                            "space": preset.PTZPosition.PanTilt.space if preset.PTZPosition.PanTilt else None,
                        },
                        "Zoom": {
                            "x": preset.PTZPosition.Zoom.x if preset.PTZPosition.Zoom else None,
                            "space": preset.PTZPosition.Zoom.space if preset.PTZPosition.Zoom else None,
                        } if preset.PTZPosition else None,
                    },
                }
                for preset in presets
            ]
            return presets_list
        except ONVIFError as e:
            raise ONVIFError(f"Error fetching presets: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error occurred: {str(e)}")