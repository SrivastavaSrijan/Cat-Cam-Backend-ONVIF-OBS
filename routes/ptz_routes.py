"""
PTZ camera control routes.
"""
from flask import Blueprint, request, jsonify
from onvif.exceptions import ONVIFError
from lib.camera.onvif_manager import ONVIFCameraInstance

ptz_bp = Blueprint('ptz', __name__, url_prefix='/ptz')


def init_ptz_routes(camera_service):
    """Initialize PTZ routes with camera service"""
    
    @ptz_bp.route("/switch_camera", methods=["POST"])
    def switch_camera():
        data = request.json or {}
        nickname = data.get("nickname")
        if not nickname:
            return jsonify({"error": "Nickname is required"}), 400
        try:
            def operation(camera: ONVIFCameraInstance):
                return {"success": f"Switched to {nickname}, {camera.host}"}
            result = camera_service.perform_operation(nickname, operation)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @ptz_bp.route("/status", methods=["GET"])
    def status():
        nickname = request.args.get("nickname")
        if not nickname:
            return jsonify({"error": "Nickname is required"}), 400
        try:
            def operation(camera: ONVIFCameraInstance):
                # Use the safe status method first
                safe_status = camera.get_status_safe()
                if not safe_status.get("online", False):
                    return safe_status
                
                # If camera is online, get detailed status
                try:
                    return camera.get_status()
                except Exception:
                    return safe_status
            
            result = camera_service.perform_operation(nickname, operation)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @ptz_bp.route("/presets", methods=["GET"])
    def get_all_presets():
        nickname = request.args.get("nickname")
        if not nickname:
            return jsonify({"error": "Nickname is required"}), 400
        try:
            def operation(camera: ONVIFCameraInstance):
                return {"presets": camera.get_presets()}
            result = camera_service.perform_operation(nickname, operation)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @ptz_bp.route("/goto_preset", methods=["POST"])
    def goto_preset():
        data = request.json or {}
        nickname = request.args.get("nickname")
        preset_token = data.get("presetToken")
        if not nickname or not preset_token:
            return jsonify({"error": "Both nickname and presetToken are required"}), 400
        try:
            def operation(camera: ONVIFCameraInstance):
                return camera.goto_preset(preset_token)
            
            result = camera_service.perform_operation(nickname, operation)
            return jsonify(result), 200
        except ONVIFError as e:
            return jsonify({"error": f"ONVIFError: {str(e)}"}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @ptz_bp.route("/move", methods=["POST"])
    def move():
        """Quick single tap movement using clean PTZ implementation"""
        data = request.json or {}
        nickname = request.args.get("nickname")
        direction = data.get("direction")
        velocity_factor = data.get("velocity_factor", 0.15)  # Small for single taps
        
        if not nickname or not direction:
            return jsonify({"error": "Both nickname and direction are required"}), 400
        
        try:
            def operation(camera: ONVIFCameraInstance):
                return camera.move_direction(direction, velocity_factor)
            
            result = camera_service.perform_operation(nickname, operation)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @ptz_bp.route("/movement_speed", methods=["POST"])
    def set_movement_speed():
        data = request.json or {}
        nickname = request.args.get("nickname")
        pan_tilt_speed = data.get("pan_tilt_speed", 0.2)
        zoom_speed = data.get("zoom_speed", 0.1)
        
        if not nickname:
            return jsonify({"error": "Nickname is required"}), 400
        
        # Validate speed values
        try:
            pan_tilt_speed = float(pan_tilt_speed)
            zoom_speed = float(zoom_speed)
            pan_tilt_speed = max(0.01, min(1.0, pan_tilt_speed))
            zoom_speed = max(0.01, min(1.0, zoom_speed))
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid speed values"}), 400
        
        try:
            def operation(camera: ONVIFCameraInstance):
                return camera.set_movement_speed(pan_tilt_speed, zoom_speed)
            
            result = camera_service.perform_operation(nickname, operation)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @ptz_bp.route("/imaging", methods=["GET"])
    def get_imaging_settings():
        nickname = request.args.get("nickname")
        if not nickname:
            return jsonify({"error": "Nickname is required"}), 400
        
        try:
            def operation(camera: ONVIFCameraInstance):
                return camera.get_imaging_settings()
            
            result = camera_service.perform_operation(nickname, operation)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @ptz_bp.route("/night_mode", methods=["POST"])
    def set_night_mode():
        data = request.json or {}
        nickname = request.args.get("nickname")
        enable = data.get("enable", True)
        
        if not nickname:
            return jsonify({"error": "Nickname is required"}), 400
        
        # Validate enable parameter
        if not isinstance(enable, bool):
            try:
                enable = str(enable).lower() in ['true', '1', 'yes', 'on']
            except:
                enable = True
        
        try:
            def operation(camera: ONVIFCameraInstance):
                return camera.set_night_mode(enable)
            
            result = camera_service.perform_operation(nickname, operation)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @ptz_bp.route("/cameras", methods=["GET"])
    def get_cameras():
        """Get list of all cameras with their status"""
        try:
            cameras = camera_service.get_camera_list()
            return jsonify({"cameras": cameras}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @ptz_bp.route("/continuous_move", methods=["POST"])
    def continuous_move():
        """Start continuous movement using clean PTZ implementation"""
        data = request.json or {}
        nickname = request.args.get("nickname")
        direction = data.get("direction")
        speed = data.get("speed", 0.6)  # Faster for continuous
        
        if not nickname or not direction:
            return jsonify({"error": "Both nickname and direction are required"}), 400
        
        direction_map = {
            "up": (0.0, speed),
            "down": (0.0, -speed),
            "left": (-speed, 0.0),
            "right": (speed, 0.0),
        }
        
        if direction not in direction_map:
            return jsonify({"error": "Invalid direction"}), 400
        
        pan_speed, tilt_speed = direction_map[direction]
        
        try:
            def operation(camera: ONVIFCameraInstance):
                return camera.continuous_move(pan_speed, tilt_speed)
            
            result = camera_service.perform_operation(nickname, operation)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @ptz_bp.route("/stop", methods=["POST"])
    def stop_movement():
        """Stop all PTZ movement using clean implementation"""
        nickname = request.args.get("nickname")
        if not nickname:
            return jsonify({"error": "Nickname is required"}), 400
        
        try:
            def operation(camera: ONVIFCameraInstance):
                return camera.stop_movement()
            
            result = camera_service.perform_operation(nickname, operation)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return ptz_bp
