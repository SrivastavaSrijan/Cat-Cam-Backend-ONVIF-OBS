"""
OBS WebSocket control routes.
"""
from flask import Blueprint, request, jsonify
from utils.response import success_response, error_response, data_response

obs_bp = Blueprint('obs', __name__, url_prefix='/obs')


def init_obs_routes(obs_service):
    """Initialize OBS routes with OBS service"""
    
    @obs_bp.route("/scenes", methods=["GET"])
    def get_all_scenes():
        try:
            result = obs_service.get_scenes()
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500
        
    @obs_bp.route("/current_scene", methods=["GET"])
    def get_current_scene():
        try:
            result = obs_service.get_current_scene()
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500

    @obs_bp.route("/switch_scene", methods=["POST"])
    def switch_scene():
        data = request.json or {}
        scene_name = data.get("scene_name")

        if not scene_name:
            return jsonify(error_response("scene_name is required")), 400

        try:
            result = obs_service.switch_scene(scene_name)
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500

    @obs_bp.route("/transform", methods=["POST"])
    def apply_transformation():
        data = request.json or {}
        transformation_type = data.get("type")
        active_source = data.get("active_source")

        if transformation_type not in ["grid", "highlight"]:
            return jsonify(error_response("Invalid transformation type")), 400
        if transformation_type == "highlight" and not active_source:
            return jsonify(error_response("active_source required for highlight transformation")), 400

        try:
            result = obs_service.apply_transformation(transformation_type, active_source)
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500
        
    @obs_bp.route("/reconnect", methods=["POST"])
    def reconnect_to_obs():
        try:
            result = obs_service.reconnect()
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500

    @obs_bp.route("/current_transformation", methods=["GET"])
    def get_current_highlighted_source():
        """Get the currently highlighted source in the Mosaic scene"""
        try:
            scene_name = request.args.get("scene_name", "Mosaic")
            result = obs_service.get_current_transformation(scene_name)
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500

    @obs_bp.route("/projector/start", methods=["POST"])
    def start_projector():
        data = request.json or {}
        source_name = data.get("source_name", "Mosaic")
        monitor_index = data.get("monitor_index", 0)

        try:
            result = obs_service.start_fullscreen_projector(source_name, monitor_index)
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500

    @obs_bp.route("/projector/close", methods=["POST"])
    def close_projector():
        data = request.json or {}
        projector_type = data.get("projector_type", "source")

        try:
            result = obs_service.close_projector(projector_type)
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500

    @obs_bp.route("/virtual_camera/start", methods=["POST"])
    def start_virtual_camera():
        try:
            result = obs_service.start_virtual_camera()
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500

    @obs_bp.route("/virtual_camera/stop", methods=["POST"])
    def stop_virtual_camera():
        try:
            result = obs_service.stop_virtual_camera()
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500

    @obs_bp.route("/virtual_camera/status", methods=["GET"])
    def get_virtual_camera_status():
        try:
            result = obs_service.get_virtual_camera_status()
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500

    return obs_bp
