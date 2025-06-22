"""
Motion detection control routes.
"""
from flask import Blueprint, request, jsonify
from utils.response import success_response, error_response, data_response

detection_bp = Blueprint('detection', __name__)


def init_detection_routes(detection_service):
    """Initialize detection routes with detection service"""
    
    @detection_bp.route("/start_detection", methods=["POST"])
    def start_detection():
        """Start the detection process"""
        try:
            result = detection_service.start_detection()
            if "error" in result:
                return jsonify(result), 400
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(f"Failed to start script: {str(e)}")), 500

    @detection_bp.route("/stop_detection", methods=["POST"])
    def stop_detection():
        """Stop the detection process"""
        try:
            result = detection_service.stop_detection()
            if "error" in result:
                return jsonify(result), 400
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(f"Failed to stop script: {str(e)}")), 500
        
    @detection_bp.route("/fetch_logs", methods=["GET"])
    def fetch_logs():
        """Fetch the logs from the detection script"""
        try:
            result = detection_service.fetch_logs()
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500

    @detection_bp.route("/poll_detection_status", methods=["GET"])
    def poll_detection_status():
        """Poll the status of the detection_to_transform script"""
        try:
            result = detection_service.get_status()
            return jsonify(result), 200
        except Exception as e:
            return jsonify(error_response(str(e))), 500

    return detection_bp
