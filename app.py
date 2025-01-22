import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from onvif.exceptions import ONVIFError
from threading import Thread, Lock
import subprocess
from movement_utils import ONVIFCameraManager, ONVIFCameraInstance
from obs_client_module import OBSWebSocketClient
from shared_config import CAMERA_CONFIGS, OBS_PASSWORD, OBS_URL, STREAMS
# Flask App Setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Global Variables
script_status = {"running": False, "exit_code": None}
lock = Lock()
process = None


obs_client = OBSWebSocketClient(OBS_URL, OBS_PASSWORD)
log_file_path="motion_detection.log"

if not (obs_client.connect()):
    logging.error("Could not connect to OBS WebSocket")



camera_manager = ONVIFCameraManager()
camera_manager.initialize_cameras(CAMERA_CONFIGS)

@app.route("/ptz/switch_camera", methods=["POST"])
def switch_camera():
    data = request.json
    nickname = data.get("nickname")
    if not nickname:
        return jsonify({"error": "Nickname is required"}), 400
    try:
        def operation(camera:ONVIFCameraInstance):
            return {"success": f"Switched to {nickname}, {camera.host}"}
        result = camera_manager.perform_operation(nickname, operation)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ptz/status", methods=["GET"])
def status():
    nickname = request.args.get("nickname")
    if not nickname:
        return jsonify({"error": "Nickname is required"}), 400
    try:
        def operation(camera:ONVIFCameraInstance):
            return camera.get_status()
        result = camera_manager.perform_operation(nickname, operation)
        return jsonify(result), 200
    except ONVIFError as e:
        return jsonify({"error": f"ONVIFError: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ptz/presets", methods=["GET"])
def get_all_presets():
    nickname = request.args.get("nickname")
    if not nickname:
        return jsonify({"error": "Nickname is required"}), 400
    try:
        def operation(camera:ONVIFCameraInstance):
            return {"presets": camera.get_presets()}
        result = camera_manager.perform_operation(nickname, operation)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ptz/goto_preset", methods=["POST"])
def goto_preset():
    data = request.json
    nickname = request.args.get("nickname")
    preset_token = data.get("presetToken")
    if not nickname or not preset_token:
        return jsonify({"error": "Both nickname and presetToken are required"}), 400
    try:
        def operation(camera:ONVIFCameraInstance):
            camera._ptz.GotoPreset({
                "ProfileToken": camera._media_profile.token,
                "PresetToken": preset_token,
            })
            return {"message": f"Moved to preset '{preset_token}' successfully!"}
        result = camera_manager.perform_operation(nickname, operation)
        return jsonify(result), 200
    except ONVIFError as e:
        return jsonify({"error": f"ONVIFError: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ptz/move", methods=["POST"])
def move():
    data = request.json
    nickname = request.args.get("nickname")
    direction = data.get("direction")
    if not nickname or not direction:
        return jsonify({"error": "Both nickname and direction are required"}), 400
    try:
        def operation(camera:ONVIFCameraInstance):
            movement_map = {
                "up": lambda: camera.move_up(),
                "down": lambda: camera.move_down(),
                "right": lambda: camera.move_left(),
                "left": lambda: camera.move_right(),
                "upright": lambda: camera.move_upleft(),
                "upleft": lambda: camera.move_upright(),
                "downright": lambda: camera.move_downleft(),
                "downleft": lambda: camera.move_downright(),
                "zoomin": lambda: camera.Zoom_in(),
                "zoomout": lambda: camera.Zoom_out(),
            }
            move_fn = movement_map.get(direction.lower())
            if not move_fn:
                raise ValueError("Invalid direction")
            camera.get_status()
            move_fn()
            return {"success": f"Moved {direction}"}
        result = camera_manager.perform_operation(nickname, operation)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/start_detection", methods=["POST"])
def start_detection():
    """
    Start the detection process.
    """
    global process

    if process and process.poll() is None:
        return jsonify({"error": "Detection script is already running"}), 400

    # Clear the log file on start
    with open(log_file_path, "w") as log_file:
        log_file.truncate(0)

    try:

        with open(log_file_path, "a") as log_file:
            process = subprocess.Popen(
                ["python", "-c", "import detection_to_transform; detection_to_transform.init()"],          
                stdout=log_file,
                stderr=log_file,
            )
        return jsonify({"success": "Detection script started"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to start script: {str(e)}"}), 500
@app.route("/stop_detection", methods=["POST"])
def stop_detection():
    """
    Stop the detection process.
    """
    global process

    if not process or process.poll() is not None:
        return jsonify({"error": "No detection script is running"}), 400

    try:
        process.terminate()
        process.wait()

        # Clear the log file on stop
        with open(log_file_path, "w") as log_file:
            log_file.truncate(0)

        return jsonify({"success": "Detection script stopped"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to stop script: {str(e)}"}), 500
    
@app.route("/fetch_logs", methods=["GET"])
def fetch_logs():
    """
    Fetch the logs from the detection script.
    """
    try:
        with open("motion_detection.log", "r") as log_file:
            logs = log_file.read()
        return jsonify({"logs": logs}), 200
    except FileNotFoundError:
        return jsonify({"logs": "Log file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/poll_detection_status", methods=["GET"])
def poll_detection_status():
    """
    Poll the status of the detection_to_transform script.
    """
    with lock:
        return jsonify(script_status), 200
# =========================
# Route: Get All Scenes
# =========================
@app.route("/obs/scenes", methods=["GET"])
def get_all_scenes():
    try:
        obs_client.retrieve_scene_sources("Mosaic")
        return jsonify({"scenes": STREAMS}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/obs/current_scene", methods=["GET"])
def get_current_scene():
    try:
        current_scene = obs_client()
        return jsonify({"current_scene": current_scene}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Route: Switch Scene
# =========================
@app.route("/obs/switch_scene", methods=["POST"])
def switch_scene():
    data = request.json
    scene_name = data.get("scene_name")

    if not scene_name:
        return jsonify({"error": "Scene name is required"}), 400

    try:
        obs_client.switch_scene(scene_name)
        return jsonify({"success": f"Switched to scene '{scene_name}'"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Route: Apply Transformation
# =========================
@app.route("/obs/transform", methods=["POST"])
def apply_transformation():
    data = request.json
    obs_client.retrieve_scene_sources("Mosaic")
    transformation_type = data.get("type")
    active_source = data.get("active_source")

    if transformation_type not in ["grid", "highlight"]:
        return jsonify({"error": "Invalid transformation type. Use 'grid' or 'highlight'"}), 400
    if transformation_type == "highlight" and not active_source:
        return jsonify({"error": "Active source is required for 'highlight' transformation"}), 400

    try:
        if transformation_type == "grid":
            # Apply 2x2 grid layout
            obs_client.update_obs_layout(scene_name="Mosaic", active_source=None)
        elif transformation_type == "highlight":
            # Apply 75% main + 25% others (stacked) layout
            obs_client.update_obs_layout(scene_name="Mosaic", active_source=active_source)
        return jsonify({"success": f"Applied transformation: {transformation_type}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)