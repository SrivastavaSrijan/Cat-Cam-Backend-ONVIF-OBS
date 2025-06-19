import logging
import time
import os
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

print("🚀 Starting Enhanced ONVIF Camera Control System...")

if not (obs_client.connect()):
    logging.error("Could not connect to OBS WebSocket")
    print("⚠️  OBS WebSocket connection failed - OBS features will not be available")
else:
    print("✅ OBS WebSocket connected successfully")

print("\n📹 Initializing cameras...")
camera_manager = ONVIFCameraManager()
camera_manager.initialize_cameras(CAMERA_CONFIGS)

# Check how many cameras were successfully initialized
initialized_count = len(camera_manager.cameras)
total_count = len(CAMERA_CONFIGS)
print(f"\n📊 Camera initialization complete: {initialized_count}/{total_count} cameras online")

if initialized_count == 0:
    print("⚠️  WARNING: No cameras were successfully initialized!")
    print("   - Check camera network connectivity")
    print("   - Verify ONVIF credentials in shared_config.py")
    print("   - Ensure cameras support ONVIF protocol")
else:
    print("✅ Camera system ready!")
    for nickname in camera_manager.cameras.keys():
        print(f"   - {nickname}: Online")

print("\n🌐 Starting Flask web server...")
print("   API will be available at: http://localhost:5000")
print("   Use Ctrl+C to stop the server")
print("-" * 50)

@app.route("/ptz/switch_camera", methods=["POST"])
def switch_camera():
    data = request.json or {}
    nickname = data.get("nickname")
    if not nickname:
        return jsonify({"error": "Nickname is required"}), 400
    try:
        def operation(camera: ONVIFCameraInstance):
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
        def operation(camera: ONVIFCameraInstance):
            # Use the safe status method first
            safe_status = camera.get_status_safe()
            if not safe_status.get("online", False):
                return safe_status
            # If camera is online, get detailed status
            try:
                return camera.get_status()
            except Exception:
                # Return the safe status if detailed status fails
                return safe_status
        
        result = camera_manager.perform_operation(nickname, operation)
        return jsonify(result), 200
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
    data = request.json or {}
    nickname = request.args.get("nickname")
    preset_token = data.get("presetToken")
    if not nickname or not preset_token:
        return jsonify({"error": "Both nickname and presetToken are required"}), 400
    try:
        def operation(camera: ONVIFCameraInstance):
            return camera.goto_preset(preset_token)
        
        result = camera_manager.perform_operation(nickname, operation)
        return jsonify(result), 200
    except ONVIFError as e:
        return jsonify({"error": f"ONVIFError: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ptz/move", methods=["POST"])
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
        current_scene = obs_client.get_current_scene()
        return jsonify({"current_scene": current_scene}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Route: Switch Scene
# =========================
@app.route("/obs/switch_scene", methods=["POST"])
def switch_scene():
    data = request.json or {}
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
    data = request.json or {}
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
    
# =========================
# Route: Reconnect to OBS
# =========================
@app.route("/obs/reconnect", methods=["POST"])
def reconnect_to_obs():
    try:
        obs_client.close();
        # Wait for 1 second before reconnecting
        time.sleep(5);
        obs_client.connect();
        return jsonify({"success": "Reconnected to OBS WebSocket"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ptz/movement_speed", methods=["POST"])
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
        
        result = camera_manager.perform_operation(nickname, operation)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ptz/imaging", methods=["GET"])
def get_imaging_settings():
    nickname = request.args.get("nickname")
    if not nickname:
        return jsonify({"error": "Nickname is required"}), 400
    
    try:
        def operation(camera: ONVIFCameraInstance):
            return camera.get_imaging_settings()
        
        result = camera_manager.perform_operation(nickname, operation)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ptz/night_mode", methods=["POST"])
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
        
        result = camera_manager.perform_operation(nickname, operation)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ptz/cameras", methods=["GET"])
def get_cameras():
    """Get list of all cameras with their status"""
    try:
        cameras = []
        for nickname, camera in camera_manager.cameras.items():
            if camera:
                safe_status = camera.get_status_safe()
                cameras.append({
                    "nickname": nickname,
                    "host": safe_status.get("host", "unknown"),
                    "port": getattr(camera, '_port', 554),  # Default ONVIF port
                    "status": "online" if safe_status.get("online", False) else "offline",
                    "error": safe_status.get("error") if not safe_status.get("online", False) else None
                })
            else:
                cameras.append({
                    "nickname": nickname,
                    "host": "unknown",
                    "port": 554,
                    "status": "offline",
                    "error": "Camera not initialized"
                })
        
        return jsonify({"cameras": cameras}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ptz/continuous_move", methods=["POST"])
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
        return jsonify({"error": f"Invalid direction: {direction}"}), 400
    
    pan_speed, tilt_speed = direction_map[direction]
    
    try:
        def operation(camera: ONVIFCameraInstance):
            return camera.continuous_move(pan_speed, tilt_speed)
        
        result = camera_manager.perform_operation(nickname, operation)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ptz/stop", methods=["POST"])
def stop_movement():
    """Stop all PTZ movement using clean implementation"""
    nickname = request.args.get("nickname")
    if not nickname:
        return jsonify({"error": "Nickname is required"}), 400
    
    try:
        def operation(camera: ONVIFCameraInstance):
            return camera.stop_movement()
        
        result = camera_manager.perform_operation(nickname, operation)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# Route: Get Current Transformation
# =========================
@app.route("/obs/current_transformation", methods=["GET"])
def get_current_highlighted_source():
    """Get the currently highlighted source in the Mosaic scene"""
    try:
        scene_name = request.args.get("scene_name", "Mosaic")
        print(obs_client.current_highlighted_source)
        # Check if OBS client has a current highlighted source stored
        if hasattr(obs_client, 'current_highlighted_source') and obs_client.current_highlighted_source:
            return jsonify({
                "success": True,
                "scene_name": scene_name,
                "highlighted_source": obs_client.current_highlighted_source,
                "layout_mode": "highlight"
            }), 200
        else:
            # No highlighted source, assume grid layout
            return jsonify({
                "success": True,
                "scene_name": scene_name,
                "highlighted_source": None,
                "layout_mode": "grid"
            }), 200
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# =========================
# Route: MJPEG Stream
# =========================
import cv2
from flask import Response

@app.route("/mjpeg")
def mjpeg_stream():
    def generate():
        # Read OBS Virtual Camera device index from environment variable
        VIDEO_DEVICE_INDEX = int(os.getenv("VIDEO_DEVICE_INDEX", "1"))
        cap = cv2.VideoCapture(VIDEO_DEVICE_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Could not open video source")

        while True:
            success, frame = cap.read()
            if not success:
                continue
            _, jpeg = cv2.imencode(".jpg", frame)
            frame_bytes = jpeg.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================
# Route: Start Fullscreen Projector
# =========================
@app.route("/obs/projector/start", methods=["POST"])
def start_projector():
    data = request.json or {}
    scene_name = data.get("scene_name", "Mosaic")
    monitor_index = data.get("monitor_index", 0)

    try:
        result = obs_client.start_fullscreen_projector(scene_name, monitor_index)
        return jsonify(result), 200 if "success" in result else 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# Route: Close Projector
# =========================
@app.route("/obs/projector/close", methods=["POST"])
def close_projector():
    data = request.json or {}
    projector_type = data.get("projector_type", "source")

    try:
        result = obs_client.close_projector(projector_type)
        return jsonify(result), 200 if "success" in result else 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# Route: Start Virtual Camera
# =========================
@app.route("/obs/virtual_camera/start", methods=["POST"])
def start_virtual_camera():
    try:
        result = obs_client.start_virtual_camera()
        return jsonify(result), 200 if "success" in result else 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# Route: Stop Virtual Camera
# =========================
@app.route("/obs/virtual_camera/stop", methods=["POST"])
def stop_virtual_camera():
    try:
        result = obs_client.stop_virtual_camera()
        return jsonify(result), 200 if "success" in result else 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# Route: Get Virtual Camera Status
# =========================
@app.route("/obs/virtual_camera/status", methods=["GET"])
def get_virtual_camera_status():
    try:
        result = obs_client.get_virtual_camera_status()
        return jsonify(result), 200 if "success" in result else 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)