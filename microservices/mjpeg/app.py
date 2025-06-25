#!/usr/bin/env python3
"""
FFmpeg MJPEG streaming service with integrated motion detection.
Streams video from OBS Virtual Camera with optional cat detection overlay.
"""

import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Detection module (class‑based).  Wrapper functions keep old API.
# ---------------------------------------------------------------------------

try:
    # Fix the import path - detection_service.py is in the same directory as app.py
    from lib.mjpeg_detection.detection_service import (
        process_frame_for_detection,
        init_detection,
        toggle_detection,
        get_detection_status,
        refresh_film_strip,
        create_detections_strip,
    )
    DETECTION_AVAILABLE = True
    print("Detection service imported successfully")
except ImportError as exc:  # pragma: no cover – detection optional
    print(f"Detection module not available: {exc}")
    DETECTION_AVAILABLE = False
except Exception as exc:
    print(f"Unexpected error importing detection: {exc}")
    DETECTION_AVAILABLE = False

# ---------------------------------------------------------------------------
# Flask setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Global FFmpeg state (kept minimal – could be refactored later)
# ---------------------------------------------------------------------------

FFMPEG_BIN = Path(os.getenv("FFMPEG_BIN", "/opt/homebrew/bin/ffmpeg"))

ffmpeg_process: Optional[subprocess.Popen] = None
streaming_active = False
latest_frame: Optional[bytes] = None

frame_lock = threading.Lock()

# Background threads
_reader_th: Optional[threading.Thread] = None
_stderr_th: Optional[threading.Thread] = None

# ---------------------------------------------------------------------------
# Init detection if available
# ---------------------------------------------------------------------------

if DETECTION_AVAILABLE:
    try:
        init_detection()  # spins up background worker threads and auto-enables
        print("Motion detection auto-initialized and enabled")
    except Exception as e:
        print(f"Failed to initialize detection: {e}")
        DETECTION_AVAILABLE = False

# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _detect_obs_cam() -> str:
    """Return avfoundation index of OBS Virtual Camera (fallback to '0')."""
    cmd = [str(FFMPEG_BIN), "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        for line in res.stderr.split("\n"):
            m = re.search(r"\[(\d+)\].*OBS Virtual Camera", line, re.I)
            if m:
                print(f"Found OBS Virtual Camera at index {m.group(1)}")
                return m.group(1)
        print("OBS Virtual Camera not found by name, using fallback")
    except Exception as e:
        print(f"Error finding OBS camera: {e}")
    return "0"

# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def _ffmpeg_cmd(cam: str) -> list[str]:
    return [
        str(FFMPEG_BIN),
        "-f", "avfoundation",
        "-framerate", "60",
        "-pixel_format", "nv12",
        "-i", cam,
        "-vf", "scale=1280:720:flags=fast_bilinear",
        "-f", "mjpeg",
        "-q:v", "6",
        "-threads", "1",
        "-fflags", "+nobuffer+flush_packets+genpts",
        "-flags", "+low_delay",
        "-strict", "experimental",
        "-avoid_negative_ts", "make_zero",
        "-tune", "zerolatency",
        "pipe:1",
    ]

def _start_ffmpeg():
    global ffmpeg_process, _reader_th, _stderr_th
    cam = _detect_obs_cam()
    print(f"Starting FFmpeg with camera {cam}")
    
    ffmpeg_process = subprocess.Popen(
        _ffmpeg_cmd(cam),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        preexec_fn=os.setsid,
    )
    
    _reader_th = threading.Thread(target=_read_frames, daemon=True)
    _reader_th.start()
    _stderr_th = threading.Thread(target=_read_stderr, daemon=True)
    _stderr_th.start()
    
    print(f"FFmpeg process started with PID {ffmpeg_process.pid}")

def _stop_ffmpeg():
    global ffmpeg_process, _reader_th, _stderr_th, latest_frame
    if not ffmpeg_process:
        return
    
    try:
        os.killpg(os.getpgid(ffmpeg_process.pid), signal.SIGTERM)
        ffmpeg_process.wait(timeout=5)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        try:
            os.killpg(os.getpgid(ffmpeg_process.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    
    ffmpeg_process = None
    
    # Wait for threads to finish
    if _reader_th and _reader_th.is_alive():
        _reader_th.join(timeout=3)
    if _stderr_th and _stderr_th.is_alive():
        _stderr_th.join(timeout=3)
    
    _reader_th = None
    _stderr_th = None
    
    with frame_lock:
        latest_frame = None
    
    print("Stopped FFmpeg stream")

# ---------------------------------------------------------------------------
# Reader threads
# ---------------------------------------------------------------------------

def _read_stderr():
    if not ffmpeg_process or not ffmpeg_process.stderr:
        return
    
    try:
        for line in ffmpeg_process.stderr:
            if not streaming_active:
                break
            decoded_line = line.decode(errors="ignore").strip()
            if decoded_line and any(keyword in decoded_line.lower() for keyword in 
                                   ['error', 'failed', 'warning', 'fps=', 'time=']):
                print(f"FFmpeg: {decoded_line}")
    except Exception as e:
        print(f"Error in stderr reader: {e}")

def _read_frames():
    global latest_frame
    if not ffmpeg_process or not ffmpeg_process.stdout:
        return
    
    frame_data = b""
    consecutive_empty_reads = 0
    max_empty_reads = 10
    
    while streaming_active and ffmpeg_process:
        try:
            chunk = ffmpeg_process.stdout.read(4096)
            if not chunk:
                consecutive_empty_reads += 1
                if consecutive_empty_reads > max_empty_reads:
                    print("Too many empty reads, FFmpeg may have stalled")
                    break
                time.sleep(0.001)
                continue
            else:
                consecutive_empty_reads = 0
            
            frame_data += chunk
            
            # Process frames immediately when found
            while True:
                start_pos = frame_data.find(b'\xff\xd8')
                if start_pos == -1:
                    break
                
                end_pos = frame_data.find(b'\xff\xd9', start_pos)
                if end_pos == -1:
                    break
                
                frame = frame_data[start_pos:end_pos + 2]
                frame_data = frame_data[end_pos + 2:]
                
                if len(frame) > 500:
                    # Send ALL frames to detection - let detection service handle skipping
                    if DETECTION_AVAILABLE:
                        frame = process_frame_for_detection(frame)
                    
                    with frame_lock:
                        latest_frame = frame
                
                # Prevent buffer from growing too large
                if len(frame_data) > 50000:
                    frame_data = frame_data[-25000:]
                    
        except Exception as e:
            print(f"Frame reading error: {e}")
            break
    
    print("Frame reader stopped")

# ---------------------------------------------------------------------------
# Flask helpers
# ---------------------------------------------------------------------------

def _base_url() -> str:
    if not request:
        return "http://localhost:8080"
    host = request.host.split(":")[0]
    scheme = request.scheme
    if host in ("localhost", "127.0.0.1"):
        return f"{scheme}://{host}:8080"
    else:
        # Use the host as-is (reverse proxy handles the domain)
        return f"{scheme}://{request.host}/stream"

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

# Streaming ------------------------------------------------------------------

@app.route("/stream")
def stream():
    if not streaming_active:
        _start_stream()
        time.sleep(1)
        if not streaming_active:
            return jsonify({"error": "Streaming is not active"}), 503
    
    return Response(_frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

def _frame_generator():
    no_frame_count = 0
    while streaming_active:
        with frame_lock:
            frame = latest_frame
        
        if frame:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(frame)).encode()
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )
            no_frame_count = 0
            time.sleep(1 / 60)
        else:
            no_frame_count += 1
            if no_frame_count > 120:
                print("No frames available, breaking connection")
                break
            time.sleep(0.008)

# Control --------------------------------------------------------------------

@app.route("/start", methods=["POST"])
def _start_stream():
    global streaming_active
    if streaming_active:
        return jsonify({"success": "MJPEG stream already running"}), 200
    
    try:
        streaming_active = True
        _start_ffmpeg()
        
        time.sleep(2)  # Wait for FFmpeg to start
        
        if ffmpeg_process and ffmpeg_process.poll() is None:
            response_data = {
                "active": True,
                "streaming": streaming_active,
                "camera_type": "real",
                "clients": 0,
                "stream_url": f"{_base_url()}/stream",
                "ffmpeg_pid": ffmpeg_process.pid
            }
            return jsonify({"success": "MJPEG stream started", "data": response_data}), 200
        else:
            streaming_active = False
            return jsonify({"error": "Failed to start stream"}), 500
            
    except Exception as e:
        streaming_active = False
        return jsonify({"error": f"Failed to start stream: {str(e)}"}), 500

@app.route("/stop", methods=["POST"])
def _stop_stream():
    global streaming_active
    streaming_active = False
    _stop_ffmpeg()
    return jsonify({"success": "MJPEG stream stopped"}), 200

# Status / health ------------------------------------------------------------

@app.route("/status")
def _status():
    status_data = {
        "active": streaming_active,
        "streaming": streaming_active,
        "camera_type": "real",
        "clients": 0,
        "stream_url": f"{_base_url()}/stream" if streaming_active else None,
        "ffmpeg_pid": ffmpeg_process.pid if ffmpeg_process else None,
        "error": None
    }
    
    if DETECTION_AVAILABLE:
        try:
            detection_status = get_detection_status()
            detection_status["available"] = True
            status_data["detection"] = detection_status
        except Exception as e:
            status_data["detection"] = {"available": False, "error": str(e)}
    
    return jsonify({"data": status_data}), 200

@app.route("/health")
def _health():
    health_data = {
        "status": "healthy",
        "service": "mjpeg-streaming", 
        "timestamp": time.time()
    }
    return jsonify({"data": health_data}), 200

# Detection control routes ---------------------------------------------------

@app.route("/detection/toggle", methods=["POST"])
def _det_toggle():
    if not DETECTION_AVAILABLE:
        return jsonify({"error": "Detection not available"}), 503
    
    try:
        enabled = request.json.get("enabled", True)
        result = toggle_detection(enabled)
        return jsonify({"success": f"Detection {'enabled' if result else 'disabled'}"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to toggle detection: {e}"}), 500

@app.route("/detection/status")
def _det_status():
    if not DETECTION_AVAILABLE:
        return jsonify({"data": {"available": False, "error": "Detection module not loaded"}}), 200
    
    try:
        status = get_detection_status()
        status["available"] = True
        return jsonify({"data": status}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to get status: {e}"}), 500

@app.route("/detection/logs")
def _det_logs():
    if not DETECTION_AVAILABLE:
        return jsonify({"error": "Detection not available"}), 503
    
    try:
        log_file = "motion_logs/events.jsonl"
        if not os.path.exists(log_file):
            return jsonify({"data": {"events": [], "count": 0}}), 200
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-50:] if len(lines) > 50 else lines
            
        events = []
        for line in recent_lines:
            try:
                events.append(json.loads(line.strip()))
            except (json.JSONDecodeError, ValueError):
                continue
        
        return jsonify({"data": {"events": events, "count": len(events)}}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to get logs: {e}"}), 500

@app.route("/detection/refresh-strip", methods=["POST"])
def _det_refresh():
    if not DETECTION_AVAILABLE:
        return jsonify({"error": "Detection not available"}), 503
    
    try:
        result = refresh_film_strip()
        if result.get("success"):
            return jsonify({"success": result["message"]}), 200
        else:
            return jsonify({"error": result.get("error", "Unknown error")}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to refresh detection strip: {e}"}), 500

@app.route("/detection/test-strip", methods=["POST"])
def _det_test():
    if not DETECTION_AVAILABLE:
        return jsonify({"error": "Detection not available"}), 503
    
    try:
        create_detections_strip()
        status = get_detection_status()
        
        return jsonify({
            "success": "Film strip generated successfully", 
            "data": {
                "strip_file": "motion_logs/detections_strip.jpg",  # Fixed: Changed from PNG to JPG
                "film_strip_info": status.get("film_strip", {}),
                "message": "Check the detection strip JPG image"  # Updated message
            }
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to generate test strip: {e}"}), 500

# Add FFmpeg logs route for debugging
@app.route('/ffmpeg/logs')
def _ffmpeg_logs():
    """Get recent FFmpeg logs"""
    try:
        with open('/tmp/ffmpeg_output.log', 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-100:] if len(lines) > 100 else lines
            return jsonify({
                "logs": ''.join(recent_lines),
                "total_lines": len(lines)
            }), 200
    except FileNotFoundError:
        return jsonify({"logs": "No logs available yet", "total_lines": 0}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to read logs: {e}"}), 500

###############################################################################
# Dev entry-point (use gunicorn in production)
###############################################################################

if __name__ == "__main__":
    print("Starting FFmpeg MJPEG service with detection on port 8080")
    if DETECTION_AVAILABLE:
        print("Motion detection auto-enabled - cat detection and smart cropping active")
        print("Cropped detections will be saved to: motion_logs/crops/")
        print("Detection strip will be saved to: motion_logs/detections_strip.jpg")
    else:
        print("Detection module not available - streaming only")
        print("Cropped detections will be saved to: motion_logs/crops/")
        print("Detection strip will be saved to: motion_logs/detections_strip.jpg")
    app.run(host="0.0.0.0", port=8081, debug=False)
