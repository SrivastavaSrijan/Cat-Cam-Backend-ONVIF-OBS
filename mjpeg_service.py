#!/usr/bin/env python3
"""
Simple MJPEG Streaming Microservice

A lightweight Flask service that manages MJPEG camera streaming.
Designed to run on the host machine to avoid Docker camera access issues.
"""

import cv2
import os
import sys
import time
import signal
import logging
import threading
import subprocess
from contextlib import contextmanager
from flask import Flask, jsonify, Response
from flask_cors import CORS
import numpy as np
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global state
streaming_state = {
    'active': False,
    'camera': None,
    'use_mock': False,
    'clients': 0,
    'start_time': None,
    'error': None
}
state_lock = threading.Lock()

class CameraManager:
    """Manages camera access with fallback to mock"""
    
    def __init__(self):
        self.cap = None
        self.use_mock = False
        
    def get_camera(self):
        """Try to get a real camera, fallback to mock"""
        VIDEO_DEVICE_INDEX = int(os.getenv("VIDEO_DEVICE_INDEX", "1"))
        
        # Try real cameras first
        for device_idx in [VIDEO_DEVICE_INDEX, 0, 1, 2]:
            cap = None
            try:
                cap = cv2.VideoCapture(device_idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        logger.info(f"Using real camera device {device_idx}")
                        # Configure camera
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        return cap, False
                    else:
                        cap.release()
            except Exception as e:
                logger.warning(f"Failed to open device {device_idx}: {e}")
                if cap:
                    cap.release()
        
        # No real camera found, use mock
        logger.info("No real camera found, using mock camera")
        return None, True
    
    def create_mock_frame(self):
        """Create a mock camera frame"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [30, 30, 30]  # Dark gray background
        
        # Add timestamp and labels
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, "Mock Camera Feed", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, timestamp, (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add moving circle
        t = time.time()
        x = int(320 + 200 * np.sin(t))
        y = int(240 + 100 * np.cos(t))
        cv2.circle(frame, (x, y), 30, (255, 0, 0), -1)
        
        return frame
    
    def get_frame(self):
        """Get a frame from camera or mock"""
        if self.use_mock:
            return True, self.create_mock_frame()
        elif self.cap:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Camera read failed, switching to mock")
                self.release()
                self.use_mock = True
                return True, self.create_mock_frame()
            return ret, frame
        else:
            return True, self.create_mock_frame()
    
    def start(self):
        """Initialize camera"""
        self.cap, self.use_mock = self.get_camera()
        return True
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None

camera_manager = CameraManager()

def generate_frames():
    """Generate MJPEG frames"""
    global streaming_state
    
    with state_lock:
        streaming_state['clients'] += 1
        logger.info(f"Client connected. Total clients: {streaming_state['clients']}")
    
    try:
        while streaming_state['active']:
            ret, frame = camera_manager.get_frame()
            
            if not ret or frame is None:
                logger.error("Failed to get frame")
                time.sleep(0.1)
                continue
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                   frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        logger.error(f"Error generating frames: {e}")
    finally:
        with state_lock:
            streaming_state['clients'] -= 1
            logger.info(f"Client disconnected. Total clients: {streaming_state['clients']}")

def is_port_in_use(port):
    """Check if a port is in use"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    except:
        return False

def kill_port_processes(port):
    """Kill processes using the specified port"""
    try:
        result = subprocess.run(['lsof', '-ti', f':{port}'], 
                               capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    logger.info(f"Killed process {pid} using port {port}")
                except (ProcessLookupError, ValueError):
                    pass
            time.sleep(2)
    except Exception as e:
        logger.warning(f"Failed to kill processes on port {port}: {e}")

# =========================
# API Routes
# =========================

@app.route('/stream')
def video_feed():
    """MJPEG video stream endpoint"""
    if not streaming_state['active']:
        return jsonify({"error": "Streaming is not active"}), 503
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start', methods=['POST'])
def start_stream():
    """Start the MJPEG stream"""
    global streaming_state
    
    with state_lock:
        if streaming_state['active']:
            return jsonify({
                "error": "Stream is already active",
                "status": "running"
            }), 400
        
        try:
            # Initialize camera
            if not camera_manager.start():
                return jsonify({"error": "Failed to initialize camera"}), 500
            
            # Update state
            streaming_state.update({
                'active': True,
                'camera': camera_manager,
                'use_mock': camera_manager.use_mock,
                'clients': 0,
                'start_time': time.time(),
                'error': None
            })
            
            camera_type = "Mock" if camera_manager.use_mock else "Real"
            logger.info(f"MJPEG stream started with {camera_type} camera")
            
            return jsonify({
                "success": "MJPEG stream started",
                "camera_type": camera_type.lower(),
                "stream_url": f"http://localhost:{os.getenv('MJPEG_PORT', 8080)}/stream",
                "start_time": streaming_state['start_time']
            }), 200
            
        except Exception as e:
            streaming_state['error'] = str(e)
            logger.error(f"Failed to start stream: {e}")
            return jsonify({"error": f"Failed to start stream: {str(e)}"}), 500

@app.route('/stop', methods=['POST'])
def stop_stream():
    """Stop the MJPEG stream"""
    global streaming_state
    
    with state_lock:
        if not streaming_state['active']:
            return jsonify({
                "error": "Stream is not active",
                "status": "stopped"
            }), 400
        
        try:
            # Release camera resources
            camera_manager.release()
            
            # Update state
            streaming_state.update({
                'active': False,
                'camera': None,
                'use_mock': False,
                'clients': 0,
                'start_time': None,
                'error': None
            })
            
            logger.info("MJPEG stream stopped")
            
            return jsonify({
                "success": "MJPEG stream stopped"
            }), 200
            
        except Exception as e:
            streaming_state['error'] = str(e)
            logger.error(f"Failed to stop stream: {e}")
            return jsonify({"error": f"Failed to stop stream: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get current stream status"""
    with state_lock:
        uptime = None
        if streaming_state['active'] and streaming_state['start_time']:
            uptime = time.time() - streaming_state['start_time']
        
        return jsonify({
            "active": streaming_state['active'],
            "camera_type": "mock" if streaming_state['use_mock'] else "real",
            "clients": streaming_state['clients'],
            "uptime_seconds": uptime,
            "stream_url": f"http://localhost:{os.getenv('MJPEG_PORT', 8080)}/stream" if streaming_state['active'] else None,
            "error": streaming_state['error']
        }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "mjpeg-streaming",
        "timestamp": time.time()
    }), 200

# =========================
# Signal Handlers
# =========================

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received shutdown signal {signum}")
    
    # Clean shutdown
    with state_lock:
        if streaming_state['active']:
            camera_manager.release()
            streaming_state['active'] = False
    
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# =========================
# Main
# =========================

if __name__ == '__main__':
    port = int(os.getenv('MJPEG_PORT', 8080))
    
    # Check if port is in use and clean it if needed
    if is_port_in_use(port):
        logger.warning(f"Port {port} is in use, attempting to clean up...")
        kill_port_processes(port)
        time.sleep(3)
    
    logger.info(f"🎥 Starting MJPEG Streaming Service on port {port}")
    logger.info(f"   Health check: http://localhost:{port}/health")
    logger.info(f"   Status: http://localhost:{port}/status")
    logger.info(f"   Stream: http://localhost:{port}/stream")
    logger.info("   Use Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        sys.exit(1)
