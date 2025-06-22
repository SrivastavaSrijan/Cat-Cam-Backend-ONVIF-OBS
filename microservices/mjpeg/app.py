#!/usr/bin/env python3
"""
S        # FFmpeg command for OBS Virtual Camera - 60fps, better quality
        cmd = [
            'ffmpeg',
            '-f', 'avfoundation',
            '-framerate', '60',  # Keep 60fps as required
            '-pixel_format', 'nv12',  # Better pixel format for OBS Virtual Camera
            '-i', '0',  # Device 0 ONLY
            '-vf', 'scale=640:360:flags=lanczos',  # Better scaling, no fps filter
            '-f', 'mjpeg',
            '-q:v', '3',  # Much better quality
            '-threads', '4',  # Multi-threading for better performance
            'pipe:1'
        ]MJPEG Streaming Service
Uses FFmpeg directly for camera access - no OpenCV bullshit
"""

import subprocess
import threading
import time
from flask import Flask, Response, jsonify
from flask_cors import CORS
import queue
import copy

app = Flask(__name__)
CORS(app)

# Global FFmpeg process
ffmpeg_process = None
streaming_active = False
frame_buffer = queue.Queue(maxsize=10)  # Buffer for frames
frame_reader_thread = None
latest_frame = None
frame_lock = threading.Lock()

def read_frames_from_ffmpeg():
    """Read frames from FFmpeg and buffer them for multiple clients"""
    global ffmpeg_process, streaming_active, latest_frame
    
    frame_data = b''
    
    while streaming_active and ffmpeg_process and ffmpeg_process.stdout:
        try:
            # Read data from FFmpeg
            chunk = ffmpeg_process.stdout.read(4096)
            if not chunk:
                print("No more data from FFmpeg")
                break
                
            frame_data += chunk
            
            # Look for JPEG start and end markers
            while True:
                # Find JPEG start marker (FFD8)
                start_pos = frame_data.find(b'\xff\xd8')
                if start_pos == -1:
                    break
                    
                # Find JPEG end marker (FFD9) after the start
                end_pos = frame_data.find(b'\xff\xd9', start_pos)
                if end_pos == -1:
                    # Incomplete frame, wait for more data
                    break
                    
                # Extract complete JPEG frame
                jpeg_frame = frame_data[start_pos:end_pos + 2]
                frame_data = frame_data[end_pos + 2:]
                
                # Store latest frame for all clients
                with frame_lock:
                    latest_frame = jpeg_frame
                    
        except Exception as e:
            print(f"Frame reading error: {e}")
            break
    
    print("Frame reader stopped")

def start_ffmpeg_stream():
    """Start FFmpeg MJPEG stream - ONLY device 0"""
    global ffmpeg_process, streaming_active
    
    if ffmpeg_process is not None:
        print("FFmpeg already running, skipping start")
        return
    
    try:
        # FFmpeg command specifically for OBS Virtual Camera - no color flashes
        cmd = [
            'ffmpeg',
            '-f', 'avfoundation',
            '-framerate', '60',  # Lower framerate for OBS stability
            '-pixel_format', 'nv12',  # Better pixel format for OBS Virtual Camera
            '-i', '0',  # Device 0 ONLY
            '-vf', 'scale=1280:720:flags=lanczos,fps=30',  # Better scaling, consistent fps
            '-f', 'mjpeg',
            '-q:v', '6',  # Balanced quality/performance
            '-threads', '4',  # Multi-threading for better performance
            'pipe:1'
        ]
        
        print(f"Starting FFmpeg with device 0: {' '.join(cmd)}")
        
        ffmpeg_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        # Wait and check if process is alive
        print(f"FFmpeg process started with PID {ffmpeg_process.pid}, waiting 3 seconds...")
        time.sleep(3)
        
        poll_result = ffmpeg_process.poll()
        print(f"FFmpeg poll result: {poll_result}")
        
        if poll_result is None:
            # Process is still running, success!
            streaming_active = True
            
            # Start frame reader thread
            frame_reader_thread = threading.Thread(target=read_frames_from_ffmpeg, daemon=True)
            frame_reader_thread.start()
            
            print(f"Device 0 started successfully with PID {ffmpeg_process.pid}")
        else:
            # Process died, get error and FAIL (no fallback)
            stdout_data = ""
            stderr_data = ""
            
            try:
                stdout_data, stderr_data = ffmpeg_process.communicate(timeout=1)
                stdout_data = stdout_data.decode() if stdout_data else ""
                stderr_data = stderr_data.decode() if stderr_data else ""
            except subprocess.TimeoutExpired:
                print("Timeout reading FFmpeg output")
            
            print(f"FFmpeg STDOUT: {stdout_data}")
            print(f"FFmpeg STDERR: {stderr_data}")
            
            ffmpeg_process = None
            streaming_active = False
            raise Exception(f"Device 0 failed to start. Exit code: {poll_result}. Error: {stderr_data}")
        
    except Exception as e:
        print(f"Failed to start device 0: {e}")
        streaming_active = False
        ffmpeg_process = None
        raise e

def stop_ffmpeg_stream():
    """Stop FFmpeg stream"""
    global ffmpeg_process, streaming_active, frame_reader_thread, latest_frame
    
    streaming_active = False  # Stop frame reader first
    
    if ffmpeg_process:
        ffmpeg_process.terminate()
        ffmpeg_process.wait()
        ffmpeg_process = None
    
    # Wait for frame reader thread to finish
    if frame_reader_thread and frame_reader_thread.is_alive():
        frame_reader_thread.join(timeout=2)
    
    frame_reader_thread = None
    
    with frame_lock:
        latest_frame = None
    
    print("Stopped FFmpeg stream")

def generate_frames():
    """Generate MJPEG frames from buffered frames - supports multiple clients"""
    global latest_frame
    
    if not streaming_active:
        return
    
    while streaming_active:
        try:
            # Get latest frame from buffer
            with frame_lock:
                current_frame = latest_frame
            
            if current_frame:
                # Yield frame with proper MJPEG boundary
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(current_frame)).encode() + b'\r\n'
                       b'\r\n' + current_frame + b'\r\n')
                
                # Small delay to control frame rate for clients
                time.sleep(1/60)  # ~60fps for clients
            else:
                # No frame available yet, wait a bit
                time.sleep(0.01)
                    
        except Exception as e:
            print(f"Frame generation error: {e}")
            break
    
    print("Frame generation stopped")

@app.route('/stream')
def stream():
    """MJPEG stream endpoint"""
    if not streaming_active:
        start_ffmpeg_stream()
        time.sleep(1)  # Give FFmpeg time to start
        
        # Check if start was successful
        if not streaming_active:
            return {"error": "Streaming is not active"}, 503
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start', methods=['POST'])
def start():
    """Start streaming"""
    try:
        start_ffmpeg_stream()
        if streaming_active:
            response_data = {
                "active": True,
                "streaming": streaming_active,
                "camera_type": "real",
                "clients": 0,
                "stream_url": f"http://localhost:8080/stream",
                "ffmpeg_pid": ffmpeg_process.pid if ffmpeg_process else None
            }
            return {
                "success": "MJPEG stream started",
                "data": response_data
            }, 200
        else:
            return {
                "error": "Failed to start stream - streaming not active after start attempt"
            }, 500
    except Exception as e:
        return {
            "error": f"Failed to start stream: {str(e)}"
        }, 500

@app.route('/stop', methods=['POST'])
def stop():
    """Stop streaming"""
    stop_ffmpeg_stream()
    return {
        "success": "MJPEG stream stopped"
    }, 200

@app.route('/status')
def status():
    """Get streaming status"""
    status_data = {
        "active": streaming_active,
        "streaming": streaming_active,  # backward compatibility
        "camera_type": "real",
        "clients": 0,  # FFmpeg doesn't track clients
        "stream_url": f"http://localhost:8080/stream" if streaming_active else None,
        "ffmpeg_pid": ffmpeg_process.pid if ffmpeg_process else None,
        "error": None
    }
    return {
        "data": status_data
    }, 200

@app.route('/health')
def health():
    """Health check endpoint"""
    health_data = {
        "status": "healthy",
        "service": "mjpeg-streaming", 
        "timestamp": time.time()
    }
    return {
        "data": health_data
    }, 200

if __name__ == '__main__':
    print("Starting simple FFmpeg MJPEG service on port 8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
