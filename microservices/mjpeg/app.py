#!/usr/bin/env python3

import subprocess
import threading
import time
import os
import signal
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import queue
import re

app = Flask(__name__)
CORS(app)

# Global FFmpeg process
ffmpeg_process = None
streaming_active = False
frame_reader_thread = None
monitor_thread = None
stderr_reader_thread = None
latest_frame = None
frame_lock = threading.Lock()
restart_lock = threading.Lock()

def get_obs_camera_index():
    """Find OBS Virtual Camera by name instead of guessing index"""
    try:
        # List all video devices
        cmd = ['/opt/homebrew/bin/ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', '']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        # Parse the output to find OBS Virtual Camera
        lines = result.stderr.split('\n')
        for line in lines:
            # Look for lines like "[0] OBS Virtual Camera"
            match = re.search(r'\[(\d+)\].*OBS Virtual Camera', line, re.IGNORECASE)
            if match:
                index = int(match.group(1))
                print(f"Found OBS Virtual Camera at index {index}")
                return str(index)
        
        # Fallback to common indices if not found by name
        print("OBS Virtual Camera not found by name, trying common indices...")
        for test_index in ['0', '1', '2']:
            if test_camera_index(test_index):
                print(f"Using camera index {test_index} as fallback")
                return test_index
        
        raise Exception("No working camera found")
        
    except Exception as e:
        print(f"Error finding OBS camera: {e}")
        # Last resort fallback
        return '0'

def test_camera_index(index):
    """Test if a camera index works"""
    try:
        cmd = [
            '/opt/homebrew/bin/ffmpeg', '-f', 'avfoundation', '-framerate', '30', 
            '-i', index, '-frames:v', '1', '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def read_ffmpeg_stderr():
    """Read and log FFmpeg stderr output"""
    global ffmpeg_process, streaming_active
    
    if not ffmpeg_process or not ffmpeg_process.stderr:
        return
    
    try:
        # Create log file for FFmpeg output
        log_file = '/tmp/ffmpeg_output.log'
        
        with open(log_file, 'a') as f:
            f.write(f"\n=== FFmpeg started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            
            while streaming_active and ffmpeg_process:
                try:
                    line = ffmpeg_process.stderr.readline()
                    if not line:
                        break
                    
                    decoded_line = line.decode('utf-8', errors='ignore').strip()
                    if decoded_line:
                        # Write to log file
                        f.write(f"{time.strftime('%H:%M:%S')}: {decoded_line}\n")
                        f.flush()
                        
                        # Also print important messages
                        if any(keyword in decoded_line.lower() for keyword in 
                               ['error', 'failed', 'warning', 'fps=', 'time=']):
                            print(f"FFmpeg: {decoded_line}")
                            
                except Exception as e:
                    f.write(f"Error reading stderr: {e}\n")
                    break
                    
    except Exception as e:
        print(f"Error in stderr reader: {e}")
    
    print("FFmpeg stderr reader stopped")

def get_base_url():
    """Get base URL from the current request - localhost gets :8080, others use their own domain"""
    if request:
        host = request.host
        scheme = request.scheme
        
        # If it's localhost, add port 8080
        if 'localhost' in host or '127.0.0.1' in host:
            # Remove any existing port and add 8080
            host_without_port = host.split(':')[0]
            return f"{scheme}://{host_without_port}:8080"
        else:
            # Use the host as-is (reverse proxy handles the domain)
            return f"{scheme}://{host}/stream"
    
    # Fallback if no request context
    external_domain = os.getenv('EXTERNAL_DOMAIN')
    if external_domain:
        return f"https://{external_domain}/stream"
    return "http://localhost:8080"

def monitor_ffmpeg():
    """Monitor FFmpeg process and restart if it dies"""
    global ffmpeg_process, streaming_active
    
    while streaming_active:
        try:
            if ffmpeg_process and ffmpeg_process.poll() is not None:
                print(f"FFmpeg died unexpectedly (exit code: {ffmpeg_process.poll()})")
                
                with restart_lock:
                    if streaming_active:  # Double check we still want to be active
                        print("Attempting to restart FFmpeg...")
                        _restart_ffmpeg_internal()
            
            time.sleep(1)  # Check every second
            
        except Exception as e:
            print(f"Monitor error: {e}")
            break
    
    print("FFmpeg monitor stopped")

def _restart_ffmpeg_internal():
    """Internal restart function - called from monitor"""
    global ffmpeg_process, frame_reader_thread, stderr_reader_thread
    
    try:
        # Stop current process
        if ffmpeg_process:
            try:
                ffmpeg_process.terminate()
                ffmpeg_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                ffmpeg_process.kill()
            ffmpeg_process = None
        
        # Wait for threads to stop
        if frame_reader_thread and frame_reader_thread.is_alive():
            frame_reader_thread.join(timeout=2)
        if stderr_reader_thread and stderr_reader_thread.is_alive():
            stderr_reader_thread.join(timeout=2)
        
        # Small delay before restart
        time.sleep(1)
        
        # Restart FFmpeg
        _start_ffmpeg_process()
        
    except Exception as e:
        print(f"Restart failed: {e}")
        global streaming_active
        streaming_active = False

def _start_ffmpeg_process():
    """Internal function to start FFmpeg process"""
    global ffmpeg_process, frame_reader_thread, stderr_reader_thread
    
    # Find OBS camera by name
    camera_index = get_obs_camera_index()
    
    cmd = [
        '/opt/homebrew/bin/ffmpeg',  # Changed this line
        '-f', 'avfoundation',
        '-framerate', '60',
        '-pixel_format', 'nv12',
        '-i', camera_index,
        '-vf', 'scale=1280:720:flags=fast_bilinear',
        '-f', 'mjpeg',
        '-q:v', '6',
        '-threads', '1',
        '-fflags', '+nobuffer+flush_packets+genpts',
        '-flags', '+low_delay',
        '-strict', 'experimental',
        '-avoid_negative_ts', 'make_zero',
        '-tune', 'zerolatency',
        'pipe:1'
    ]
    
    print(f"Starting FFmpeg with camera {camera_index}: {' '.join(cmd)}")
    
    ffmpeg_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        preexec_fn=os.setsid
    )
    
    # Start frame reader thread
    frame_reader_thread = threading.Thread(target=read_frames_from_ffmpeg, daemon=True)
    frame_reader_thread.start()
    
    # Start stderr reader thread for logging
    stderr_reader_thread = threading.Thread(target=read_ffmpeg_stderr, daemon=True)
    stderr_reader_thread.start()

def read_frames_from_ffmpeg():
    """Read frames from FFmpeg and buffer them for multiple clients"""
    global ffmpeg_process, streaming_active, latest_frame
    
    frame_data = b''
    consecutive_empty_reads = 0
    max_empty_reads = 10
    
    while streaming_active and ffmpeg_process:
        try:
            if not ffmpeg_process.stdout:
                break
                
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
                    
                jpeg_frame = frame_data[start_pos:end_pos + 2]
                frame_data = frame_data[end_pos + 2:]
                
                if len(jpeg_frame) > 500:
                    with frame_lock:
                        latest_frame = jpeg_frame
                
                if len(frame_data) > 50000:
                    frame_data = frame_data[-25000:]
                    
        except Exception as e:
            print(f"Frame reading error: {e}")
            break
    
    print("Frame reader stopped")

def start_ffmpeg_stream():
    """Start FFmpeg MJPEG stream - Find OBS Virtual Camera by name"""
    global ffmpeg_process, streaming_active, monitor_thread
    
    if ffmpeg_process is not None:
        print("FFmpeg already running, skipping start")
        return
    
    try:
        streaming_active = True
        
        # Start FFmpeg process
        _start_ffmpeg_process()
        
        # Wait and check if process is alive
        print(f"FFmpeg process started with PID {ffmpeg_process.pid}")
        time.sleep(2)
        
        poll_result = ffmpeg_process.poll()
        print(f"FFmpeg poll result: {poll_result}")
        
        if poll_result is None:
            # Start monitor thread
            monitor_thread = threading.Thread(target=monitor_ffmpeg, daemon=True)
            monitor_thread.start()
            
            print(f"OBS Virtual Camera started successfully with PID {ffmpeg_process.pid}")
            print(f"FFmpeg logs available at: /tmp/ffmpeg_output.log")
        else:
            # Process died, get error
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
            raise Exception(f"OBS Virtual Camera failed to start. Exit code: {poll_result}. Error: {stderr_data}")
        
    except Exception as e:
        print(f"Failed to start OBS Virtual Camera: {e}")
        streaming_active = False
        ffmpeg_process = None
        raise e

def stop_ffmpeg_stream():
    """Stop FFmpeg stream"""
    global ffmpeg_process, streaming_active, frame_reader_thread, monitor_thread, stderr_reader_thread, latest_frame
    
    streaming_active = False
    
    # Stop monitor thread
    if monitor_thread and monitor_thread.is_alive():
        monitor_thread.join(timeout=2)
    monitor_thread = None
    
    # Stop FFmpeg process
    if ffmpeg_process:
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
    if frame_reader_thread and frame_reader_thread.is_alive():
        frame_reader_thread.join(timeout=3)
    if stderr_reader_thread and stderr_reader_thread.is_alive():
        stderr_reader_thread.join(timeout=3)
    
    frame_reader_thread = None
    stderr_reader_thread = None
    
    with frame_lock:
        latest_frame = None
    
    print("Stopped FFmpeg stream")

# Add new route to view FFmpeg logs
@app.route('/ffmpeg/logs')
def ffmpeg_logs():
    """Get recent FFmpeg logs"""
    try:
        with open('/tmp/ffmpeg_output.log', 'r') as f:
            # Get last 100 lines
            lines = f.readlines()
            recent_lines = lines[-100:] if len(lines) > 100 else lines
            return {
                "logs": ''.join(recent_lines),
                "total_lines": len(lines)
            }, 200
    except FileNotFoundError:
        return {"logs": "No logs available yet", "total_lines": 0}, 200
    except Exception as e:
        return {"error": f"Failed to read logs: {e}"}, 500

# ...rest of your existing routes stay the same...

def generate_frames():
    """Generate MJPEG frames from buffered frames - supports multiple clients"""
    global latest_frame
    
    if not streaming_active:
        return
    
    no_frame_count = 0
    
    while streaming_active:
        try:
            with frame_lock:
                current_frame = latest_frame
            
            if current_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(current_frame)).encode() + b'\r\n'
                       b'\r\n' + current_frame + b'\r\n')
                
                no_frame_count = 0
                time.sleep(1/60)
            else:
                no_frame_count += 1
                if no_frame_count > 120:
                    print("No frames available, breaking connection")
                    break
                time.sleep(0.008)
                    
        except Exception as e:
            print(f"Frame generation error: {e}")
            break
    
    print("Frame generation stopped")

@app.route('/stream')
def stream():
    """MJPEG stream endpoint"""
    if not streaming_active:
        start_ffmpeg_stream()
        time.sleep(1)
        
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
        time.sleep(2)
        if streaming_active:
            response_data = {
                "active": True,
                "streaming": streaming_active,
                "camera_type": "real",
                "clients": 0,
                "stream_url": f"{get_base_url()}/stream",
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
        "streaming": streaming_active,
        "camera_type": "real",
        "clients": 0,
        "stream_url": f"{get_base_url()}/stream" if streaming_active else None,
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