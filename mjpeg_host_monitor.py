#!/usr/bin/env python3
import os
import time
import subprocess
import json
import signal
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

BACKEND_PATH = "/Users/srijansrivastava/Documents/Personal/ssvcam/backend"
SIGNAL_FILE = os.path.join(BACKEND_PATH, "mjpeg_control.json")
STATUS_FILE = os.path.join(BACKEND_PATH, "mjpeg_status.json")
LOG_FILE = os.path.join(BACKEND_PATH, "mjpeg_streamer.log")

mjpeg_process = None

def update_status(running, pid=None, port=None, error=None):
    status_data = {
        "running": running,
        "pid": pid,
        "port": port,
        "error": error,
        "timestamp": time.time()
    }
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(status_data, f)
    except Exception as e:
        logging.error(f"Failed to update status: {e}")

def signal_handler(sig, frame):
    global mjpeg_process
    logging.info("Received shutdown signal")
    if mjpeg_process:
        mjpeg_process.terminate()
        try:
            mjpeg_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mjpeg_process.kill()
    update_status(False)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print("🎥 MJPEG Host Monitor started...")
update_status(False)

while True:
    try:
        if os.path.exists(SIGNAL_FILE):
            with open(SIGNAL_FILE, 'r') as f:
                command_data = json.load(f)
            
            action = command_data.get('action')
            
            if action == "start" and (not mjpeg_process or mjpeg_process.poll() is not None):
                print("🚀 Starting MJPEG streamer...")
                
                # Clear log file
                try:
                    with open(LOG_FILE, 'w') as f:
                        f.truncate(0)
                except:
                    pass
                
                # Start with logging
                try:
                    with open(LOG_FILE, 'a') as log_file:
                        mjpeg_process = subprocess.Popen([
                            "python3", "mjpeg_streamer.py"
                        ], cwd=BACKEND_PATH, stdout=log_file, stderr=log_file)
                    
                    # Wait a bit and check if it started
                    time.sleep(3)
                    if mjpeg_process.poll() is None:
                        print(f"✅ MJPEG streamer started (PID: {mjpeg_process.pid})")
                        update_status(True, mjpeg_process.pid, 8080)
                    else:
                        print("❌ MJPEG streamer failed to start")
                        update_status(False, None, None, "Failed to start")
                        mjpeg_process = None
                        
                except Exception as e:
                    print(f"❌ Error starting MJPEG: {e}")
                    update_status(False, None, None, str(e))
                    mjpeg_process = None
                
            elif action == "stop" and mjpeg_process and mjpeg_process.poll() is None:
                print("🛑 Stopping MJPEG streamer...")
                try:
                    mjpeg_process.terminate()
                    try:
                        mjpeg_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print("Force killing MJPEG process...")
                        mjpeg_process.kill()
                        mjpeg_process.wait()
                    
                    print("✅ MJPEG streamer stopped")
                    update_status(False)
                    mjpeg_process = None
                except Exception as e:
                    print(f"❌ Error stopping MJPEG: {e}")
                    update_status(False, None, None, str(e))
            
            elif action == "status":
                # Just update status
                running = mjpeg_process and mjpeg_process.poll() is None
                pid = mjpeg_process.pid if running else None
                update_status(running, pid, 8080 if running else None)
            
            os.remove(SIGNAL_FILE)
        
        # Check if process died unexpectedly
        if mjpeg_process and mjpeg_process.poll() is not None:
            print("⚠️  MJPEG process died unexpectedly")
            update_status(False, None, None, "Process died unexpectedly")
            mjpeg_process = None
            
    except Exception as e:
        print(f"❌ Monitor error: {e}")
    
    time.sleep(0.5)