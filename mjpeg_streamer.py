import cv2
import socket
import time
import logging
import os
import signal
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)

def kill_port(port):
    """Kill any process using the specified port"""
    try:
        import subprocess
        # Find process using the port
        result = subprocess.run(['lsof', '-ti', f':{port}'], 
                               capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    logging.info(f"Killed process {pid} using port {port}")
                    time.sleep(1)  # Give more time to cleanup
                except ProcessLookupError:
                    pass  # Process already dead
                except Exception as e:
                    logging.warning(f"Failed to kill process {pid}: {e}")
            
            # Wait a bit more and force kill if still running
            time.sleep(2)
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                   capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        logging.info(f"Force killed process {pid} using port {port}")
                    except ProcessLookupError:
                        pass
                    except Exception as e:
                        logging.warning(f"Failed to force kill process {pid}: {e}")
                        
    except Exception as e:
        logging.warning(f"Failed to kill port {port}: {e}")

def stream_mjpeg():
    """Simple MJPEG streamer without Flask overhead"""
    VIDEO_DEVICE_INDEX = int(os.getenv("VIDEO_DEVICE_INDEX", "1"))
    PORT = 8080
    
    # Kill any existing process on port 8080
    kill_port(PORT)
    time.sleep(2)  # Give more time for cleanup
    
    # Initialize variables
    cap = None
    server_socket = None
    
    try:
        # Create socket server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', PORT))
        server_socket.listen(5)
        logging.info(f"MJPEG Streamer listening on port {PORT}...")
        logging.info(f"Stream available at: http://localhost:{PORT}")
        
        # Initialize camera
        for device_idx in [VIDEO_DEVICE_INDEX, 0, 1, 2]:
            try:
                cap = cv2.VideoCapture(device_idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        logging.info(f"Using video device {device_idx}")
                        break
                    else:
                        cap.release()
                        cap = None
            except Exception as e:
                logging.warning(f"Failed to open device {device_idx}: {e}")
                if cap:
                    cap.release()
                cap = None
        
        if not cap or not cap.isOpened():
            logging.error("No video device found!")
            return
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            try:
                client_socket, addr = server_socket.accept()
                logging.info(f"Client connected: {addr}")
                
                # Send HTTP headers
                response = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                    "Connection: close\r\n"
                    "Cache-Control: no-cache\r\n"
                    "Pragma: no-cache\r\n\r\n"
                )
                client_socket.send(response.encode())
                
                # Stream frames
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logging.warning("Failed to read frame")
                        break
                        
                    # Encode frame as JPEG
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_data = (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + 
                        jpeg.tobytes() + b'\r\n'
                    )
                    
                    try:
                        client_socket.send(frame_data)
                        time.sleep(0.033)  # ~30 FPS
                    except (BrokenPipeError, ConnectionResetError):
                        logging.info("Client disconnected")
                        break
                    
            except Exception as e:
                logging.error(f"Error serving client: {e}")
            finally:
                if 'client_socket' in locals():
                    try:
                        client_socket.close()
                    except:
                        pass
                        
    except Exception as e:
        logging.error(f"Server error: {e}")
    finally:
        # Clean up resources
        if cap:
            cap.release()
        if server_socket:
            server_socket.close()
        logging.info("MJPEG Streamer stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logging.info("Received shutdown signal")
    sys.exit(0)

if __name__ == '__main__':
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    stream_mjpeg()