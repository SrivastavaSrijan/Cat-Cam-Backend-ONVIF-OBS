import cv2
import socket
import time
import logging
import os
import signal
import sys
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)

def create_mock_frame():
    """Create a mock camera frame when no real camera is available"""
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

def kill_port(port):
    """Kill any process using the specified port"""
    try:
        import subprocess
        result = subprocess.run(['lsof', '-ti', f':{port}'], 
                               capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    logging.info(f"Killed process {pid} using port {port}")
                    time.sleep(2)
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logging.warning(f"Failed to kill process {pid}: {e}")
            
            time.sleep(3)
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

def get_camera():
    """Try to get a real camera, fallback to mock"""
    VIDEO_DEVICE_INDEX = int(os.getenv("VIDEO_DEVICE_INDEX", "1"))
    
    # Try real cameras first
    for device_idx in [VIDEO_DEVICE_INDEX, 0, 1, 2]:
        try:
            cap = cv2.VideoCapture(device_idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    logging.info(f"Using real camera device {device_idx}")
                    return cap, False
                else:
                    cap.release()
        except Exception as e:
            logging.warning(f"Failed to open device {device_idx}: {e}")
            if 'cap' in locals():
                cap.release()
    
    # No real camera found, use mock
    logging.info("No real camera found, using mock camera")
    return None, True

def stream_mjpeg():
    """Simple MJPEG streamer with mock fallback"""
    PORT = 8080
    
    # Kill any existing process on port 8080
    kill_port(PORT)
    time.sleep(3)  # Give more time for cleanup
    
    # Initialize variables
    cap = None
    server_socket = None
    use_mock = False
    
    try:
        # Get camera (real or mock)
        cap, use_mock = get_camera()
        
        if not use_mock:
            # Set camera properties for real camera
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create socket server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', PORT))
        server_socket.listen(5)
        
        camera_type = "Mock" if use_mock else "Real"
        logging.info(f"MJPEG Streamer listening on port {PORT} with {camera_type} camera")
        logging.info(f"Stream available at: http://localhost:{PORT}")
        
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
                try:
                    while True:
                        if use_mock:
                            frame = create_mock_frame()
                            ret = True
                        else:
                            ret, frame = cap.read()
                        
                        if not ret or frame is None:
                            logging.warning("Failed to read frame")
                            if not use_mock:
                                # Switch to mock if real camera fails
                                logging.info("Switching to mock camera")
                                if cap:
                                    cap.release()
                                use_mock = True
                                continue
                            else:
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
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            logging.info("Client disconnected")
                            break
                            
                except Exception as e:
                    logging.warning(f"Error streaming to client: {e}")
                finally:
                    try:
                        client_socket.close()
                    except:
                        pass
                    
            except KeyboardInterrupt:
                logging.info("Received keyboard interrupt")
                break
            except Exception as e:
                logging.error(f"Error accepting client: {e}")
                time.sleep(1)  # Brief pause before continuing
                
    except Exception as e:
        logging.error(f"Server error: {e}")
    finally:
        # Clean up resources
        if cap and not use_mock:
            cap.release()
        if server_socket:
            server_socket.close()
        logging.info("MJPEG Streamer stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logging.info(f"Received shutdown signal {signum}")
    sys.exit(0)

if __name__ == '__main__':
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        stream_mjpeg()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)