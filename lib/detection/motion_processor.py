import cv2
import torch
import numpy as np
import threading
import time
import logging
import warnings
import os
from datetime import datetime
from ultralytics import YOLO
from lib.obs.websocket_client import OBSWebSocketClient
from config import STREAMS

# =========================
# Configuration
# =========================
OBS_URL = "ws://103.227.96.47:4455"
OBS_PASSWORD = ""


# Set environment variable for FFmpeg to use TCP for RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


SAVE_TIME = 30  # Save detection frame every 30 seconds
device = "mps" if torch.backends.mps.is_available() else "cpu"
YOLO_MODEL = YOLO("yolov8l.pt").to(device)  # Replace with your preferred YOLO model
CHECK_INTERVAL = 0.1  # Seconds between frame checks
NO_CAT_TIMEOUT = 20
cv2.ocl.setUseOpenCL(True)

os.makedirs("motion_logs", exist_ok=True)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("motion_detection.log"), logging.StreamHandler()],
)
warnings.filterwarnings("ignore", message="Failed to load image Python extension")


# =========================
# YOLO-Based Detection
# =========================



def detect_objects(frame, cat_class_id=15, conf=0.3, upscale_factor=1.5):
    """
    Detects cats in a single frame using YOLOv8.
    :param frame: BGR image from OpenCV
    :param cat_class_id: The numerical class ID for 'cat' in the model (COCO: 15)
    :param conf: Confidence threshold for detection
    :param upscale_factor: Resize factor for the frame (improves detection of small cats)
    :return: List of (x1, y1, x2, y2) bounding boxes for each detected cat
    """
    # Optionally upscale
    if upscale_factor != 1.0:
        frame = cv2.resize(frame, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)

    results = YOLO_MODEL.predict(frame, conf=conf, iou=0.45)  # Adjust iou as needed
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            if cls_id == cat_class_id and confidence > conf:
                # Scale coordinates back if you upscaled the frame
                if upscale_factor != 1.0:
                    x1 = int(x1 / upscale_factor)
                    y1 = int(y1 / upscale_factor)
                    x2 = int(x2 / upscale_factor)
                    y2 = int(y2 / upscale_factor)
                detections.append((x1, y1, x2, y2))

    return detections


def save_detection_frame(frame, name, detections):
    """Save the frame with detected cats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("motion_logs", name)
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"cat_detected_{timestamp}.jpg")

    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(filename, frame)
    logging.info(f"Saved detection frame: {filename}")


# =========================
# Layout Update for OBS
# =========================



import cv2
import numpy as np
import time
import logging

MAX_STREAM_RETRIES = 50       # example limit on how many times to retry an offline stream
import cv2
import numpy as np
import time
import logging

# Constants (tweak as you like)
CHECK_INTERVAL = 0.2         # seconds between frame checks
NO_CAT_TIMEOUT = 10.0        # revert to Mosaic if no cats seen in this many seconds
SAVE_TIME = 5.0              # only save frames if at least 5s passed since last save
MAX_STREAM_RETRIES = 5       # how many times to retry if the stream is offline

def detect_motion(prev_gray, current_gray, threshold_value=25):
    """
    Detects motion between two consecutive frames using frame differencing, 
    thresholding, and morphological filtering to reduce noise.
    Returns True if motion is detected, False otherwise.
    """
    frame_diff = cv2.absdiff(prev_gray, current_gray)
    blurred = cv2.GaussianBlur(frame_diff, (5, 5), 0)

    # Morphological open to remove small specks
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Sum-based measure of motion; you can also do contour-based logic
    motion_score = np.sum(morphed) / (morphed.shape[0] * morphed.shape[1])
    return motion_score > 0.01

def open_stream_with_retries(url, max_retries=MAX_STREAM_RETRIES, delay=3):
    """
    Attempt to open the video capture with a certain number of retries.
    Return the cv2.VideoCapture object if successful, otherwise None.
    """
    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            logging.info(f"Successfully opened stream after {attempt} attempt(s).")
            return cap
        logging.warning(f"Failed to open stream (attempt {attempt}/{max_retries}). Retrying in {delay}s...")
        time.sleep(delay)

    logging.error("Max retries reached. Could not open stream.")
    return None

def process_stream(stream, obs_client):
    """
    Main loop that processes a single stream:
      - Uses motion detection to decide when to run YOLO.
      - Pins the camera if a cat is detected until a cat is detected in another camera
        OR we haven't detected any cat for NO_CAT_TIMEOUT seconds (revert to mosaic).
    """
    name = stream["name"]
    url = stream["url"]

    # Try opening the stream with retries
    cap = open_stream_with_retries(url, max_retries=MAX_STREAM_RETRIES, delay=3)
    if cap is None or not cap.isOpened():
        logging.error(f"Unable to open stream {name} after retries. Exiting.")
        return

    logging.info(f"Opened stream {name}: {url}")

    # Keep track of the last time we saw a cat (from any camera).
    last_cat_time = time.time()

    # Keep track of which camera is currently pinned (active). Start with Mosaic.
    current_active_camera = "Mosaic"
    last_saved_frame_time = None

    # For motion detection
    prev_gray_frame = None
    init_done = False

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Stream {name} disconnected. Reconnecting in 3s...")
            cap.release()
            time.sleep(3)
            cap = open_stream_with_retries(url, max_retries=MAX_STREAM_RETRIES, delay=3)
            if cap is None or not cap.isOpened():
                logging.error(f"Unable to reconnect stream {name}. Exiting loop.")
                break
            prev_gray_frame = None  # reset after reconnection
            continue

        # Convert to grayscale for motion detection
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            logging.warning(f"Invalid frame received from {name}. Skipping...")
            continue

        # ----------------- INIT FRAME LOGIC -----------------
        # If this is the very first frame after opening or reconnect,
        # run an optional one-time detection to handle a cat that is already there (but not moving).
        if prev_gray_frame is None:
            prev_gray_frame = gray_frame
            if not init_done:
                init_done = True
                detections = detect_objects(frame, conf=0.5, upscale_factor=1.5)
                if detections:
                    last_cat_time = time.time()
                    logging.info(f"INIT detection found cat(s) in {name}. Pinning {name}.")
                    obs_client.update_obs_layout(active_source=name)
                    current_active_camera = name
                    save_detection_frame(frame, name, detections)
            continue

        # ----------------- MOTION DETECTION -----------------
        motion_detected = detect_motion(prev_gray_frame, gray_frame)
        prev_gray_frame = gray_frame  # update for next iteration

        current_time = time.time()

        # ----------------- ONLY RUN YOLO IF MOTION DETECTED -----------------
        if motion_detected:
            detections = detect_objects(frame, conf=0.5, upscale_factor=1.5)

            if detections:
                # If we see cats in this camera, we "pin" this camera as active
                last_cat_time = current_time

                if current_active_camera != name:
                    logging.info(f"Detected cats in {name}. Switching layout from {current_active_camera} to {name}.")
                    obs_client.update_obs_layout(active_source=name)
                    current_active_camera = name

                # Save frames periodically
                if (not last_saved_frame_time) or ((current_time - last_saved_frame_time) > SAVE_TIME):
                    save_detection_frame(frame, name, detections)
                    last_saved_frame_time = current_time

            else:
                # We have motion, but no cat detections for this camera
                # That alone doesn't cause us to revert from an already pinned camera
                # However, we DO check if we've been cat-free for too long
                pass

        # ----------------- CHECK TIMEOUT FOR NO CATS ANYWHERE -----------------
        # If we currently have a pinned camera, but it's been NO_CAT_TIMEOUT
        # seconds since we last saw ANY cat, revert to mosaic.
        if current_active_camera != "Mosaic":
            if (current_time - last_cat_time) > NO_CAT_TIMEOUT:
                logging.info(f"No cats seen for {NO_CAT_TIMEOUT}s. Reverting to Mosaic.")
                obs_client.update_obs_layout(active_source=None)
                current_active_camera = "Mosaic"

        # Sleep a bit before the next iteration
        time.sleep(CHECK_INTERVAL)

    cap.release()
    logging.info(f"Exiting process_stream for {name}.")


# =========================
# Main Function
# =========================

def init():
    obs_client = OBSWebSocketClient(OBS_URL, OBS_PASSWORD)

    if not (obs_client.connect()):
        logging.error("Could not connect to OBS WebSocket")
    obs_client.retrieve_scene_sources("Mosaic")
    obs_client.update_obs_layout(active_source=None)
    threads = []
    for stream in STREAMS:
        t = threading.Thread(target=process_stream, args=(stream, obs_client), daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Exiting...")
