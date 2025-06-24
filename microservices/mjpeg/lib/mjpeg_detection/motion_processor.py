"""
Motion detection and cat detection module for MJPEG streaming service.
Provides real-time motion detection with YOLO-based cat detection overlay.
"""

# Standard library imports
import json
import os
import time
from datetime import datetime

# Third-party imports
import cv2
import numpy as np
import glob  # For finding crop files

import threading
import queue

# Detection settings and global state
DETECTION_ENABLED = True  # Auto-enable when available
YOLO_MODEL = None
PREV_GRAY = None

# Save management for film strip timing
LAST_SAVE_TIME = 0
FILM_STRIP_INTERVAL = 30.0  # Save detection images every 30 seconds
FILM_STRIP_WINDOW = 180.0  # Keep images for 3 minutes (6 images at 30s intervals)
PERIODIC_CHECK_INTERVAL = 30.0  # Check for motion/cats every 30 seconds regardless of movement

# Async processing queue and thread
detection_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues
detection_thread = None
detection_thread_running = False
periodic_check_thread = None
periodic_check_running = False
latest_frame_for_periodic = None  # Store latest frame for periodic checks

# Configuration constants
MOTION_THRESHOLD = 25
MOTION_LOG_DIR = "motion_logs"
CROP_SAVE_DIR = "motion_logs/crops"  # Directory for cropped detections
DETECTION_STRIP_FILE = "motion_logs/detections_strip.png"  # OBS image source file (PNG for transparency)
YOLO_MODEL_PATH = "/Users/srijansrivastava/Documents/Personal/ssvcam/backend/data/yolov8l.pt"  # Use existing model in data folder
YOLO_MODEL_FALLBACK = "yolov8n.pt"  # Fallback to nano model (downloads automatically)
CAT_CLASS_ID = 15  # COCO dataset class ID for cats
MOTION_PIXEL_THRESHOLD = 1000
JPEG_QUALITY = 85

# Strip generation settings
STRIP_WIDTH = 1920  # Target width for the film strip (OBS canvas width)
STRIP_HEIGHT = 270  # Height of the detection strip (updated to match available space)
STRIP_MAX_IMAGES = 6  # Number of images in 3-minute window (30s intervals)
STRIP_MARGIN = 15  # Margin between images (increased for better spacing)
STRIP_TIMESTAMP_HEIGHT = 35  # Height reserved for timestamp text with background
STRIP_LEFT_MARGIN = 20  # Left margin to start images away from edge

def start_detection_thread():
    """Start the async detection processing thread"""
    global detection_thread, detection_thread_running
    
    if detection_thread and detection_thread.is_alive():
        return
    
    detection_thread_running = True
    detection_thread = threading.Thread(target=_detection_worker, daemon=True)
    detection_thread.start()
    print("Async detection thread started")

def stop_detection_thread():
    """Stop the async detection processing thread"""
    global detection_thread_running
    detection_thread_running = False
    if detection_thread:
        detection_thread.join(timeout=2)

def _detection_worker():
    """Background worker that processes frames for detection"""
    global detection_thread_running
    
    while detection_thread_running:
        try:
            # Get frame from queue (timeout to allow thread to exit)
            jpeg_frame = detection_queue.get(timeout=1.0)
            
            if jpeg_frame is None:  # Poison pill to stop thread
                break
                
            _process_frame_async(jpeg_frame)
            detection_queue.task_done()
            
        except queue.Empty:
            continue  # Timeout, check if we should still be running
        except Exception as e:
            print(f"Detection worker error: {e}")

def _process_frame_async(jpeg_frame):
    """Process frame for detection in background thread"""
    try:
        # Convert JPEG to OpenCV frame
        frame_array = np.frombuffer(jpeg_frame, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        # Store frame for periodic checking (keep latest frame for sleeping cat detection)
        set_latest_frame_for_periodic_check(frame.copy())
        
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check for motion
        motion_detected = detect_motion(gray)
        
        if motion_detected:
            # Check if it's time to save a new film strip image
            current_time = time.time()
            global LAST_SAVE_TIME
            time_since_last_save = current_time - LAST_SAVE_TIME
            
            # Save immediately on first detection, or after interval
            should_save = (LAST_SAVE_TIME == 0 or time_since_last_save >= FILM_STRIP_INTERVAL)
            
            log_detection_event("motion")
            
            # Only run YOLO if motion detected
            cat_detections = detect_cats(frame)
            
            if cat_detections:
                log_detection_event("cat_detected", len(cat_detections))
                
                # Save cropped detections if it's time for a new film strip entry
                if should_save:
                    save_cropped_detections(frame, cat_detections)
        
    except Exception as e:
        print(f"Async frame processing error: {e}")

def init_detection():
    """Initialize detection components - safe initialization"""
    global YOLO_MODEL, DETECTION_ENABLED
    
    try:
        # Create logs directory
        os.makedirs(MOTION_LOG_DIR, exist_ok=True)
        os.makedirs(CROP_SAVE_DIR, exist_ok=True)  # Create crops directory
        
        # Try to load YOLO model
        try:
            from ultralytics import YOLO  # pylint: disable=import-outside-toplevel
            
            # Try to load existing model first
            if os.path.exists(YOLO_MODEL_PATH):
                YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
                print(f"YOLO model loaded from {YOLO_MODEL_PATH}")
            else:
                # Fallback to downloading nano model (smaller, faster)
                print(f"Model not found at {YOLO_MODEL_PATH}, using fallback {YOLO_MODEL_FALLBACK}")
                YOLO_MODEL = YOLO(YOLO_MODEL_FALLBACK)
                print("YOLO nano model loaded successfully")
                
        except ImportError:
            print("ultralytics package not installed - YOLO detection disabled")
            YOLO_MODEL = None
        except Exception as e:
            print(f"YOLO model loading failed: {e}")
            print("Falling back to motion detection only")
            YOLO_MODEL = None
        
        DETECTION_ENABLED = True
        print("Motion detection initialized")
        
        # Create initial detection strip
        create_detections_strip()
        
        # Start async processing thread
        start_detection_thread()
        
        # Start periodic checking thread for sleeping cats
        start_periodic_check()
        
    except Exception as e:
        print(f"Detection initialization failed: {e}")
        DETECTION_ENABLED = False

def detect_motion(current_gray):
    """Simple motion detection between frames"""
    global PREV_GRAY
    
    try:
        if PREV_GRAY is None:
            PREV_GRAY = current_gray.copy()
            return False
        
        # Calculate frame difference
        diff = cv2.absdiff(PREV_GRAY, current_gray)
        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv2.threshold(blur, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Count non-zero pixels
        motion_pixels = cv2.countNonZero(thresh)
        motion_detected = motion_pixels > MOTION_PIXEL_THRESHOLD  # Configurable threshold
        
        PREV_GRAY = current_gray.copy()
        return motion_detected
        
    except Exception as e:
        print(f"Motion detection error: {e}")
        return False

def detect_cats(frame):
    """Detect cats using YOLO - returns bounding boxes"""
    if not YOLO_MODEL:
        return []
    
    try:
        results = YOLO_MODEL.predict(frame, conf=0.5, classes=[CAT_CLASS_ID], verbose=False)
        detections = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append((int(x1), int(y1), int(x2), int(y2)))
        
        return detections
        
    except Exception as e:
        print(f"Cat detection error: {e}")
        return []

def save_cropped_detections(frame, detections):
    """Save cropped bounding box images for film strip timeline"""
    global LAST_SAVE_TIME
    
    current_time = time.time()
    
    # Check if enough time has passed based on film strip interval
    if current_time - LAST_SAVE_TIME < FILM_STRIP_INTERVAL:
        return
    
    try:
        # Clean up old images first (older than 3 minutes)
        cleanup_old_detections()
        
        # Save the best detection from this interval
        if detections:
            # Take the largest detection (assuming it's the most significant)
            largest_detection = max(detections, key=lambda det: (det[2] - det[0]) * (det[3] - det[1]))
            x1, y1, x2, y2 = largest_detection
            
            # Add some padding to the crop
            padding = 20
            h, w = frame.shape[:2]
            
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(w, x2 + padding)
            crop_y2 = min(h, y2 + padding)
            
            # Crop the detection
            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if cropped.size > 0:
                # Save with timestamp for film strip ordering
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"filmstrip_{timestamp}.jpg"
                filepath = os.path.join(CROP_SAVE_DIR, filename)
                
                cv2.imwrite(filepath, cropped, [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(f"Saved film strip detection: {filename}")
        
        # Update save timing
        LAST_SAVE_TIME = current_time
        
        # Update the detection strip for OBS
        create_detections_strip()
        
    except Exception as e:
        print(f"Error saving film strip crops: {e}")

def cleanup_old_detections():
    """Remove detection images older than the film strip window"""
    try:
        current_time = time.time()
        crop_pattern = os.path.join(CROP_SAVE_DIR, "filmstrip_*.jpg")
        crop_files = glob.glob(crop_pattern)
        
        for crop_file in crop_files:
            file_mtime = os.path.getmtime(crop_file)
            if current_time - file_mtime > FILM_STRIP_WINDOW:
                os.remove(crop_file)
                print(f"Removed old detection image: {os.path.basename(crop_file)}")
                
    except Exception as e:
        print(f"Error cleaning up old detections: {e}")

def reset_save_interval():
    """Reset save interval - no longer needed with fixed film strip timing"""
    pass

def create_detections_strip():
    """Create a rolling film strip of detection images for OBS with transparency and optimized layout"""
    try:
        # Get film strip files (chronological order, oldest first for left-to-right display)
        crop_pattern = os.path.join(CROP_SAVE_DIR, "filmstrip_*.jpg")
        crop_files = sorted(glob.glob(crop_pattern), key=os.path.getmtime)
        
        if not crop_files:
            # Create transparent strip with minimal "waiting" message
            strip = np.zeros((STRIP_HEIGHT, STRIP_WIDTH, 4), dtype=np.uint8)  # RGBA for transparency
            
            # Add a subtle waiting message with background
            message = "Waiting for detections..."
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = STRIP_LEFT_MARGIN
            text_y = STRIP_HEIGHT // 2 + 10
            
            # Add semi-transparent background for text
            cv2.rectangle(strip, (text_x - 5, text_y - 25), (text_x + text_size[0] + 10, text_y + 10), 
                         (0, 0, 0, 180), -1)  # Semi-transparent black
            
            # Add white text
            strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGBA2BGR)
            cv2.putText(strip_bgr, message, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert back to RGBA and set alpha
            strip = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGBA)
            strip[:, :, 3] = np.where(np.any(strip[:, :, :3] != 0, axis=2), 255, 0)  # Transparent where black
            
            cv2.imwrite(DETECTION_STRIP_FILE, strip)
            return
        
        # Calculate optimal image dimensions for left-aligned layout
        num_images = min(len(crop_files), STRIP_MAX_IMAGES)
        
        # Use available width more efficiently - calculate based on actual content
        available_width = STRIP_WIDTH - STRIP_LEFT_MARGIN - (STRIP_MARGIN * (num_images - 1))
        ideal_cell_width = available_width // num_images if num_images > 0 else 200
        
        # Ensure reasonable minimum and maximum cell widths
        cell_width = max(150, min(ideal_cell_width, 300))
        
        # Create transparent strip (RGBA)
        strip = np.zeros((STRIP_HEIGHT, STRIP_WIDTH, 4), dtype=np.uint8)
        
        # Take the most recent images (up to STRIP_MAX_IMAGES)
        recent_crops = crop_files[-num_images:] if num_images > 0 else []
        
        # Process and place images left to right (oldest to newest)
        current_x = STRIP_LEFT_MARGIN
        
        for i, crop_file in enumerate(recent_crops):
            try:
                img = cv2.imread(crop_file)
                if img is None:
                    continue
                
                # Get image timestamp from filename
                filename = os.path.basename(crop_file)
                timestamp_str = filename.replace("filmstrip_", "").replace(".jpg", "")
                try:
                    timestamp_dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    # Shorter format: "HH:mm AM/PM"
                    time_text = timestamp_dt.strftime("%I:%M %p")
                except ValueError:
                    time_text = "??:??"
                
                # Calculate image area (reserve space for timestamp)
                img_height = STRIP_HEIGHT - STRIP_TIMESTAMP_HEIGHT - 10  # 10px padding
                
                # Resize image to fit cell while maintaining aspect ratio
                h, w = img.shape[:2]
                
                # Scale to fit both width and height constraints
                width_scale = cell_width / w
                height_scale = img_height / h
                scale = min(width_scale, height_scale)
                
                new_width = int(w * scale)
                new_height = int(h * scale)
                
                if new_width > 0 and new_height > 0:
                    resized = cv2.resize(img, (new_width, new_height))
                    
                    # Position image in the cell (top-aligned for better layout)
                    img_x = current_x
                    img_y = STRIP_TIMESTAMP_HEIGHT + 5  # Start below timestamp area
                    
                    # Place the image in the strip
                    end_x = min(img_x + new_width, STRIP_WIDTH)
                    end_y = min(img_y + new_height, STRIP_HEIGHT)
                    
                    if img_x < STRIP_WIDTH and img_y < STRIP_HEIGHT:
                        # Convert to RGBA and place image
                        resized_rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
                        resized_rgba[:, :, 3] = 255  # Fully opaque
                        
                        strip[img_y:end_y, img_x:end_x] = resized_rgba[:end_y-img_y, :end_x-img_x]
                        
                        # Add timestamp with background for better readability
                        text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                        text_x = img_x + (new_width - text_size[0]) // 2  # Center under image
                        text_y = 25
                        
                        # Add semi-transparent background for timestamp
                        bg_padding = 3
                        cv2.rectangle(strip, 
                                    (text_x - bg_padding, text_y - 18), 
                                    (text_x + text_size[0] + bg_padding, text_y + 5), 
                                    (0, 0, 0, 200), -1)  # Semi-transparent black
                        
                        # Convert to BGR for text rendering, then back to RGBA
                        strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGBA2BGR)
                        cv2.putText(strip_bgr, time_text, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        strip = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGBA)
                        
                        # Restore alpha channel (preserve transparency)
                        strip[:, :, 3] = np.where(np.any(strip[:, :, :3] != 0, axis=2), 255, 0)
                        
                        print(f"Added detection {i+1}/{num_images} at {time_text}")
                
                # Move to next position with smart spacing
                current_x += new_width + STRIP_MARGIN
                
            except Exception as e:
                print(f"Error processing film strip image {crop_file}: {e}")
                continue
        
        # Add minimal info overlay (bottom right, with background)
        if num_images > 0:
            info_text = f"{num_images} recent"
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            info_x = STRIP_WIDTH - text_size[0] - 20
            info_y = STRIP_HEIGHT - 10
            
            # Add background for info text
            cv2.rectangle(strip, 
                        (info_x - 5, info_y - 15), 
                        (info_x + text_size[0] + 5, info_y + 3), 
                        (0, 0, 0, 150), -1)  # Semi-transparent black
            
            # Add info text
            strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGBA2BGR)
            cv2.putText(strip_bgr, info_text, (info_x, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            strip = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGBA)
            
            # Restore alpha channel
            strip[:, :, 3] = np.where(np.any(strip[:, :, :3] != 0, axis=2), 255, 0)
        
        # Save strip for OBS (PNG with transparency)
        cv2.imwrite(DETECTION_STRIP_FILE, strip)
        print(f"Updated optimized film strip with {num_images} detections (left-aligned, transparent PNG)")
        
    except Exception as e:
        print(f"Error creating optimized film strip: {e}")

def log_detection_event(event_type, detections_count=0):
    """Log detection events to file"""
    try:
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "detections": detections_count
        }
        
        log_file = os.path.join(MOTION_LOG_DIR, "events.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
            
    except Exception as e:
        print(f"Logging error: {e}")

def process_frame_for_detection(jpeg_frame):
    """Queue frame for async detection - returns original frame immediately for zero latency"""
    if not DETECTION_ENABLED:
        return jpeg_frame
    
    try:
        # Add frame to processing queue (non-blocking)
        if not detection_queue.full():
            detection_queue.put_nowait(jpeg_frame)
    except queue.Full:
        # Queue is full, skip this frame (detection will continue with next frames)
        pass
    except Exception as e:
        print(f"Error queuing frame for detection: {e}")
    
    # ALWAYS return original frame immediately - ZERO processing delay
    return jpeg_frame

def toggle_detection(enabled):
    """Enable/disable detection"""
    global DETECTION_ENABLED
    DETECTION_ENABLED = enabled
    
    if enabled:
        start_detection_thread()
        start_periodic_check()
    else:
        stop_detection_thread()
        stop_periodic_check()
    
    return DETECTION_ENABLED

def get_detection_status():
    """Get current detection status"""
    try:
        # Count film strip images
        crop_pattern = os.path.join(CROP_SAVE_DIR, "filmstrip_*.jpg")
        film_strip_count = len(glob.glob(crop_pattern))
        
        # Get time since last save
        time_since_last = time.time() - LAST_SAVE_TIME if LAST_SAVE_TIME > 0 else 0
        next_save_in = max(0, FILM_STRIP_INTERVAL - time_since_last)
        
        return {
            "enabled": DETECTION_ENABLED,
            "yolo_available": YOLO_MODEL is not None,
            "log_dir": MOTION_LOG_DIR,
            "crop_dir": CROP_SAVE_DIR,
            "async_processing": detection_thread_running,
            "queue_size": detection_queue.qsize() if detection_queue else 0,
            "film_strip": {
                "images_count": film_strip_count,
                "max_images": STRIP_MAX_IMAGES,
                "interval_seconds": FILM_STRIP_INTERVAL,
                "window_seconds": FILM_STRIP_WINDOW,
                "next_save_in_seconds": round(next_save_in, 1),
                "strip_file": DETECTION_STRIP_FILE
            }
        }
    except Exception as e:
        print(f"Error getting detection status: {e}")
        return {
            "enabled": DETECTION_ENABLED,
            "error": str(e)
        }

def refresh_film_strip():
    """Manually refresh the detection strip - useful for testing"""
    try:
        create_detections_strip()
        return {"success": True, "message": "Film strip refreshed"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def start_periodic_check():
    """Start periodic motion/cat checking thread"""
    global periodic_check_thread, periodic_check_running
    
    if periodic_check_thread and periodic_check_thread.is_alive():
        return
    
    periodic_check_running = True
    periodic_check_thread = threading.Thread(target=_periodic_check_worker, daemon=True)
    periodic_check_thread.start()
    print("Periodic motion check thread started (checking every 30s)")

def stop_periodic_check():
    """Stop periodic motion checking thread"""
    global periodic_check_running
    periodic_check_running = False
    if periodic_check_thread:
        periodic_check_thread.join(timeout=2)

def _periodic_check_worker():
    """Background worker that performs periodic checks for sleeping cats"""
    global periodic_check_running, latest_frame_for_periodic
    
    while periodic_check_running:
        try:
            time.sleep(PERIODIC_CHECK_INTERVAL)  # Wait 30 seconds
            
            if not periodic_check_running:
                break
                
            # Try to get the latest frame for checking
            if latest_frame_for_periodic is not None:
                frame = latest_frame_for_periodic
                
                print("Performing periodic check for sleeping cats...")
                
                # Always run cat detection on periodic check (even if no motion)
                cat_detections = detect_cats(frame)
                
                if cat_detections:
                    print(f"Periodic check found {len(cat_detections)} cat(s) - saving to film strip")
                    log_detection_event("periodic_cat_detected", len(cat_detections))
                    
                    # Save to film strip if it's time
                    current_time = time.time()
                    if current_time - LAST_SAVE_TIME >= FILM_STRIP_INTERVAL:
                        save_cropped_detections(frame, cat_detections)
                else:
                    print("Periodic check - no cats detected")
                    
        except Exception as e:
            print(f"Periodic check worker error: {e}")

def set_latest_frame_for_periodic_check(frame):
    """Set the latest frame for periodic checking"""
    global latest_frame_for_periodic
    latest_frame_for_periodic = frame