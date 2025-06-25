"""detection_service.py – drop‑in replacement for the old
`lib.mjpeg_detection.motion_processor` module.

Highlights
==========
* **All logic lives inside a `DetectionService` class** → no global state races.
* **Public functional wrappers** at the bottom keep the *exact* old API so
  `app.py` (or any other caller) can continue `from detection_service import
  process_frame_for_detection, init_detection, …` with zero code changes.
* Background threads are started *once* (lazy‑initialised) and shut down cleanly
  on `toggle_detection(False)` or program exit.
* Uses `logging` instead of naked `print` for easier integration with whatever
  log stack you prefer.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover – YOLO optional
    YOLO = None

log = logging.getLogger("detection")

###############################################################################
# Config
###############################################################################

@dataclass
class Config:
    # Motion
    motion_threshold: int = int(os.getenv("MOTION_THRESHOLD", 25))
    pixel_threshold: int = int(os.getenv("MOTION_PIXEL_THRESHOLD", 1000))

    # YOLO
    model_path: str = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    cat_class_id: int = int(os.getenv("CAT_CLASS_ID", 15))
    yolo_conf: float = float(os.getenv("YOLO_CONF", 0.5))

    # Film strip timing
    film_interval: float = float(os.getenv("FILM_STRIP_INTERVAL", 30))  # s
    film_window: float = float(os.getenv("FILM_STRIP_WINDOW", 180))

    # Film strip layout
    strip_width: int = 1920
    strip_height: int = 270
    strip_max_images: int = 6
    strip_margin: int = 15
    strip_left_margin: int = 20

    # Sleeping‑cat periodic scan
    periodic_interval: float = 30.0  # s

    # I/O root
    root_dir: Path = Path(os.getenv("MOTION_ROOT", "motion_logs"))

    @property
    def crop_dir(self) -> Path:
        return self.root_dir / "crops"

    @property
    def strip_file(self) -> Path:
        return self.root_dir / "detections_strip.jpg"  # Changed from PNG to JPG

###############################################################################
# Internal helpers
###############################################################################

def _makedirs(cfg: Config):
    cfg.root_dir.mkdir(parents=True, exist_ok=True)
    cfg.crop_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up any old PNG files
    old_png = cfg.root_dir / "detections_strip.png"
    if old_png.exists():
        old_png.unlink()
        print(f"Removed old PNG file: {old_png}")

###############################################################################
# Motion detector
###############################################################################

class MotionDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prev: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def __call__(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with self._lock:
            if self.prev is None:
                self.prev = gray
                return False
            diff = cv2.absdiff(self.prev, gray)
            self.prev = gray
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thr = cv2.threshold(diff, self.cfg.motion_threshold, 255, cv2.THRESH_BINARY)
        return cv2.countNonZero(thr) > self.cfg.pixel_threshold

###############################################################################
# Cat detector
###############################################################################

class CatDetector:
    def __init__(self, cfg: Config):
        if YOLO:
            try:
                self.model = YOLO(cfg.model_path)
                log.info("YOLO model loaded: %s", cfg.model_path)
            except Exception as e:
                log.warning("Failed to load YOLO model %s: %s", cfg.model_path, e)
                self.model = None
        else:
            self.model = None
            log.warning("ultralytics not installed – cat detection disabled")
        self.cfg = cfg

    def __call__(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if not self.model:
            return []
        try:
            result = self.model.predict(frame, conf=self.cfg.yolo_conf, classes=[self.cfg.cat_class_id], verbose=False)
            boxes: List[Tuple[int, int, int, int]] = []
            for r in result:
                if r.boxes is not None:
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                        boxes.append((int(x1), int(y1), int(x2), int(y2)))
            return boxes
        except Exception as e:
            log.error("Cat detection error: %s", e)
            return []

###############################################################################
# Film strip writer
###############################################################################

class FilmStrip:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_save = 0.0
        _makedirs(cfg)
        
        # Ensure we remove any old PNG file
        old_png = self.cfg.root_dir / "detections_strip.png"
        if old_png.exists():
            old_png.unlink()
            print(f"Cleaned up old PNG: {old_png}")
        
        self._render_blank()

    # ---------- public
    def maybe_save(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]):
        now = time.time()
        if not boxes or now - self.last_save < self.cfg.film_interval:
            return
        self._purge_old()
        self._save_largest(frame, boxes)
        self.last_save = now
        self._render()

    def refresh(self):
        self._render()

    # ---------- helpers
    def _purge_old(self):
        cutoff = time.time() - self.cfg.film_window
        # Use consistent filename pattern - filmstrip_ not film_
        for p in self.cfg.crop_dir.glob("filmstrip_*.jpg"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    log.debug("Removed old detection image: %s", p.name)
            except OSError:
                pass

    def _save_largest(self, frame, boxes):
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        pad = 20
        h, w = frame.shape[:2]
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(w, x2 + pad)
        crop_y2 = min(h, y2 + pad)
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return
        fname = f"filmstrip_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(self.cfg.crop_dir / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        log.debug("saved crop %s", fname)

    def _render_blank(self):
        # Create black background JPG instead of transparent PNG
        blank = np.zeros((self.cfg.strip_height, self.cfg.strip_width, 3), dtype=np.uint8)
        
        # Make sure we're writing to JPG file
        jpg_file = self.cfg.strip_file
        print(f"Creating blank film strip at: {jpg_file}")
        
        cv2.imwrite(str(jpg_file), blank, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def _render(self):
        # Use consistent filename pattern - filmstrip_ not film_
        files = sorted(self.cfg.crop_dir.glob("filmstrip_*.jpg"), key=lambda p: p.stat().st_mtime)[-self.cfg.strip_max_images:]
        
        # Create black background JPG canvas
        canvas = np.zeros((self.cfg.strip_height, self.cfg.strip_width, 3), dtype=np.uint8)
        
        if not files:
            # Create waiting message on black background
            message = "Waiting for detections..."
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = self.cfg.strip_left_margin
            text_y = self.cfg.strip_height // 2 + 10
            
            # Add white text directly on black canvas
            cv2.putText(canvas, message, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imwrite(str(self.cfg.strip_file), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            log.debug("Created blank film strip")
            return
        
        # Render images on black background
        x = self.cfg.strip_left_margin
        for i, fp in enumerate(files):
            try:
                img = cv2.imread(str(fp))
                if img is None:
                    log.warning("Could not load image: %s", fp)
                    continue
                
                # Get timestamp from filename
                ts = fp.stem.replace("filmstrip_", "")
                try:
                    txt = datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%I:%M %p")
                except ValueError:
                    txt = "??:??"
                
                # Calculate dimensions - reserve more space for timestamp overlay
                img_height = self.cfg.strip_height - 50  # More space for text overlay
                h, w = img.shape[:2]
                
                # Scale to fit
                available_width = (self.cfg.strip_width - self.cfg.strip_left_margin - 
                                 (self.cfg.strip_margin * (len(files) - 1))) // len(files)
                cell_width = max(150, min(available_width, 300))
                
                width_scale = cell_width / w
                height_scale = img_height / h
                scale = min(width_scale, height_scale)
                
                new_width = int(w * scale)
                new_height = int(h * scale)
                
                if new_width > 0 and new_height > 0:
                    resized = cv2.resize(img, (new_width, new_height))
                    
                    # Position image in canvas
                    img_y = 10  # Start from top with some margin
                    end_x = min(x + new_width, self.cfg.strip_width)
                    end_y = min(img_y + new_height, self.cfg.strip_height - 40)  # Leave space for text
                    
                    if x < self.cfg.strip_width:
                        # Place image directly on black canvas
                        canvas[img_y:end_y, x:end_x] = resized[:end_y-img_y, :end_x-x]
                        
                        # Add timestamp overlay with black background directly on the image
                        text_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                        text_x = x + (new_width - text_size[0]) // 2
                        text_y = end_y + 25  # Below the image
                        
                        # Draw black rectangle for text background
                        cv2.rectangle(canvas, 
                                    (text_x - 5, text_y - 18), 
                                    (text_x + text_size[0] + 5, text_y + 5), 
                                    (0, 0, 0), -1)
                        
                        # Add white text on black background
                        cv2.putText(canvas, txt, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        log.debug("Added detection %d/%d at %s", i+1, len(files), txt)
                    
                    x += new_width + self.cfg.strip_margin
                    
            except Exception as e:
                log.error("Error processing film strip image %s: %s", fp, e)
                continue
        
        # Add info text in bottom right corner
        if len(files) > 0:
            info_text = f"{len(files)} recent detections"
            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            info_x = self.cfg.strip_width - text_size[0] - 20
            info_y = self.cfg.strip_height - 10
            
            # Black background for info text
            cv2.rectangle(canvas, 
                        (info_x - 5, info_y - 15), 
                        (info_x + text_size[0] + 5, info_y + 3), 
                        (0, 0, 0), -1)
            
            # White text
            cv2.putText(canvas, info_text, (info_x, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Make sure we're writing to JPG file
        jpg_file = self.cfg.strip_file
        print(f"Updating film strip at: {jpg_file}")
        
        # Save as high-quality JPG
        cv2.imwrite(str(jpg_file), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        log.debug("Updated film strip with %d detections", len(files))

###############################################################################
# DetectionService orchestrator
###############################################################################

class DetectionService:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.motion = MotionDetector(self.cfg)
        self.cats = CatDetector(self.cfg)
        self.film = FilmStrip(self.cfg)

        # Smaller queue with smart dropping strategy
        self.queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=5)
        self.running = False
        self._worker: Optional[threading.Thread] = None
        self._periodic: Optional[threading.Thread] = None
        self._last_motion = 0.0
        self._latest: Optional[np.ndarray] = None
        
        # Frame dropping stats
        self._frames_processed = 0
        self._frames_dropped = 0
        self._last_stats_time = time.time()

    # ------------ control --------------
    def start(self):
        if self.running:
            return
        self.running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._periodic = threading.Thread(target=self._periodic_loop, daemon=True)
        self._periodic.start()
        log.info("Detection threads started")

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.queue.put(None)
        if self._worker:
            self._worker.join(timeout=2)
        if self._periodic:
            self._periodic.join(timeout=2)
        log.info("Detection threads stopped")

    def enqueue(self, jpeg: bytes):
        """Smart frame enqueueing with adaptive dropping"""
        try:
            # Try to put frame without blocking
            self.queue.put_nowait(jpeg)
            self._frames_processed += 1
        except queue.Full:
            # Queue is full - implement smart dropping
            self._frames_dropped += 1
            
            # Clear old frames and add new one (keep only latest)
            try:
                # Remove oldest frame if queue is full
                self.queue.get_nowait()
                self.queue.put_nowait(jpeg)
                self._frames_processed += 1
            except queue.Empty:
                # Queue became empty between checks, just add the frame
                try:
                    self.queue.put_nowait(jpeg)
                    self._frames_processed += 1
                except queue.Full:
                    # Still full, drop this frame
                    pass
            
            # Log stats periodically instead of every drop
            now = time.time()
            if now - self._last_stats_time > 10:  # Log every 10 seconds
                total = self._frames_processed + self._frames_dropped
                drop_rate = (self._frames_dropped / total * 100) if total > 0 else 0
                log.info("Detection stats: %d processed, %d dropped (%.1f%% drop rate)", 
                        self._frames_processed, self._frames_dropped, drop_rate)
                self._last_stats_time = now
                # Reset counters
                self._frames_processed = 0
                self._frames_dropped = 0

    def status(self) -> dict:
        total = self._frames_processed + self._frames_dropped
        drop_rate = (self._frames_dropped / total * 100) if total > 0 else 0
        
        return {
            "enabled": self.running,
            "yolo_available": self.cats.model is not None,
            "log_dir": str(self.cfg.root_dir),
            "crop_dir": str(self.cfg.crop_dir),
            "async_processing": self.running,
            "queue_size": self.queue.qsize(),
            "processing_stats": {
                "frames_processed": self._frames_processed,
                "frames_dropped": self._frames_dropped,
                "drop_rate_percent": round(drop_rate, 1)
            },
            "film_strip": {
                "images_count": len(list(self.cfg.crop_dir.glob("filmstrip_*.jpg"))),
                "max_images": self.cfg.strip_max_images,
                "interval_seconds": self.cfg.film_interval,
                "window_seconds": self.cfg.film_window,
                "next_save_in_seconds": max(0, self.cfg.film_interval - (time.time() - self.film.last_save)),
                "strip_file": str(self.cfg.strip_file)
            }
        }

    # ------------ worker loops ---------
    def _worker_loop(self):
        while self.running:
            try:
                # Shorter timeout to process frames faster
                item = self.queue.get(timeout=0.5)
                if item is None:  # Poison pill
                    break
                self._process_frame(item)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                log.error("Worker loop error: %s", e)

    def _process_frame(self, jpeg: bytes):
        try:
            frame_array = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                return
            
            # Always update latest frame for periodic checks
            self._latest = frame.copy()
            
            # Skip motion detection on every Nth frame to reduce load
            skip_motion = (self._frames_processed % 3 != 0)  # Only check motion every 3rd frame
            
            if not skip_motion:
                # Check motion
                motion = self.motion(frame)
                if motion:
                    self._last_motion = time.time()
                    self._log_event("motion")
                    
                    # Check for cats only when motion is detected
                    boxes = self.cats(frame)
                    if boxes:
                        self._log_event("cat_detected", len(boxes))
                        self.film.maybe_save(frame, boxes)
                    
        except Exception as e:
            log.error("Frame processing error: %s", e)

    def _periodic_loop(self):
        while self.running:
            try:
                time.sleep(self.cfg.periodic_interval)
                if not self.running or self._latest is None:
                    continue
                    
                log.debug("Performing periodic cat check")
                boxes = self.cats(self._latest)
                if boxes:
                    log.info("Periodic check found %d cat(s)", len(boxes))
                    self._log_event("periodic_cat_detected", len(boxes))
                    self.film.maybe_save(self._latest, boxes)
                    
            except Exception as e:
                log.error("Periodic loop error: %s", e)

    def _log_event(self, event_type: str, count: int = 0):
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "detections": count
            }
            log_file = self.cfg.root_dir / "events.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            log.error("Event logging error: %s", e)

###############################################################################
# Backward compatibility API
###############################################################################

# Global service instance
_service: Optional[DetectionService] = None

def init_detection():
    """Initialize detection components - backward compatibility"""
    global _service
    try:
        if _service is not None:
            log.info("Detection service already initialized")
            return
            
        _service = DetectionService()
        _service.start()  # Auto-start the service
        log.info("Detection service initialized and auto-enabled")
        print("Detection service initialized and auto-enabled")  # Keep print for app.py
    except Exception as e:
        log.error("Detection initialization failed: %s", e)
        print(f"Detection initialization failed: {e}")  # Keep print for app.py
        raise e

def process_frame_for_detection(jpeg_frame: bytes) -> bytes:
    """Queue frame for async detection - returns original frame immediately"""
    if _service and _service.running:
        _service.enqueue(jpeg_frame)
    return jpeg_frame

def toggle_detection(enabled: bool) -> bool:
    """Enable/disable detection"""
    global _service
    if enabled:
        if not _service:
            init_detection()  # This will auto-start
        elif not _service.running:
            _service.start()
    else:
        if _service:
            _service.stop()
    return _service.running if _service else False

def get_detection_status() -> dict:
    """Get current detection status"""
    if _service:
        return _service.status()
    return {"enabled": False, "error": "Service not initialized"}

def refresh_film_strip() -> dict:
    """Manually refresh the detection strip"""
    try:
        if _service:
            _service.film.refresh()
            return {"success": True, "message": "Film strip refreshed"}
        return {"success": False, "error": "Service not initialized"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Legacy functions for compatibility
def detect_motion(gray_frame):
    """Legacy function - use DetectionService instead"""
    if _service:
        return _service.motion(cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR) if len(gray_frame.shape) == 2 else gray_frame)
    return False

def detect_cats(frame):
    """Legacy function - use DetectionService instead"""
    if _service:
        return _service.cats(frame)
    return []

def log_detection_event(event_type: str, count: int = 0):
    """Legacy function - use DetectionService instead"""
    if _service:
        _service._log_event(event_type, count)

def create_detections_strip():
    """Legacy function - use DetectionService instead"""
    if _service:
        _service.film.refresh()
