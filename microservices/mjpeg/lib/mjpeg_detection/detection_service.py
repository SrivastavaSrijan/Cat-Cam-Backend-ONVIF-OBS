
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
    # YOLO
    model_path: str = os.getenv("YOLO_MODEL_PATH", "yolov8m.pt")
    cat_class_id: int = int(os.getenv("CAT_CLASS_ID", 15))
    yolo_conf: float = float(os.getenv("YOLO_CONF", 0.5))

    # time‑bucket size for film‑strip (seconds)
    film_interval: float = float(os.getenv("FILM_STRIP_INTERVAL", 30))
    film_window: float = float(os.getenv("FILM_STRIP_WINDOW", 180))

    # Film strip layout
    strip_width: int = 1920
    strip_height: int = 270
    strip_max_images: int = 6
    strip_margin: int = 15
    strip_left_margin: int = 20

    # Sleeping‑cat periodic scan
    periodic_interval: float = 30.0  # s

    # YOLO only every N frames  (60 = 1 Hz on typical 60 fps feed)
    detect_every: int = int(os.getenv("DETECT_EVERY", 60))

    # Optional second‑pass upscale factor for tiny cameras (e.g. 2 → 2×)
    upscale_factor: int = int(os.getenv("UPSCALE_FACTOR", 1))

    # Pixels to crop from the bottom of every frame (exclude detection strip)
    bottom_crop_px: int = int(os.getenv("BOTTOM_CROP_PX", 270))

    # Log summary every N seconds
    log_interval: int = int(os.getenv("LOG_INTERVAL", 10))

    # I/O root
    root_dir: Path = Path(os.getenv("MOTION_ROOT", "motion_logs"))

    # --- debugging ----------------------------------------------------
    # Enable to dump every YOLO input frame to disk (rotating buffer of 50)
    debug_frames: bool = bool(int(os.getenv("DEBUG_FRAMES", "0")))
    debug_dir: Path = Path(os.getenv("DEBUG_DIR", "motion_logs/debug"))

    @property
    def crop_dir(self) -> Path:
        return self.root_dir / "crops"

    @property
    def strip_file(self) -> Path:
        return self.root_dir / "detections_strip.jpg"

###############################################################################
# Internal helpers
###############################################################################

def _makedirs(cfg: Config):
    cfg.root_dir.mkdir(parents=True, exist_ok=True)
    cfg.crop_dir.mkdir(parents=True, exist_ok=True)
    if cfg.debug_frames:
        cfg.debug_dir.mkdir(parents=True, exist_ok=True)
    # Clean up any old PNG files
    old_png = cfg.root_dir / "detections_strip.png"
    if old_png.exists():
        old_png.unlink()
        print(f"Removed old PNG file: {old_png}")

###############################################################################
# Cat detector (no more motion detector!)
###############################################################################

class CatDetector:
    def __init__(self, cfg: Config):
        if YOLO:
            try:
                self.model = YOLO(cfg.model_path)
                log.info("YOLO model loaded: %s", cfg.model_path)
                print(f"YOLO model loaded: {cfg.model_path}")
            except Exception as e:
                log.warning("Failed to load YOLO model %s: %s", cfg.model_path, e)
                print(f"Failed to load YOLO model: {e}")
                self.model = None
        else:
            self.model = None
            log.warning("ultralytics not installed – cat detection disabled")
            print("ultralytics not installed – cat detection disabled")
        self.cfg = cfg

    def __call__(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # Optionally crop bottom strip (e.g. 270 px filmstrip)
        if self.cfg.bottom_crop_px and frame.shape[0] > self.cfg.bottom_crop_px:
            frame = frame[:-self.cfg.bottom_crop_px, :]
        # save raw YOLO input for inspection
        if self.cfg.debug_frames:
            ts = time.strftime("%H%M%S")
            cv2.imwrite(str(self.cfg.debug_dir / f"yolo_in_{ts}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            # keep only last 50 files
            files = sorted(self.cfg.debug_dir.glob("yolo_in_*.jpg"))
            for old in files[:-50]:
                old.unlink(missing_ok=True)
        """Run YOLO once; if nothing is found and an upscale_factor > 1 is
        configured, run a second pass on an up‑sampled copy to catch small
        cats in 320‑pixel sub‑views."""
        if not self.model:
            # save an annotated preview
            if self.cfg.debug_frames and []:
                preview = frame.copy()
                ts = time.strftime("%H%M%S")
                for (x1,y1,x2,y2) in []:
                    ts = time.strftime("%H%M%S")
                    cv2.rectangle(preview, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.imwrite(str(self.cfg.debug_dir / f"yolo_out_{ts}.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return []

        boxes = self._detect(frame)
        if boxes or self.cfg.upscale_factor <= 1:
            # save an annotated preview
            if self.cfg.debug_frames and boxes:
                preview = frame.copy()
                ts = time.strftime("%H%M%S")
                for (x1,y1,x2,y2) in boxes:
                    ts = time.strftime("%H%M%S")
                    cv2.rectangle(preview, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.imwrite(str(self.cfg.debug_dir / f"yolo_out_{ts}.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return boxes

        # second‑pass upscale
        try:
            up = cv2.resize(frame, None, fx=self.cfg.upscale_factor,
                            fy=self.cfg.upscale_factor,
                            interpolation=cv2.INTER_CUBIC)
            boxes_up = self._detect(up)
            # map back to original coords
            scale = 1 / self.cfg.upscale_factor
            mapped_boxes = [(int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale))
                    for x1, y1, x2, y2 in boxes_up]
            # save an annotated preview
            if self.cfg.debug_frames and mapped_boxes:
                preview = frame.copy()
                for (x1,y1,x2,y2) in mapped_boxes:
                    cv2.rectangle(preview, (x1,y1), (x2,y2), (0,255,0), 2)
                ts = time.strftime("%H%M%S")
                cv2.imwrite(str(self.cfg.debug_dir / f"yolo_out_{ts}.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return mapped_boxes
        except Exception as e:
            log.error("Upscale detection error: %s", e)
            # save an annotated preview
            if self.cfg.debug_frames and boxes:
                preview = frame.copy()
                for (x1,y1,x2,y2) in boxes:
                    cv2.rectangle(preview, (x1,y1), (x2,y2), (0,255,0), 2)
                ts = time.strftime("%H%M%S")
                cv2.imwrite(str(self.cfg.debug_dir / f"yolo_out_{ts}.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return boxes  # fall back to first‑pass result

    # --- helper -----------------------------------------------------------
    def _detect(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            res = self.model.predict(img,
                                     conf=self.cfg.yolo_conf,
                                     classes=[self.cfg.cat_class_id],
                                     verbose=False)
            out: List[Tuple[int, int, int, int]] = []
            for r in res:
                for b in r.boxes or []:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    out.append((int(x1), int(y1), int(x2), int(y2)))
            return out
        except Exception as e:
            log.error("YOLO predict error: %s", e)
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
    def maybe_save(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> bool:
        if not boxes:
            return False
        self._save_largest(frame, boxes)
        self._purge_old()
        self._render()
        return True

    def refresh(self):
        self._render()

    # ---------- helpers
    def _purge_old(self):
        files = sorted(self.cfg.crop_dir.glob("filmstrip_*.jpg"),
                       key=lambda p: p.stat().st_mtime)
        cutoff = time.time() - self.cfg.film_window  # 180 s default
        # 1) age‑based purge
        for p in files:
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
        # 2) hard‑cap newest 200 to avoid disk bloat
        files = sorted(self.cfg.crop_dir.glob("filmstrip_*.jpg"),
                       key=lambda p: p.stat().st_mtime)
        for old in files[:-200]:
            old.unlink(missing_ok=True)

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
        blank = np.zeros((self.cfg.strip_height, self.cfg.strip_width, 3), dtype=np.uint8)
        jpg_file = self.cfg.strip_file
        print(f"Creating blank film strip at: {jpg_file}")
        cv2.imwrite(str(jpg_file), blank, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def _render(self):
        all_files = sorted(self.cfg.crop_dir.glob("filmstrip_*.jpg"),
                           key=lambda p: p.stat().st_mtime, reverse=True)

        # pick newest file in each film_interval bucket, up to strip_max_images
        files: List[Path] = []
        seen_buckets = set()
        for fp in all_files:
            bucket = int(fp.stat().st_mtime // self.cfg.film_interval)
            if bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)
            files.append(fp)
            if len(files) == self.cfg.strip_max_images:
                break

        files = list(reversed(files))  # oldest -> newest for left→right rendering
        canvas = np.zeros((self.cfg.strip_height, self.cfg.strip_width, 3), dtype=np.uint8)
        
        if not files:
            message = "Waiting for detections..."
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = self.cfg.strip_left_margin
            text_y = self.cfg.strip_height // 2 + 10
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
                
                # Calculate dimensions
                img_height = self.cfg.strip_height - 50
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
                    img_y = 10
                    end_x = min(x + new_width, self.cfg.strip_width)
                    end_y = min(img_y + new_height, self.cfg.strip_height - 40)
                    
                    if x < self.cfg.strip_width:
                        # Place image directly on black canvas
                        canvas[img_y:end_y, x:end_x] = resized[:end_y-img_y, :end_x-x]
                        
                        # Add timestamp
                        text_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                        text_x = x + (new_width - text_size[0]) // 2
                        text_y = end_y + 25
                        
                        cv2.rectangle(canvas, 
                                    (text_x - 5, text_y - 18), 
                                    (text_x + text_size[0] + 5, text_y + 5), 
                                    (0, 0, 0), -1)
                        
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
            
            cv2.rectangle(canvas, 
                        (info_x - 5, info_y - 15), 
                        (info_x + text_size[0] + 5, info_y + 3), 
                        (0, 0, 0), -1)
            
            cv2.putText(canvas, info_text, (info_x, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        jpg_file = self.cfg.strip_file
        print(f"Updating film strip at: {jpg_file}")
        cv2.imwrite(str(jpg_file), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        log.debug("Updated film strip with %d detections", len(files))

###############################################################################
# DetectionService orchestrator
###############################################################################

class DetectionService:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.cats = CatDetector(self.cfg)  # Only cat detector now!
        self.film = FilmStrip(self.cfg)

        # Bigger queue - no more artificial bottleneck
        self.queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=50)
        self.running = False
        self._worker: Optional[threading.Thread] = None
        self._latest: Optional[np.ndarray] = None
        
        # Frame stats - NEW: track frames seen
        self._frames_seen = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._last_stats_time = time.time()
        # new counters for logging frequency
        self._det_checks = 0
        self._film_saves = 0
        self._log_interval = self.cfg.log_interval

    # ------------ control --------------
    def start(self):
        if self.running:
            return
        self.running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        log.info("Detection threads started")
        print(
            f"Detection service started – running YOLO every {self.cfg.detect_every} "
            f"frames (~{60/self.cfg.detect_every:.1f} detections/sec at 60 fps)"
        )

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.queue.put(None)
        if self._worker:
            self._worker.join(timeout=2)
        log.info("Detection threads stopped")

    def enqueue(self, jpeg: bytes):
        """Smart frame enqueueing with adaptive dropping"""
        try:
            self.queue.put_nowait(jpeg)
            self._frames_processed += 1
        except queue.Full:
            self._frames_dropped += 1
            
            # Clear old frames and add new one (keep only latest)
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(jpeg)
                self._frames_processed += 1
            except queue.Empty:
                try:
                    self.queue.put_nowait(jpeg)
                    self._frames_processed += 1
                except queue.Full:
                    pass
            
            # Log stats periodically
            now = time.time()
            if now - self._last_stats_time > 10:
                total = self._frames_processed + self._frames_dropped
                drop_rate = (self._frames_dropped / total * 100) if total > 0 else 0
                log.info("Detection stats: %d seen, %d processed, %d dropped (%.1f%% drop rate)", 
                        self._frames_seen, self._frames_processed, self._frames_dropped, drop_rate)
                self._last_stats_time = now
                # Reset counters
                self._frames_seen = 0
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
            "detect_every": self.cfg.detect_every,
            "processing_stats": {
                "frames_seen": self._frames_seen,  # NEW
                "frames_processed": self._frames_processed,
                "frames_dropped": self._frames_dropped,
                "drop_rate_percent": round(drop_rate, 1),
                "detections_per_sec_est": round(60 / self.cfg.detect_every, 1),
                "yolo_runs_last_interval": self._det_checks,
                "film_saves_last_interval": self._film_saves,
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
                item = self.queue.get(timeout=0.5)
                if item is None:  # Poison pill
                    break
                self._process_frame(item)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                log.error("Worker loop error: %s", e)
            # periodic stats log
            if time.time() - self._last_stats_time >= self._log_interval:
                det_rate = self._det_checks / self._log_interval
                save_rate = self._film_saves / self._log_interval
                log.info(
                    "Detection summary: %.1f YOLO runs/sec, %.2f strip saves/sec "
                    "(queue %d)", det_rate, save_rate, self.queue.qsize()
                )
                self._det_checks = 0
                self._film_saves = 0
                self._last_stats_time = time.time()

    def _process_frame(self, jpeg: bytes):
        try:
            frame_array = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if frame is None:
                return

            # always remember latest (for periodic thread)
            self._latest = frame

            # ➜ frame-skipping: heavy work only every Nth frame
            self._frames_seen += 1
            # count how many times we actually run YOLO
            if self._frames_seen % self.cfg.detect_every:
                return  # skip this one – just counted

            self._det_checks += 1  # YOLO run

            # run YOLO directly – no motion first
            boxes = self.cats(frame)
            if boxes:
                self._log_event("cat_detected", len(boxes))
                saved = self.film.maybe_save(frame, boxes)
                if saved:
                    self._film_saves += 1

        except Exception as e:
            log.error("Frame processing error: %s", e)

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
        print("Detection service initialized and auto-enabled")
    except Exception as e:
        log.error("Detection initialization failed: %s", e)
        print(f"Detection initialization failed: {e}")
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
