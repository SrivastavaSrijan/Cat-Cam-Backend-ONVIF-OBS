"""
Detection service for managing motion detection processes.
"""
import subprocess
import os
from threading import Lock


class DetectionService:
    """Service for managing motion detection operations"""
    
    def __init__(self, log_file_path="motion_detection.log"):
        self.log_file_path = log_file_path
        self.process = None
        self.lock = Lock()
    
    def start_detection(self):
        """Start the detection process"""
        with self.lock:
            if self.process and self.process.poll() is None:
                return {"error": "Detection script is already running"}

            # Clear the log file on start
            try:
                with open(self.log_file_path, "w") as log_file:
                    log_file.truncate(0)
            except Exception as e:
                return {"error": f"Failed to clear log file: {str(e)}"}

            try:
                with open(self.log_file_path, "a") as log_file:
                    self.process = subprocess.Popen(
                        ["python", "-c", "import detection_to_transform; detection_to_transform.init()"],          
                        stdout=log_file,
                        stderr=log_file,
                    )
                return {"success": "Detection script started"}
            except Exception as e:
                return {"error": f"Failed to start script: {str(e)}"}
    
    def stop_detection(self):
        """Stop the detection process"""
        with self.lock:
            if not self.process or self.process.poll() is not None:
                return {"error": "No detection script is running"}

            try:
                self.process.terminate()
                self.process.wait()

                # Clear the log file on stop
                with open(self.log_file_path, "w") as log_file:
                    log_file.truncate(0)

                return {"success": "Detection script stopped"}
            except Exception as e:
                return {"error": f"Failed to stop script: {str(e)}"}
    
    def fetch_logs(self):
        """Fetch the logs from the detection script"""
        try:
            with open(self.log_file_path, "r") as log_file:
                logs = log_file.read()
            return {"logs": logs}
        except FileNotFoundError:
            return {"logs": "No logs available"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_status(self):
        """Get the status of the detection process"""
        with self.lock:
            if self.process and self.process.poll() is None:
                return {"running": True, "exit_code": None}
            elif self.process:
                return {"running": False, "exit_code": self.process.poll()}
            else:
                return {"running": False, "exit_code": None}
