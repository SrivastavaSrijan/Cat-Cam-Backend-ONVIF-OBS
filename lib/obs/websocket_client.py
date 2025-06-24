import json
import websocket
import threading
import time
import logging

from config import CANVAS_HEIGHT, CANVAS_WIDTH


class OBSWebSocketClient:
    def __init__(self, url, password, max_retries=5, retry_backoff=2):
        self.url = url
        self.password = password

        # WebSocket references
        self.ws = None

        # State flags
        self.connected = False
        self._connecting = False  # Guard so we don’t spawn multiple run_forever
        self._reconnecting = False  # Guard for reconnect logic

        # Thread safety
        self._lock = threading.Lock()

        # Keep-alive management
        self.keep_alive_thread = None
        self.keep_alive_running = False

        # Reconnection config
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.retry_count = 0

        # You might also have these if used below
        self.scene_item_ids = {}  # example from your code
        self.current_highlighted_source = None  # Track currently highlighted source

    # ------------------------------------------------------
    #               WebSocket Event Callbacks
    # ------------------------------------------------------
    def on_open(self, ws):
        logging.info("WebSocket connection opened.")
        with self._lock:
            self.connected = True
            self._connecting = False
            self._reconnecting = False
            self.retry_count = 0

        

    def on_close(self, ws, close_status_code, close_msg):
        logging.warning("WebSocket connection closed.")
        with self._lock:
            self.connected = False
            self._connecting = False
        self.stop_keep_alive()
        self.reconnect_if_needed()

    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")
        with self._lock:
            self.connected = False
            self._connecting = False
        self.stop_keep_alive()
        self.reconnect_if_needed()

    # ------------------------------------------------------
    #                Keep Alive Management
    # ------------------------------------------------------
    def start_keep_alive(self):
        """
        Ensures there is exactly one keep-alive thread running.
        """
        with self._lock:
            # Stop an old keep-alive thread if it’s still around
            self.stop_keep_alive()

            if not self.connected:
                return

            self.keep_alive_running = True
            self.keep_alive_thread = threading.Thread(
                target=self._keep_alive_loop,
                daemon=True
            )
            self.keep_alive_thread.start()

    def stop_keep_alive(self):
        """
        Stop the KeepAlive thread if running.
        """
        self.keep_alive_running = False
        if self.keep_alive_thread and self.keep_alive_thread.is_alive():
            self.keep_alive_thread.join(timeout=5)
        self.keep_alive_thread = None

    def _keep_alive_loop(self):
        """
        Sends periodic pings or small messages to keep the WebSocket alive.
        """
        while True:
            with self._lock:
                if not self.keep_alive_running or not self.connected:
                    break

            try:
                # self.ws.send("ping")
                logging.debug("Keep-alive ping sent.")
            except Exception as e:
                logging.error(f"Keep-alive ping failed: {e}")
                # If we can’t send, likely disconnected – stop and reconnect
                with self._lock:
                    self.connected = False
                self.stop_keep_alive()
                self.reconnect_if_needed()
                break

            time.sleep(15)  # Adjust keep-alive frequency as needed

    # ------------------------------------------------------
    #         Connection / Reconnection Logic
    # ------------------------------------------------------
    def connect(self, timeout=10):
        """
        Establish the WebSocket connection (once). Returns True if connected.
        Waits up to `timeout` seconds for on_open to set `connected`.
        """
        with self._lock:
            if self.connected:
                logging.debug("Already connected.")
                return True
            if self._connecting:
                logging.debug("Connection attempt already in progress.")
                return False

            self._connecting = True

        logging.info(f"Starting WebSocket connect to {self.url} ...")
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_error=self.on_error,
            on_close=self.on_close,
            on_message=self.on_message,  # keep from your existing code
        )
        wst = threading.Thread(target=self.ws.run_forever, daemon=True)
        wst.start()

        # Wait for on_open to set self.connected, or for timeout
        start_time = time.time()
        while not self.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if self.connected:
            logging.info("Successfully connected to OBS WebSocket.")
            return True
        else:
            logging.error("Failed to connect within timeout.")
            with self._lock:
                self._connecting = False
            return False

    def reconnect_if_needed(self):
        """
        Attempt a re-connect with exponential backoff. Ensures we do not
        spawn multiple overlaps or keep retrying if max_retries is reached.
        """
        with self._lock:
            if self.connected:
                return  # No need to reconnect
            if self._reconnecting:
                return  # Already reconnecting
            if self.retry_count >= self.max_retries:
                logging.error("Max reconnection retries reached. Not reconnecting.")
                return
            self._reconnecting = True
            self.retry_count += 1
            backoff = self.retry_backoff ** self.retry_count

        logging.warning(f"Reconnecting attempt #{self.retry_count} in {backoff}s...")
        time.sleep(backoff)

        if self.connect():
            logging.info("Reconnected successfully.")
        else:
            with self._lock:
                self._reconnecting = False
            # Recursively keep trying until max_retries is reached
            self.reconnect_if_needed()

    def close(self):
        """
        Gracefully close the WebSocket connection.
        """
        with self._lock:
            self.connected = False
            self._connecting = False
            self._reconnecting = False
            self.stop_keep_alive()
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass
            self.ws = None

    def on_message(self, ws, message):
        try:
            response = json.loads(message)
            op = response.get("op")
            if op == 0:  # Hello message
                identify_payload = {
                    "op": 1,
                    "d": {"rpcVersion": 1, "eventSubscriptions": 33},
                }
                ws.send(json.dumps(identify_payload))
                logging.debug("Sent Identify payload.")
            elif op == 2:  # Identified
                self.connected = True
                self.start_keep_alive()
                self.start_refresh_thread();
                logging.info("Connected and identified with OBS WebSocket.")
            elif op == 7:  # RequestResponse
                self.handle_request_response(response)
        except Exception as e:
            logging.error(f"Error in WebSocket message handling: {e}")

    def handle_request_response(self, response):
        request_type = response["d"].get("requestType")
        request_status = response["d"].get("requestStatus", {})
        request_id = response["d"].get("requestId", "")
        
        if request_status.get("result"):
            logging.info(f"Successfully executed request: {request_type}")
            
            if request_type == "GetSceneItemList":
                items = response["d"]["responseData"].get("sceneItems", [])
                self.scene_item_ids.clear()
                # Remove sources we don't want to transform: Background, Please wait, and Detections
                items = [item for item in items if item["sourceName"] not in ["Background", "Please wait!", "Detections"]]
                for item in items:
                    self.scene_item_ids[item["sourceName"]] = item["sceneItemId"]
                logging.info(f"Retrieved scene sources: {self.scene_item_ids}")
                self.connected = True  # Signal that the response has been processed
                
            elif request_type == "GetSceneItemTransform":
                # Handle transform responses
                if hasattr(self, '_pending_transform_requests') and request_id in self._pending_transform_requests:
                    transform_data = response["d"]["responseData"].get("sceneItemTransform", {})
                    self._pending_transform_requests[request_id] = transform_data
                    
        else:
            logging.error(f"Failed request: {request_type}, Reason: {request_status.get('comment')}")
            
            # Mark failed transform requests
            if request_type == "GetSceneItemTransform" and hasattr(self, '_pending_transform_requests') and request_id in self._pending_transform_requests:
                self._pending_transform_requests[request_id] = {"error": request_status.get('comment', 'Unknown error')}

    def switch_scene(self, scene_name):
        """
        Switch the active OBS scene with retries.
        """
        if not self.connected:
            logging.warning("Not connected to OBS WebSocket.")
            return

        payload = {
            "op": 6,
            "d": {
                "requestType": "SetCurrentProgramScene",
                "requestId": f"switch_scene_{int(time.time())}",
                "requestData": {"sceneName": scene_name},
            },
        }

        for attempt in range(self.max_retries):
            try:
                with self._lock:
                    self.ws.send(json.dumps(payload))
                logging.info(f"Switched scene to: {scene_name}")
                return
            except Exception as e:
                logging.error(f"Failed to switch scene. Attempt {attempt + 1}: {e}")
                time.sleep(self.retry_backoff ** (attempt + 1))

        logging.error("Failed to switch scene after maximum retries.")

    def retrieve_scene_sources(self, scene_name="Mosaic"):
        """
        Retrieve all sources in the specified OBS scene with retries.
        """
        if not self.connected:
            logging.warning("Not connected to OBS WebSocket.")
            return

        payload = {
            "op": 6,
            "d": {
                "requestType": "GetSceneItemList",
                "requestId": f"get_scene_items_{scene_name}",
                "requestData": {"sceneName": scene_name},
            },
        }

        for attempt in range(self.max_retries):
            try:
                with self._lock:
                    self.ws.send(json.dumps(payload))
                logging.info("Scene sources requested.")
                time.sleep(0.5)  # Allow time for the response
                return
            except Exception as e:
                logging.error(f"Failed to retrieve scene sources. Attempt {attempt + 1}: {e}")
                time.sleep(self.retry_backoff ** (attempt + 1))

        logging.error("Failed to retrieve scene sources after maximum retries.")

    def set_transform(self, transform_data):
        """
        Apply a transform to a scene item with retries.
        """
        if not self.connected:
            logging.warning("Not connected to OBS WebSocket.")
            return

        payload = {
            "op": 6,
            "d": {
                "requestType": "SetSceneItemTransform",
                "requestId": f"transform_{int(time.time())}",
                "requestData": {
                    "sceneName": "Mosaic",
                    "sceneItemId": transform_data.get("sceneItemId"),
                    "sceneItemTransform": transform_data,
                },
            },
        }

        for attempt in range(self.max_retries):
            try:
                with self._lock:
                    self.ws.send(json.dumps(payload))
                logging.info(f"Applied transform: {transform_data}")
                return
            except Exception as e:
                logging.error(f"Failed to apply transform. Attempt {attempt + 1}: {e}")
                time.sleep(self.retry_backoff ** (attempt + 1))

        logging.error("Failed to apply transform after maximum retries.")
    
    def update_obs_layout(self, scene_name="Mosaic", active_source=None):
        """
        Arrange sources at absolute positions/sizes using bounding boxes.
        :param scene_name: Name of the scene to modify.
        :param active_source: Optional name of a "detected" source to highlight.
        """
        try:
            # Track the currently highlighted source
            self.current_highlighted_source = active_source
            
            # Retrieve all sources from the given scene
            sources = list(self.scene_item_ids.items())  # Convert to list of tuples
            if not sources:
                logging.error("No sources found to layout.")
                return

            # Use the utility function to calculate layout
            layout_instructions = self._calculate_layout_transforms(scene_name, active_source)
            
            if not layout_instructions:
                logging.error("Failed to calculate layout instructions.")
                return

            # Apply each transform via obs_client
            for src_name, transform in layout_instructions.items():
                try:
                    self.set_transform(transform)
                    logging.info(f"Applied bounding-box transform for {src_name}: {transform}")
                except Exception as e:
                    logging.error(f"Failed to apply transform to {src_name}: {e}")

        except Exception as ex:
            logging.error(f"An unexpected error occurred during layout update: {str(ex)}")
            self.reconnect_if_needed()



    def start_refresh_thread(self):
        """
        Starts the refresh_sources function in a background thread if not already running.
        Ensures only one thread exists to prevent duplicate toggling.
        """
        with self._lock:
            if hasattr(self, "refresh_thread") and self.refresh_thread and self.refresh_thread.is_alive():
                logging.info("Refresh thread is already running. Skipping new thread creation.")
                return

            self.refresh_thread_running = True
            self.refresh_thread = threading.Thread(target=self._refresh_sources_loop, daemon=True)
            self.refresh_thread.start()
            logging.info("Refresh thread started successfully.")

    def stop_refresh_thread(self):
        """
        Stops the refresh_sources thread gracefully.
        """
        with self._lock:
            self.refresh_thread_running = False

        if hasattr(self, "refresh_thread") and self.refresh_thread and self.refresh_thread.is_alive():
            self.refresh_thread.join(timeout=5)
            logging.info("Refresh thread stopped.")

    def _refresh_sources_loop(self):
        """
        Continuously toggles visibility for all sources every 30 seconds,
        making them invisible for 100ms before restoring visibility.
        """
        while self.refresh_thread_running:
          
            try:
                # Make the source invisible
                self.switch_scene("Please Wait")
                logging.info(f"Temporarily hid Mosaic")

                # Wait for 3s before restoring visibility
                time.sleep(3)

                # Make the source visible again
                self.switch_scene("Mosaic")
                logging.info(f"Restored visibility for Mosaic")

            except Exception as e:
                logging.error(f"Failed to toggle visibility for Mosaic: {e}")

            # Wait for 30 mins before toggling again
            time.sleep(60 * 10)

    def start_fullscreen_projector(self, scene_name, monitor_index=0):
        """
        Start a fullscreen projector for a specific scene.
        :param scene_name: Name of the scene to project
        :param monitor_index: Monitor index (0 for primary, 1 for secondary, etc.)
        """
        if not self.connected:
            logging.warning("Not connected to OBS WebSocket.")
            return {"error": "Not connected to OBS"}

        payload = {
            "op": 6,
            "d": {
                "requestType": "OpenSourceProjector",
                "requestId": f"projector_{int(time.time())}",
                "requestData": {
                    "sourceName": scene_name,
                    "monitorIndex": monitor_index,
                    "projectorGeometry": None  # Fullscreen
                },
            },
        }

        try:
            with self._lock:
                self.ws.send(json.dumps(payload))
            logging.info(f"Started fullscreen projector for scene: {scene_name}")
            return {"success": f"Fullscreen projector started for {scene_name}"}
        except Exception as e:
            logging.error(f"Failed to start projector: {e}")
            return {"error": str(e)}

    def close_projector(self, projector_type="source"):
        """
        Close all projectors of specified type.
        :param projector_type: "source", "scene", "multiview", or "all"
        """
        if not self.connected:
            logging.warning("Not connected to OBS WebSocket.")
            return {"error": "Not connected to OBS"}

        payload = {
            "op": 6,
            "d": {
                "requestType": "CloseProjector",
                "requestId": f"close_projector_{int(time.time())}",
                "requestData": {
                    "projectorType": projector_type
                },
            },
        }

        try:
            with self._lock:
                self.ws.send(json.dumps(payload))
            logging.info(f"Closed {projector_type} projectors")
            return {"success": f"Closed {projector_type} projectors"}
        except Exception as e:
            logging.error(f"Failed to close projector: {e}")
            return {"error": str(e)}

    def start_virtual_camera(self):
        """
        Start the OBS virtual camera.
        """
        if not self.connected:
            logging.warning("Not connected to OBS WebSocket.")
            return {"error": "Not connected to OBS"}

        payload = {
            "op": 6,
            "d": {
                "requestType": "StartVirtualCam",
                "requestId": f"start_vcam_{int(time.time())}",
                "requestData": {},
            },
        }

        try:
            with self._lock:
                self.ws.send(json.dumps(payload))
            logging.info("Started virtual camera")
            return {"success": "Virtual camera started"}
        except Exception as e:
            logging.error(f"Failed to start virtual camera: {e}")
            return {"error": str(e)}

    def stop_virtual_camera(self):
        """
        Stop the OBS virtual camera.
        """
        if not self.connected:
            logging.warning("Not connected to OBS WebSocket.")
            return {"error": "Not connected to OBS"}

        payload = {
            "op": 6,
            "d": {
                "requestType": "StopVirtualCam",
                "requestId": f"stop_vcam_{int(time.time())}",
                "requestData": {},
            },
        }

        try:
            with self._lock:
                self.ws.send(json.dumps(payload))
            logging.info("Stopped virtual camera")
            return {"success": "Virtual camera stopped"}
        except Exception as e:
            logging.error(f"Failed to stop virtual camera: {e}")
            return {"error": str(e)}

    def get_virtual_camera_status(self):
        """
        Get the status of the virtual camera.
        """
        if not self.connected:
            logging.warning("Not connected to OBS WebSocket.")
            return {"error": "Not connected to OBS"}

        payload = {
            "op": 6,
            "d": {
                "requestType": "GetVirtualCamStatus",
                "requestId": f"vcam_status_{int(time.time())}",
                "requestData": {},
            },
        }

        try:
            with self._lock:
                self.ws.send(json.dumps(payload))
            logging.info("Requested virtual camera status")
            return {"success": "Virtual camera status requested"}
        except Exception as e:
            logging.error(f"Failed to get virtual camera status: {e}")
            return {"error": str(e)}

    def _calculate_layout_transforms(self, scene_name="Mosaic", active_source=None):
        """
        Calculate what the layout transforms should be based on the current state.
        This is the same logic as update_obs_layout but without actually applying them.
        
        :param scene_name: Name of the scene
        :param active_source: Optional active source for highlighted layout
        :return: Dictionary of layout instructions
        """
        if not self.scene_item_ids:
            return {}
            
        sources = list(self.scene_item_ids.items())
        if not sources:
            return {}

        layout_instructions = {}


        if active_source and active_source in self.scene_item_ids:
            # Active-source layout with space for detection strip
            active_item_id = self.scene_item_ids.get(active_source)
            active_width = int(CANVAS_WIDTH * 0.75)  # 960px
            active_height = int(CANVAS_HEIGHT * 0.75)  # ~690px (75% of available height)

            layout_instructions[active_source] = {
                "sceneItemId": active_item_id,
                "positionX": 0,
                "positionY":  0, # Top-left corner
                "boundsType": "OBS_BOUNDS_SCALE_INNER",
                "boundsWidth": active_width,
                "boundsHeight": active_height,
            }
            
            others = [src for src, _ in sources if src != active_source]
            side_width = CANVAS_WIDTH - active_width  # ~320px
            side_height_each = CANVAS_HEIGHT // len(others) if others else 0

            for idx, src_name in enumerate(others):
                item_id = self.scene_item_ids.get(src_name)
                if item_id:
                    layout_instructions[src_name] = {
                        "sceneItemId": item_id,
                        "positionX": active_width,
                        "positionY": idx * side_height_each,
                        "boundsType": "OBS_BOUNDS_SCALE_INNER",
                        "boundsWidth": side_width,
                        "boundsHeight": side_height_each,
                    }
        else:
            # Default 2×2 Grid Layout with space for detection strip
            cell_width = CANVAS_WIDTH // 2   # 640px
            cell_height = CANVAS_HEIGHT // 2  # ~405px (half of available height)

            for idx, (src_name, scene_item_id) in enumerate(sources):
                col = idx % 2  # 0 or 1
                row = idx // 2  # 0 or 1
                layout_instructions[src_name] = {
                    "sceneItemId": scene_item_id,
                    "positionX": col * cell_width,
                    "positionY": row * cell_height,
                    "boundsType": "OBS_BOUNDS_SCALE_INNER",
                    "boundsWidth": cell_width,
                    "boundsHeight": cell_height,
                }

        return layout_instructions

    def get_current_highlighted_source(self):
        """
        Determine the currently highlighted source by actually querying OBS for current transforms
        and finding which one has the largest area.
        
        :return: Dictionary with highlighted source name or None if grid layout
        """
        if not self.connected or not self.ws:
            return {"error": "Not connected to OBS"}
            
        if not self.scene_item_ids:
            self.retrieve_scene_sources()
            
        if not self.scene_item_ids:
            return {"error": "No scene items available"}
        
        # Get actual transforms from OBS for all sources
        actual_transforms = {}
        for source_name, item_id in self.scene_item_ids.items():
            transform = self._get_actual_transform_from_obs(source_name, item_id)
            if transform and "error" not in transform:
                actual_transforms[source_name] = transform
        
        if not actual_transforms:
            return {"error": "Could not retrieve any transforms from OBS"}
        
        # Find the source with the largest area
        max_area = 0
        highlighted_source = None
        
        for source_name, transform in actual_transforms.items():
            width = transform.get("boundsWidth", transform.get("scaleX", 1) * 100)  # Fallback to scale
            height = transform.get("boundsHeight", transform.get("scaleY", 1) * 100)
            
            # Convert to numbers if they're strings
            try:
                width = float(width)
                height = float(height)
                area = width * height
                
                if area > max_area:
                    max_area = area
                    highlighted_source = source_name
            except (ValueError, TypeError):
                continue
        
        if highlighted_source:
            # Check if this source is significantly larger than others (highlighted)
            expected_highlighted_area = (CANVAS_WIDTH * 0.75) * (CANVAS_HEIGHT * 0.75)
            
            if max_area > expected_highlighted_area * 0.8:  # 80% of expected highlighted size
                return {"highlighted_source": highlighted_source}
        
        return {"highlighted_source": None, "layout": "grid"}
    
    def _get_actual_transform_from_obs(self, source_name, item_id):
        """
        Actually query OBS for the current transform of a specific scene item.
        """
        if not self.connected or not self.ws:
            return {"error": "Not connected to OBS"}
            
        payload = {
            "op": 6,
            "d": {
                "requestType": "GetSceneItemTransform",
                "requestId": f"get_transform_{source_name}_{int(time.time())}",
                "requestData": {
                    "sceneName": "Mosaic",
                    "sceneItemId": item_id
                },
            },
        }
        
        # Store the response in a class variable that the message handler can update
        self._pending_transform_requests = getattr(self, '_pending_transform_requests', {})
        request_id = payload["d"]["requestId"]
        self._pending_transform_requests[request_id] = None
        
        try:
            with self._lock:
                self.ws.send(json.dumps(payload))
            
            # Wait for response (up to 2 seconds)
            for _ in range(20):  # 20 * 0.1 = 2 seconds
                if request_id in self._pending_transform_requests and self._pending_transform_requests[request_id] is not None:
                    result = self._pending_transform_requests[request_id]
                    del self._pending_transform_requests[request_id]
                    return result
                time.sleep(0.1)
            
            # Timeout - clean up
            if request_id in self._pending_transform_requests:
                del self._pending_transform_requests[request_id]
            return {"error": f"Timeout getting transform for {source_name}"}
            
        except Exception as e:
            return {"error": f"Failed to get transform for {source_name}: {str(e)}"}
        
    def get_highlighted_source_name(self):
        """
        Simple method to just get the name of the currently highlighted source.
        
        :return: Source name string or None if not found
        """
        result = self.get_current_highlighted_source()
        if "error" in result:
            return None
        return result.get("highlighted_source")


