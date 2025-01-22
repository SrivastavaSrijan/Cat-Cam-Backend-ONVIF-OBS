import json
import websocket
import threading
import time
import logging

from shared_config import CANVAS_HEIGHT, CANVAS_WIDTH



class OBSWebSocketClient:
  import threading
import time
import websocket
import logging

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
                logging.info("Connected and identified with OBS WebSocket.")
            elif op == 7:  # RequestResponse
                self.handle_request_response(response)
        except Exception as e:
            logging.error(f"Error in WebSocket message handling: {e}")

    def handle_request_response(self, response):
        request_type = response["d"].get("requestType")
        request_status = response["d"].get("requestStatus", {})
        if request_status.get("result"):
            logging.info(f"Successfully executed request: {request_type}")
            if request_type == "GetSceneItemList":
                items = response["d"]["responseData"].get("sceneItems", [])
                self.scene_item_ids.clear()
                # Remove sourceName "Background"
                items = [item for item in items if item["sourceName"] != "Background"]
                for item in items:
                    self.scene_item_ids[item["sourceName"]] = item["sceneItemId"]
                logging.info(f"Retrieved scene sources: {self.scene_item_ids}")
                self.connected = True  # Signal that the response has been processed
        else:
            logging.error(
                f"Failed request: {request_type}, Reason: {request_status.get('comment')}"
            )

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
            # Retrieve all sources from the given scene
            sources = list(self.scene_item_ids.items())  # Convert to list of tuples
            if not sources:
                logging.error("No sources found to layout.")
                return

            layout_instructions = {}

            if active_source:
                #
                # --- Active-source layout ---
                #
                active_item_id = self.scene_item_ids.get(active_source)
                if not active_item_id:
                    logging.error(f"Active source '{active_source}' not found in scene items.")
                    return

                active_width = int(CANVAS_WIDTH * 0.75)  # 960
                active_height = int(CANVAS_HEIGHT * 0.75)  # 540

                layout_instructions[active_source] = {
                    "sceneItemId": active_item_id,
                    "positionX": 0,
                    "positionY": (CANVAS_HEIGHT - active_height) // 4,  # Center vertically
                    "boundsType": "OBS_BOUNDS_SCALE_INNER",
                    "boundsWidth": active_width,
                    "boundsHeight": active_height,
                }
                others = [src for src, _ in sources if src != active_source]
                side_width = CANVAS_WIDTH - active_width  # ~320
                side_height_each = CANVAS_HEIGHT // len(others) if others else 0

                for idx, src_name in enumerate(others):
                    item_id = self.scene_item_ids.get(src_name)
                    if not item_id:
                        logging.warning(f"Scene item ID not found for source: {src_name}")
                        continue
                    layout_instructions[src_name] = {
                        "sceneItemId": item_id,
                        "positionX": active_width,  # ~960
                        "positionY": idx * side_height_each,  # Stack vertically
                        "boundsType": "OBS_BOUNDS_SCALE_INNER",
                        "boundsWidth": side_width,  # ~320
                        "boundsHeight": side_height_each,
                    }
            else:
                #
                # --- Default 2×2 Grid Layout ---
                #
                cell_width = CANVAS_WIDTH // 2   # 640
                cell_height = CANVAS_HEIGHT // 2  # 360

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
