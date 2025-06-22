// ============================================================================
// API Request & Response Types
// ============================================================================

// Base Response Types
export interface ApiResponse<T = unknown> {
  success?: string;
  message?: string;
  data?: T;
  error?: string;
}

export interface ApiError {
  error: string;
  message?: string;
  status?: number;
}

// ============================================================================
// Camera API Types
// ============================================================================

export interface PTZPosition {
  PanTilt: {
    x: number;
    y: number;
    space?: string;
  };
  Zoom: {
    x: number;
    space?: string;
  };
}

export interface MovementLimits {
  x_max: number;
  x_min: number;
  y_max: number;
  y_min: number;
  max_velocity: number;
}

export interface CameraStatus {
  PTZPosition: PTZPosition;
  limits: MovementLimits;
  online?: boolean;
  host?: string;
  error?: string;
}

export interface CameraInfo {
  nickname: string;
  host: string;
  port: number;
  status: "online" | "offline" | "unknown";
  error?: string;
  limits?: MovementLimits;
  current_position?: PTZPosition;
  online: boolean;
}

export interface Preset {
  Name: string;
  Token: string;
  PTZPosition?: PTZPosition;
}

export interface ImagingSettings {
  brightness?: number;
  contrast?: number;
  saturation?: number;
  sharpness?: number;
  ir_cut_filter?: string;
  video_source_token?: string;
}

export interface MovementSpeed {
  panTilt: number;
  zoom: number;
}

export type MovementDirection =
  | "up"
  | "down"
  | "left"
  | "right"
  | "upleft"
  | "upright"
  | "downleft"
  | "downright"
  | "zoom_in"
  | "zoom_out";

// Request Types
export interface SwitchCameraRequest {
  nickname: string;
}

export interface GotoPresetRequest {
  presetToken: string;
}

export interface MoveRequest {
  direction: MovementDirection;
  velocity_factor?: number;
}

export interface ContinuousMoveRequest {
  direction: MovementDirection;
  speed?: number;
}

export interface MovementSpeedRequest {
  pan_tilt_speed?: number;
  zoom_speed?: number;
}

export interface NightModeRequest {
  enable: boolean;
}

// Response Types
export interface CameraStatusResponse extends ApiResponse<CameraStatus> {}
export interface CameraListResponse
  extends ApiResponse<{ cameras: CameraInfo[] }> {}
export interface PresetListResponse
  extends ApiResponse<{ presets: Preset[] }> {}
export interface ImagingSettingsResponse extends ApiResponse<ImagingSettings> {}

// ============================================================================
// OBS API Types
// ============================================================================

export interface Scene {
  name: string;
  obs_scene: string;
}

export interface CurrentScene {
  name: string;
  current_scene: string;
}

export interface TransformationState {
  layout_mode: "grid" | "highlight";
  highlighted_source: string;
}

export interface VirtualCameraStatus {
  active: boolean;
  status: string;
}

// Request Types
export interface SwitchSceneRequest {
  scene_name: string;
}

export interface TransformRequest {
  type: "grid" | "highlight";
  active_source?: string;
}

export interface StartProjectorRequest {
  source_name: string;
  monitor_index?: number;
}

export interface CloseProjectorRequest {
  projector_type: string;
}

// Response Types
export interface ScenesResponse extends ApiResponse<{ scenes: Scene[] }> {}
export interface CurrentSceneResponse extends ApiResponse<CurrentScene> {}
export interface TransformationStateResponse
  extends ApiResponse<TransformationState> {}
export interface VirtualCameraStatusResponse
  extends ApiResponse<VirtualCameraStatus> {}

// ============================================================================
// MJPEG Stream API Types
// ============================================================================

export interface MjpegStreamStatus {
  active: boolean;
  streaming: boolean;
  camera_type: string;
  clients: number;
  stream_url?: string;
  ffmpeg_pid?: number;
  error?: string;
}

export interface MjpegHealthStatus {
  status: "healthy" | "unhealthy";
  service: string;
  timestamp: number;
}

// Response Types
export interface MjpegStreamStatusResponse
  extends ApiResponse<MjpegStreamStatus> {}
export interface MjpegHealthResponse extends ApiResponse<MjpegHealthStatus> {}

// ============================================================================
// Detection Script API Types
// ============================================================================

export interface DetectionStatus {
  running: boolean;
  exit_code?: number;
}

export interface DetectionLogs {
  logs: string[];
}

// Response Types
export interface DetectionStatusResponse extends ApiResponse<DetectionStatus> {}
export interface DetectionLogsResponse extends ApiResponse<DetectionLogs> {}

// ============================================================================
// API Client Configuration
// ============================================================================

export interface ApiClientConfig {
  baseUrl: string;
  mjpegBaseUrl: string;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
}

export interface RequestOptions {
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  signal?: AbortSignal;
}

// ============================================================================
// Hook Return Types
// ============================================================================

export interface UseApiReturn {
  loading: boolean;
  error: string | null;
  clearError: () => void;
}

export interface UseCameraControlReturn extends UseApiReturn {
  // Camera operations
  getCameraStatus: (nickname: string) => Promise<CameraStatus>;
  getCameraPresets: (nickname: string) => Promise<Preset[]>;
  getCameraImaging: (nickname: string) => Promise<ImagingSettings>;
  getAllCameras: () => Promise<CameraInfo[]>;
  switchCamera: (nickname: string) => Promise<void>;
  gotoPreset: (nickname: string, presetToken: string) => Promise<void>;
  moveCamera: (
    nickname: string,
    direction: MovementDirection,
    velocityFactor?: number
  ) => Promise<void>;
  continuousMove: (
    nickname: string,
    direction: MovementDirection,
    speed?: number
  ) => Promise<void>;
  stopMove: (nickname: string) => Promise<void>;
  setMovementSpeed: (
    nickname: string,
    panTiltSpeed?: number,
    zoomSpeed?: number
  ) => Promise<void>;
  toggleNightMode: (nickname: string, enable: boolean) => Promise<void>;
}

export interface UseOBSControlReturn extends UseApiReturn {
  // OBS operations
  getScenes: () => Promise<Scene[]>;
  getCurrentScene: () => Promise<CurrentScene>;
  switchScene: (sceneName: string) => Promise<void>;
  getTransformationState: () => Promise<TransformationState>;
  applyTransformation: (
    type: "grid" | "highlight",
    activeSource?: string
  ) => Promise<void>;
  reconnect: () => Promise<void>;
  startVirtualCamera: () => Promise<void>;
  stopVirtualCamera: () => Promise<void>;
  getVirtualCameraStatus: () => Promise<VirtualCameraStatus>;
  startProjector: (sourceName: string) => Promise<void>;
  closeProjector: () => Promise<void>;
}

export interface UseMjpegStreamReturn extends UseApiReturn {
  // MJPEG operations
  getHealth: () => Promise<MjpegHealthStatus>;
  getStatus: () => Promise<MjpegStreamStatus>;
  startStream: () => Promise<void>;
  stopStream: () => Promise<void>;
}
