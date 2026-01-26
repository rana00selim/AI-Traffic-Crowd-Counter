"""
Configuration constants for AI Traffic - Crowd Counter
Centralizes magic numbers and repeated values
"""

# ==================== VIDEO ====================
DEFAULT_FRAME_WIDTH = 854
DEFAULT_FRAME_HEIGHT = 480
ASPECT_RATIO = "16:9"

# ==================== COLORS (BGR format) ====================
TEAL_BGR = (212, 245, 0)         # Electric teal for lines/zones
GREEN_BGR = (0, 255, 0)           # Inside zone / crossing flash
GRAY_BGR = (128, 128, 128)        # Outside zone
YELLOW_BGR = (0, 255, 255)        # Default line color
WHITE_BGR = (255, 255, 255)       # Text

# ==================== DETECTION ====================
DEFAULT_CONFIDENCE = 0.3
DEFAULT_SKIP_FRAMES = 1
TRACK_HISTORY_LENGTH = 30
FLASH_DURATION_FRAMES = 15

# ==================== UI ====================
CHART_UPDATE_INTERVAL = 5  # Update chart every N frames to reduce flickering

# ==================== MODEL ====================
DEFAULT_MODEL = "yolo26l.pt"
AVAILABLE_MODELS = ["yolo26l.pt", "yolov8l.pt", "yolov8n.pt"]

# ==================== MESSAGES ====================
ERROR_NO_VIDEO = "No valid video source"
ERROR_NO_SAMPLES = "No sample videos found"
PREVIEW_TEXT = "PREVIEW - Press Start"
WEBCAM_PREVIEW_TEXT = "WEBCAM PREVIEW - Press Start"
