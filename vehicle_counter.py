"""
YOLO Object Counter with Line Crossing Detection
Counts objects that cross a virtual line, with visual feedback
"""

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import supervision as sv
from typing import Dict, Tuple, Optional, List
from collections import defaultdict, deque
from dataclasses import dataclass, field
import time


@st.cache_resource
def load_yolo_model(model_name: str) -> YOLO:
    """Cache the YOLO model to prevent reloading on every UI interaction"""
    return YOLO(model_name)


@dataclass
class DetectionCount:
    """Detection and counting results"""
    total: int = 0
    in_count: int = 0
    out_count: int = 0
    counts_by_class: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    zone_occupancy: int = 0


class LineZone:
    """Custom line crossing detection using geometric cross-product method"""
    
    def __init__(self, start: Tuple[int, int], end: Tuple[int, int]):
        self.start = np.array(start)
        self.end = np.array(end)
        self.vector = self.end - self.start
        
    def is_crossing(self, prev_pos: np.ndarray, curr_pos: np.ndarray) -> Optional[str]:
        """Check if movement from prev_pos to curr_pos crossed the line"""
        def side(point):
            return (point[1] - self.start[1]) * self.vector[0] - (point[0] - self.start[0]) * self.vector[1]
        
        prev_side = side(prev_pos)
        curr_side = side(curr_pos)
        
        # If signs differ, trajectory crossed the line
        if prev_side * curr_side < 0:
            # Check if crossing point is within line segment bounds
            t = prev_side / (prev_side - curr_side)
            cross_point = prev_pos + t * (curr_pos - prev_pos)
            
            min_x, max_x = min(self.start[0], self.end[0]), max(self.start[0], self.end[0])
            min_y, max_y = min(self.start[1], self.end[1]), max(self.start[1], self.end[1])
            
            if min_x <= cross_point[0] <= max_x and min_y <= cross_point[1] <= max_y:
                return 'in' if curr_side > 0 else 'out'
        return None


class RegionZone:
    """Rectangular region for occupancy tracking"""
    
    def __init__(self, x1, y1, x2, y2):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        
    def is_inside(self, point: np.ndarray) -> bool:
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def get_coords(self):
        return (self.x1, self.y1), (self.x2, self.y2)


class YOLOProcessor:
    """
    YOLO processor with line crossing detection and zone occupancy.
    Tracks objects and counts line crossings with visual feedback.
    """
    
    CONFIGS = {
        'vehicle': {
            'classes': {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'},
            'colors': {
                'car': (255, 107, 107),
                'motorcycle': (78, 205, 196),
                'bus': (69, 183, 209),
                'truck': (150, 206, 180)
            }
        },
        'person': {
            'classes': {0: 'person'},
            'colors': {'person': (255, 150, 255)}
        }
    }
    
    def __init__(self, model_name: str = "yolo26l.pt", config_name: str = "vehicle", 
                 mode: str = "line", confidence: float = 0.3):
        self.model = load_yolo_model(model_name)
        self.confidence = confidence
        self.mode = mode  # 'line' or 'zone'
        
        # Load config
        self.config = self.CONFIGS.get(config_name, self.CONFIGS['vehicle'])
        self.target_classes = self.config['classes']
        self.target_ids = list(self.target_classes.keys())
        self.colors = self.config.get('colors', {})
        
        # Tracker
        self.tracker = sv.ByteTrack()
        
        # Position history for crossing detection (bounded deque for memory efficiency)
        self.track_history: Dict[int, deque] = {}
        self._max_history_length = 30
        
        # Counting state
        self.counted_ids: set = set()  # Objects that have already crossed
        self.flash_timers: Dict[int, int] = {}  # Visual feedback timers
        self.inside_zone_ids: set = set()  # Objects currently inside zone
        
        # Geometry
        self.line_zone: Optional[LineZone] = None
        self.region_zone: Optional[RegionZone] = None
        
        # Results
        self.current_counts = DetectionCount()
        self.start_time = time.time()
    
    def _prune_stale_tracks(self, active_ids: set):
        """Remove tracking data for objects no longer in frame (memory optimization)"""
        stale_ids = [tid for tid in self.track_history if tid not in active_ids]
        for tid in stale_ids:
            del self.track_history[tid]
            self.flash_timers.pop(tid, None)
    
    def reset_counts(self):
        """Reset all counters"""
        self.counted_ids.clear()
        self.track_history.clear()
        self.flash_timers.clear()
        self.current_counts = DetectionCount()
        
    def set_line(self, start: Tuple[int, int], end: Tuple[int, int]):
        """Set up line crossing detection"""
        self.line_zone = LineZone(start, end)
        self.mode = "line"
        
    def set_region(self, x1: int, y1: int, x2: int, y2: int):
        """Set up zone occupancy detection"""
        self.region_zone = RegionZone(x1, y1, x2, y2)
        self.mode = "zone"
    
    def set_zone(self, polygon: np.ndarray, frame_size: Tuple[int, int]):
        """Set up zone occupancy from polygon array (dashboard compatibility)"""
        # Extract bounding box from polygon
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        x1, x2 = int(x_coords.min()), int(x_coords.max())
        y1, y2 = int(y_coords.min()), int(y_coords.max())
        self.set_region(x1, y1, x2, y2)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionCount]:
        """Process a single frame and return annotated frame with counts"""
        
        # 1. Run YOLO detection
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # 2. Filter by target classes
        mask = np.isin(detections.class_id, self.target_ids)
        detections = detections[mask]
        
        # 3. Track objects
        detections = self.tracker.update_with_detections(detections)
        
        # 4. Process based on mode
        if self.mode == "line" and self.line_zone:
            self._process_line_crossing(detections)
        elif self.mode == "zone" and self.region_zone:
            self._process_zone_occupancy(detections)
        else:
            # Default: count all unique objects
            self._process_simple_count(detections)
        
        # 5. Draw annotations
        annotated = self._annotate(frame, detections)
        
        # 6. Prune stale tracks to prevent memory leaks
        if detections.tracker_id is not None:
            active_ids = set(detections.tracker_id)
            self._prune_stale_tracks(active_ids)
        
        return annotated, self.current_counts
    
    def _process_line_crossing(self, detections: sv.Detections):
        """Detect line crossings using position history"""
        if detections.tracker_id is None:
            return
            
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue
            
            # Calculate center point of bounding box
            bbox = detections.xyxy[i]
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            
            # Store position history (using bounded deque for memory efficiency)
            if tracker_id not in self.track_history:
                self.track_history[tracker_id] = deque(maxlen=self._max_history_length)
            self.track_history[tracker_id].append(center)
            
            # Check for line crossing (only if not already counted)
            if len(self.track_history[tracker_id]) >= 2 and tracker_id not in self.counted_ids:
                pts = self.track_history[tracker_id]
                
                # Check last few segments for crossing
                for j in range(1, min(5, len(pts))):
                    start_pt = pts[-(j+1)]
                    end_pt = pts[-j]
                    crossing = self.line_zone.is_crossing(start_pt, end_pt)
                    
                    if crossing:
                        # Mark as counted
                        self.counted_ids.add(tracker_id)
                        self.flash_timers[tracker_id] = 15  # Green flash frames
                        
                        # Get class name
                        class_name = self.target_classes.get(detections.class_id[i], 'unknown')
                        
                        # Update counts
                        self.current_counts.total += 1
                        self.current_counts.counts_by_class[class_name] += 1
                        
                        if crossing == 'in':
                            self.current_counts.in_count += 1
                        else:
                            self.current_counts.out_count += 1
                        break
    
    def _process_zone_occupancy(self, detections: sv.Detections):
        """Count objects currently inside the zone"""
        current_zone_total = 0
        current_class_counts = defaultdict(int)
        self.inside_zone_ids.clear()
        
        if detections.tracker_id is None:
            self.current_counts.total = 0
            self.current_counts.zone_occupancy = 0
            self.current_counts.counts_by_class = current_class_counts
            return
        
        for i, tracker_id in enumerate(detections.tracker_id):
            bbox = detections.xyxy[i]
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            
            if self.region_zone.is_inside(center):
                current_zone_total += 1
                class_name = self.target_classes.get(detections.class_id[i], 'unknown')
                current_class_counts[class_name] += 1
                self.inside_zone_ids.add(tracker_id)
        
        self.current_counts.total = current_zone_total
        self.current_counts.zone_occupancy = current_zone_total
        self.current_counts.counts_by_class = current_class_counts
    
    def _process_simple_count(self, detections: sv.Detections):
        """Simple unique object counting"""
        if detections.tracker_id is None:
            return
            
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id not in self.counted_ids:
                self.counted_ids.add(tracker_id)
                class_name = self.target_classes.get(detections.class_id[i], 'unknown')
                self.current_counts.total += 1
                self.current_counts.counts_by_class[class_name] += 1
    
    def _annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Draw bounding boxes, labels, and geometry on frame"""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        scale = max(w, h) / 1920.0
        scale = max(0.5, min(scale, 2.0))
        
        # Colors for different states
        inside_color = (0, 255, 0)      # Green for inside zone
        outside_color = (128, 128, 128)  # Gray for outside zone
        flash_color = (0, 255, 0)        # Green flash for line crossing
        
        # Draw detections
        if detections.tracker_id is not None:
            for i, (bbox, cls_id, trk_id) in enumerate(zip(detections.xyxy, detections.class_id, detections.tracker_id)):
                cls_name = self.target_classes.get(cls_id, 'unknown')
                x1, y1, x2, y2 = map(int, bbox)
                
                # Determine color and thickness based on mode
                if self.mode == "zone":
                    if trk_id in self.inside_zone_ids:
                        color = inside_color
                        thickness = 4
                    else:
                        color = outside_color
                        thickness = 2
                else:
                    # Line mode: Use flash effect or base color
                    base_color = self.colors.get(cls_name, (200, 200, 200))
                    if trk_id in self.flash_timers and self.flash_timers[trk_id] > 0:
                        color = flash_color
                        thickness = 4
                        self.flash_timers[trk_id] -= 1
                    else:
                        color = base_color
                        thickness = 2
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label
                label = f"#{trk_id} {cls_name}"
                font_scale = 0.6 * scale
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                
                cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (255, 255, 255), 2)
        
        # Draw line or zone
        if self.mode == "line" and self.line_zone:
            cv2.line(annotated, 
                    tuple(self.line_zone.start.astype(int)), 
                    tuple(self.line_zone.end.astype(int)), 
                    (0, 255, 255), 3)  # Yellow line
        elif self.mode == "zone" and self.region_zone:
            pt1, pt2 = self.region_zone.get_coords()
            # Semi-transparent overlay
            overlay = annotated.copy()
            cv2.rectangle(overlay, pt1, pt2, (255, 100, 0), -1)
            cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
            cv2.rectangle(annotated, pt1, pt2, (255, 100, 0), 3)
        
        return annotated
