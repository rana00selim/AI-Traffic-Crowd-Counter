"""
Simple YOLO Object Counter with Cumulative Tracking
Counts total unique objects from start to stop
"""

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from typing import Dict, Tuple
from collections import defaultdict

class YOLOProcessor:
    """
    YOLO processor with cumulative counting.
    Tracks unique objects and maintains running totals.
    """
    
    # Predefined configs
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
    
    def __init__(self, model_name="yolo26l.pt", config_name="vehicle", confidence=0.3):
        self.model = YOLO(model_name)
        self.confidence = confidence
        
        # Load config
        self.config = self.CONFIGS.get(config_name, self.CONFIGS['vehicle'])
        self.target_classes = self.config['classes']
        self.target_ids = list(self.target_classes.keys())
        self.colors = self.config['colors']
        
        # Tracking
        self.tracker = sv.ByteTrack()
        
        # Cumulative state
        self.reset_counts()
    
    def reset_counts(self):
        """Reset cumulative counts (called when starting new session)"""
        self.seen_ids = set()  # Track unique object IDs
        self.cumulative_counts = defaultdict(int)  # Total counts by class
        self.cumulative_counts['total'] = 0
    
    
    def set_zone(self, polygon: np.ndarray, frame_resolution_wh: Tuple[int, int]):
        """Set the active zone for occupancy tracking"""
        self.zone_polygon = polygon
        self.zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=frame_resolution_wh)
        self.zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.zone, 
            color=sv.Color.WHITE, 
            thickness=2,
            text_thickness=2,
            text_scale=1.0
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process frame and maintain cumulative counts.
        
        Returns:
            annotated_frame: Frame with bounding boxes
            counts: {'total': int, 'car': int, 'zone_occupancy': int, ...}
        """
        # 1. Run YOLO detection
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # 2. Filter by target classes
        mask = np.isin(detections.class_id, self.target_ids)
        detections = detections[mask]
        
        # 3. Track objects to get unique IDs
        detections = self.tracker.update_with_detections(detections)
        
        # 4. Update cumulative counts for NEW unique objects
        if detections.tracker_id is not None:
            for i, tracker_id in enumerate(detections.tracker_id):
                if tracker_id not in self.seen_ids:
                    # New unique object detected
                    self.seen_ids.add(tracker_id)
                    
                    class_id = detections.class_id[i]
                    class_name = self.target_classes.get(class_id, 'unknown')
                    
                    self.cumulative_counts['total'] += 1
                    self.cumulative_counts[class_name] += 1
        
        # 5. Zone Occupancy Analysis
        zone_occupancy = 0
        annotated = frame.copy()
        
        if hasattr(self, 'zone') and self.zone is not None:
            # Trigger zone check
            zone_mask = self.zone.trigger(detections=detections)
            zone_occupancy = sum(zone_mask)
            
            # Add occupancy to counts
            self.cumulative_counts['zone_occupancy'] = zone_occupancy
            
            # Annotate zone
            annotated = self.zone_annotator.annotate(scene=annotated)
        
        # 6. General Annotation (on top of zone)
        annotated = self._annotate(annotated, detections)
        
        # Return cumulative counts
        return annotated, dict(self.cumulative_counts)
    
    def _annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Draw bounding boxes with tracking IDs"""
        annotated = frame # Already copied in process_frame if zone is active, else need copy?
        if not hasattr(self, 'zone') or self.zone is None:
             annotated = frame.copy()

        h, w = frame.shape[:2]
        scale = max(w, h) / 1920.0
        scale = max(0.5, min(scale, 2.0))
        
        if len(detections) == 0:
            return annotated
        
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            class_id = detections.class_id[i]
            class_name = self.target_classes.get(class_id, 'unknown')
            
            # Get tracker ID if available
            tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else None
            
            # Get color
            color = self.colors.get(class_name, (200, 200, 200))
            
            # Draw box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label with ID
            if tracker_id is not None:
                label = f"#{tracker_id} {class_name}"
            else:
                label = class_name
                
            font_scale = 0.6 * scale
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            
            # Label background
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 255), 2)
            
        return annotated

