import cv2
import numpy as np
from collections import Counter

class CarColorDetector:
    """Simple car color detector."""
    
    def __init__(self):
        # Define color ranges in HSV
        self.color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255)],
            'red2': [(170, 50, 50), (180, 255, 255)],  # Red wraps around
            'blue': [(100, 50, 50), (130, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'yellow': [(20, 50, 50), (35, 255, 255)],
            'orange': [(10, 50, 50), (20, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)],
            'black': [(0, 0, 0), (180, 255, 50)],
            'silver': [(0, 0, 100), (180, 50, 200)],
            'purple': [(130, 50, 50), (160, 255, 255)]
        }
    
    def detect_color(self, car_crop):
        """Detect the dominant color of a car."""
        if car_crop.size == 0:
            return 'unknown'
        
        # Convert to HSV
        hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        
        # Count pixels for each color
        color_counts = {}
        
        for color_name, (lower, upper) in self.color_ranges.items():
            if color_name == 'red2':  # Handle red specially
                continue
                
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Also check second red range
            if color_name == 'red':
                mask2 = cv2.inRange(hsv, np.array(self.color_ranges['red2'][0]), 
                                    np.array(self.color_ranges['red2'][1]))
                mask = cv2.bitwise_or(mask, mask2)
            
            color_counts[color_name] = cv2.countNonZero(mask)
        
        # Get dominant color
        if max(color_counts.values()) > 100:  # Minimum pixels threshold
            dominant_color = max(color_counts, key=color_counts.get)
            return dominant_color
        return 'unknown'
    
    def add_color_label(self, frame, detections, car_crops):
        """Add color labels to detections."""
        for i, (det, crop) in enumerate(zip(detections, car_crops)):
            if crop is not None:
                color = self.detect_color(crop)
                x1, y1, x2, y2 = map(int, det['bbox'])
                
                # Add color label
                label = f"{color} {det['class_name']}"
                cv2.putText(frame, label, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame

# Usage example:
# color_detector = CarColorDetector()
# car_crop = frame[y1:y2, x1:x2]
# color = color_detector.detect_color(car_crop)