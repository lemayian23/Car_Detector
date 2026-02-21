import csv
import os
from datetime import datetime
from pathlib import Path

class SimpleLogger:
    """Log detections to CSV file."""
    
    def __init__(self, filename='car_log.csv'):
        self.filename = filename
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
        self.filepath = Path("logs") / filename
        self.create_file()
    
    def create_file(self):
        """Create CSV file with headers if it doesn't exist."""
        if not self.filepath.exists():
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Car Type', 'Confidence', 'X', 'Y', 'Width', 'Height', 'Color'])
    
    def log_detection(self, detection):
        """Log a single detection."""
        try:
            with open(self.filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                bbox = detection['bbox']
                
                # Get color if available
                color = detection.get('color', 'unknown')
                
                writer.writerow([
                    timestamp,
                    detection['class_name'],
                    f"{detection['confidence']:.2f}",
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2] - bbox[0]),
                    int(bbox[3] - bbox[1]),
                    color
                ])
        except Exception as e:
            print(f"Error logging detection: {e}")
    
    def get_today_summary(self):
        """Get summary of today's detections."""
        today = datetime.now().strftime('%Y-%m-%d')
        counts = {}
        
        try:
            with open(self.filepath, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row and row[0].startswith(today):
                        car_type = row[1]
                        counts[car_type] = counts.get(car_type, 0) + 1
        except:
            pass
        
        return counts