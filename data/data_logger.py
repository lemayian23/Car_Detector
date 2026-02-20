import csv
from datetime import datetime

class SimpleLogger:
    """Log detection to csv file. """
    def __init__(self, filename = 'car_log.csv'):
        self.filename = filename
        #Write header if file doesn't exist
        try:
            with open(self.filename, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'car_type', 'confidence'])
        except FileExistsError:
            pass

    def create_file(self):
        """Craete a csv file with headers. """
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Car Type', 'Confidence', 'X', 'Y', 'Width', 'Height'])

    def log_detection(self, detections):
        """Log a single detection. """:
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            bbox = detections['bbox']
            writer.writerow([timestamp, detections['class_name'], detections['confidence'], bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])

    def get_today_summary(self):
        """Get summary of today's detections. """
        today = datetime.now().date()
        summary = {}
        with open(self.filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S")
                if timestamp.date() == today:
                    car_type = row['car_type']
                    summary[car_type] = summary.get(car_type, 0) + 1
        return summary