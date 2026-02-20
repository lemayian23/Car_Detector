import cv2
import numpy as np


class CarEmojiOverlay:
    """Add emoji overlays to deteced cars. """

    def __init(self):
            self.emoji_map = {
            'car': '🚗',
            'truck': '🚛',
            'bus': '🚌',
            'motorcycle': '🏍️',
            'bicycle': '🚲',
            'default': '🚘'
        }

        self.speed_emojis = ['💨', '💨💨', '💨💨💨', '💨💨💨💨']
        self.happy_emojis = ['😊', '😎', '🌟', '✨']

    def add_emoji(self, frame, detections):
        """Add emoji based on car type. """
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])

            #Get emoji for car type
            car_type = det['class_name'].lower()
            emoji = self.emoji_map.get(car_type, self.emoji_map['default'])

            #Add emoji above car
            cv2.putText(frame, emoji, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255,255), 2)

                   # Add speed lines for high confidence
            if det['confidence'] > 0.8:
                speed_emoji = np.random.choice(self.speed_emojis)
                cv2.putText(frame, speed_emoji, (x2, y1 - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        return frame

    def add_reaction(self, frame, detections):
        ""Add reaction emoji when many cars detected. """
        if len(detections) > 5:
                       cv2.putText(frame, '🤯 SO MANY CARS!', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        elif len(detections) > 3:
            cv2.putText(frame, '😎 COOL!', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        elif len(detections) > 0:
            cv2.putText(frame, '👋 Found you!', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

# Usage example:
# emoji = CarEmojiOverlay()
# frame = emoji.add_emoji(frame, detections)                     