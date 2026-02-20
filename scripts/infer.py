import cv2
from src.models.detector import CarDetector
from src.utils.helpers import load_config
import argparse


#Keyboard shortcuts:
def setup_keyboard_controls():
    """Add fun keyboard shortcuts. """
    print("\n🎮 Keyboard Controls:")
    print("  q - Quit")
    print("  s - Save screenshot")
    print("  c - Change color (toggle filters)")
    print("  f - Toggle FPS display")
    print("  h - Honk horn")
    print("  m - Mirror mode")
    print("  r - Reset all effects")
    print("  ? - Show this help")

def realtime_detection(source=0):
    config = load_config()
    
    # Initialize detector
    detector = CarDetector(
        model_type=config['detection']['model_type'],
        config=config
    )
    
    # Load model
    detector.load_model('models/weights/car_detector.pt')
    
    # Open video source
    cap = cv2.VideoCapture(source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect cars
        detections = detector.detect(
            frame, 
            confidence=config['detection']['confidence_threshold']
        )
        
        # Draw detections
        for det in detections:

            filter_mode = 0
            filters = ['ormal', 'night_vision', 'thermal', 'xray', 'old_movie',]
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"screnshot_{time.time()}.jpg", frame)
                print("📸 Screenshot saved!")
            elif key == ord('c'):
                filter_mode = (filter_mode + 1) % len(filters)
                print(f"🎨 Filter: {filters[filter_mode]}")
            elif key == ord('h'):
                winsound.Beep(400, 500) # Honk!
            elif key == ord('m'):
                frame = cv2.flip(frame, 1) # Mirror mode
            elif key == ord('?'):
                setup_keyboard_controls()
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Car Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0',
                       help='0 for webcam, or path to video file')
    args = parser.parse_args()
    
    realtime_detection(int(args.source) if args.source.isdigit() else args.source)