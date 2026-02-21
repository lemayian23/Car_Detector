import argparse
from src.utils.helpers import load_config, setup_logging, set_seed
from src.data.data_loader import create_dataloaders
from src.models.detector import CarDetector
import torch

def main():
    parser = argparse.ArgumentParser(description='Car Detection and Classification')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'infer'])
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--image', type=str, help='Path to image for inference')
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    logger = setup_logging()
    set_seed(42)
    device = torch.device(config['training']['device'])
    
    if args.mode == 'train':
        logger.info("Starting training...")
        
        # Create data loaders
        train_loader, val_loader, _ = create_dataloaders(config)
        
        # Initialize detector
        detector = CarDetector(
            model_type=config['detection']['model_type'],
            config=config
        )
        
        # Train
        detector.train(
            train_data=config['paths']['train_data'],
            val_data=config['paths']['val_data'],
            epochs=config['training']['epochs']
        )
        
        logger.info("Training completed!")
    
    elif args.mode == 'infer' and args.image:
        logger.info("Running inference...")
        
        # Initialize detector
        detector = CarDetector(
            model_type=config['detection']['model_type'],
            config=config
        )
        
        # Load trained weights
        detector.load_model('models/weights/car_detector.pt')
        
        # Run inference
        import cv2
        image = cv2.imread(args.image)
        detections = detector.detect(
            image, 
            confidence=config['detection']['confidence_threshold']
        )
        logger.info(f"Detections: {detections}")

        # In your main detection script
from src.utils.car_sound import CarSoundEffects
from src.utils.car_emoji import CarEmojiOverlay
from src.utils.car_counter import SimpleCarCounter
from src.utils.car_color import CarColorDetector
from src.utils.fun_filters import QuickFilters
from src.utils.simple_logger import SimpleLogger

# Initialize features
sound = CarSoundEffects()
emoji = CarEmojiOverlay()
counter = SimpleCarCounter()
color_detector = CarColorDetector()
logger = SimpleLogger()
filter_mode = 'normal'

# In your detection loop:
for det in detections:
    # Play sound (optional)
    if det['confidence'] > 0.7:
        sound.play_detection_sound(det['class_name'])
    
    # Log detection
    logger.log_detection(det)
    
    # Get car crop for color detection
    x1, y1, x2, y2 = map(int, det['bbox'])
    car_crop = frame[y1:y2, x1:x2]
    color = color_detector.detect_color(car_crop)

# Update counter
counter.update(detections)

# Add emoji overlays
frame = emoji.add_emoji(frame, detections)

# Apply filter
if filter_mode == 'night_vision':
    frame = QuickFilters.night_vision(frame)
elif filter_mode == 'thermal':
    frame = QuickFilters.thermal(frame)

# Show stats
print(counter.get_stats())
    else:
        logger.error("Invalid mode or missing image path for inference.")
if __name__ == "__main__":
    main()