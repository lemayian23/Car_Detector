import argparse
from src.utils.helpers import load_config, setup_logging, set_seed
from src.data.data_loader import create_dataloaders
from src.models.detector import CarDetector
import torch
import cv2
import os
from pathlib import Path

# Import fun features (optional - wrap in try/except to avoid errors if files don't exist)
try:
    from src.utils.car_sound import CarSoundEffects
    from src.utils.car_emoji import CarEmojiOverlay
    from src.utils.car_counter import SimpleCarCounter
    from src.utils.car_color import CarColorDetector
    from src.utils.fun_filters import QuickFilters
    from src.utils.simple_logger import SimpleLogger
    FUN_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some fun features not available: {e}")
    FUN_FEATURES_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description='Car Detection and Classification')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'infer', 'fun'])
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--image', type=str, help='Path to image for inference')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source for fun mode (0 for webcam or path to video)')
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    logger = setup_logging()
    set_seed(42)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if args.mode == 'train':
        logger.info("Starting training...")
        
        # Check if data directories exist
        train_path = Path(config['paths']['train_data'])
        if not train_path.exists():
            logger.error(f"Training data path {train_path} does not exist!")
            return
        
        # Create data loaders
        train_loader, val_loader, _ = create_dataloaders(config)
        
        # Initialize detector
        detector = CarDetector(
            model_type=config['detection']['model_type'],
            config=config
        )
        
        # Train
        detector.train(
            train_data=str(train_path),
            val_data=str(Path(config['paths']['val_data'])),
            epochs=config['training']['epochs']
        )
        
        logger.info("Training completed!")
    
    elif args.mode == 'infer':
        if not args.image:
            logger.error("Please provide an image path with --image")
            return
        
        logger.info(f"Running inference on {args.image}...")
        
        # Check if image exists
        if not os.path.exists(args.image):
            logger.error(f"Image {args.image} not found!")
            return
        
        # Initialize detector
        detector = CarDetector(
            model_type=config['detection']['model_type'],
            config=config
        )
        
        # Load trained weights if they exist
        weights_path = 'models/weights/car_detector.pt'
        if os.path.exists(weights_path):
            detector.load_model(weights_path)
            logger.info("Loaded trained weights")
        else:
            logger.warning("No trained weights found, using pre-trained model")
        
        # Run inference
        image = cv2.imread(args.image)
        if image is None:
            logger.error(f"Failed to load image {args.image}")
            return
            
        detections = detector.detect(
            image, 
            confidence=config['detection']['confidence_threshold']
        )
        
        logger.info(f"Found {len(detections)} detections:")
        for i, det in enumerate(detections):
            logger.info(f"  {i+1}. {det['class_name']} - Confidence: {det['confidence']:.2f}")
        
        # Draw detections on image
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save result
        output_path = 'results/detections/output.jpg'
        os.makedirs('results/detections', exist_ok=True)
        cv2.imwrite(output_path, image)
        logger.info(f"Result saved to {output_path}")
        
        # Display image (optional)
        cv2.imshow('Detections', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif args.mode == 'fun':
        if not FUN_FEATURES_AVAILABLE:
            logger.error("Fun features not available. Make sure all modules are created.")
            return
        
        logger.info("Starting fun mode with cool features!")
        run_fun_mode(args.source, config, logger)
    
    else:
        logger.error(f"Invalid mode: {args.mode}")

def run_fun_mode(source, config, logger):
    """Run detection with fun features."""
    
    # Initialize features
    try:
        sound = CarSoundEffects()
        emoji = CarEmojiOverlay()
        counter = SimpleCarCounter()
        color_detector = CarColorDetector()
        simple_logger = SimpleLogger()
        filter_mode = 'normal'
    except Exception as e:
        logger.error(f"Failed to initialize fun features: {e}")
        return
    
    # Initialize detector
    detector = CarDetector(
        model_type=config['detection']['model_type'],
        config=config
    )
    
    # Load weights if available
    weights_path = 'models/weights/car_detector.pt'
    if os.path.exists(weights_path):
        detector.load_model(weights_path)
    
    # Open video source
    try:
        if source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
    except:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logger.error("Failed to open video source")
        return
    
    logger.info("🎮 Fun mode controls:")
    logger.info("  q - Quit")
    logger.info("  s - Save screenshot")
    logger.info("  f - Change filter")
    logger.info("  m - Toggle mirror mode")
    logger.info("  c - Show stats")
    logger.info("  h - Honk horn")
    
    mirror_mode = False
    filters = ['normal', 'night_vision', 'thermal', 'xray', 'old_movie']
    filter_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect cars
        detections = detector.detect(
            frame,
            confidence=config['detection']['confidence_threshold']
        )
        
        # Apply fun features
        if detections:
            # Play sound for high confidence detections
            for det in detections:
                if det['confidence'] > 0.7:
                    sound.play_detection_sound(det['class_name'])
                
                # Log detection
                simple_logger.log_detection(det)
                
                # Detect color
                x1, y1, x2, y2 = map(int, det['bbox'])
                if y2 > y1 and x2 > x1:  # Valid crop
                    car_crop = frame[y1:y2, x1:x2]
                    if car_crop.size > 0:
                        color = color_detector.detect_color(car_crop)
                        # Add color to detection
                        det['color'] = color
            
            # Update counter
            counter.update(detections)
            
            # Add emoji overlays
            frame = emoji.add_emoji(frame, detections)
        
        # Apply filter
        if filter_mode != 'normal':
            try:
                if filter_mode == 'night_vision':
                    frame = QuickFilters.night_vision(frame)
                elif filter_mode == 'thermal':
                    frame = QuickFilters.thermal(frame)
                elif filter_mode == 'xray':
                    frame = QuickFilters.xray(frame)
                elif filter_mode == 'old_movie':
                    frame = QuickFilters.old_movie(frame)
            except Exception as e:
                logger.error(f"Filter error: {e}")
        
        # Mirror mode
        if mirror_mode:
            frame = cv2.flip(frame, 1)
        
        # Add info overlay
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Filter: {filter_mode}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Cars: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Fun Car Detection', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_path = f"results/screenshots/fun_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            os.makedirs('results/screenshots', exist_ok=True)
            cv2.imwrite(screenshot_path, frame)
            logger.info(f"📸 Screenshot saved: {screenshot_path}")
        elif key == ord('f'):
            filter_idx = (filter_idx + 1) % len(filters)
            filter_mode = filters[filter_idx]
            logger.info(f"🎨 Filter: {filter_mode}")
        elif key == ord('m'):
            mirror_mode = not mirror_mode
            logger.info(f"🪞 Mirror mode: {'ON' if mirror_mode else 'OFF'}")
        elif key == ord('c'):
            logger.info(counter.get_stats())
        elif key == ord('h'):
            try:
                sound.play_horn()
            except:
                pass
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Show final stats
    logger.info("\n" + "="*50)
    logger.info("FUN MODE SESSION SUMMARY")
    logger.info("="*50)
    logger.info(counter.get_stats())

if __name__ == "__main__":
    main()