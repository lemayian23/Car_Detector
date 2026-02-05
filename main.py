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
    else:
        logger.error("Invalid mode or missing image path for inference.")
if __name__ == "__main__":
    main()