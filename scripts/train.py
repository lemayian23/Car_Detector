import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from src.helpers import load_config, setup_logging, set_seed
from src.data.data_loader import create_dataloaders

def train_model():
    config = load_config()
    logger = setup_logging()
    set_seed(42)

    logger.info("Initializing Car Detector...")
    detector = CarDetector(
        model_type=config['detection']['model_type'],
        config=config
    )
    logger.info("Starting training...")
    results = detector.train(
        train_data=config['paths']['train_data'],
        val_data=config['paths']['val_data'],
        epochs=config['training']['epochs']
    )
    logger.info("Saving Model...")
    detector.save_model('models/weights/car_detector_final.pt')
    logger.info("Training completed!")

if __name__ == "__main__":
    train_model()