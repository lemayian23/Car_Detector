import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import json
from src.utils.helpers import load_config, setup_logging
from src.models.detector import CarDetector
from src.data.data_loader import CarDataset, get_transforms
from torch.utils.data import DataLoader

class ModelEvaluator:
    def __init__(self, config_path='config/config.yaml'):
        self.config = load_config(config_path)
        self.logger = setup_logging()
        self.device = torch.device(self.config['training']['device'])
        
    def evaluate_detector(self, model_path='models/weights/car_detector.pt'):
        """Evaluate the car detector."""
        self.logger.info("Evaluating detector...")
        
        # Initialize detector
        detector = CarDetector(
            model_type=self.config['detection']['model_type'],
            config=self.config
        )
        detector.load_model(model_path)
        
        # Load validation dataset
        val_transform = get_transforms(
            train=False, 
            img_size=self.config['detection']['input_size'][0]
        )
        
        val_dataset = CarDataset(
            image_dir=self.config['paths']['val_data'],
            label_dir=self.config['paths']['val_data'] / 'labels',
            transform=val_transform,
            is_train=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        # Evaluation metrics
        metrics = {
            'total_images': 0,
            'total_gt_boxes': 0,
            'total_detections': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'per_class_metrics': {},
            'confidence_scores': [],
            'inference_times': []
        }
        
        # Initialize per class metrics
        for class_name in self.config['detection']['classes']:
            metrics['per_class_metrics'][class_name] = {
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        # Evaluate
        detector.model.eval()
        import time
        
        for batch_idx, (images, targets) in enumerate(val_loader):
            metrics['total_images'] += 1
            
            image = images[0].numpy().transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
            
            target = targets[0]
            gt_boxes = target['boxes'].numpy() if 'boxes' in target else []
            gt_labels = target['labels'].numpy() if 'labels' in target else []
            
            metrics['total_gt_boxes'] += len(gt_boxes)
            
            # Run inference
            start_time = time.time()
            detections = detector.detect(
                image,
                confidence=self.config['detection']['confidence_threshold']
            )
            inference_time = time.time() - start_time
            metrics['inference_times'].append(inference_time)
            
            metrics['total_detections'] += len(detections)
            
            # Store confidence scores
            for det in detections:
                metrics['confidence_scores'].append(det['confidence'])
            
            # Match detections with ground truth
            if len(gt_boxes) > 0 and len(detections) > 0:
                # Calculate IoU matrix
                iou_matrix = self._calculate_iou_matrix(gt_boxes, detections)
                
                # Match using threshold
                matches = self._match_detections(
                    iou_matrix, 
                    self.config['detection']['iou_threshold']
                )
                
                for gt_idx, det_idx in matches.items():
                    if det_idx is not None:
                        metrics['true_positives'] += 1
                        class_name = self.config['detection']['classes'][gt_labels[gt_idx]]
                        metrics['per_class_metrics'][class_name]['tp'] += 1
                    else:
                        metrics['false_negatives'] += 1
                        class_name = self.config['detection']['classes'][gt_labels[gt_idx]]
                        metrics['per_class_metrics'][class_name]['fn'] += 1
                