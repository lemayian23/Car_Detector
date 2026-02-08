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
                
                # False positives (detections without matching ground truth)
                matched_det_indices = set(matches.values())
                for det_idx, det in enumerate(detections):
                    if det_idx not in matched_det_indices:
                        metrics['false_positives'] += 1
                        class_name = det['class_name']
                        if class_name in metrics['per_class_metrics']:
                            metrics['per_class_metrics'][class_name]['fp'] += 1
            
            if (batch_idx + 1) % 10 == 0:
                self.logger.info(f"Processed {batch_idx + 1} images")
        
        # Calculate final metrics
        metrics = self._calculate_final_metrics(metrics)
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _calculate_iou_matrix(self, gt_boxes, detections):
        """Calculate IoU between ground truth boxes and detections."""
        iou_matrix = np.zeros((len(gt_boxes), len(detections)))
        
        for i, gt_box in enumerate(gt_boxes):
            for j, det in enumerate(detections):
                det_box = det['bbox']
                iou = self._calculate_iou(gt_box, det_box)
                iou_matrix[i, j] = iou
        
        return iou_matrix
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _match_detections(self, iou_matrix, iou_threshold):
        """Match detections to ground truth using IoU."""
        matches = {}
        
        for gt_idx in range(iou_matrix.shape[0]):
            best_iou = iou_threshold
            best_det_idx = None
            
            for det_idx in range(iou_matrix.shape[1]):
                if iou_matrix[gt_idx, det_idx] > best_iou:
                    best_iou = iou_matrix[gt_idx, det_idx]
                    best_det_idx = det_idx
            
            matches[gt_idx] = best_det_idx
        
        return matches
    
    def _calculate_final_metrics(self, metrics):
        """Calculate precision, recall, and F1 score."""
        # Overall metrics
        metrics['precision'] = metrics['true_positives'] / max(1, metrics['true_positives'] + metrics['false_positives'])
        metrics['recall'] = metrics['true_positives'] / max(1, metrics['true_positives'] + metrics['false_negatives'])
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / max(1e-10, metrics['precision'] + metrics['recall'])
        
        metrics['avg_inference_time'] = np.mean(metrics['inference_times']) if metrics['inference_times'] else 0
        metrics['fps'] = 1 / metrics['avg_inference_time'] if metrics['avg_inference_time'] > 0 else 0
        
        # Per class metrics
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            tp = class_metrics['tp']
            fp = class_metrics['fp']
            fn = class_metrics['fn']
            
            class_metrics['precision'] = tp / max(1, tp + fp)
            class_metrics['recall'] = tp / max(1, tp + fn)
            class_metrics['f1'] = 2 * (class_metrics['precision'] * class_metrics['recall']) / max(1e-10, class_metrics['precision'] + class_metrics['recall'])
        
        return metrics
    
    def _save_results(self, metrics):
        """Save evaluation results."""
        results_dir = Path('results/evaluations')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = results_dir / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    json_metrics[key] = value.tolist()
                elif isinstance(value, np.generic):
                    json_metrics[key] = value.item()
                elif isinstance(value, dict):
                    json_metrics[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            json_metrics[key][sub_key] = {}
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                if isinstance(sub_sub_value, np.generic):
                                    json_metrics[key][sub_key][sub_sub_key] = sub_sub_value.item()
                                else:
                                    json_metrics[key][sub_key][sub_sub_key] = sub_sub_value
                        else:
                            json_metrics[key][sub_key] = sub_value
                else:
                    json_metrics[key] = value
            
            json.dump(json_metrics, f, indent=4)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total Images: {metrics['total_images']}")
        print(f"Total Ground Truth Boxes: {metrics['total_gt_boxes']}")
        print(f"Total Detections: {metrics['total_detections']}")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Average Inference Time: {metrics['avg_inference_time']:.4f} seconds")
        print(f"FPS: {metrics['fps']:.2f}")
        print(f"\nPer Class Metrics:")
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall: {class_metrics['recall']:.4f}")
            print(f"    F1: {class_metrics['f1']:.4f}")
        
        self.logger.info(f"Results saved to: {json_path}")

def main():
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_detector()

if __name__ == "__main__":
    main()