import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import yaml

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.raw_data_dir = Path(config['paths']['raw_data'])
        self.processed_dir = Path(config['paths']['processed_data'])
        
    def organize_dataset(self):
        """Organize raw dataset into proper structure."""
        print("Organizing dataset...")
        
        # Create necessary directories
        (self.processed_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.processed_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.raw_data_dir.rglob(f'*{ext}')))
            image_files.extend(list(self.raw_data_dir.rglob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_files)} images")
        
        # Copy images to processed directory
        for img_path in image_files:
            # Find corresponding label file
            label_path = img_path.with_suffix('.txt')
            if not label_path.exists():
                # Try with different naming conventions
                possible_names = [
                    img_path.with_suffix('.txt'),
                    img_path.parent / f"{img_path.stem}.txt",
                    img_path.parent / 'labels' / f"{img_path.stem}.txt",
                    img_path.parent.parent / 'labels' / f"{img_path.stem}.txt"
                ]
                
                label_path = None
                for name in possible_names:
                    if name.exists():
                        label_path = name
                        break
            
            # Copy image
            new_img_name = f"{img_path.stem}_{hash(img_path) % 10000:04d}{img_path.suffix}"
            new_img_path = self.processed_dir / 'images' / new_img_name
            shutil.copy2(img_path, new_img_path)
            
            # Copy label if exists
            if label_path and label_path.exists():
                new_label_name = new_img_path.with_suffix('.txt')
                shutil.copy2(label_path, new_label_name)
        
        print("Dataset organization complete!")
    
    def create_yolo_dataset(self):
        """Create YOLO format dataset for training."""
        print("Creating YOLO dataset...")
        
        # Split data into train/val/test
        all_images = list((self.processed_dir / 'images').glob('*.*'))
        
        # Filter only images with labels
        images_with_labels = []
        for img_path in all_images:
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                images_with_labels.append(img_path)
        
        print(f"Found {len(images_with_labels)} images with labels")
        
        # Split dataset
        train_val, test = train_test_split(
            images_with_labels, test_size=0.1, random_state=42
        )
        train, val = train_test_split(
            train_val, test_size=0.1, random_state=42
        )
        
        # Create dataset structure
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            split_dir = Path(self.config['paths'][f'{split_name}_data'])
            (split_dir / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
            
            for img_path in split_data:
                # Copy image
                shutil.copy2(img_path, split_dir / 'images' / img_path.name)
                
                # Copy label
                label_path = img_path.with_suffix('.txt')
                if label_path.exists():
                    shutil.copy2(label_path, split_dir / 'labels' / label_path.name)
            
            print(f"{split_name.capitalize()}: {len(split_data)} images")
        
        # Create dataset.yaml for YOLO
        self.create_yolo_config()
        
        print("YOLO dataset creation complete!")
    
    def create_yolo_config(self):
        """Create YOLO dataset configuration file."""
        config = {
            'path': str(Path(self.config['paths']['train_data']).parent.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.config['detection']['classes']),
            'names': self.config['detection']['classes']
        }
        
        yolo_config_path = self.processed_dir.parent / 'dataset.yaml'
        with open(yolo_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"YOLO config created at: {yolo_config_path}")
    
    def analyze_dataset(self):
        """Analyze dataset statistics."""
        print("Analyzing dataset...")
        
        stats = {
            'total_images': 0,
            'images_with_labels': 0,
            'total_bboxes': 0,
            'class_distribution': {},
            'bbox_size_distribution': [],
            'image_sizes': []
        }
        
        # Initialize class distribution
        for class_name in self.config['detection']['classes']:
            stats['class_distribution'][class_name] = 0
        
        # Analyze training data
        train_images_dir = Path(self.config['paths']['train_data']) / 'images'
        train_labels_dir = Path(self.config['paths']['train_data']) / 'labels'
        
        for img_path in train_images_dir.glob('*.*'):
            stats['total_images'] += 1
            
            # Get image size
            img = cv2.imread(str(img_path))
            if img is not None:
                stats['image_sizes'].append(img.shape[:2])
            
            # Check for labels
            label_path = train_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                stats['images_with_labels'] += 1
                
                with open(label_path, 'r') as f:
                    for line in f:
                        stats['total_bboxes'] += 1
                        
                        # Parse YOLO format
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_name = self.config['detection']['classes'][class_id]
                            stats['class_distribution'][class_name] += 1
                            
                            # Get bbox size
                            _, x_center, y_center, width, height = map(float, parts[:5])
                            stats['bbox_size_distribution'].append((width, height))
        
        # Calculate statistics
        if stats['image_sizes']:
            avg_size = np.mean(stats['image_sizes'], axis=0)
            print(f"\nDataset Statistics:")
            print(f"Total images: {stats['total_images']}")
            print(f"Images with labels: {stats['images_with_labels']}")
            print(f"Percentage labeled: {(stats['images_with_labels']/stats['total_images'])*100:.2f}%")
            print(f"Total bounding boxes: {stats['total_bboxes']}")
            print(f"Average image size: {avg_size[0]:.0f}x{avg_size[1]:.0f}")
            print(f"\nClass Distribution:")
            for class_name, count in stats['class_distribution'].items():
                percentage = (count/stats['total_bboxes'])*100 if stats['total_bboxes'] > 0 else 0
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return stats
    
    def validate_labels(self):
        """Validate YOLO format labels."""
        print("Validating labels...")
        
        issues = []
        
        for split in ['train', 'val']:
            labels_dir = Path(self.config['paths'][f'{split}_data']) / 'labels'
            
            for label_file in labels_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    
                    # Check format
                    if len(parts) != 5:
                        issues.append(f"{label_file.name}: Line {i+1} has {len(parts)} parts (expected 5)")
                        continue
                    
                    # Check class ID
                    try:
                        class_id = int(parts[0])
                        if not (0 <= class_id < len(self.config['detection']['classes'])):
                            issues.append(f"{label_file.name}: Line {i+1} invalid class ID {class_id}")
                    except ValueError:
                        issues.append(f"{label_file.name}: Line {i+1} invalid class ID format")
                    
                    # Check coordinates
                    for j, coord in enumerate(parts[1:]):
                        try:
                            val = float(coord)
                            if not (0 <= val <= 1):
                                issues.append(f"{label_file.name}: Line {i+1} coord {j+1} out of range: {val}")
                        except ValueError:
                            issues.append(f"{label_file.name}: Line {i+1} invalid coordinate format")
        
        if issues:
            print(f"Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("All labels are valid!")
        
        return issues