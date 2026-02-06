import albumentations as A
import cv2
import numpy as np
from typing import List, Tuple

class CarAugmentation:
    def __init__(self, config):
        self.config = config
        self.img_size = config['detection']['input_size'][0]
        
    def get_train_augmentations(self):
        """Get training augmentations."""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                p=0.5
            ),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.3),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.3),
                A.HueSaturationValue(p=0.3),
            ], p=0.5),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_visibility=0.3,
            label_fields=['class_labels']
        ))
    
    def get_val_augmentations(self):
        """Get validation augmentations."""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def apply_augmentations(self, image, bboxes=None, labels=None, is_train=True):
        """Apply augmentations to image and bounding boxes."""
        if is_train:
            transform = self.get_train_augmentations()
        else:
            transform = self.get_val_augmentations()
        
        if bboxes is not None and labels is not None:
            augmented = transform(
                image=image,
                bboxes=bboxes,
                class_labels=labels
            )
            return augmented['image'], augmented['bboxes'], augmented['class_labels']
        else:
            augmented = transform(image=image)
            return augmented['image'], None, None
    
    def visualize_augmentations(self, image, bboxes, labels, num_samples=5):
        """Visualize augmented samples."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(1, num_samples + 1, figsize=(20, 4))
        
        # Original image
        axes[0].imshow(image)
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            axes[0].add_patch(rect)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Augmented images
        for i in range(num_samples):
            aug_img, aug_bboxes, _ = self.apply_augmentations(
                image, bboxes, labels, is_train=True
            )
            
            # Convert normalized image back
            aug_img_vis = self.denormalize_image(aug_img)
            axes[i+1].imshow(aug_img_vis)
            
            if aug_bboxes:
                for bbox in aug_bboxes:
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    axes[i+1].add_patch(rect)
            
            axes[i+1].set_title(f'Augmented {i+1}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def denormalize_image(self, image):
        """Denormalize image for visualization."""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)
        
        image = image * std + mean
        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)