import os
import cv2
import torch
from torch.utils.data import Dataset, Dataloader
from pathlib import Path
import alumentations as A
from alumentations.pytorch import ToTensorV2

class CarDataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transform=None, is_train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.is_train = is_train

        #Get all images files
        self.image_paths = list(self.image_dir.glob("*.jpg")) +\
                            list(self.image_dir.glob("*.png")) +\
                            list(self.image_dir.glob("*.jpeg"))
                            
        if label_dir:
            self.label_files = [self.label_dir / f"{img.stem}.txt"
                                for img in self.image_files]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        #Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize variables
        boxed = []
        labels = []

        #Load labels if available
        if self.label_dir and self.is_train:
            label_path = self.label_files[idx]
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        data = line.strip().split()
                        if len(data) == 5: #YOLO format
                            label = int(data[0])
                            x_center, y_center, width, height = map(float, data[1:])

                            #Covert YOLO TO Pascal VOC format
                            x_min = (x_center - width/2) * image.shape[1]
                            y_min = (y_center - height/2) * image.shape[0]
                            x_max = (x_center + width/2) * image.shape[1]
                            y_max = (y_center + height/2) * image.shape[0]
                            labels.append(label)

        #Apply transformations
        if self.transform:
            if boxes:
                transformed = self.transform(
                    image=image, 
                    bboxes=boxes, 
                    class_labels=labels
                )
                image = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']
            else:
                transformed = self.transform(image=image)
                image = transformed['image']

        #Convert boxes and labels to tensors
        image_tensor = ToTensorV2()(image=image)['image']

        if self.is_train and boxes:
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            }
            return image_tensor, target
        else:
            return image_tensor, str(img_path)

def get_transforms(train= True, img_size = 640):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
        ])

def create_dataloaders(config):
    """Create train, validation and test dataloaders."""

    train_transform = get_transforms(train=True,
                                     img_size=config['detection']['input_size'][0])
    val_transform = get_transforms(train=False,
                                   img_size=config['detection']['input_size'][0])

    #Create Datasets
    train_dataset = CarDataset(
        image_dir=config['paths']['train_data'],
        label_dir=config['paths']['train_data'] / 'labels',
        transform=train_transform,
        is_train=True
    )
    val_dataset = CarDataset(
        image_dir=config['paths']['val_data'],
        label_dir=config['paths']['val_data'] / 'labels',
        transform=val_transform,
        is_train=True
    )

    test_dataset = CarDataset(
        image_dir=config['paths']['test_data'],
        label_dir=None,
        transform=val_transform,
        is_train=False
    )

    #Create Dataloaders
    train_loader = Dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))  #Custom collate function to handle variable number of boxes
    )

    test_loader = Dataloader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader