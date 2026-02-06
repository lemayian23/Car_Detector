import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

class CarClassifier:
    def __init__(self, num_classes=10, model_type='resnet50', config=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.model_type = model_type
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _create_model(self):
        """Create classification model based on type."""
        if self.model_type == 'resnet50':
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_type == 'resnet18':
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_type == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        elif self.model_type == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
        
        return model
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Train the classifier."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {100.*train_correct/train_total:.2f}%")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, "
                  f"Val Acc: {100.*val_correct/val_total:.2f}%")
    
    def classify(self, image_crop):
        """Classify a single car crop."""
        self.model.eval()
        
        # Convert numpy array to PIL Image
        if isinstance(image_crop, np.ndarray):
            if len(image_crop.shape) == 3 and image_crop.shape[2] == 3:
                image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_crop)
        else:
            image_pil = image_crop
        
        # Apply transformations
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get class name
        class_id = predicted.item()
        class_name = self.config['classification']['car_types'][class_id]
        
        return {
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence.item()
        }
    
    def classify_batch(self, image_crops):
        """Classify multiple car crops."""
        results = []
        for crop in image_crops:
            results.append(self.classify(crop))
        return results
    
    def save_model(self, path='models/weights/car_classifier.pth'):
        """Save the classifier."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'config': self.config
        }, path)
    
    def load_model(self, path='models/weights/car_classifier.pth'):
        """Load the classifier."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()