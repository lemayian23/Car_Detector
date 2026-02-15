import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

class FunImageProcessor:
    """Add fun effects to car detection images."""
    
    def __init__(self):
        self.filters = {
            'cartoon': self.cartoon_effect,
            'vintage': self.vintage_effect,
            'neon': self.neon_effect,
            'thermal': self.thermal_effect,
            'pixelate': self.pixelate_effect,
            'sketch': self.sketch_effect,
            'oil_painting': self.oil_painting_effect,
            'emboss': self.emboss_effect
        }
        
        self.stickers = ['🚗', '🚕', '🚙', '🚌', '🚎', '🏎', '🚓', '🚑', '🚒', '🚐']
    
    def cartoon_effect(self, image):
        """Apply cartoon effect."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply median blur
        gray = cv2.medianBlur(gray, 5)
        
        # Detect edges
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        
        # Color quantization
        color = cv2.bilateralFilter(image, 9, 300, 300)
        
        # Combine
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
        return cartoon
    
    def vintage_effect(self, image):
        """Apply vintage sepia effect."""
        # Create sepia kernel
        kernel = np.array([[0.393, 0.769, 0.189],
                          [0.349, 0.686, 0.168],
                          [0.272, 0.534, 0.131]])
        
        sepia = cv2.transform(image, kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        
        # Add vignette effect
        rows, cols = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, 200)
        kernel_y = cv2.getGaussianKernel(rows, 200)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        
        for i in range(3):
            sepia[:, :, i] = sepia[:, :, i] * mask
        
        return sepia
    
    def neon_effect(self, image):
        """Apply neon effect."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Create neon effect
        neon = np.zeros_like(image)
        
        # Random colors for edges
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0),
                 (0, 255, 0), (255, 0, 0), (0, 0, 255)]
        
        for i in range(3):
            neon[:, :, i] = edges * random.randint(100, 255)
        
        return neon
    
    def thermal_effect(self, image):
        """Apply thermal camera effect."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply colormap
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        return thermal
    
    def pixelate_effect(self, image, pixel_size=10):
        """Apply pixelation effect."""
        height, width = image.shape[:2]
        
        # Resize down
        temp = cv2.resize(image, (width // pixel_size, height // pixel_size),
                         interpolation=cv2.INTER_LINEAR)
        
        # Resize up
        pixelated = cv2.resize(temp, (width, height),
                              interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    def sketch_effect(self, image):
        """Apply pencil sketch effect."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Invert
        inverted = cv2.bitwise_not(gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        
        # Divide
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        
        # Stack to 3 channels
        sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        
        return sketch
    
    def oil_painting_effect(self, image):
        """Apply oil painting effect."""
        # Apply bilateral filter multiple times
        result = image.copy()
        for _ in range(3):
            result = cv2.bilateralFilter(result, 9, 75, 75)
        
        return result
    
    def emboss_effect(self, image):
        """Apply emboss effect."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Emboss kernel
        kernel = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]])
        
        emboss = cv2.filter2D(gray, -1, kernel)
        emboss = cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)
        
        return emboss
    
    def add_stickers(self, image, detections):
        """Add fun stickers to detected cars."""
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Random sticker
            sticker = random.choice(self.stickers)
            
            # Add sticker using PIL
            pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            