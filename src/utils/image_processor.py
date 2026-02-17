import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random 

class FunImageProcessor:
    """Add fun effects to car detection images. """

    def __init__(self):
        self.filters = {
            'cartoon': self.cartoon_effect,
            'vintage': self.vintage_effect,
            'neon': self.neon_effect,
            'thermal': self.thermal_effect,
            'pixelate': self.pixelate_effect,
            'oil_painting': self.oil_painting_effect,
            'emboss': self.emboss_effect
        }

        self.stickers = ['🚗', '🚕', '🚙', '🚌', '🚎', '🏎', '🚓', '🚑', '🚒', '🚐']

    def cartoon_effect(self, image):
        """Apply cartoon effect. """
        #Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Apply median blur
        gray = cv2.medianBlur(gray, 5)

        #Detect edges
        edges = cv2.adaptiveThreshold(gray, 255, 
                                      cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 9)

        #Color quantization
        color = cv2.bilateralFilter(image, 9, 300, 300)

        #Combine
        cartoon = cv2.bitwise_and(color, color, mask=edges)

        return cartoon

    def vintage_effect(self, image):
        """Apply vintage sepia effect."""

        #Create sepia kernel
        kernel = np.array([[0.393, 0.769, 0.189],
                           [0.349, 0.686, 0.168],
                           [0.272, 0.534, 0.131]])

        sepia = cv2.transform(image, kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)

        #Add  vignette effect
        rows, cols = sepia.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, 200)
        kernel_y = cv2.getGaussianKernel(rows, 200)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)

        for i in range(3):
            sepia[:,:,i] = sepia[:,:,i] * mask

        return sepia
    def neon_effect(self, image):
        """Apply neo effect. """
        #Convert tograyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Detect edges
        edges = cv2.Canny(gray, 50, 150)

        #Dilate edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        #Create neon effect
        neon = np.zeros_like(image
        
        #Random colors for edges
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

        for i in range(3):
            neon[:,:,i] = edges * random.randint(100, 255)
        return neon

    def thermal_effect(self, image):
        """Apply thermal camera effect. """
        #Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Apply color map
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        return thermal

    def pixelate_effect(self, image, pixel_size=10):
        """Apply pixelation effect. """
        height, width = image.shape[:2]

        #Resize down
        temp = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)

        #Resize up
        pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
        return pixelated

        def sketch_effect(self,image):
            """Apply pencil sketch effect. """
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #Invert
            inverted = cv2.bitwise_not(gray)

            #Apply Gaussian blur
            blurred = cv2.GaussianBlur(inverted, (21, 21), 0)

            #Sketch to 3 channels
            sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
            return sketch

    def oil_painting_effect(self, image):
        """Apply oil painting effect. """

        #Apply bilateral filter multiple times
        result = image.copy()

    