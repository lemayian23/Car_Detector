import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from skimage import exposure

class CarFeatureExtractor:
    """Extract various features from car images for learning purposes."""
    
    def __init__(self):
        self.feature_methods = {
            'hog': self.extract_hog,
            'lbp': self.extract_lbp,
            'color_histogram': self.extract_color_histogram,
            'edges': self.extract_edges,
            'keypoints': self.extract_keypoints,
            'contours': self.extract_contours
        }
    
    def extract_hog(self, image, visualize=True):
        """Extract Histogram of Oriented Gradients (HOG) features."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize for consistent HOG computation
        gray = cv2.resize(gray, (128, 128))
        
        # Compute HOG
        features, hog_image = hog(
            gray, 
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=visualize,
            block_norm='L2-Hys'
        )
        
        if visualize:
            # Rescale for display
            hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            return features, hog_image
        return features
    
    def extract_lbp(self, image, radius=1, n_points=8):
        """Extract Local Binary Pattern features."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Compute LBP
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Compute histogram
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float) / hist.sum()
        
        return hist, lbp
    
    def extract_color_histogram(self, image, bins=32):
        """Extract color histogram features."""
        # Initialize histogram
        hist_features = []
        hist_images = []
        
        # Compute histogram for each channel
        for i, color in enumerate(['Blue', 'Green', 'Red']):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
            
            # Create histogram image for visualization
            hist_img = np.zeros((200, 256, 3), dtype=np.uint8)
            cv2.normalize(hist, hist, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX)
            
            for j in range(bins - 1):
                cv2.line(hist_img, 
                        (j*8, 200 - int(hist[j])), 
                        ((j+1)*8, 200 - int(hist[j+1])), 
                        (255, 255, 255), 2)
            hist_images.append(hist_img)
        
        return np.array(hist_features), hist_images
    
    def extract_edges(self, image):
        """Extract edge features using Canny edge detector."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        # Detect edges with different thresholds
        edges1 = cv2.Canny(blurred, 50, 150)
        edges2 = cv2.Canny(blurred, 100, 200)
        
        # Compute edge density
        edge_density = np.sum(edges1 > 0) / edges1.size
        
        return edge_density, edges1, edges2
    
    def extract_keypoints(self, image):
        """Extract keypoints using SIFT or ORB."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Try SIFT (might need opencv-contrib-python)
        try:
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
        except:
            # Fallback to ORB
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        # Draw keypoints
        keypoint_image = cv2.drawKeypoints(
            image, keypoints, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return keypoints, descriptors, keypoint_image
    
    def extract_contours(self, image):
        """Extract contour features."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw contours
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        
        # Extract contour properties
        contour_features = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                contour_features.append({
                    'area': area,
                    'perimeter': perimeter,
                    'circularity': circularity
                })
        
        return contour_features, contour_image
    
    def visualize_all_features(self, image, save_path=None):
        """Visualize all features in one plot."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # HOG
        _, hog_img = self.extract_hog(image)
        axes[0, 1].imshow(hog_img, cmap='gray')
        axes[0, 1].set_title('HOG Features')
        axes[0, 1].axis('off')
        
        # LBP
        _, lbp_img = self.extract_lbp(image)
        axes[0, 2].imshow(lbp_img, cmap='gray')
        axes[0, 2].set_title('LBP Features')
        axes[0, 2].axis('off')
        
        # Color histograms
        _, hist_imgs = self.extract_color_histogram(image)
        for i, hist_img in enumerate(hist_imgs):
            axes[1, i].imshow(hist_img)
            axes[1, i].set_title(f'{["Blue", "Green", "Red"][i]} Histogram')
            axes[1, i].axis('off')
        
        # Edges
        _, edges1, edges2 = self.extract_edges(image)
        axes[2, 0].imshow(edges1, cmap='gray')
        axes[2, 0].set_title('Edges (low threshold)')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(edges2, cmap='gray')
        axes[2, 1].set_title('Edges (high threshold)')
        axes[2, 1].axis('off')
        
        # Keypoints
        _, _, keypoint_img = self.extract_keypoints(image)
        axes[2, 2].imshow(cv2.cvtColor(keypoint_img, cv2.COLOR_BGR2RGB))
        axes[2, 2].set_title('Keypoints')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()