import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.colors = self._generate_colors()
        
    def _generate_colors(self, num_colors=20):
        """Generate distinct colors for different classes."""
        cmap = plt.cm.get_cmap('tab20', num_colors)
        colors = [cmap(i)[:3] for i in range(num_colors)]
        return [(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
    
    def plot_image_with_bboxes(self, image, detections, save_path=None, figsize=(12, 8)):
        """Plot image with bounding boxes."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Convert image if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_img = image
        
        ax.imshow(display_img)
        
        # Draw bounding boxes
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Convert to integers
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get color based on class
            class_id = det.get('class_id', 0)
            color = self.colors[class_id % len(self.colors)]
            
            # Create rectangle
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor=np.array(color)/255,
                           facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            ax.text(x1, y1-10, label,
                   color='white', fontsize=10,
                   bbox=dict(facecolor=np.array(color)/255, alpha=0.7))
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_detection_grid(self, images_list, detections_list, titles=None, 
                           save_path=None, figsize=(15, 10)):
        """Plot multiple images with detections in a grid."""
        n_images = len(images_list)
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_images == 1:
            axes = np.array([axes])
        
        axes = axes.flatten()
        
        for i, (image, detections) in enumerate(zip(images_list, detections_list)):
            ax = axes[i]
            
            # Convert image if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                display_img = image
            
            ax.imshow(display_img)
            
            # Draw bounding boxes
            for det in detections:
                bbox = det['bbox']
                class_name = det['class_name']
                confidence = det['confidence']
                
                x1, y1, x2, y2 = map(int, bbox)
                class_id = det.get('class_id', 0)
                color = self.colors[class_id % len(self.colors)]
                
                rect = Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor=np.array(color)/255,
                               facecolor='none')
                ax.add_patch(rect)
                
                label = f"{class_name}: {confidence:.2f}"
                ax.text(x1, y1-10, label,
                       color='white', fontsize=8,
                       bbox=dict(facecolor=np.array(color)/255, alpha=0.7))
            
            if titles and i < len(titles):
                ax.set_title(titles[i])
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_class_distribution(self, class_counts, save_path=None):
        """Plot class distribution bar chart."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Sort by count
        sorted_indices = np.argsort(counts)[::-1]
        classes = [classes[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        bars = ax.bar(classes, counts, color='skyblue')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        ax.set_xticklabels(classes, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_confidence_distribution(self, confidences, save_path=None):
        """Plot confidence score distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Detection Confidence Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_bbox_size_distribution(self, bbox_sizes, save_path=None):
        """Plot bounding box size distribution."""
        if not bbox_sizes:
            print("No bounding boxes to plot")
            return None
        
        widths = [w for w, h in bbox_sizes]
        heights = [h for w, h in bbox_sizes]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].hist(widths, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('Width (normalized)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Bounding Box Width Distribution')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(heights, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1].set_xlabel('Height (normalized)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Bounding Box Height Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def create_detection_video(self, video_path, detector, output_path='results/detection_video.mp4'):
        """Create video with detection overlay."""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = detector.detect(
                frame, 
                confidence=self.config['detection']['confidence_threshold']
            )
            
            # Draw detections
            for det in detections:
                bbox = det['bbox']
                class_name = det['class_name']
                confidence = det['confidence']
                
                x1, y1, x2, y2 = map(int, bbox)
                class_id = det.get('class_id', 0)
                color = self.colors[class_id % len(self.colors)]
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1-label_size[1]-10),
                            (x1+label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Video saved to: {output_path}")