import cv2
import numpy as np


class QuickFilters:
    """Quick and fun image filters. """

    @staticmethod
    def night_vision(frame):
        """Night vision effect. """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Add green tint
        night_vision = cv2.applyColorMap(gray, cv2.COLORMAP_SUMMER)

        #Add crosshair
        h, w = frame.shape[:2]
        cv2.line(night_vision, (w//2, 0), (w//2, h), (0, 255, 0), 1)
        cv2.line(night_vision, (0, h//2), (w, h//2), (0, 255, 0), 1)

        return night_vision

    @staticmethod
    def thermal(frame):
        """Thermal camera effect. """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def xray(frame):
        """X- ray effect (inverted). """
        return cv2.bitwise_not(frame)

    @staticmethod
    def old_movie(frame):
        """Old movie effect with noise. """

        #covert to sepia
        kernel =np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        sepia = cv2.transform(frame, kernel)
        sepia = cv2.clip(sepia, 0, 255).astype(np.uint8)

        #Add noise
        noise = np.random.randint(0,30, frame.shape, dtype=np.uint8)
        sepia = cv2.add(sepia, noise)

        return sepia

    @staticmethod
        def cartoon(frame):
        """Simple cartoon effect."""
        # Simplify colors
        data = frame.reshape((-1, 3))
        data = np.float32(data)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        simplified = centers[labels.flatten()].reshape(frame.shape)

        # Add edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 5)

        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        #Combine
        return cv2.bitwise_and(simplified, edges)
    @staticmethod
    def mirror(frame, mode='horizontal'):
        """Mirror effect. """
        if mode == 'horizontal':
            return cv2.flip(frame, 1)
        elif mode == 'vertical':
            return cv2.flip(frame, 0)
        else:
            raise frame
    
#Quick usage in detection loop:
# frame = QuickFilters.night_vision(frame)
# frame = QuickFilters.thermal(frame)