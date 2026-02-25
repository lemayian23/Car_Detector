"""
Module 1: Data Ingestion
Objective: Load data from various sources (images, videos, streams)
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Generator
import logging

class DataIngestor:
    """Ingest data from multiple sources."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.sources = {
            'image': self._ingest_image,
            'video': self._ingest_video,
            'webcam': self._ingest_webcam,
            'directory': self._ingest_directory
        }
    
    def ingest(self, source: Union[str, int], source_type: str = 'auto') -> Generator:
        """
        Main ingestion method - yields frames/images.
        
        Args:
            source: Path to file/directory or camera ID
            source_type: 'image', 'video', 'webcam', 'directory', or 'auto'
        
        Yields:
            tuple: (frame, metadata)
        """
        if source_type == 'auto':
            source_type = self._detect_source_type(source)
        
        if source_type in self.sources:
            yield from self.sources[source_type](source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def _detect_source_type(self, source):
        """Auto-detect source type."""
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            return 'webcam'
        
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        
        if path.is_file():
            ext = path.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                return 'image'
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                return 'video'
        elif path.is_dir():
            return 'directory'
        
        raise ValueError(f"Could not detect source type for: {source}")
    
    def _ingest_image(self, path: str):
        """Ingest single image."""
        img = cv2.imread(str(path))
        if img is not None:
            metadata = {
                'source': str(path),
                'type': 'image',
                'shape': img.shape,
                'filename': Path(path).name
            }
            yield img, metadata
    
    def _ingest_video(self, path: str):
        """Ingest video file frame by frame."""
        cap = cv2.VideoCapture(str(path))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            metadata = {
                'source': str(path),
                'type': 'video',
                'frame': frame_count,
                'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC)
            }
            yield frame, metadata
            frame_count += 1
        
        cap.release()
    
    def _ingest_webcam(self, camera_id: int):
        """Ingest from webcam."""
        cap = cv2.VideoCapture(int(camera_id))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            metadata = {
                'source': f'webcam_{camera_id}',
                'type': 'webcam',
                'frame': frame_count,
                'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC)
            }
            yield frame, metadata
            frame_count += 1
        
        cap.release()
    
    def _ingest_directory(self, path: str):
        """Ingest all images from directory."""
        path = Path(path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for img_path in path.iterdir():
            if img_path.suffix.lower() in image_extensions:
                yield from self._ingest_image(img_path)