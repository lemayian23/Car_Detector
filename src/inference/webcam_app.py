import cv2
import numpy as np
from tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import time
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.detector import CarDetector
from src.models.classifier import CarClassifier
from src.utils.helpers import load_config

class CarDetectionApp:
    """Fun webcam application for car detection. """

    def __init__(self):
        self.config = load_config()
        self.window = tk.Tk()
        self.window.title("🚗 Car Detection Fun App 🚗")
        self.window.geometry("1200x800")

        #Initialize models
        self.detector = None
        self.classifier = None
        self.camera = None
        self.is_running = None
        self.current_frame = None


        #Statistics
        self.detection_count = 0
        self.car_types_count = {}

        self.setup_ui()
        self.load_models()

    def setup_ui(self):
        """Setup the user interface. """
        #Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column = 0, sticky=(tk.W, tk.E, tk.N, tk.S))

        #Left panel = Video feed
        left_frame = tk.Frame(main_frame)
        left_frame.grid(row=0, column = 0, padx=5, pady = 5)

        self.video_label = ttk.Frame(left_frame)
        control_frame.grid(row=0, column= 0)

        #   Control buttons
        control_frame = ttk.Frame(left_frame)
        control_frame.grid(row=1, column=0, pady = 10)

        self.start_btn = ttk.Button(
            control_frame,
            text = "▶ Start Camera",
            command = self.toggle_camera,
            width = 15
        )
        self.start_btn.grid(row=0, column=0, padx=5)

        self.screenshot_btn= ttk.Button(
            control_frame,
            text= "📸 Screenshot",
            command = self.take_screenshot,
            width = 15,
            state = 'disabled'
        
        )
        self.screenshot_btn.grid(row=0, column =1, padx =5)

        #Right panel -Info and stats
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.grid(row=0, column=1, padx=5, pady=5. sticky=(tk.N,tk.S))

        #Detection settings
        settings_frame = ttk.LabelFrame(right_frame, text=)