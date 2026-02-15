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
        settings_frame = ttk.LabelFrame(right_frame, text="⚙ Settings", padding = "10")
        settings_frame.grid(row=0 column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Label(settings_frame, text="Confidence:".grid(row=0, column=0, sticky=tk.W))
        self.confidence_var = tk.Double(value=0.5)
        self.confidence_scale = ttk.Scale(
            settings_frame,
            from_ 0.1, to=1.0,
            variable=self.confidence_var,
            orient=tk.HORIZONTAL,
                length=200
        )
        self.confidence_scale.grid(row=0, column=1,padx=5)
        self.confidence_label= ttk.Label(settings_frame, text="0.5")
        self.confidence_label.grid(row=0, column=2)

        self.confidence_scale.configure(command=lambda x: self.confidence_label.configure(text=f"{float(x):.2f}"))


        #Live stats
        stats_frame = ttk.LabelFrame(right_frame, text="📊 Live Stats", padding="10")
        stats_frame.grid(row=1, column=0, pady=10, sticky=(tk.W, tk.E))

        self.stats_text = tk.text(stats_frame,height=10.width=40,font=("Courier", 10))
        self.stats_text.grid(row=0,column=0)

        #Detection log
        log_frame = ttk.LabelFrame(right_frame, text="📝 Detection Log", padding="10")
        log_frame.grid(row=2, column=0, pady=5,sticky=(tk.W, tk.E, tk.S))

        self.log_text=tk.Text(log_frame, height=15, width=40, font=("Courier", 9))
        self.log_text.grid(row=0, column=0)

        scrollbar.grid(row=0,column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=scrollbar.set)

        #Fun facts
        self.fact_label = ttk.Label(
            right_frame,
            text ="Did you know? The fastest car in the world is the SSC Tuatara at 316 mph!",
            wraplength=350,
            font=("Arial", 10, "italic")
        )
        self.fact_label.grid(row=3, column =0, pady =10)

        #Bind close event
        self.window.protocal("WM_DELETE_WINDOW", self.on_closing)

        def_load_models(self):
            """Load detection and classification models. """
            try:
                self.detector = CarDetector(
                    model_type=self.config['detection']['model_type'],
                    config=self.config
                )
                self.log("✅ Detector loaded successfully!")
            except Exception as e:
                self.log(f"⚠ Detector not loaded:{e}")

            try:
                self.classifier = CarClasifier(
                    num_classes= len(self.config['classification']['car_types']),
                    model_type=self.config['classification']['model_type'],
                    config=self.config
                )
                self.log("✅ Classifier loaded successfully!")
            except Exception as e:
                self.log(f"⚠ Classifier not loaded: {e}")

        def toggle_camera(self):
            """Start or stop the camera."""
            if not self.is_running:
                self.start_camera()
            else:
                self.stop_camera()

        def start_camera(self):
            """Sart the camera feed. """
            self.camera= cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.log("❌ Could not open camera!")
                return

            self.is_running = Trueself.start_btn.configure(text="|| Stop Camera")
            self.screenshot_btn.configure(state='normal')
            self.update_frame()

        def stop_camera(self):
            """Stop the camera feed. """
            self.is_running = False
            if self.camera:
                self.camera.release()
            self.start_btn_configure(text="▶ Start Camera")
            self.screenshot_btn.configure(state='disabled')
            self.video_label_configure(image='')

        def update_frame(self):
            """Update video frame. """
            if self.is_running:
                ret, frame = self.camera.read()
                if ret:
                    self.current_frame = frame.copy()

                    #Detect cars
                    if self.detector:
                        detections = self.detector.detect(
                            frame,
                            confidence=self.confidence_var.get()
                        )

                    #Draw detection
                    frame = self.draw_detections(frame, detections)

                    #Update stats
                    self.update_stats(detections)

                #Convert for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                self.window.after(10, self.update_frame)

        def draw_detections(self, frame, detections):
            """Draw detection boxes and labels. """
            colors = [(255,0,0), (0,255,0), (0,0,255),
                        (255, 255, 0), (255, 0, 255),(0, 255, 255)]

            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                class_name = det['class_name']
                confidence = det['confidence']

                #Random color based on class
                color = colors[hash(class_name) % len(colors)]

                #Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                #Draw label background

                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame,
                              (x1, y1 - label_size[1] - 10),
                              (x1+ label_size[0], y1),
                              color, -1)


                #Draw label text
                
