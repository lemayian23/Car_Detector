import cv2
import numpy as np
import time
import random
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.detector import CarDetector
from src.utils.helper import load_config

class __init__(self):
    self.config =  load_config()
    self.detector = CarDetector(
        model_type = self.config['detection']['model_type'],
        config = self.config
    )

    self.score = 0
    self.level = 1
    self.target_car = None
    self.targets_found = False
    self.game_active = False
    self.start_time = None
    self.time_limit = 60 #60 seconds

    #Car types for targets
    self.car_type = self.config['detection']['classes']

    #Fun facts about cars
    self.fun_facts = [
        "The first car was invented in 1886!",
        "The average car has 30,000 parts! ",
        "The fastest production car goes 304mph!",
        "There are over 1 billion cars in the world!",
        "The longest car is 100 feet long!",
        "Most cars beep in the key of F!",
        "The first speeding ticket was issued in 1902!",
        "Acar uses half its fuel just getting to the store!",
        "The most expensive car ever sold was $70milliom!",
        "Car radios were invented in 1929!"
    ]
def start_game(self, source = 0):
    """Start the I Spy car game."""
    print("\n"+"="*50)
    print("🚗 I SPY CAR DETECTION GAME 🚗")
    print("="*50)
    print("\nHow to play:")
    print("1. Find and detect the target car type")
    print("2. Keep the car in frame to score points")
    print("3. Level up every 5 targets found!")
    print("\nPress 'q' to quit 'p' to pause")
    print("="*50)

    #Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Could not open camera!")
        return

    self.game_active = True
    self.start_time = time.time()
    self.set_new_target()


    #FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    while self.game_active:
        ret, frame = cap.read()
        if not ret:
            break

        #Calculate FPS
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.time()

        #Detect cars
        detection = self.detector.detect(
            frame,
            confidence = 0.5
        )
        #Check game conditions
        time_left = max(0, self.time_limit - (time.time() - self.start_time))

        if time_left <= 0:
            self.game_over("Time's up!")
            break

        #Process detection for game
        frame, found_target = self.process_game_frame(frame, detections)

        #Add game overlay
        frame = self.add_game_overlay(frame, time_left, fps)

        #Display 
        cv2.imshow("I Spy Car Game", frame)

        #Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0) #Pause
        elif key == ord('n'):
            self.set_new_target() #New target

    cap.release()
    cv2.destroyAllWindows()

def process_game_frame(self, frame, detections):
    """Process frame for game logic. """
    found_target = False

    for det in detections:
        x1, y1,x2, y2 = map(int, det['bbox'])
        class_name = det['class_name']
        
        #Check if this is our target
        if class_name.lower() ==self.target_car.lower():
            found_target = True

            #Draw special target box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            #Add sparkle effect
            self.add_sparkle_effect(frame, x1, y1, x2, y2)

            #Increment score while target is visivle
            self.score += 1

            #Level up every 100 points
            if self.score % 100 == 0:
                self.level_up()
        else:
            #Draw normal box for other cars
            cv2.rectangle(frame, (x1,y1), (x2, y2), (0, 255, 0), 2)

        #Add  label
        label = f"{class_name}: {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #Update target progress
    if found_target:
        self.targets_found = min(self.targets_found + 1, 100)

    return frame, found_target

def add_game_overlay(self, frame, time_left, fps):
    """Add game information overlay. """
    overlay = frame.copy()
    height, width = frame.shape[:2]

    #Semi-transparent overlay for text
    cv2.rectangle(overlay, (0,0), (width, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    #Game info
    info_text = f"Score: {self.score} Level: {self.level} Time: {int(time_left)}s"
    cv2.putText(frame, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    #Target info
    target_text = f"Find: {self.target_car}"
    target_size = cv2.getTextSize(target_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    target_x = width - target_size[0] - 10
    cv2.putText(frame, target_text, (target_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    #Progress bar
    bar_width = 200
    bar_height = 20
    bar_x = (widt - bar_width) //2
    bar_y = height - 40

    #Background
    cv2.rectangle(frame, (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (100, 100,100), -1)
    
    #Progress
    progress_width = int(bar_width * (self.targets_found / 100))
    cv2.rectangle(frame, (bar_x, bar_y),
                    (bar_x + progress_width, bar_y + bar_height),
                    (0, 255, 0), -1)

    #FPS
    cv2.putText(frame, f"FPS: {fps}", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)

    return frame

def add_sparkle_effect(self, frame, x1, y1, x2, y2):
    """Add sparkle effect around target. """
    center_x = (x1 + x2) //2
    center_y = (y1 + y2) //2

    for _ in range(5):
        angle = random.uniform(0, 2 * np.pi)
        distance = random.randint(20,40)

        sparkle_x = int(center_x + distance * np.cos(angle))
        sparkle_y = int(center_y + distance * np.sin(angle))

        #Draw sparkle
        
