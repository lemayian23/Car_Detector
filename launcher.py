#!/usr/bin/env python
"""
Car Detection Project Launcher
Choose what you want to do!
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header():
    print("\n" + "="*60)
    print("🚗 CAR DETECTION PROJECT LAUNCHER 🚗")
    print("="*60)
    print("\nWhat would you like to do?\n")

def print_menu():
    menu = """
    [1] 🎯 Train the model
    [2] 📊 Evaluate the model
    [3] 📸 Run inference on an image
    [4] 🎥 Run inference on a video
    [5] 📹 Real-time webcam detection
    [6] 🎮 Play I Spy Car Game!
    [7] 🎨 Try fun image filters
    [8] 📓 Open Jupyter notebook for analysis
    [9] 🖥️ Launch GUI application
    [10] 📝 Prepare dataset
    [11] 🧹 Clean project
    [12] 🚪 Exit
    
    Enter your choice (1-12): """
    return menu

def run_command(command):
    """Run a command and wait for it to complete."""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        input("\nPress Enter to continue...")

def main():
    while True:
        print_header()
        choice = input(print_menu())
        
        if choice == '1':
            print("\n🚀 Starting training...")
            run_command("python scripts/train.py")
        
        elif choice == '2':
            print("\n📊 Evaluating model...")
            run_command("python scripts/evaluate.py")
        
        elif choice == '3':
            image_path = input("\nEnter path to image (or drag and drop): ").strip().strip('"')
            if image_path:
                run_command(f'python main.py --mode infer --image "{image_path}"')
        
        elif choice == '4':
            video_path = input("\nEnter path to video: ").strip().strip('"')
            if video_path:
                run_command(f'python scripts/infer.py --source "{video_path}"')
        
        elif choice == '5':
            print("\n📹 Starting webcam detection...")
            run_command("python scripts/infer.py --source 0")
        
        elif choice == '6':
            print("\n🎮 Starting I Spy Car Game...")
            run_command("python src/inference/car_spy_game.py")
        
        elif choice == '7':
            print("\n🎨 Fun filters coming soon!")
            print("Check out notebooks/fun_filters.ipynb for now!")
            input("\nPress Enter to continue...")
        
        elif choice == '8':
            print("\n📓 Starting Jupyter notebook...")
            run_command("jupyter notebook notebooks/")
        
        elif choice == '9':
            print("\n🖥️ Launching GUI application...")
            run_command("python src/inference/webcam_app.py")
        
        elif choice == '10':
            print("\n📝 Preparing dataset...")
            from src.data.data_preprocessor import DataPreprocessor
            from src.utils.helpers import load_config
            
            config = load_config()
            preprocessor = DataPreprocessor(config)
            
            print("\nWhat would you like to do?")
            print("1. Organize raw dataset")
            print("2. Create YOLO dataset")
            print("3. Analyze dataset")
            print("4. Validate labels")
            
            sub_choice = input("\nEnter choice (1-4): ")
            
            if sub_choice == '1':
                preprocessor.organize_dataset()
            elif sub_choice == '2':
                preprocessor.create_yolo_dataset()
            elif sub_choice == '3':
                preprocessor.analyze_dataset()
            elif sub_choice == '4':
                preprocessor.validate_labels()
            
            input("\nPress Enter to continue...")
        
        elif choice == '11':
            print("\n🧹 Cleaning project...")
            confirm = input("This will remove all __pycache__, .pyc files, and logs. Continue? (y/n): ")
            if confirm.lower() == 'y':
                # Remove Python cache
                run_command("find . -type d -name __pycache__ -exec rm -rf {} +")
                run_command("find . -type f -name '*.pyc' -delete")
                # Clear logs
                run_command("rm -rf logs/*")
                print("✅ Project cleaned!")
            input("\nPress Enter to continue...")
        
        elif choice == '12':
            print("\n👋 Goodbye!")
            sys.exit(0)
        
        else:
            print("\n❌ Invalid choice. Please try again.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()