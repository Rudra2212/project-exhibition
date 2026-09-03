import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import copy
import itertools
from collections import Counter
import csv
import sys
import os

# Add the current directory to path so it can find keypoint_classifier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from keypoint_classifier import KeyPointClassifier

class SignLanguageApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Load Labels
        self.labels = []
        labels_path = os.path.join(os.path.dirname(__file__), 'model', 'keypoint_classifier_label.csv')
        with open(labels_path, encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            self.labels = [row[0] for row in reader]
            
        self.labels.append("not recognised")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize Classifier
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'keypoint_classifier.tflite')
        self.classifier = KeyPointClassifier(model_path=model_path)
        
        # Video Capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # UI Elements
        self.main_frame = ttk.Frame(window, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel (Video)
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.grid(row=0, column=0, rowspan=4, padx=10, pady=10)
        
        # Right Panel (Controls & Text)
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, sticky="n")
        
        ttk.Label(self.right_frame, text="Detected Letter:", font=("Helvetica", 14)).pack(anchor="w", pady=5)
        self.current_letter_var = tk.StringVar(value="-")
        self.current_letter_label = ttk.Label(self.right_frame, textvariable=self.current_letter_var, font=("Helvetica", 48, "bold"), foreground="blue")
        self.current_letter_label.pack(pady=10)
        
        ttk.Label(self.right_frame, text="Sentence:", font=("Helvetica", 14)).pack(anchor="w", pady=5)
        self.sentence_text = tk.Text(self.right_frame, height=5, width=30, font=("Helvetica", 14))
        self.sentence_text.pack(pady=10)
        
        # Buttons
        self.btn_frame = ttk.Frame(self.right_frame)
        self.btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(self.btn_frame, text="Add Letter (Space)", command=self.add_letter).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.btn_frame, text="Add Space", command=self.add_space).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.btn_frame, text="Backspace", command=self.backspace).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(self.btn_frame, text="Clear All", command=self.clear_text).grid(row=1, column=1, padx=5, pady=5)
        
        self.window.bind('<space>', lambda e: self.add_letter())
        self.window.bind('<BackSpace>', lambda e: self.backspace())
        
        # Smoothing mechanism
        self.history = []
        self.current_stable_char = ""
        
        # Auto-add mechanism
        self.auto_add_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.right_frame, text="Auto-type (hold sign for 1.5s)", variable=self.auto_add_var).pack(pady=10)
        self.hold_count = 0
        
        self.update_frame()
        
    def add_letter(self):
        char = self.current_letter_var.get()
        if char and len(char) == 1 and char != '-':
            self.sentence_text.insert(tk.END, char)
            
    def add_space(self):
        self.sentence_text.insert(tk.END, " ")
        
    def backspace(self):
        current_text = self.sentence_text.get("1.0", tk.END)
        if len(current_text) > 1:
            self.sentence_text.delete("end-2c", tk.END)
            
    def clear_text(self):
        self.sentence_text.delete("1.0", tk.END)
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            debug_image = copy.deepcopy(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            detected_char = "-"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on frame
                    self.mp_draw.draw_landmarks(debug_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Preprocess for model
                    landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                    
                    # Predict
                    hand_sign_id = self.classifier(pre_processed_landmark_list)
                    if hand_sign_id == -1:
                        detected_char = "-"
                    else:
                        detected_char = self.labels[hand_sign_id]
            
            # Smoothing logic
            self.history.append(detected_char)
            if len(self.history) > 10:
                self.history.pop(0)
                
            most_common = Counter(self.history).most_common(1)[0][0]
            self.current_letter_var.set(most_common)
            
            # Auto-add logic
            if self.auto_add_var.get():
                if most_common == self.current_stable_char and most_common != "-":
                    self.hold_count += 1
                else:
                    self.current_stable_char = most_common
                    self.hold_count = 0
                    
                # 30 frames ~ 1.0 - 1.5 seconds at 30 fps
                if self.hold_count == 30:
                    if most_common == 'space':
                        self.add_space()
                    elif most_common == 'del':
                        self.backspace()
                    else:
                        self.sentence_text.insert(tk.END, most_common)
                    self.hold_count = -20 # Cooldown so it doesn't spam
            
            # Convert to Tkinter image
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(debug_image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        self.window.after(10, self.update_frame)

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
            
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list)))
        if max_value == 0:
            max_value = 1
        temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
        return temp_landmark_list

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root, "ASL Alphabet Detector")
    root.mainloop()
