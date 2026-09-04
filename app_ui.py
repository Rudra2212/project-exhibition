import customtkinter as ctk
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

# Set Modern UI Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SignLanguageApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1100x700")
        self.window.minsize(950, 650)
        
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
        
        # ----------------------------------------------------
        # UI LAYOUT WITH BACKGROUND DESIGN
        # ----------------------------------------------------
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Main Background Wrapper (Gives the app a nice outer canvas)
        self.main_bg = ctk.CTkFrame(self.window, fg_color=("#f0f2f5", "#141416"), corner_radius=0)
        self.main_bg.grid(row=0, column=0, sticky="nsew")
        
        self.main_bg.grid_columnconfigure(0, weight=1)
        self.main_bg.grid_columnconfigure(1, weight=0)
        self.main_bg.grid_rowconfigure(0, weight=1)
        
        # --- LEFT PANEL (Header & Camera) ---
        self.left_panel = ctk.CTkFrame(self.main_bg, fg_color="transparent")
        self.left_panel.grid(row=0, column=0, padx=25, pady=25, sticky="nsew")
        self.left_panel.grid_rowconfigure(1, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        # Header inside Left Panel
        self.left_header = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.left_header.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        # Dark/Light Mode Toggle Switch
        self.theme_switch_var = ctk.StringVar(value="Dark")
        self.theme_switch = ctk.CTkSwitch(self.left_header, text="Dark Mode", command=self.toggle_theme,
                                          variable=self.theme_switch_var, onvalue="Dark", offvalue="Light",
                                          font=ctk.CTkFont(size=14, weight="bold"))
        self.theme_switch.pack(side="left")
        
        # Camera Title (Centered above camera)
        self.camera_title = ctk.CTkLabel(self.left_header, text="Sign Language Detector", 
                                         font=ctk.CTkFont(size=32, weight="bold", family="Helvetica"))
        self.camera_title.pack(side="left", expand=True, padx=(0, 100)) # Offset slightly to center visually
        
        # Video Frame Wrapper (Design element with borders)
        self.video_frame = ctk.CTkFrame(self.left_panel, corner_radius=20, 
                                        fg_color=("#ffffff", "#1e1e21"), 
                                        border_width=2, border_color=("#d1d5db", "#333333"))
        self.video_frame.grid(row=1, column=0, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=15, pady=15)
        
        # --- RIGHT PANEL (Controls & Sidebar) ---
        self.right_frame = ctk.CTkFrame(self.main_bg, width=380, corner_radius=20, 
                                        fg_color=("#ffffff", "#1e1e21"),
                                        border_width=2, border_color=("#d1d5db", "#333333"))
        self.right_frame.grid(row=0, column=1, padx=(0, 25), pady=25, sticky="nsew")
        self.right_frame.grid_propagate(False)
        
        # Title inside Controls
        self.title_label = ctk.CTkLabel(self.right_frame, text="Live Translation", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(25, 20))
        
        # Detected Letter Widget
        self.letter_title = ctk.CTkLabel(self.right_frame, text="Current Letter", font=ctk.CTkFont(size=14), text_color="gray")
        self.letter_title.pack(pady=(10, 0))
        
        self.current_letter_var = ctk.StringVar(value="-")
        self.current_letter_label = ctk.CTkLabel(self.right_frame, textvariable=self.current_letter_var, 
                                                 font=ctk.CTkFont(size=90, weight="bold"), text_color="#3a7ebf")
        self.current_letter_label.pack(pady=(0, 20))
        
        # Sentence Textbox
        self.sentence_title = ctk.CTkLabel(self.right_frame, text="Word / Sentence", font=ctk.CTkFont(size=14), text_color="gray")
        self.sentence_title.pack(pady=(10, 5), anchor="w", padx=25)
        
        self.sentence_text = ctk.CTkTextbox(self.right_frame, height=140, font=ctk.CTkFont(size=22), 
                                            corner_radius=12, border_width=1, border_color=("#cccccc", "#444444"),
                                            fg_color=("#f9fafb", "#18181a"))
        self.sentence_text.pack(fill="x", padx=25, pady=(0, 20))
        
        # Settings
        self.auto_add_var = ctk.BooleanVar(value=True)
        self.auto_checkbox = ctk.CTkCheckBox(self.right_frame, text="Auto-type (hold sign for 1.5s)", variable=self.auto_add_var,
                                             font=ctk.CTkFont(size=14))
        self.auto_checkbox.pack(pady=(0, 25), padx=25, anchor="w")
        
        # Button Grid
        self.btn_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=20)
        
        self.btn_frame.grid_columnconfigure(0, weight=1)
        self.btn_frame.grid_columnconfigure(1, weight=1)
        
        self.btn_add = ctk.CTkButton(self.btn_frame, text="Type Letter", height=45, font=ctk.CTkFont(size=14, weight="bold"), command=self.add_letter)
        self.btn_add.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.btn_space = ctk.CTkButton(self.btn_frame, text="Space", height=45, font=ctk.CTkFont(size=14, weight="bold"), command=self.add_space, fg_color=("#71717a", "#444444"), hover_color=("#52525b", "#555555"))
        self.btn_space.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.btn_back = ctk.CTkButton(self.btn_frame, text="Backspace", height=45, font=ctk.CTkFont(size=14, weight="bold"), command=self.backspace, fg_color="#ef4444", hover_color="#dc2626")
        self.btn_back.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.btn_clear = ctk.CTkButton(self.btn_frame, text="Clear All", height=45, font=ctk.CTkFont(size=14, weight="bold"), command=self.clear_text, fg_color="#ef4444", hover_color="#dc2626")
        self.btn_clear.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Keyboard Bindings
        self.window.bind('<space>', lambda e: self.add_space())
        self.window.bind('<BackSpace>', lambda e: self.backspace())
        self.window.bind('<Return>', lambda e: self.add_letter())
        
        # Logic Variables
        self.history = []
        self.current_stable_char = ""
        self.hold_count = 0
        
        # Start Loop
        self.update_frame()
        
    def toggle_theme(self):
        theme = self.theme_switch_var.get()
        ctk.set_appearance_mode(theme)
        if theme == "Light":
            self.theme_switch.configure(text="Light Mode")
        else:
            self.theme_switch.configure(text="Dark Mode")
            
    def add_letter(self):
        char = self.current_letter_var.get()
        if char and len(char) == 1 and char != '-':
            self.sentence_text.insert("end", char)
            
    def add_space(self):
        self.sentence_text.insert("end", " ")
        
    def backspace(self):
        current_text = self.sentence_text.get("1.0", "end-1c")
        if len(current_text) > 0:
            self.sentence_text.delete("end-2c", "end-1c")
            
    def clear_text(self):
        self.sentence_text.delete("1.0", "end")
        
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
                        self.sentence_text.insert("end", most_common)
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
    root = ctk.CTk()
    app = SignLanguageApp(root, "ASL Translator - Pro Edition")
    root.mainloop()
