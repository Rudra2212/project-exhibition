# Sign Language Alphabet Detector

This project is a real-time computer vision application that detects American Sign Language (ASL) alphabets using a webcam. 

It tracks hand joints using **Google MediaPipe** and classifies the ASL gestures into letters using a lightweight **TensorFlow Lite** Machine Learning model. The application features a smooth graphical user interface (GUI) built with Tkinter.

## Features
* **Real-Time Detection:** Tracks hand landmarks at high FPS using your webcam.
* **Full ASL Alphabet:** Detects A-Z, as well as "Space" and "Delete" gestures.
* **Auto-Typing:** Hold a gesture steady for ~1.5 seconds, and the app will automatically "type" the letter into the sentence box to help you spell out words easily.
* **Lightweight:** Uses an optimized TFLite model, meaning it doesn't require a high-end GPU to run smoothly.

## Prerequisites
Make sure you have Python installed. You can install all the required dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
*(Dependencies include: `opencv-python`, `mediapipe`, `tensorflow`, `numpy`, and `Pillow`)*

## Usage

Simply run the main application file from your terminal:

```bash
python app_ui.py
```

1. **Video Panel:** Your webcam feed will appear on the left with skeletal landmarks drawn over your hand.
2. **Current Letter:** The right panel will continuously display the alphabet you are currently signing.
3. **Sentence Box:** The text box at the bottom will collect your letters to form words.

## Project Structure
* `app_ui.py`: The main Tkinter application script handling the UI, webcam stream, and prediction logic.
* `keypoint_classifier.py`: A Python wrapper class that handles loading and running the TensorFlow Lite model.
* `model/`: Contains the pre-trained `keypoint_classifier.tflite` model, the label mapping (`keypoint_classifier_label.csv`), and the original training data (`keypoint.csv`).
* `requirements.txt`: The list of Python library dependencies.
