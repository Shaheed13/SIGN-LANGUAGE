import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import requests
import tkinter as tk
from PIL import Image, ImageTk
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load Trained Model
model = tf.keras.models.load_model('sign_language_model.h5')

# Class Labels
labels_dict = {0: "Hello", 1: "Thank You", 2: "Yes", 3: "No", 4: "Please"}

# ThingSpeak API
THINGSPEAK_URL = "https://api.thingspeak.com/update"
THINGSPEAK_API_KEY = "YOUR_THINGSPEAK_API_KEY"

# GUI for displaying GIFs
class GestureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")
        self.label = tk.Label(root)
        self.label.pack()
        self.current_gif = None
        self.gif_frames = []
        self.frame_index = 0

    def load_gif(self, gif_path):
        self.gif_frames.clear()
        image = Image.open(gif_path)
        try:
            while True:
                frame = image.copy()
                self.gif_frames.append(ImageTk.PhotoImage(frame))
                image.seek(len(self.gif_frames))
        except EOFError:
            pass
        self.frame_index = 0
        self.update_gif()

    def update_gif(self):
        if self.gif_frames:
            self.label.configure(image=self.gif_frames[self.frame_index])
            self.frame_index = (self.frame_index + 1) % len(self.gif_frames)
            self.root.after(100, self.update_gif)

def send_to_thingspeak(message):
    payload = {"api_key": THINGSPEAK_API_KEY, "field1": message}
    try:
        requests.get(THINGSPEAK_URL, params=payload)
    except requests.RequestException as e:
        print("Error sending to ThingSpeak:", e)

# Capture Hand Gestures
cap = cv2.VideoCapture(0)
gui = GestureGUI(tk.Tk())

def process_frame():
    ret, frame = cap.read()
    if not ret:
        return
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract Features
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(1, -1)
            
            # Predict Gesture
            prediction = model.predict(landmarks)
            gesture_index = np.argmax(prediction)
            gesture_text = labels_dict.get(gesture_index, "Unknown")
            
            # Display Gesture
            cv2.putText(frame, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Send Data to ThingSpeak
            send_to_thingspeak(gesture_text)
            
            # Update GUI with GIF
            gif_path = f"gifs/{gesture_text}.gif"
            gui.load_gif(gif_path)
    
    cv2.imshow("Hand Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        gui.root.quit()
        return
    
    gui.root.after(10, process_frame)

gui.root.after(10, process_frame)
gui.root.mainloop()
