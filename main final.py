from gtts import gTTS
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import urllib.request
import random
import json
import string
import threading
from PIL import Image
import requests

# ThingSpeak Channel Settings
THINGSPEAK_API_KEY = "IP6U7YPZJW9WNANU"  # Replace with your API key
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/466832/feeds/last.json?api_key={THINGSPEAK_API_KEY}"

# List of predefined words that have corresponding GIFs
isl_gif = [
    'any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
    'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 
    'do you have money', 'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 
    'dont worry', 'flower is beautiful', 'good afternoon', 'good evening', 'good morning', 'good night', 
    'good question', 'had your lunch', 'happy journey', 'hello what is your name', 'how many people are there in your family', 
    'i am a clerk', 'i am bore doing nothing', 'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 
    'i dont understand anything', 'i go to a theatre', 'i love to shop', 'i had to say something but i forgot', 
    'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker', 
    'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later', 
    'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 
    'please wait for sometime', 'shall I help you', 'shall we go together tomorrow', 'sign language interpreter', 
    'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking', 'what are you doing', 
    'what is the problem', 'what is todays date', 'what is your father do', 'what is your job', 
    'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when we will go', 
    'where do you stay', 'where is the bathroom', 'where is the police station', 'you are wrong'
]


def send_to_thingspeak(message):
        mytext = message
        encoded_text = urllib.parse.quote(mytext)
        print(encoded_text)
        wp = urllib.request.urlopen("https://api.thingspeak.com/update?api_key=LNXWKWVLZ8LL8EQX&field1=" + encoded_text)
        
# Alphabet list
arr = list(string.ascii_lowercase)

# Global variables for threading
latest_text = ""
gif_frame = None

def fetch_text_from_thingspeak():
    """Fetches the latest text from ThingSpeak"""
    global latest_text
    while True:
        try:
            response = urllib.request.urlopen(THINGSPEAK_URL)
            data = json.load(response)
            text = data["field1"]  # Assuming text is stored in 'field1'
            latest_text = text.strip().lower()
        except Exception as e:
            print(f"Error fetching data from ThingSpeak: {e}")
        time.sleep(2)  # Fetch every 2 seconds

def display_gif_or_letters():
    """Displays GIF or letters based on the latest text"""
    global latest_text, gif_frame
    while True:
        if latest_text:
            text = latest_text
            for c in string.punctuation:
                text = text.replace(c, "")

            if text in isl_gif:
                gif_path = f'ISL_Gifs/{text}.gif'
                if os.path.exists(gif_path):
                    gif = Image.open(gif_path)
                    gif_frame = np.array(gif.convert('RGB'))
                else:
                    print("GIF not found for:", text)
                    gif_frame = None
            else:
                gif_frame = None
                for char in text:
                    if char in arr:
                        image_address = f'letters/{char}.jpg'
                        if os.path.exists(image_address):
                            image = cv2.imread(image_address)
                            if gif_frame is None:
                                gif_frame = image
                            else:
                                gif_frame = np.hstack((gif_frame, image))
                        else:
                            print("Letter image not found for:", char)
        time.sleep(1)  # Check for updates every second

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognition model
model = load_model('model.hdf5')

# Load gesture names
with open('gesture.names8', 'r') as f:
    classNames = f.read().split('\n')
print(classNames)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Start threads for ThingSpeak and GIF/letter display
threading.Thread(target=fetch_text_from_thingspeak, daemon=True).start()
threading.Thread(target=display_gif_or_letters, daemon=True).start()

while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    image_width, image_height = frame.shape[1], frame.shape[0]
    frame = cv2.flip(frame, 1)

    # Get hand landmark prediction
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for i, lm in enumerate(handslms.landmark):
                lmx = int(lm.x * image_width)
                lmy = int(lm.y * image_height)
                if i == 0:
                    temp_x = lmx
                    temp_y = lmy
                    lmx = 0
                    lmy = 0
                else:
                    lmx = lmx - temp_x
                    lmy = lmy - temp_y
                landmarks.append(lmx)
                landmarks.append(lmy)

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            landmarks = np.array(landmarks)
            landmarks = landmarks / abs(max(landmarks, key=abs))
            landmarks = landmarks.reshape(42, 1)
            prediction = model.predict(np.array([landmarks]))
            classID = np.argmax(prediction)
            className = classNames[classID]

            # Display emoji overlay
            emoji_path = f"emojis/{classID}.png"
            if os.path.exists(emoji_path):
                overlay = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                if overlay.shape[2] == 4:  # Check if the image has an alpha channel
                    overlay = cv2.resize(overlay, (100, 100))
                    x_offset, y_offset = 10, 10
                    for c in range(0, 3):
                        frame[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1], c] = \
                            overlay[:, :, c] * (overlay[:, :, 3] / 255.0) + \
                            frame[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1], c] * (1.0 - overlay[:, :, 3] / 255.0)
                else:
                    overlay = cv2.resize(overlay, (100, 100))
                    x_offset, y_offset = 10, 10
                    frame[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]] = overlay

    # Add the recognized gesture text to the frame
    cv2.putText(frame, className, (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    if(className !=""):
        send_to_thingspeak(className)
        print('Sending to thingspeak')
    # Create a blank frame for the second functionality (GIF/letters)
    if gif_frame is not None:
        gif_frame_resized = cv2.resize(gif_frame, (image_width, image_height))
    else:
        gif_frame_resized = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Combine the webcam feed and GIF/letter display vertically
    combined_frame = np.hstack((frame, gif_frame_resized))

    # Show the final output
    cv2.imshow("Output", combined_frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
