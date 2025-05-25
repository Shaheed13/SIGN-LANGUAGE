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



import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easygui import *
import os
from PIL import Image, ImageTk
from itertools import count
import tkinter as tk
import string
import time

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

# Alphabet list
arr = list(string.ascii_lowercase)

def fetch_text_from_thingspeak():
    """Fetches the latest text from ThingSpeak"""
    try:
        response = urllib.request.urlopen(THINGSPEAK_URL)
        data = json.load(response)
        text = data["field1"]  # Assuming text is stored in 'field1'
        return text.strip().lower()
    except Exception as e:
        print(f"Error fetching data from ThingSpeak: {e}")
        return None

class ImageLabel(tk.Label):
    """A label that displays images and plays them if they are GIFs"""
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        self.loc = 0
        self.frames = []
        try:
            for i in count(1):
                self.frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass

        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100

        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.loc += 1
            self.loc %= len(self.frames)
            self.config(image=self.frames[self.loc])
            self.after(self.delay, self.next_frame)

def display_gif(text):
    """Displays the GIF corresponding to the detected text"""
    root = tk.Tk()
    lbl = ImageLabel(root)
    lbl.pack()
    gif_path = f'ISL_Gifs/{text}.gif'
    if os.path.exists(gif_path):
        lbl.load(gif_path)
    else:
        print("GIF not found for:", text)
    root.mainloop()

def display_alphabets(text):
    """Displays images corresponding to each letter in the detected text"""
    for char in text:
        if char in arr:
            image_address = f'letters/{char}.jpg'
            if os.path.exists(image_address):
                image_itself = Image.open(image_address)
                image_numpy_format = np.asarray(image_itself)
                plt.imshow(image_numpy_format)
                plt.draw()
                plt.pause(0.8)
            else:
                print("Letter image not found for:", char)
    plt.close()

    

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def gausss(image, var=0.01):
    row, col, ch = image.shape
    mean = 0
    # var = 0.01 ##the less the less noise
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = 255 * gauss  # Now scale by 255
    gauss = gauss.astype(np.uint8)
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy




def callback(x):
    pass  
def add_noise(pixels, img):
   
    row, col, depth = img.shape

    for i in range(round(pixels/2)):
      
        y_coord = random.randint(0, row - 1)

       
        x_coord = random.randint(0, col - 1)

       
        img[y_coord][x_coord][0] = 255
        img[y_coord][x_coord][1] = 255
        img[y_coord][x_coord][2] = 255

    for i in range(round(pixels/2)):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord][0] = 0
        img[y_coord][x_coord][1] = 0
        img[y_coord][x_coord][2] = 0

    return img
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


model = load_model('model.hdf5')


f = open('gesture.names8', 'r')

classNames = f.read().split('\n')
f.close()
print(classNames)

cv2.namedWindow('trackbar', 2)
cv2.resizeWindow("trackbar", 550, 10);


cv2.createTrackbar('s&p', 'trackbar', 0, 300000, callback)
cv2.createTrackbar('brightness', 'trackbar', 0, 255, callback)
cv2.createTrackbar('gauss', 'trackbar', 0, 100, callback)


cv2.createTrackbar('Median filter', 'trackbar', 0, 1, callback)
cv2.createTrackbar('Gaussian filter', 'trackbar', 0, 1, callback)

# Initialize the webcam
cap = cv2.VideoCapture(0)
i = 0
while True:
    # Read each frame from the webcam


    last_text = None  # Store the last processed message

    if 1:
        print("Fetching text from ThingSpeak...")
        text = fetch_text_from_thingspeak()
        
        if text and text != last_text:  # Only process if the message is new
            print(f"New message received: {text}")
            
            # Remove punctuation
            for c in string.punctuation:
                text = text.replace(c, "")

            if text in ["goodbye", "good bye", "bye"]:
                print("Time to say goodbye!")
                break

            elif text in isl_gif:
                display_gif(text)
            else:
                display_alphabets(text)

            last_text = text  # Update last processed message

          
    _, frame = cap.read()

    # x, y, c = frame.shape

    image_width, image_height = frame.shape[1], frame.shape[0]

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)



    # Get hand landmark prediction
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    
    className = ''
    kk=0

    # post process the result
    if result.multi_hand_landmarks:

        landmarks = []
        for handslms in result.multi_hand_landmarks:
            i = 0
            for lm in handslms.landmark:
                # print(id, lm)
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

                # landmarks.append([lmx, lmy])
                landmarks.append(lmx)
                landmarks.append(lmy)

                i = i + 1


            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            landmarks = np.array(landmarks)

            landmarks = landmarks/abs(max(landmarks, key=abs))

            landmarks = landmarks.reshape(42, 1)
            prediction = model.predict(np.array([landmarks]))


            classID = np.argmax(prediction)
            className = classNames[classID]


            emoji_path = "emojis/" + str(classID) + ".png"

            overlay = cv2.imread(emoji_path)
            if classID == 0:
                overlay = cv2.resize(overlay, (150, 150))
            else:
                overlay = cv2.resize(overlay, (100, 100))
            h, w = overlay.shape[:2]
            shapes = np.zeros_like(frame, np.uint8)
            shapes[0:h, 0:w] = overlay
            alpha = 0.8
            mask = shapes.astype(bool)
            frame[mask] = cv2.addWeighted(shapes, alpha, shapes, 1 - alpha, 0)[mask]

            kk=1
    # show the prediction on the frame
    cv2.putText(frame, className, (250, 200), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv2.LINE_AA)



    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break
    if(kk==1):
        mytext = className
        encoded_text = urllib.parse.quote(mytext)
        print(encoded_text)
        wp = urllib.request.urlopen("https://api.thingspeak.com/update?api_key=LNXWKWVLZ8LL8EQX&field1=" + encoded_text)
        time.sleep(2)
        

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
