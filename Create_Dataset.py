import os
import cv2
import mediapipe as mp
import pickle

# Define relevant variables
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = '.\data'

data = []
labels = []

# Loop through all folders inside the "data" folder (each folder has a different hand pose)
for dir_ in os.listdir(DATA_DIR):
    # Loop through all the images inside the folder
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  # Save image in this variable
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Recolor image to RGB

        results = hands.process(img_rgb)  # Process image using mediapipe and save the data into this variable

        # Check if only one hand is detected with 21 landmarks
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1 and len(results.multi_hand_landmarks[0].landmark) == 21:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Iterate through each individual landmark
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)  # Save landmark's x position
                data_aux.append(landmark.y)  # Save landmark's y position

            # Save data and label in the respective arrays
            data.append(data_aux)
            labels.append(dir_)

# Use pickle to save the arrays as a file in other to use them in other scripts later
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
