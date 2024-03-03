import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize variables
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Dictionary for the classification
labels_dict = {0: 'Thumbs Up', 1: 'V', 2: 'Ok'}

# Loop through each frame of the webcam
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    height, width, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Recolor frame to RGB

    results = hands.process(frame_rgb)  # Process frame using mediapipe and save the data into this variable

    if results.multi_hand_landmarks:
        # Draw landmarks on the screen
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Iterate through landmarks and save their data on the "data_aux" array
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        # Define borders of the rectangle around hand detected
        x1 = int(min(x_) * width) - 20
        y1 = int(min(y_) * height) - 20

        x2 = int(max(x_) * width) + 20
        y2 = int(max(y_) * height) + 20

        # If more than one hand is shown in the camera at once, it will throw an exception and skip current frame
        try:
            prediction = model.predict([np.asarray(data_aux)])  # Make prediction
            predicted_symbol = labels_dict[int(prediction[0])]  # Use dictionary

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # Draw the rectangle around detected hand
            cv2.putText(frame, predicted_symbol, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)  # Write the predicted hand gesture/pose on top of the rectangle
        except Exception as e:
            continue

    cv2.imshow('webcam', frame)  # Show frame on the screen
    cv2.waitKey(1)

cap.release()
cap.destroyAllWindows()
