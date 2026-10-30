import cv2
import mediapipe as mp
import numpy as np
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Enter gesture label (e.g. A, B, HELLO): ")

with open("data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                data = []

                for lm in handLms.landmark:
                    data.extend([lm.x, lm.y, lm.z])

                data.append(label)
                writer.writerow(data)

                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()