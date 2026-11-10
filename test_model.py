import cv2
import numpy as np
import pickle
import time

model = pickle.load(open("model.pkl", "rb"))

labels = ['A','B','C','D','E','F','G','H','I','K','L','M',
          'N','O','P','Q','R','S','T','U','V','W','X','Y']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera warming up...")
for i in range(10):
    cap.read()
time.sleep(2)
print("Ready!")

sentence = ""
last_letter = ""
letter_count = 0
CONFIRM_FRAMES = 20  # 20 frames same letter வந்தா add ஆகும்

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    x1, y1, x2, y2 = 200, 100, 500, 400
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, "Put hand here", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    flattened = resized.flatten().reshape(1, -1)

    prediction = model.predict(flattened)[0]
    letter = labels[prediction] if prediction < len(labels) else str(prediction)

    # Same letter count பண்ணு
    if letter == last_letter:
        letter_count += 1
    else:
        last_letter = letter
        letter_count = 0

    # 20 frames same letter → sentence-ல add பண்ணு
    if letter_count == CONFIRM_FRAMES:
        sentence += letter
        letter_count = 0

    # Display
    cv2.putText(frame, f"Letter: {letter}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
    cv2.putText(frame, f"Sentence: {sentence}", (50, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

    # Controls info
    cv2.putText(frame, "C=Clear  SPACE=Space  Q=Quit", (50, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.imshow("Sign Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""  # Clear
    elif key == ord(' '):
        sentence += " "  # Space

cap.release()
cv2.destroyAllWindows()