import cv2
import numpy as np
import pickle

# load model
model = pickle.load(open("model.pkl", "rb"))

# webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Webcam not opening")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    # 🔥 DRAW BOX (IMPORTANT FOR USER GUIDANCE)
    cv2.rectangle(frame, (150, 100), (450, 400), (0, 255, 0), 2)

    # crop ROI (ONLY HAND AREA)
    roi = frame[100:400, 150:450]

    # convert to gray
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # resize to model input
    img = cv2.resize(gray, (28, 28))

    # normalize
    img = img / 255.0

    # invert (MNIST style match)
    img = 1 - img

    # reshape for model
    img = img.flatten().reshape(1, -1)

    # prediction
    prediction = model.predict(img)

    # show result
    cv2.putText(frame, f"Prediction: {prediction[0]}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # display
    cv2.imshow("Sign Detection", frame)
    cv2.imshow("ROI", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
