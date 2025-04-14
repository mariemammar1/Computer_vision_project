import cv2
from keras.models import load_model
import numpy as np

model = load_model("eye_tracking_model.h5")
threshold = 0.6

def detect_eye_movement(eye_image):
    eye_image = cv2.resize(eye_image, (48,48))
    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    eye_image = np.expand_dims(eye_image, axis=0)
    eye_image = np.expand_dims(eye_image, axis=-1)
    eye_image = eye_image.astype("float32")/255

    prob_right_movement = model.predict(eye_image)[0][1]
    if prob_right_movement > threshold:
        return True
    else:
        return False
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    for (x,y,w,h) in faces:
        eye_region = frame[y:y+h, x:x+w]
        movement = detect_eye_movement(eye_region)
        if movement:
            cv2.putText(frame, "Suspicious behaviour is detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        else:
            cv2.putText(frame, " ", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()