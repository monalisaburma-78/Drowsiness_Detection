import cv2
import os
import numpy as np
import dlib
from scipy.spatial import distance
from keras.models import load_model
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

lbl = ['Close', 'Open']

model = load_model('models.h5')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

drowsy_counter = 0  
drowsy_threshold = 10  

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = []
        right_eye = []

        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))

        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if ear < 0.25:  
            drowsy_counter += 1
            if drowsy_counter >= drowsy_threshold:
                cv2.putText(frame, "Drowsiness Detected", (10, height-20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

                if not mixer.music.get_busy():
                    try:
                        sound.play()
                    except Exception as e:
                        print(f"Error: {e}")

        
                cv2.rectangle(frame, (left_eye[0][0] - 5, left_eye[1][1] - 5), (right_eye[3][0] + 5, right_eye[4][1] + 5), (0, 0, 255), 2)

        else:
            drowsy_counter = 0  
            cv2.putText(frame, "Open", (10, height-20), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.rectangle(frame, (left_eye[0][0] - 5, left_eye[1][1] - 5), (right_eye[3][0] + 5, right_eye[4][1] + 5), (0, 255, 0), 2)

            if mixer.music.get_busy():
                sound.stop()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
