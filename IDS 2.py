import cv2 
import numpy as numpy
import dlib
from math import hypot
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\Dropbox\Skole\Datalogi\Interactive digital systems\shape_predictor_68_face_landmarks.dat")
#File path of the shape predeiction path
#C:\users\bau\OneDrive - IDA\Desktop\VS\openCV\shape_predictor_68_face_landmarks.dat
#Download link at: https://github.com/GuoQuanhao/68_points/blob/master/shape_predictor_68_face_landmarks.dat
while True: 
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        #print(face)
        landmarks = predictor(gray_frame, face)
        top_nose = (landmarks.part(27).x, landmarks.part(27).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        bottom_nose = (landmarks.part(33).x, landmarks.part(33 ).y)

        #nose_width = hypot(left_nose[0]- right_nose[0], left_nose[1] - right_nose[1])
        #print(nose_width)


        cv2.circle(frame, top_nose, 3, (255,0,0), 3)

    cv2.imshow("Frame", frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break