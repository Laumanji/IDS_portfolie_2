import cv2 
import numpy as numpy
import dlib
from math import hypot
cap = cv2.VideoCapture(0)
nose_image = cv2.imread("dick_placeholder.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
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
         
        
        nose_width = int(hypot(left_nose[0]-right_nose[0],
                            left_nose[1]-right_nose[1]))
        #print(nose_width)

        notADick =cv2.resize(nose_image, (nose_width + 50, nose_width + 50))
    

        """
        cv2.circle(frame, top_nose, 2, (255,0,0), 1)
        cv2.circle(frame, left_nose, 2, (255,0,0), 1)
        cv2.circle(frame, right_nose, 2, (255,0,0), 1)
        cv2.circle(frame, bottom_nose, 2, (255,0,0), 1)
        """

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Frame", frame) 
    #cv2.imshow("new nose", nose_image)
    cv2.imshow("nose_image", notADick)


    key = cv2.waitKey(1)

   
