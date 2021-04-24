import cv2
import numpy as numpy
import dlib
from math import hypot
cap = cv2.VideoCapture(0)

##Made with help from https://pysource.com/2019/03/25/pigs-nose-instagram-face-filter-opencv-with-python/

#image
#load image with alpha channel.  use IMREAD_UNCHANGED to ensure loading of alpha channel
nose_image = cv2.imread("test.png")
mask_image = cv2.imread("Mask.png")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True: 
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        bottom_nose = (landmarks.part(33).x, landmarks.part(33 ).y)

        left_face = (landmarks.part(0).x, landmarks.part(0).y)
        right_face = (landmarks.part(16).x, landmarks.part(16).y)
        center_face = (landmarks.part(27).x, landmarks.part(27).y)

        #calculates the distance between right and left nose to scale the filter image

        nose_width = int(hypot(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]) * 1.7)

        face_width = int(hypot(left_face[0] - right_face[0], left_face[1] - right_face[1]))

        #finding the height, we take the proportion of the image size and multiply it with the width
        # 660/900px                  
        nose_height = int(nose_width * 0.73)

        face_height = int (face_width * 0.10+40)

        #debugging for nose image width
        #print(nose_width)
        #print(nose_height)

        #shows facemarkers facemarkers
        #cv2.circle(frame, top_nose, 3, (255,0,0), 3)

        #draw retancle from center point to insert image
        
        #top_left = (int(center_nose[0] - nose_width / 2), int(center_nose[1] - nose_height / 2))
        #bottom_right = (int(center_nose[0] + nose_width / 2), int(center_nose[1] + nose_height / 2))

        top_left = (int(center_face[0] - face_width / 2),
                              int(center_face[1] - face_height / 2))
        bottom_right = (int(center_face[0] + face_width / 2),
                       int(center_face[1] + face_height / 2))
        

        #rezise image
        nose_pig = cv2.resize(nose_image, (nose_width, nose_height))

        #mask_height = int(face_width* 0.20)
        face_guy = cv2.resize(mask_image, (face_width, face_height))
        
        
        

        #remove background from image
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(nose_area_no_nose, nose_pig)


        face_guy_gray = cv2.cvtColor(face_guy, cv2.COLOR_BGR2GRAY)
        _, guy_mask = cv2.threshold(face_guy_gray, 25, 255, cv2.THRESH_BINARY_INV)
        face_area = frame[top_left[1]: top_left[1] + face_height,
                    top_left[0]: top_left[0] + face_width]
        face_area_no_guy = cv2.bitwise_and(face_area, face_area, mask=guy_mask)
        final_guy = cv2.add(face_area_no_guy, face_guy)



        
        #frame[top_left[1]: top_left[1] + nose_height, top_left[0]: top_left[0] + nose_width] = final_nose

        frame[top_left[1]: top_left[1] + face_height, top_left[0]: top_left[0] + face_width] = final_guy

        cv2.imshow("face mask", face_guy)

        cv2.imshow("face mask gray", face_guy_gray)
        cv2.imshow("no mask", guy_mask)
        

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break