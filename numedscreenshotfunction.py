import cv2 
import numpy as np
import dlib
from math import hypot
from time import sleep
from scipy.spatial import distance as dist  


#Call the webcam feed
cap = cv2.VideoCapture(0)
webcam = cv2.VideoCapture(0)
#Import chosen photo
nose_image = cv2.imread("test.png")
# call face detector and shapepredictor from dlib
detector = dlib.get_frontal_face_detector()
#File path of the shape prediction path
#Download link at: https://github.com/GuoQuanhao/68_points/blob/master/shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
_, frame = cap.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)

while True: 
    _, frame = cap.read()
    nose_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)

    for face in faces:
        #find the facial landmarks, set up as tuples
        landmarks = predictor(gray_frame, face) 
        #Get nose position
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        top_nose = (landmarks.part(27).x, landmarks.part(27).y)
        bottom_nose = (landmarks.part(33).x, landmarks.part(33 ).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        """
         #View the points adjust if needed
        cv2.circle(frame, top_nose, 2, (255,0,0), 1)
        cv2.circle(frame, left_nose, 2, (255,0,0), 1)
        cv2.circle(frame, right_nose, 2, (255,0,0), 1)
        cv2.circle(frame, bottom_nose, 2, (255,0,0), 1)
        """
          
        #Here using the distance formula to calculate the distance in euclidean space
        nose_width = int(hypot(left_nose[0]-right_nose[0], 
            left_nose[1]-right_nose[1])+ 30)
        nose_height = int(nose_width)
        #Translating cartesion to euclidean
        start_point = (int(center_nose[0] - nose_width/2), int(center_nose[1] - nose_width/2)) #int(landmarks.part(31).x, landmarks.parts(27).y)
        end_point = (int(center_nose[0] + nose_width / 2), int(center_nose[1] + nose_height / 2)) #int(landmarks.part(35).x, landmarks.part(33).y)

        #Resize to the width of the original nose
        resize_nose = cv2.resize(nose_image, (nose_width, nose_height))
        #remove background
        """
        nose_gray = cv2.cvtColor(resize_nose, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_gray, 25, 200, cv2.THRESH_BINARY_INV)
        nose_area = frame[start_point[1]: start_point[1] + nose_height,
                        start_point[0]: start_point[0] + nose_width] 
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(nose_area_no_nose, resize_nose)
        """
        nose_gray = cv2.cvtColor(resize_nose, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_gray, 25, 255, cv2.THRESH_BINARY_INV)
        nose_area = frame[start_point[1]: start_point[1] + nose_height,
                    start_point[0]: start_point[0] + nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)


        final_nose = cv2.add(nose_area_no_nose, resize_nose)
        frame[start_point[1]: start_point[1] + nose_height,
                    start_point[0]: start_point[0] + nose_width] = final_nose


        #cv2.imshow("new nose", nose_image)
        #cv2.imshow("nose_image", resize_nose)
        #cv2.imshow('nosegray', nose_gray)
        #cv2.imshow('nose', nose_mask)
    
    cv2.imshow("Frame", frame)
    



    #frame[start_point[1]: start_point[1] + nose_width,
    #   start_point[0]: start_point[0] + nose_width] = final_nose

    #Overlay image on the livefeed, using centernose as origin(Origo)
    #using https://theailearner.com/2019/03/18/add-image-to-a-live-camera-feed-using-opencv-python/#:~:text=%20Steps%3A%20%201%20Take%20an%20image%20which,%28%29%206%20Press%20%E2%80%98q%E2%80%99%20to%20break%20More%20
    #for reference
    
     

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('s'): 
        cv2.imwrite(filename='saved_img.jpg', img=frame)
        webcam.release()
        """
        print("Processing image...")
        img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
        print("Converting RGB image to grayscale...")
        gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        print("Converted RGB image to grayscale...")
        print("Resizing image to 28x28 scale...")
        img_ = cv2.resize(gray,(28,28))
        print("Resized...")
        img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
        """
        print("Image saved!")
        
        break
        
    elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
