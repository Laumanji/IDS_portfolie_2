import cv2 
import dlib
import keyboard
from math import hypot


#Made with help from https://pysource.com/2019/03/25/pigs-nose-instagram-face-filter-opencv-with-python/

#Call the webcam feed
cap = cv2.VideoCapture(0)

#Import chosen photo for masking
nose_image = cv2.imread("test.png")
mask_image = cv2.imread("Mask.png")

# call face detector and shapepredictor from dlib
#Download link at: https://github.com/GuoQuanhao/68_points/blob/master/shape_predictor_68_face_landmarks.dat
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True: 
    parse, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame)

    for face in faces:
        def noseMask():
            """Creates a mask above the nose"""
            #find the facial landmarks, set up as tuples
            landmarks = predictor(gray_frame, face) 
            #Get nose position
            left_nose = (landmarks.part(31).x, landmarks.part(31).y)
            right_nose = (landmarks.part(35).x, landmarks.part(35).y)
            top_nose = (landmarks.part(27).x, landmarks.part(27).y)
            bottom_nose = (landmarks.part(33).x, landmarks.part(33 ).y)
            center_nose = (landmarks.part(30).x, landmarks.part(30).y)

            #Here using the distance formula to calculate the distance in euclidean space
            nose_width = int(hypot(left_nose[0]-right_nose[0], 
                                    left_nose[1]-right_nose[1])+ 30)
            nose_height = int(nose_width)

            #Translating cartesion to euclidean
            start_point = (int(center_nose[0] - nose_width/2), int(center_nose[1] - nose_width/2)) #int(landmarks.part(31).x, landmarks.parts(27).y)
            end_point = (int(center_nose[0] + nose_width / 2), int(center_nose[1] + nose_height / 2)) #int(landmarks.part(35).x, landmarks.part(33).y)
            #Resize to the width of the original nose
            resize_nose = cv2.resize(nose_image, (nose_width, nose_height))
            #Overlay image on the livefeed, using centernose as origin(Origo)
            #using https://theailearner.com/2019/03/18/add-image-to-a-live-camera-feed-using-opencv-python/#:~:text=%20Steps%3A%20%201%20Take%20an%20image%20which,%28%29%206%20Press%20%E2%80%98q%E2%80%99%20to%20break%20More%20
        
            #remove background
            nose_gray = cv2.cvtColor(resize_nose, cv2.COLOR_BGR2GRAY)
            parse, nose_mask = cv2.threshold(nose_gray, 25, 255, cv2.THRESH_BINARY_INV)
            nose_area = frame[start_point[1]: start_point[1] + nose_height,
                            start_point[0]: start_point[0] + nose_width]
            nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(nose_area_no_nose, resize_nose)
            frame[start_point[1]: start_point[1] + nose_height,
                        start_point[0]: start_point[0] + nose_width] = final_nose

        def eyeMask():
            """Create a mask over the eyes"""
            #find the facial landmarks, set up as tuples
            landmarks = predictor(gray_frame, face)
            #Landmarks for the facemask
            left_face = (landmarks.part(17).x, landmarks.part(17).y)
            right_face = (landmarks.part(26).x, landmarks.part(26).y)
            center_face = (landmarks.part(27).x, landmarks.part(27).y)

            #Calculating the face width
            face_width = int(hypot(left_face[0] - right_face[0], left_face[1] - right_face[1]) * 1.2)
            #face height
            face_height = int (face_width * 0.31)

            #Cartesion to euclidean conversion for face landmarks
            top_left = (int(center_face[0] - face_width / 2),
                        int(center_face[1] - face_height / 2))
            bottom_right = (int(center_face[0] + face_width / 2),
                        int(center_face[1] + face_height / 2))
            #Resize facemask image
            eyes_censor = cv2.resize(mask_image, (face_width, face_height))

            #Remove background for the face
            eyes_censor_gray = cv2.cvtColor(eyes_censor, cv2.COLOR_BGR2GRAY)
            parse, eyes_mask = cv2.threshold(eyes_censor_gray, 255, 25, cv2.THRESH_BINARY)
            face_area = frame[top_left[1]: top_left[1] + face_height,
                        top_left[0]: top_left[0] + face_width]
            mask_area_no_mask = cv2.bitwise_and(face_area, face_area, mask=eyes_mask)
            final_eye_mask = cv2.add(mask_area_no_mask, eyes_censor)
            frame[top_left[1]: top_left[1] + face_height, top_left[0]: top_left[0] + face_width] = final_eye_mask
    
    
    key = cv2.waitKey(1)
    if keyboard.is_pressed('w'): 
        noseMask()
    if keyboard.is_pressed('e'):
        eyeMask()
    
    cv2.imshow("Capturing", frame)
    #key = cv2.waitKey(1)
    if keyboard.is_pressed('s'): 
        cv2.imwrite(filename='saved_img.jpg', img=frame)
        cap.release()
        print("Image saved!")   
        break  
    elif cv2.waitKey(1) == ord('q'):
        break
