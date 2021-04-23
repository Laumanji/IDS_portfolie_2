import cv2 
import numpy as np
import dlib
from math import hypot
from scipy.spatial import distance as dist  
from scipy.spatial import ConvexHull  


cap = cv2.VideoCapture(0) 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\Dropbox\Skole\Datalogi\Interactive digital systems\shape_predictor_68_face_landmarks.dat")
#File path of the shape predeiction path
#Download link at: https://github.com/GuoQuanhao/68_points/blob/master/shape_predictor_68_face_landmarks.dat

JAWLINE_POINTS = list(range(0, 17))  
RIGHT_EYEBROW_POINTS = list(range(17, 22))  
LEFT_EYEBROW_POINTS = list(range(22, 27))  
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_OUTLINE_POINTS = list(range(48, 61))  
MOUTH_INNER_POINTS = list(range(61, 68))  

nose_image = cv2.imread(r"D:\Downloads\nose-png.png", -1)

orig_mask = nose_image[:,:,3]

orig_mask_inv = cv2.bitwise_not(orig_mask)

nose_image = nose_image[:,:,0:3]

orig_nose_height, orig_nose_width = nose_image.shape[:2]

def nose_size(nose):  
   noseWidth = dist.euclidean(nose[0], nose[3])  
   hull = ConvexHull(nose)  
   noseCenter = np.mean(nose[hull.vertices, :], axis=0)  
   
   noseCenter = noseCenter.astype(int)  
   
   return int(noseWidth), noseCenter  
 

def place_nose(frame, noseCenter, noseSize):  
   noseSize = int(noseSize * 1.5)  
   
   x1 = int(noseCenter[0,0] - (noseSize/2))  
   x2 = int(noseCenter[0,0] + (noseSize/2))  
   y1 = int(noseCenter[0,1] - (noseSize/2))  
   y2 = int(noseCenter[0,1] + (noseSize/2))  
   
   h, w = frame.shape[:2]  
   
   # check for clipping  
   if x1 < 0:  
     x1 = 0  
   if y1 < 0:  
     y1 = 0  
   if x2 > w:  
     x2 = w  
   if y2 > h:  
     y2 = h  
   
   # re-calculate the size to avoid clipping  
   noseOverlayWidth = x2 - x1  
   noseOverlayHeight = y2 - y1  
   
   # calculate the masks for the overlay  
   noseOverlay = cv2.resize(nose_image, (noseOverlayWidth,noseOverlayHeight), interpolation = cv2.INTER_AREA)  
   mask = cv2.resize(orig_mask, (noseOverlayWidth,noseOverlayHeight), interpolation = cv2.INTER_AREA)  
   mask_inv = cv2.resize(orig_mask_inv, (noseOverlayWidth,noseOverlayHeight), interpolation = cv2.INTER_AREA)  
   
   # take ROI for the verlay from background, equal to size of the overlay image  
   roi = frame[y1:y2, x1:x2]  
   
   # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.  
   roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)  
   
   # roi_fg contains the image pixels of the overlay only where the overlay should be  
   roi_fg = cv2.bitwise_and(noseOverlay,noseOverlay,mask = mask)  
   
   # join the roi_bg and roi_fg  
   dst = cv2.add(roi_bg,roi_fg)  
   
   # place the joined image, saved to dst back over the original image  
   frame[y1:y2, x1:x2] = dst  


while True: 
    #vid feed
    ret, frame = cap.read()

    if ret:
        #greyscale img of vid feed
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray_frame, 0)
        
        for (face) in faces:
            x = face.left()
            y = face.top()
            x1 = face.right()
            y1 = face.bottom()

            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, face).parts()])

            nose = landmarks[NOSE_POINTS]

            noseSize, noseCenter = nose_size(nose) 

            place_nose(frame, noseCenter, noseSize)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Frame", frame) 
    #cv2.imshow("new nose", nose_image)
    #cv2.imshow("nose_image", notADick)


    key = cv2.waitKey(1)







    


"""
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

        cv2.circle(frame, top_nose, 2, (255,0,0), 1)
        cv2.circle(frame, left_nose, 2, (255,0,0), 1)
        cv2.circle(frame, right_nose, 2, (255,0,0), 1)
        cv2.circle(frame, bottom_nose, 2, (255,0,0), 1)
   """