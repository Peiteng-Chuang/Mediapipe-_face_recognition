#跑得動版
import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load image using OpenCV
silhouette =  [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
  ]


rightEyeUpper0 =  [246, 161, 160, 159, 158, 157, 173]
rightEyeUpper1 = [247, 30, 29, 27, 28, 56, 190]
rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
rightEyeLower1 = [130, 25, 110, 24, 23, 22, 26, 112, 243]
rightEyeLower3 = [143, 111, 117, 118, 119, 120, 121, 128, 245]
rightEyebrowLower = [ 124, 46, 53, 52, 65, 193]

leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
leftEyeUpper1 = [467, 260, 259, 257, 258, 286, 414]
leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
leftEyeLower1 = [359, 255, 339, 254, 253, 252, 256, 341, 463]
leftEyeLower3 = [372, 340, 346, 347, 348, 349, 350, 357, 465]

rightEyeLower0.reverse()
rightEyeLower1.reverse()
rightEyeLower3.reverse()
leftEyeLower0.reverse()
leftEyeLower1.reverse()
leftEyeLower3.reverse()

leftEyebrowLower = [276, 283, 282, 295, 285]

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image=frame
    
    righteyeout_position = []
    righteyein_position = []
    lefteyeout_position = []
    lefteyein_position = []
    righteyemargin_position = []
    lefteyemargin_position = []
# Convert the BGR image to RGB
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image to find facial landmarks
    results = face_mesh.process(rgb_image)
    # Check if landmarks were detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw facial landmarks on the image
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Right Eye out region
            for i in rightEyebrowLower+rightEyeLower3:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                righteyeout_position.append([x,y])
            righteyeout_position = np.array(righteyeout_position, dtype=np.int32).reshape((-1, 1, 2))
            
            # Right Eye margin region        
            for i in rightEyeUpper1+rightEyeLower1:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                righteyemargin_position.append([x,y])
            righteyemargin_position = np.array(righteyemargin_position, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [righteyemargin_position],(255,255,255))  
            
            # Right Eye in region
            for i in rightEyeLower0+rightEyeUpper0:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                righteyein_position.append([x,y])
            righteyein_position = np.array(righteyein_position, dtype=np.int32).reshape((-1, 1, 2))        
            cv2.fillPoly(image, [righteyein_position],(0, 0, 0))      
                    
            # Left Eye out region
            for i in leftEyebrowLower+leftEyeLower3:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                lefteyeout_position.append([x,y])
            lefteyeout_position = np.array(lefteyeout_position, dtype=np.int32).reshape((-1, 1, 2))
            
            # Left Eye margin region        
            for i in leftEyeUpper1+leftEyeLower1:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                lefteyemargin_position.append([x,y])
            lefteyemargin_position = np.array(lefteyemargin_position, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [lefteyemargin_position],(255,255,255))        
            
            for i in leftEyeLower0+leftEyeUpper0:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                lefteyein_position.append([x,y])
            lefteyein_position = np.array(lefteyein_position, dtype=np.int32).reshape((-1, 1, 2))        
            cv2.fillPoly(image, [lefteyein_position],(0, 0, 0))       
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, [righteyeout_position], (255, 255, 255))
            cv2.fillPoly(mask, [lefteyeout_position], (255, 255, 255))
            image = cv2.bitwise_and(image, mask)        
            mask2 = np.zeros_like(image)
            condition = (image != [0, 0, 0])
            condition2 = (image == [255, 255, 255])
            condition = condition.all(axis=-1)
            condition2 = condition2.all(axis=-1)
            mask2[condition] = [128,128,128]
            mask2[condition2] = [255,255,255]
            
            inverted_mask2 = cv2.bitwise_not(mask2)      
            image = cv2.bitwise_and(image, inverted_mask2)        
    
    cv2.imshow('Pose Estimation', mask2)
    
    if cv2.waitKey(10) & 0xFF == 27:
        print("Detected 'esc' key press. Exiting loop.")
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()