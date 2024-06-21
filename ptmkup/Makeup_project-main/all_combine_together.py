import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# 這一區塊是切出區域

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



# Replace with your image path

# Defien all original settings
# for i in os.listdir(image_path):

def eye_region_generator(image):
# def trimap_generator(image)
# def trimap_generator(image,image_path,i =''):
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
            # print(face_landmarks.landmark)
            # Print the landmark coordinates
            # for id, lm in enumerate(face_landmarks.landmark):
            #     x, y = int(lm.x * width*4), int(lm.y * height*4)
            #     if id in leftEyebrowLower+leftEyeLower3:
            #         cv2.putText(image,f'{id}',(x,y),cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            #         cv2.circle(image,(x,y), 2, color=(0, 0, 255))

            
            
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
            
            # Left Eye in region
            for i in leftEyeLower0+leftEyeUpper0:
                lm = face_landmarks.landmark[i]
                x, y = int(lm.x * width), int(lm.y * height)
                lefteyein_position.append([x,y])
            lefteyein_position = np.array(lefteyein_position, dtype=np.int32).reshape((-1, 1, 2))        
            cv2.fillPoly(image, [lefteyein_position],(0, 0, 0))       
            
            # cv2.fillPoly(image, [rightEyebrowLower_position],(0, 0, 0))
            # Create a mask of the same size as the image, filled with zeros (black)
            
            mask = np.zeros_like(image)
            

            # # Fill the right eyebrow lower position on the mask with white color
            cv2.fillPoly(mask, [righteyeout_position], (255, 255, 255))
            cv2.fillPoly(mask, [lefteyeout_position], (255, 255, 255))
            # # Apply the inverted mask to the image
            image = cv2.bitwise_and(image, mask)        
                
            # Create mask2 to cover the image on color not black or white
            mask2 = np.zeros_like(image)
            # Create condition for non-black and non-white pixels
            condition = (image != [0, 0, 0])
            condition2 = (image == [255, 255, 255])
            condition = condition.all(axis=-1)
            condition2 = condition2.all(axis=-1)
            mask2[condition] = [255,255,255]
            mask2[condition2] = [255,255,255]
    face_mesh.close()
    cv2.destroyAllWindows()
    return mask2




# image_path = 'C:/Users/User/Makeup_project/256img_lst' 
# index = '001.jpg'
# output_image_path = os.path.join('C:/Users/User/Makeup_project/', 'test.jpg')


image_path = 'C:/Users/User/Desktop/Tony' 
index = 'processed_photo1.jpg'
output_image_path = os.path.join('C:/Users/User/Desktop/Tony/256img_lst', 'test.jpg')

image = cv2.imread(image_path+'/'+ index)
processed_image_cv = eye_region_generator(image)

# 將 OpenCV 圖像轉換為 PIL 圖像
processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image_cv, cv2.COLOR_BGR2RGB))


# 這一區塊是使用模型預測
img  = processed_image_pil.convert('L')  # 確保圖像是灰度格式
# folder = 'C:/Users/User/Desktop/Tony/256img_lst'
# filename = 'test.jpg'
# img_path = os.path.join(folder, filename)
#img = Image.open(img_path)
#img = img.convert('L')  # Ensure the image is in gray format
img = img.resize((128, 128))  # Resize image to a fixed size
img_array = np.array(img)
x_images = img_array.astype('float32') / 255.0
x_images = x_images.reshape((1, 128, 128))


new_model = keras.models.load_model('saved_model')
encoded_imgs = new_model.encoder(x_images).numpy()
result = new_model.decoder(encoded_imgs).numpy()
result = (result * 255).astype(np.uint8)

# Remove the batch dimension and reshape if necessary
result = result.reshape((128, 128))

# Convert to an image and save
output_image = Image.fromarray(result)
output_image.save("outcome.jpg")

# 這一區塊是上色

# Initialize MediaPipe Face Mesh and drawing utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Define the makeup effect function
def apply_eyeshade(image, mask, color,region_rate ,const = 0.9 ):
    
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask1 = np.all(mask >255-region_rate*255/100, axis=-1)

    mask2 = image.copy()
    mask2[mask1] = gray_mask[mask1][:, np.newaxis] *color/255

# Parameters for the GuidedFilter
    radius = 10         # Radius of the guided filter
    eps = 0.1         # Regularization parameter (epsilon)

    guided_filter = cv2.ximgproc.createGuidedFilter(guide=gray_mask, radius=radius, eps=eps)
    filtered_mask = guided_filter.filter(mask2)
    result = cv2.addWeighted(image, const, filtered_mask, 1-const, 0)

    return result

# Create trackbars to adjust lip color
def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.createTrackbar('R', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('G', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('B', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('level', 'Trackbars', 0, 100, nothing)
cv2.createTrackbar('region_rate', 'Trackbars', 0, 100, nothing)

# Path to the image
# image_path = './256img_lst/006.jpg'
# rwmask_path ='./pymatting_outcome_rw/006.jpg'

image_path = './processed_photo1.jpg'
rwmask_path ='./outcome.jpg'

# Read the image
image = cv2.imread(image_path)
image = cv2.resize(image, (128, 128))
rwmask = cv2.imread(rwmask_path)
rwmask = cv2.resize(rwmask, (128, 128))

if image is None:
    print(f"Failed to load image: {image_path}")
    exit()

if rwmask is None:
    print(f"Failed to load image: {rwmask_path}")
    exit()

# Detect face in the image
results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        height, width, _ = image.shape
        landmarks = [(int(point.x * width), int(point.y * height)) for point in face_landmarks.landmark]

while True:
    # Get the current positions of the trackbars
    r = cv2.getTrackbarPos('R', 'Trackbars')
    g = cv2.getTrackbarPos('G', 'Trackbars')
    b = cv2.getTrackbarPos('B', 'Trackbars')
    const = cv2.getTrackbarPos('level', 'Trackbars')
    region_rate = cv2.getTrackbarPos('region_rate', 'Trackbars')


    const = -(float(const)-50)/500+0.9
    img = np.zeros((100, 100, 3), np.uint8) 
    img[:] = [b,g,r]
    # Apply the makeup effect with the current color
    if results.multi_face_landmarks:
        output_image = apply_eyeshade(image.copy(),rwmask.copy() , (b, g, r),region_rate,const)
    else:
        output_image = image.copy()
    output_image = cv2.resize(output_image, (512, 512))
    # Display the result
    cv2.imshow('Lipstick Application', output_image)
    cv2.imshow('color plate',img)

    # Exit on 'esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
face_mesh.close()