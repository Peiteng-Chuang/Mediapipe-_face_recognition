import os
import csv
import cv2
import numpy as np
import mediapipe as mp

# 初始化MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# 特徵索引定義
LEFT_EYE_INDEX = [33, 133]
RIGHT_EYE_INDEX = [362, 263]
LEFT_EYE_INNER_INDEX = 133
RIGHT_EYE_INNER_INDEX = 362
LEFT_EYE_OUTER_INDEX = 33
RIGHT_EYE_OUTER_INDEX = 263
LEFT_BROW_INDEX = [70, 105]
RIGHT_BROW_INDEX = [336, 300]
NOSE_TIP_INDEX = 1
NOSE_BOTTOM_INDEX = 4
LEFT_NOSE_INDEX = 197
RIGHT_NOSE_INDEX = 429
UPPER_LIP_INDEX = 13
LOWER_LIP_INDEX = 14
FACE_WIDTH_INDEX = [234, 454]
FACE_HEIGHT_INDEX = [10, 152]

# 讀取圖片並擷取特徵值
image_folder = './test_mediapipe/preprocessed_images'
features = []

for image_name in os.listdir(image_folder):
    if image_name.endswith('.jpeg'):
        print(f'{image_name} has found.')
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            def get_distance(point1, point2):
                return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))
            
            left_eye_inner = landmarks[LEFT_EYE_INNER_INDEX]
            right_eye_inner = landmarks[RIGHT_EYE_INNER_INDEX]
            left_eye_outer = landmarks[LEFT_EYE_OUTER_INDEX]
            right_eye_outer = landmarks[RIGHT_EYE_OUTER_INDEX]
            left_brow_inner = landmarks[LEFT_BROW_INDEX[0]]
            left_brow_outer = landmarks[LEFT_BROW_INDEX[1]]
            right_brow_inner = landmarks[RIGHT_BROW_INDEX[0]]
            right_brow_outer = landmarks[RIGHT_BROW_INDEX[1]]
            nose_tip = landmarks[NOSE_TIP_INDEX]
            nose_bottom = landmarks[NOSE_BOTTOM_INDEX]
            left_nose = landmarks[LEFT_NOSE_INDEX]
            right_nose = landmarks[RIGHT_NOSE_INDEX]
            upper_lip = landmarks[UPPER_LIP_INDEX]
            lower_lip = landmarks[LOWER_LIP_INDEX]
            face_left = landmarks[FACE_WIDTH_INDEX[0]]
            face_right = landmarks[FACE_WIDTH_INDEX[1]]
            face_top = landmarks[FACE_HEIGHT_INDEX[0]]
            face_bottom = landmarks[FACE_HEIGHT_INDEX[1]]
            
            # 計算特徵值
            eye_distance = get_distance(left_eye_inner, right_eye_inner)
            eye_ball_distance = get_distance(left_eye_inner, right_eye_inner)
            eye_width = get_distance(left_eye_outer, left_eye_inner)
            face_width = get_distance(face_left, face_right)
            nose_length = get_distance(nose_tip, nose_bottom)
            nose_width = get_distance(left_nose, right_nose)
            lip_thickness = get_distance(upper_lip, lower_lip)
            left_brow_width = get_distance(left_brow_inner, left_brow_outer)
            right_brow_width = get_distance(right_brow_inner, right_brow_outer)
            brow_length = (left_brow_width + right_brow_width) / 2
            face_length = get_distance(face_top, face_bottom)
            
            features.append([
                image_name, 
                eye_ball_distance, 
                eye_distance, 
                eye_width, 
                face_width, 
                nose_length, 
                nose_width, 
                lip_thickness, 
                left_brow_width, 
                brow_length, 
                face_length
            ])
            print(f'{image_name} features has read.')

# 寫入CSV檔案
with open('face_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Eye Ball Distance', 'Eye Distance', 'Eye Width', 'Face Width', 'Nose Length', 'Nose Width', 'Lip Thickness', 'Left Brow Width', 'Brow Length', 'Face Length'])
    writer.writerows(features)

print("Features have been written to face_features.csv")
