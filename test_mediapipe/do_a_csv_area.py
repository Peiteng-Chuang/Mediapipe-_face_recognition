import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import math

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

LEFT_BROW_INDEX = [70, 107]
RIGHT_BROW_INDEX = [336, 300]

NOSE_TIP_INDEX = 1
NOSE_TOP_INDEX = 168  
LEFT_NOSE_INDEX = 115
RIGHT_NOSE_INDEX = 344

UPPER_LIP_INDEX = [0, 13]
LOWER_LIP_INDEX = [14, 16]

FACE_WIDTH_INDEX = [234, 454]
FACE_HEIGHT_INDEX = [10, 152]

# 要抓取xyz軸的landmark
rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
rightEyeLower3 = [143, 111, 117, 118, 119, 120, 121, 128, 245]
leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]
leftEyeLower3 = [372, 340, 346, 347, 348, 349, 350, 357, 465]
rightEyebrowLower = [124, 46, 53, 52, 65, 193]
leftEyebrowLower = [265, 353, 276, 283, 282, 295, 285]
data = {
    "rightEyeUpper0": rightEyeUpper0,
    "rightEyeLower0": rightEyeLower0,
    "rightEyeLower3": rightEyeLower3,
    "leftEyeUpper0": leftEyeUpper0,
    "leftEyeLower0": leftEyeLower0,
    "leftEyeLower3": leftEyeLower3,
    "rightEyebrowLower": rightEyebrowLower,
    "leftEyebrowLower": leftEyebrowLower
}
result = ['Image', "Eye Ball Distance", "Eye Distance", "Eye Width", "Face Width",
    "Nose Length", "Nose Width", "Lip Thickness", "Brow Width",
    "Face Length", "Face area", "Area Rate"]

for name, values in data.items():
    for value in values:
        result.append(f"{name}_{value}_x")
        result.append(f"{name}_{value}_y")
        result.append(f"{name}_{value}_z")

# 輪廓的索引
silhouette = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

def get_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

def calculate_face(landmarks):
    left_face = landmarks[FACE_WIDTH_INDEX[0]]
    right_face = landmarks[FACE_WIDTH_INDEX[1]]
    up_face = landmarks[FACE_HEIGHT_INDEX[0]]
    lo_face = landmarks[FACE_HEIGHT_INDEX[1]]
    face_width = math.sqrt((right_face.x - left_face.x) ** 2 + (right_face.y - left_face.y) ** 2)
    face_length = math.sqrt((up_face.x - lo_face.x) ** 2 + (up_face.y - lo_face.y) ** 2)
    return face_width, face_length

def calculate_area_rate(face_landmarks, image_shape):
    face_width, face_length = calculate_face(face_landmarks)
    landmarks = np.zeros((len(silhouette), 2), dtype=np.float32)
    for i, idx in enumerate(silhouette):
        landmarks[i] = (face_landmarks[idx].x * image_shape[1], face_landmarks[idx].y * image_shape[0])
    area = cv2.contourArea(landmarks)
    area_rate = area / ((256 * face_width) * (256 * face_length)) * 25
    return area, face_width, face_length, area_rate

def Coordinate_normalization(value, standard_value):
    return standard_value * value

# 讀取圖片並擷取特徵值
image_folder = 'test_mediapipe/256img_lst'
features = []

for image_name in os.listdir(image_folder):
    if image_name.endswith('.jpg'):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye_inner = landmarks[LEFT_EYE_INNER_INDEX]
            right_eye_inner = landmarks[RIGHT_EYE_INNER_INDEX]
            left_eye_outer = landmarks[LEFT_EYE_OUTER_INDEX]
            right_eye_outer = landmarks[RIGHT_EYE_OUTER_INDEX]
            left_brow_inner = landmarks[LEFT_BROW_INDEX[0]]
            left_brow_outer = landmarks[LEFT_BROW_INDEX[1]]
            right_brow_inner = landmarks[RIGHT_BROW_INDEX[0]]
            right_brow_outer = landmarks[RIGHT_BROW_INDEX[1]]
            nose_tip = landmarks[NOSE_TIP_INDEX]
            nose_top = landmarks[NOSE_TOP_INDEX]
            left_nose = landmarks[LEFT_NOSE_INDEX]
            right_nose = landmarks[RIGHT_NOSE_INDEX]
            upper_lip_up = landmarks[UPPER_LIP_INDEX[0]]
            upper_lip_lo = landmarks[UPPER_LIP_INDEX[1]]
            lower_lip_up = landmarks[LOWER_LIP_INDEX[0]]
            lower_lip_lo = landmarks[LOWER_LIP_INDEX[1]]
            face_left = landmarks[FACE_WIDTH_INDEX[0]]
            face_right = landmarks[FACE_WIDTH_INDEX[1]]
            face_top = landmarks[FACE_HEIGHT_INDEX[0]]
            face_bottom = landmarks[FACE_HEIGHT_INDEX[1]]

            eye_distance = get_distance(left_eye_inner, right_eye_inner)
            eye_ball_distance = get_distance(left_eye_inner, right_eye_inner)
            eye_width = get_distance(left_eye_outer, left_eye_inner)
            face_width = get_distance(face_left, face_right)
            nose_length = get_distance(nose_tip, nose_top)
            nose_width = get_distance(left_nose, right_nose)
            lip_up = get_distance(upper_lip_up, upper_lip_lo)
            lip_lo = get_distance(lower_lip_up, lower_lip_lo)
            lip_thickness = lip_up + lip_lo
            left_brow_width = get_distance(left_brow_inner, left_brow_outer)
            right_brow_width = get_distance(right_brow_inner, right_brow_outer)
            brow_length = (left_brow_width + right_brow_width) / 2
            face_length = get_distance(face_top, face_bottom)
            brow_width = max(left_brow_width, right_brow_width)

            standard_value = 256 * 0.079
            eye_distance = Coordinate_normalization(eye_distance, standard_value)
            eye_ball_distance = Coordinate_normalization(eye_ball_distance, standard_value)
            eye_width = Coordinate_normalization(eye_width, standard_value)
            face_width = Coordinate_normalization(face_width, standard_value)
            nose_length = Coordinate_normalization(nose_length, standard_value)
            nose_width = Coordinate_normalization(nose_width, standard_value)
            lip_thickness = Coordinate_normalization(lip_thickness, standard_value)
            brow_width = Coordinate_normalization(brow_width, standard_value)
            face_length = Coordinate_normalization(face_length, standard_value)

            area, _, _, area_rate = calculate_area_rate(landmarks, image.shape)

            x = [
                image_name, 
                eye_ball_distance, 
                eye_distance, 
                eye_width, 
                face_width, 
                nose_length, 
                nose_width, 
                lip_thickness, 
                brow_width, 
                face_length,
                area, 
                area_rate
            ]
            n = []
            for name, values in data.items():
                for value in values:
                    n.append(landmarks[value].x)
                    n.append(landmarks[value].y)
                    n.append(landmarks[value].z)
            x.extend(n)
            features.append(x)
        print(f"{image_name} features append success.")

# 將特徵值寫入CSV文件
output_csv = 'test_mediapipe/csv_file/face_data_area.csv'
with open(output_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(result)
    for row in features:
        csvwriter.writerow(row)

