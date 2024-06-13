# import cv2
# import mediapipe as mp
# import numpy as np
# import math

# # 初始化MediaPipe的臉部關鍵點檢測模組
# mp_face_mesh = mp.solutions.face_mesh

# # 初始化MediaPipe
# drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# # 輪廓的索引
# silhouette = [
#     10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
#     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
#     172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
# ]
# def calculate_face(landmarks):
#     # 臉部寬度索引
#     LEFT_FACE_INDEX = 234
#     RIGHT_FACE_INDEX = 454
#     UPPER_FACE_INDEX = 10
#     LOWER_FACE_INDEX = 175
    
#     left_face = landmarks[LEFT_FACE_INDEX]
#     right_face = landmarks[RIGHT_FACE_INDEX]
#     up_face = landmarks[UPPER_FACE_INDEX]
#     lo_face = landmarks[LOWER_FACE_INDEX]
    
#     # 計算臉部寬度2D
#     face_width = math.sqrt((right_face.x - left_face.x) ** 2 + (right_face.y - left_face.y) ** 2)
#     face_length =math.sqrt((up_face.x - lo_face.x) ** 2 + (up_face.y - lo_face.y) ** 2)
    
#     # 計算臉部寬度3D版
#     # face_width = math.sqrt(math.sqrt((right_face.x - left_face.x) ** 2 + (right_face.y - left_face.y) ** 2)**2+(right_face.z-left_face.z)**2)
#     # face_length =math.sqrt(math.sqrt((up_face.x - lo_face.x) ** 2 + (up_face.y - lo_face.y) ** 2)**2+(up_face.z-lo_face.z)**2)
#     return face_width, face_length
# # 初始化攝像頭
# cap = cv2.VideoCapture(0)

# # 獲取攝像頭的幀寬和幀高
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# # 當攝像頭打開時
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # 將影像轉換為 RGB 格式
#     image = cv2.flip(frame,1)
    
#     # 執行面部關鍵點檢測
#     with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
#         results = face_mesh.process(image)
        
#         # 檢查是否檢測到了面部關鍵點
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # 將面部關鍵點轉換為NumPy數組
#                 face_width, face_length = calculate_face(face_landmarks.landmark)
#                 landmarks = np.zeros((len(silhouette), 2), dtype=np.float32)
#                 for i, idx in enumerate(silhouette):
#                     landmarks[i] = (face_landmarks.landmark[idx].x * image.shape[1],
#                                     face_landmarks.landmark[idx].y * image.shape[0])
#                 # 使用OpenCV計算面部區域的面積
#                 area = cv2.contourArea(landmarks)
#                 area_rate=area/((256*face_width)*(256*face_length))*25
#                 # 在圖像上繪製面部輪廓
#                 for landmark in landmarks:
#                     cv2.circle(image, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1)
#                 # 在圖像上顯示面部區域的面積
#                 cv2.putText(image, f'Face Area: {area:.0f}', (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
#                 cv2.putText(image, f'Face width: {face_width:.3f}', (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
#                 cv2.putText(image, f'Face length: {face_length:.3f}', (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
#                 cv2.putText(image, f'Area_rate: {area_rate:.2f}', (10, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                
    
#     # 顯示結果影像
#     cv2.imshow('Pose Estimation', image)
    
#     if cv2.waitKey(10) & 0xFF == 27:
#         print("Detected 'esc' key press. Exiting loop.")
#         break

# # 釋放資源
# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import math

# 初始化MediaPipe的臉部關鍵點檢測模組
mp_face_mesh = mp.solutions.face_mesh

# 初始化MediaPipe
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
#=====================================================
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

UPPER_LIP_INDEX = 13
LOWER_LIP_INDEX = 14

UPPER_LIP_INDEX = [0,13]
LOWER_LIP_INDEX = [14,16]

# FACE_WIDTH_INDEX = [234, 454]
# FACE_HEIGHT_INDEX = [10, 152]

LEFT_FACE_INDEX = 234       # 臉部長寬度索引
RIGHT_FACE_INDEX = 454
UPPER_FACE_INDEX = 10
LOWER_FACE_INDEX = 175

#=====================================================
# 輪廓的索引
silhouette = [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
]
#======================================================
def get_distance(point1, point2):
    return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))
def calculate_face(landmarks):
    # 計算臉部長寬度
    
    left_face = landmarks[LEFT_FACE_INDEX]
    right_face = landmarks[RIGHT_FACE_INDEX]
    up_face = landmarks[UPPER_FACE_INDEX]
    lo_face = landmarks[LOWER_FACE_INDEX]
    
    # 計算臉部長寬2D
    face_width = math.sqrt((right_face.x - left_face.x) ** 2 + (right_face.y - left_face.y) ** 2)
    face_length = math.sqrt((up_face.x - lo_face.x) ** 2 + (up_face.y - lo_face.y) ** 2)
    
    # 計算臉部長寬3D版
    # face_width = math.sqrt(math.sqrt((right_face.x - left_face.x) ** 2 + (right_face.y - left_face.y) ** 2)**2+(right_face.z-left_face.z)**2)
    # face_length =math.sqrt(math.sqrt((up_face.x - lo_face.x) ** 2 + (up_face.y - lo_face.y) ** 2)**2+(up_face.z-lo_face.z)**2)
    return face_width, face_length

def calculate_area_rate(face_landmarks, image_shape):
    face_width, face_length = calculate_face(face_landmarks)
    landmarks = np.zeros((len(silhouette), 2), dtype=np.float32)
    for i, idx in enumerate(silhouette):
        landmarks[i] = (face_landmarks[idx].x * image_shape[1], face_landmarks[idx].y * image_shape[0])
    
    # 使用OpenCV計算面部區域的面積
    area = cv2.contourArea(landmarks)
    area_rate = area / ((256 * face_width) * (256 * face_length)) * 25
    
    return area, face_width, face_length, area_rate

#======================================================
# 初始化攝像頭
cap = cv2.VideoCapture(0)

# 獲取攝像頭的幀寬和幀高
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 當攝像頭打開時
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 將影像轉換為 RGB 格式
    image = cv2.flip(frame, 1)
    
    # 執行面部關鍵點檢測
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image)
        
        # 檢查是否檢測到了面部關鍵點
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                area, face_width, face_length, area_rate = calculate_area_rate(face_landmarks.landmark, image.shape)
                width_rate=area/face_length**2/1024
                length_rate=area/face_width**2/2048
                # 在圖像上繪製面部輪廓
                for idx in silhouette:
                    x = int(face_landmarks.landmark[idx].x * image.shape[1])
                    y = int(face_landmarks.landmark[idx].y * image.shape[0])
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                
                # 在圖像上顯示面部區域的面積
                #==================================================未處理數值(綠)
                cv2.putText(image, f'Face Area: {area:.0f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(image, f'Face width: {face_width:.3f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(image, f'Face length: {face_length:.3f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                #==================================================標準化數值(紅)
                cv2.putText(image, f'Area_rate: {area_rate:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(image, f'Width_rate: {width_rate:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(image, f'Length_rate: {length_rate:.2f}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                
                cv2.putText(image, f'eyes: {0}', (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    # 顯示結果影像
    cv2.imshow('Pose Estimation', image)
    
    if cv2.waitKey(10) & 0xFF == 27:
        print("Detected 'esc' key press. Exiting loop.")
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()


