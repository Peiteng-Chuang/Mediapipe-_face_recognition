import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe面部偵測器和面部標誌模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

lip_u=[0,37,39,185,80,81,82,13,312,311,310,409,270,269,267]

lip_l=[14,87,178,88,95,146,91,181,84,17,314,405,321,375,324,318,402,317]

# 定義化妝效果函數
def apply_lipstick(image, landmarks, color):
    
    lips_points_u = np.array([landmarks[i] for i in lip_u])  # 嘴唇的點
    lips_points_l = np.array([landmarks[i] for i in lip_l])  # 嘴唇的點
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [lips_points_u], color)
    cv2.fillPoly(mask, [lips_points_l], color)
    result = cv2.addWeighted(image, 1, mask, 0.4, 0)
    return result

# 讀取圖像
# 開啟攝像頭
cap = cv2.VideoCapture(0)

# 獲取攝像頭的幀寬和幀高
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# creat track bars
def nothing(x):
    pass
cv2.namedWindow('Bar')
cv2.createTrackbar('bar_R', 'Bar', 0, 255, nothing)   # in 'Track Bar' windows
cv2.createTrackbar('bar_G', 'Bar', 0, 255, nothing)   # in 'Track Bar' windows
cv2.createTrackbar('bar_B', 'Bar', 0, 255, nothing)   # in 'Track Bar' windows

while cap.isOpened():
    bar_red = cv2.getTrackbarPos('bar_R', 'Bar')
    bar_green = cv2.getTrackbarPos('bar_G', 'Bar')
    bar_blue = cv2.getTrackbarPos('bar_B', 'Bar')
    
    ret, frame = cap.read()
    if not ret:
        break
    frame=cv2.flip(frame, 1)
    # 偵測面部
    results = face_mesh.process(frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 提取面部特徵點
            height, width, _ = frame.shape
            landmarks = [(int(point.x * width), int(point.y * height)) for point in face_landmarks.landmark]

            # 應用化妝效果（例如口紅）
            frame = apply_lipstick(frame, landmarks, (bar_blue, bar_green, bar_red))  # 紅色口紅

    # 顯示結果影像
    cv2.imshow('Bar', frame)
    
    if cv2.waitKey(10) & 0xFF == 27:
        print("Detected 'esc' key press. Exiting loop.")
        break


cv2.destroyAllWindows()

# 釋放資源
face_mesh.close()