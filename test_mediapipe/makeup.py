import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe面部偵測器和面部標誌模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 定義化妝效果函數
def apply_lipstick(image, landmarks, color):
    lips_points = np.array([landmarks[i] for i in [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]])  # 嘴唇的點
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [lips_points], color)
    result = cv2.addWeighted(image, 1, mask, 0.4, 0)
    return result

# 讀取圖像
# 開啟攝像頭
cap = cv2.VideoCapture(0)

# 獲取攝像頭的幀寬和幀高
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 偵測面部
    results = face_mesh.process(frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 提取面部特徵點
            height, width, _ = frame.shape
            landmarks = [(int(point.x * width), int(point.y * height)) for point in face_landmarks.landmark]

            # 應用化妝效果（例如口紅）
            frame = apply_lipstick(frame, landmarks, (0, 0, 255))  # 紅色口紅

    # 顯示結果影像
    cv2.imshow('Pose Estimation', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Detected 'q' key press. Exiting loop.")
        break


cv2.destroyAllWindows()

# 釋放資源
face_mesh.close()
