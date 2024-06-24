import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe面部偵測器和面部標誌模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

lip_u = [0, 37, 39, 185, 80, 81, 82, 13, 312, 311, 310, 409, 270, 269, 267]
lip_l = [14, 87, 178, 88, 95, 146, 91, 181, 84, 17, 314, 405, 321, 375, 324, 318, 402, 317]

# 定義化妝效果函數
def apply_lipstick(image, landmarks, color):
    lips_points_u = np.array([landmarks[i] for i in lip_u])  # 嘴唇的點
    lips_points_l = np.array([landmarks[i] for i in lip_l])  # 嘴唇的點

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [lips_points_u], color)
    cv2.fillPoly(mask, [lips_points_l], color)
    result = cv2.addWeighted(image, 1, mask, 0.4, 0)
    return result

# 讀取圖片
image_path = 'a.jpg'
image = cv2.imread(image_path)
if image is None:
    print("Failed to load image.")
    exit()

# 應用口紅效果的函數
def apply_lipstick_effect(image, color):
    # 偵測面部
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 提取面部特徵點
            height, width, _ = image.shape
            landmarks = [(int(point.x * width), int(point.y * height)) for point in face_landmarks.landmark]

            # 應用化妝效果（例如口紅）
            image = apply_lipstick(image, landmarks, color)  # 口紅顏色

    return image

# 设置口红颜色
bar_red = 100
bar_green = 58
bar_blue = 54

# 应用口红效果
processed_image = apply_lipstick_effect(image, (bar_blue, bar_green, bar_red))

# 保存结果图像
output_path = 'b.jpg'
cv2.imwrite(output_path, processed_image)

# 釋放資源
face_mesh.close()
print(f"Processed image saved as {output_path}")
