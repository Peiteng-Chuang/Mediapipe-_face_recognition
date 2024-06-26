# import cv2
# import mediapipe as mp

# # 初始化MediaPipe Face Detection
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils

# # 開啟攝像頭
# cap = cv2.VideoCapture(0)

# # 獲取攝像頭的幀寬和幀高
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# # 初始化Face Detection
# with mp_face_detection.FaceDetection(
#     model_selection=0, min_detection_confidence=0.5) as face_detection:

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 將影像轉換為RGB格式
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # 偵測臉部
#         results = face_detection.process(image)

#         # 如果有偵測到臉部
#         if results.detections:
#             for detection in results.detections:
#                 # 繪製臉部偵測框
#                 bboxC = detection.location_data.relative_bounding_box
#                 ih, iw, _ = frame.shape
#                 x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # 顯示結果影像
#         cv2.imshow('Pose Estimation', frame)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             print("Detected 'q' key press. Exiting loop.")
#             break

# # 釋放資源
# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# 初始化MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 載入模型
model = tf.keras.models.load_model("C:/DL_model/face_classfier_V0_4.h5")

# 開啟攝像頭（如果有多個攝像頭，請嘗試更改索引號0, 1, 2...）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# 獲取攝像頭的幀寬和幀高
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 初始化Face Detection
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # 將影像轉換為RGB格式
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 偵測臉部
        results = face_detection.process(image)

        # 如果有偵測到臉部
        if results.detections:
            for detection in results.detections:
                # 繪製臉部偵測框
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 提取臉部區域並調整大小到256x256
                face = frame[y:y + h, x:x + w]
                
                face_gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (150, 150))
                
                face_resized = np.expand_dims(face_resized, axis=0)  # 增加批次維度
                face_resized = face_resized / 255.0  # 正規化

                # 使用模型進行預測
                prediction = model.predict(face_resized)
                predicted_class = np.argmax(prediction)

                # 在框外顯示預測結果
                cv2.putText(frame, f"type : {str(predicted_class)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 顯示結果影像
        cv2.imshow('Pose Estimation', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Detected 'q' key press. Exiting loop.")
            break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
