import cv2
import mediapipe as mp

# 初始化 mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 開啟攝像頭
cap = cv2.VideoCapture(0)

# 獲取攝像頭的幀寬和幀高
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 設定視頻編碼格式和保存路徑
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

# 定義要連接的關鍵點對（按火柴人形式連接）
POSE_CONNECTIONS = [
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER),
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE_INNER),
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.MOUTH_LEFT),
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.MOUTH_RIGHT),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 將影像轉換為 RGB 格式
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 將影像輸入 mediapipe
    results = pose.process(image)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for connection in POSE_CONNECTIONS:
            start_idx = connection[0].value
            end_idx = connection[1].value
            start_landmark = landmarks[start_idx]
            end_landmark = landmarks[end_idx]
            if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
                start_point = (int(start_landmark.x * frame_width), int(start_landmark.y * frame_height))
                end_point = (int(end_landmark.x * frame_width), int(end_landmark.y * frame_height))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    
    # 寫入視頻文件
    out.write(frame)
    
    # 顯示結果影像
    cv2.imshow('Pose Estimation', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("Detected 'q' key press. Exiting loop.")
        break

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
