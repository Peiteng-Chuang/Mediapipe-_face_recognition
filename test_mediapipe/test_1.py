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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 將影像轉換為 RGB 格式
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 將影像輸入 mediapipe
    results = pose.process(image)
    
    # 繪製姿態估計結果
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
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
