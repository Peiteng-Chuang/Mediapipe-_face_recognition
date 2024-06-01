import cv2
import mediapipe as mp
import math

# 初始化 mediapipe pose 和 face mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh()

# 開啟攝像頭
cap = cv2.VideoCapture(0)

# 獲取攝像頭的幀寬和幀高w
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 設定視頻編碼格式和保存路徑
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

def calculate_eye_distance(landmarks):
    # 眼睛索引
    LEFT_EYE_INDEXES = [33, 133]
    RIGHT_EYE_INDEXES = [362, 263]
    
    left_eye = [landmarks[LEFT_EYE_INDEXES[0]], landmarks[LEFT_EYE_INDEXES[1]]]
    right_eye = [landmarks[RIGHT_EYE_INDEXES[0]], landmarks[RIGHT_EYE_INDEXES[1]]]
    
    # 計算兩眼之間的距離
    left_eye_center = ((left_eye[0].x + left_eye[1].x) / 2, (left_eye[0].y + left_eye[1].y) / 2)
    right_eye_center = ((right_eye[0].x + right_eye[1].x) / 2, (right_eye[0].y + right_eye[1].y) / 2)
    
    distance = math.sqrt((right_eye_center[0] - left_eye_center[0]) ** 2 + (right_eye_center[1] - left_eye_center[1]) ** 2)
    
    return distance, left_eye, right_eye

def calculate_face_width(landmarks):
    # 臉部寬度索引
    LEFT_FACE_INDEX = 234
    RIGHT_FACE_INDEX = 454
    
    left_face = landmarks[LEFT_FACE_INDEX]
    right_face = landmarks[RIGHT_FACE_INDEX]
    
    # 計算臉部寬度
    face_width = math.sqrt((right_face.x - left_face.x) ** 2 + (right_face.y - left_face.y) ** 2)
    
    return face_width, left_face, right_face

def calculate_lip_thick(landmarks):
    # 嘴唇索引test
    UPPER_LIP_INDEX = [0,13]
    LOWER_LIP_INDEX = [14,16]
    
    upper_lip = [landmarks[UPPER_LIP_INDEX[0]],landmarks[UPPER_LIP_INDEX[1]]]
    lower_lip = [landmarks[LOWER_LIP_INDEX[0]],landmarks[LOWER_LIP_INDEX[1]]]
    
    # 計算唇部厚度
    up_lip_thick = math.sqrt((upper_lip[0].x - upper_lip[1].x) ** 2 + (upper_lip[0].y - upper_lip[1].y) ** 2)
    lo_lip_thick = math.sqrt((lower_lip[0].x - lower_lip[1].x) ** 2 + (lower_lip[0].y - lower_lip[1].y) ** 2)
    lip_thick = (up_lip_thick + lo_lip_thick)/2
    return lip_thick,upper_lip, lower_lip


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 將影像轉換為 RGB 格式
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 將影像輸入 mediapipe
    results_pose = pose.process(image)
    results_face = face_mesh.process(image)
    
    # 繪製姿態估計結果
    # if results_pose.pose_landmarks:
    #     mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # 計算並顯示眼睛距離和臉寬
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            eye_distance, left_eye, right_eye = calculate_eye_distance(face_landmarks.landmark)
            face_width, left_face, right_face = calculate_face_width(face_landmarks.landmark)
            lip_thick,upper_lip, lower_lip = calculate_lip_thick(face_landmarks.landmark)
            # 顯示眼睛距離和臉寬
            cv2.putText(frame,'face_model_v1.0',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Eye Distance: {eye_distance:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Face Width: {face_width:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Lip Thick: {lip_thick:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 畫綠點在眼睛和臉部寬度的點上
            for point in [left_eye[0], left_eye[1], right_eye[0], right_eye[1], left_face, right_face,
                          upper_lip[0],upper_lip[1],lower_lip[0],lower_lip[1]]:
                x, y = int(point.x * frame_width), int(point.y * frame_height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    # 寫入視頻文件
    out.write(frame)
    
    # 顯示結果影像
    cv2.imshow('Pose and Face Mesh Estimation', frame)
    
    # 等待按鍵輸入
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Detected 'q' key press. Exiting loop.")
        break

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
