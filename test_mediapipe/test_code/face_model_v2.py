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

def calculate_a(landmarks):
    ford_head=landmarks[54,284]
    fhx0,fhx1=ford_head[0].x,ford_head[1].x
    fhy0,fhy1=ford_head[0].y,ford_head[1].y
    fhz0,fhz1=ford_head[0].z+1,ford_head[1].z+1
    fw_wid=math.sqrt((fhx0 - fhx1) ** 2 + 
                    (fhy0 - fhy1) ** 2 +
                    (fhz0 - fhz1))
    return fw_wid

def calculate_b(landmarks):
    face_width=landmarks[234,454]
    fwx0,fwx1=face_width[0].x,face_width[1].x
    fwy0,fwy1=face_width[0].y,face_width[1].y
    fwz0,fwz1=face_width[0].z+1,face_width[1].z+1
    fw_wid=math.sqrt((fwx0 - fwx1) ** 2 + 
                    (fwy0 - fwy1) ** 2 +
                    (fwz0 - fwz1))
    return fw_wid

def calculate_c(landmarks):
    low_jaw=landmarks[172,397]
    ljx0,ljx1=low_jaw[0].x,low_jaw[1].x
    ljy0,ljy1=low_jaw[0].y,low_jaw[1].y
    ljz0,ljz1=low_jaw[0].z+1,low_jaw[1].z+1
    lj_wid=math.sqrt((ljx0 - ljx1) ** 2 + 
                    (ljy0 - ljy1) ** 2 +
                    (ljz0 - ljz1))
    return lj_wid

def calculate_d(landmarks):
    face_length=landmarks[175,10]
    
    flx0,flx1=face_length[0].x,face_length[1].x
    fly0,fly1=face_length[0].y,face_length[1].y
    flz0,flz1=face_length[0].z+1,face_length[1].z+1
    fl_wid=math.sqrt((flx0 - flx1) ** 2 + 
                    (fly0 - fly1) ** 2 +
                    (flz0 - flz1))
    return fl_wid

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
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # 計算並顯示眼睛距離和臉寬
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            dis_A=calculate_a(face_landmarks.landmark)
            dis_B=calculate_b(face_landmarks.landmark)
            dis_C=calculate_c(face_landmarks.landmark)
            dis_D=calculate_d(face_landmarks.landmark)
            # 顯示眼睛距離和臉寬
            cv2.putText(frame,'face_model_v1.0',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Eye Distance: {eye_distance:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Face Width: {face_width:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 畫綠點在眼睛和臉部寬度的點上
            for point in [left_eye[0], left_eye[1], right_eye[0], right_eye[1], left_face, right_face]:
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
