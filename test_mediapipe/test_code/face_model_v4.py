import cv2
import mediapipe as mp
import math

# 初始化 mediapipe pose 和 face mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh()

# 開啟攝像頭
cap = cv2.VideoCapture(0)

# 獲取攝像頭的幀寬和幀高w
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 設定視頻編碼格式和保存路徑


def calculate_a(landmark):
    ford_head=[landmark[54],landmark[284]]
    al,ar=ford_head[0],ford_head[1]
    fhx0,fhx1=ford_head[0].x,ford_head[1].x
    fhy0,fhy1=ford_head[0].y,ford_head[1].y
    fhz0,fhz1=ford_head[0].z+1,ford_head[1].z+1
    fh_wid=math.sqrt((fhx0 - fhx1) ** 2 + 
                    (fhy0 - fhy1) ** 2 +
                    (fhz0 - fhz1) ** 2 )
    return fh_wid,al,ar

def calculate_b(landmark):
    face_width=[landmark[234],landmark[454]]
    bl,br=face_width[0],face_width[1]
    fwx0,fwx1=face_width[0].x,face_width[1].x
    fwy0,fwy1=face_width[0].y,face_width[1].y
    fwz0,fwz1=face_width[0].z+1,face_width[1].z+1
    fw_wid=math.sqrt((fwx0 - fwx1) ** 2 + 
                    (fwy0 - fwy1) ** 2 +
                    (fwz0 - fwz1) ** 2 )
    return fw_wid,bl,br

def calculate_c(landmark):
    low_jaw=[landmark[172],landmark[397]]
    cl,cr=low_jaw[0],low_jaw[1]
    ljx0,ljx1=low_jaw[0].x,low_jaw[1].x
    ljy0,ljy1=low_jaw[0].y,low_jaw[1].y
    ljz0,ljz1=low_jaw[0].z+1,low_jaw[1].z+1
    lj_wid=math.sqrt((ljx0 - ljx1) ** 2 + 
                    (ljy0 - ljy1) ** 2 +
                    (ljz0 - ljz1) ** 2 )
    return lj_wid,cl,cr

def calculate_d(landmark):
    face_length=[landmark[175],landmark[10]]
    dl,dr=face_length[0],face_length[1]
    flx0,flx1=face_length[0].x,face_length[1].x
    fly0,fly1=face_length[0].y,face_length[1].y
    flz0,flz1=face_length[0].z+1,face_length[1].z+1
    fl_wid=math.sqrt((flx0 - flx1) ** 2 + 
                    (fly0 - fly1) ** 2 +
                    (flz0 - flz1) ** 2 )
    return fl_wid,dl,dr

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 將影像轉換為 RGB 格式
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 將影像輸入 mediapipe
    results_face = face_mesh.process(image)
    
    # 繪製姿態估計結果
    # if results_pose.pose_landmarks:
    #     mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # 計算並顯示眼睛距離和臉寬
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            dis_A,al,ar=calculate_a(face_landmarks.landmark)
            dis_B,bl,br=calculate_b(face_landmarks.landmark)
            dis_C,cl,cr=calculate_c(face_landmarks.landmark)
            dis_D,dl,dr=calculate_d(face_landmarks.landmark)
            #====================================================判斷臉型
            long_face_gate=1.65
            AB_rate=1.05
            BCrate=1.2
            if dis_A*AB_rate>dis_B and dis_B>dis_C and dis_D/dis_B<long_face_gate:face_type =0                          # 0 = heart
            elif round(dis_B*100)/100==round(dis_C*100)/100 and dis_D/dis_B<long_face_gate:face_type=1              # 1 = square
            elif dis_A*AB_rate<dis_B and dis_C*BCrate<dis_B and dis_D/dis_B<long_face_gate:face_type=2                     # 2 = round
            
            elif dis_A*AB_rate>dis_B and dis_B>dis_C and dis_D/dis_B>=long_face_gate:face_type =3                        # 3 = diamond
            elif round(dis_B*100)/100==round(dis_C*100)/100 and dis_D/dis_B>=long_face_gate:face_type=4              # 4 = retangel
            else :face_type=5                                                                                           # 5 = oval
            
            #====================================================
            
            draw_point=[al,ar,bl,br,cl,cr,dl,dr]
            # 顯示眼睛距離和臉寬
            cv2.putText(frame,f'Face Type : {face_type}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'A : {dis_A:.4f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'B : {dis_B:.4f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'C : {dis_C:.4f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'D : {dis_D:.4f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 畫綠點在眼睛和臉部寬度的點上
            for point in draw_point:
                x, y = int(point.x * frame_width), int(point.y * frame_height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    
    # 顯示結果影像
    cv2.imshow('Pose and Face Mesh Estimation', frame)
    
    # 等待按鍵輸入
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Detected 'q' key press. Exiting loop.")
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
