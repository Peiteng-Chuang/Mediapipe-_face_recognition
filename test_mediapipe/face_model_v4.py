import numpy as np
import mediapipe as mp
import math,time,threading,cv2

# 初始化 mediapipe face mesh

mp_face_mesh = mp.solutions.face_mesh
face_mesh1 = mp_face_mesh.FaceMesh()
face_mesh2 = mp_face_mesh.FaceMesh()

# 開啟攝像頭
cap = cv2.VideoCapture(0)

# 獲取攝像頭的幀寬和幀高w
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#製作功能
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
    fwz0,fwz1=(face_width[0].z+1)/2,(face_width[1].z+1)/2
    fw_wid=math.sqrt((fwx0 - fwx1) ** 2 + 
                    (fwy0 - fwy1) ** 2 +
                    (fwz0 - fwz1) ** 2 )
    return fw_wid,bl,br

def calculate_c(landmark):
    low_jaw=[landmark[172],landmark[397]]
    cl,cr=low_jaw[0],low_jaw[1]
    ljx0,ljx1=low_jaw[0].x,low_jaw[1].x
    ljy0,ljy1=low_jaw[0].y,low_jaw[1].y
    ljz0,ljz1=(low_jaw[0].z+1)/2,(low_jaw[1].z+1)/2
    lj_wid=math.sqrt((ljx0 - ljx1) ** 2 + 
                    (ljy0 - ljy1) ** 2 +
                    (ljz0 - ljz1) ** 2 )
    return lj_wid,cl,cr

def calculate_d(landmark):
    face_length=[landmark[175],landmark[10]]
    dl,dr=face_length[0],face_length[1]
    flx0,flx1=face_length[0].x,face_length[1].x
    fly0,fly1=face_length[0].y,face_length[1].y
    flz0,flz1=(face_length[0].z+1)/2,(face_length[1].z+1)/2
    fl_wid=math.sqrt((flx0 - flx1) ** 2 + 
                    (fly0 - fly1) ** 2 +
                    (flz0 - flz1) ** 2 )
    return fl_wid,dl,dr

def calculate_face_type(dis_A,dis_B,dis_C,dis_D):#以abcd四條線計算臉部分類
    long_face_gate=1.65
    AB_rate=1.05
    BC_rate=1.05
    if dis_A*AB_rate>dis_B and dis_B>dis_C and dis_D/dis_D<long_face_gate:face_type =0                       # 0 = heart
    elif round(dis_B*100)/100==round(dis_C*100)/100 and dis_D/dis_D<long_face_gate:face_type=1               # 1 = square
    elif dis_A*AB_rate<dis_B and dis_C*BC_rate<dis_B and dis_D/dis_D<long_face_gate:face_type=2               # 2 = round
    elif dis_A*AB_rate>dis_B and dis_B>dis_C and dis_D/dis_D>=long_face_gate:face_type =3                    # 3 = diamond
    elif round(dis_B*100)/100==round(dis_C*100)/100 and dis_D/dis_D>=long_face_gate:face_type=4              # 4 = retangel
    else :face_type=5                                                                                        # 5 = oval
    return face_type       

def check_face_direction(landmark,is_front=False,err_code=0): #以臉寬跟臉長的點確認臉部正向
    face_width=[landmark[234],landmark[454]]
    face_length=[landmark[175],landmark[10]]
    ddfl,udfl=face_length[0].z,face_length[1].z
    ldfw,rdfw=face_width[0].z,face_width[1].z
    l_max,l_min,w_max,w_min=max(ddfl,udfl),min(ddfl,udfl),max(ldfw,rdfw),min(ldfw,rdfw)
    if l_max*l_min>0 and l_min-l_max<=0.01 and w_max-w_min<=0.03:
        is_front=True
    else:
        if l_max*l_min>0 and l_min-l_max<=0.01:         #上下符合
            if ldfw>rdfw:err_code=1                     #上下符合，臉偏左
            else:err_code=2                             #上下符合，臉偏右
        elif w_max-w_min<=0.03:                         #左右符合
            if ddfl+1>udfl+1:err_code=3                 #左右符合，臉偏下
            else:err_code=4                             #左右符合，臉偏上
        elif ddfl+1>udfl+1 and ldfw>rdfw:err_code=5     #臉偏下,臉偏左
        elif ddfl+1>udfl+1 and ldfw<rdfw:err_code=6     #臉偏下,臉偏右
        elif ddfl+1<udfl+1 and ldfw>rdfw:err_code=7     #臉偏上,臉偏左
        elif ddfl+1<udfl+1 and ldfw<rdfw:err_code=8     #臉偏上,臉偏右
        else :err_code=9
    return [ddfl,udfl,ldfw,rdfw],is_front,err_code

# 全局变量用于同步监控
global_err_code = -1
capture_in_progress = False
start_countdown = False
countdown_start_time = 0

def get_face_box_and_aligned(frame, face_landmarks):
    height, width, _ = frame.shape
    x_min = width
    y_min = height
    x_max = y_max = 0

    for lm in face_landmarks.landmark:
        x, y = int(lm.x * width), int(lm.y * height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y

    # 计算正方形区域
    face_width = x_max - x_min
    face_height = y_max - y_min
    face_size = max(face_width, face_height)
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

    # 确保正方形区域不超出边界
    left = max(center_x - face_size // 2, 0)
    right = min(center_x + face_size // 2, width)
    top = max(center_y - face_size // 2, 0)
    bottom = min(center_y + face_size // 2, height)

    # 裁剪人脸区域并调整大小
    face_crop = frame[top:bottom, left:right]

    # 使用 landmark[175] 和 landmark[10] 校准人脸角度
    jaw_point = face_landmarks.landmark[175]
    forehead_point = face_landmarks.landmark[10]
    jaw_x, jaw_y = int(jaw_point.x * width), int(jaw_point.y * height)
    forehead_x, forehead_y = int(forehead_point.x * width), int(forehead_point.y * height)
    
    angle = np.arctan2(forehead_y - jaw_y, forehead_x - jaw_x) * 180 / np.pi
    M = cv2.getRotationMatrix2D((face_size // 2, face_size // 2), angle+90, 1)
    aligned_face = cv2.warpAffine(face_crop, M, (face_size, face_size))

    face_crop_resized = cv2.resize(aligned_face, (256, 256))
    
    return face_crop_resized, left, top, right, bottom

def countdown_and_capture(frame, face_box, countdown_time=3):               # 定义拍照倒计时函数
    global capture_in_progress, start_countdown, countdown_start_time
    
    while time.time() - countdown_start_time < countdown_time:
        if global_err_code != 0:
            capture_in_progress = False
            start_countdown = False
            print("Countdown paused due to err_code change.")
            return None
        
        # 在倒计时过程中等待主线程更新显示
        time.sleep(0.1)
    
    # 拍摄照片
    photo = face_box.copy()
    cv2.imwrite('captured_face.jpg', photo)
    capture_in_progress = False
    start_countdown = False
    print("Photo captured.")
    return photo

def monitor_and_capture(frame, face_box):
    global capture_in_progress, start_countdown, countdown_start_time
    if not capture_in_progress:
        capture_in_progress = True
        start_countdown = True
        countdown_start_time = time.time()
        threading.Thread(target=countdown_and_capture, args=(frame, face_box)).start()

#長方形版
def get_rotated_face_box_and_aligned_1(frame, face_landmarks):
    height, width, _ = frame.shape

    # 使用 landmark[175] 和 landmark[10] 校准人脸角度
    jaw_point = face_landmarks.landmark[175]
    forehead_point = face_landmarks.landmark[10]
    jaw_x, jaw_y = int(jaw_point.x * width), int(jaw_point.y * height)
    forehead_x, forehead_y = int(forehead_point.x * width), int(forehead_point.y * height)

    angle = np.arctan2(jaw_y - forehead_y, jaw_x - forehead_x) * 180 / np.pi - 90

    # 计算人脸框的四个顶点
    x_min = width
    y_min = height
    x_max = y_max = 0

    for lm in face_landmarks.landmark:
        x, y = int(lm.x * width), int(lm.y * height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y

    # 计算人脸框的中心点
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

    # 构建旋转矩阵
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)

    # 对原图进行旋转
    rotated_frame = cv2.warpAffine(frame, M, (width, height))

    # 根据人脸框在旋转后的图像中裁剪人脸区域
    rotated_face_crop = rotated_frame[y_min:y_max, x_min:x_max]

    # 调整大小为256x256
    aligned_face_resized = cv2.resize(rotated_face_crop, (256, 256))

    # 计算旋转后人脸框的顶点坐标
    rect = ((center_x, center_y), (x_max - x_min, y_max - y_min), angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    return aligned_face_resized, box

#正方形版
def get_rotated_face_box_and_aligned(frame, face_landmarks):
    height, width, _ = frame.shape

    # 使用 landmark[175] 和 landmark[10] 校准人脸角度
    jaw_point = face_landmarks.landmark[175]
    forehead_point = face_landmarks.landmark[10]
    jaw_x, jaw_y = int(jaw_point.x * width), int(jaw_point.y * height)
    forehead_x, forehead_y = int(forehead_point.x * width), int(forehead_point.y * height)

    angle = np.arctan2(jaw_y - forehead_y, jaw_x - forehead_x) * 180 / np.pi - 90

    # 计算人脸框的四个顶点
    x_min = width
    y_min = height
    x_max = y_max = 0

    for lm in face_landmarks.landmark:
        x, y = int(lm.x * width), int(lm.y * height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y

    # 计算人脸框的中心点
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

    # 构建旋转矩阵
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)

    # 对原图进行旋转
    rotated_frame = cv2.warpAffine(frame, M, (width, height))

    # 根据旋转后的人脸框尺寸调整为正方形
    face_width = x_max - x_min
    face_height = y_max - y_min
    face_size = max(face_width, face_height)

    # 计算正方形区域的左上角和右下角坐标
    face_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
    face_box_min = (int(face_center[0] - face_size // 2), int(face_center[1] - face_size // 2))
    face_box_max = (int(face_center[0] + face_size // 2), int(face_center[1] + face_size // 2))

    # 裁剪旋转后的人脸区域，并调整大小为256x256
    rotated_face_crop = rotated_frame[face_box_min[1]:face_box_max[1], face_box_min[0]:face_box_max[0]]
    aligned_face_resized = cv2.resize(rotated_face_crop, (256, 256))

    # 计算旋转后人脸框的顶点坐标
    rect = ((center_x, center_y), (face_size, face_size), angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    return aligned_face_resized, box

def read_captured_face():
    file_name="./captured_face.jpg"
    captured_face=cv2.imread(file_name)
    return captured_face

# 在主循环中使用新的函数
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face1 = face_mesh1.process(image)
    #=====================================================================第一個face_mesh，處理固定照
    
    if results_face1.multi_face_landmarks:
        for face_landmarks in results_face1.multi_face_landmarks:
            
            face_box, box_points = get_rotated_face_box_and_aligned(frame, face_landmarks)
            array_cfd, is_front, err_code = check_face_direction(face_landmarks.landmark)
            global_err_code = err_code
            error_text = ['Good', 'Turn >', 'Turn <', 'Turn ^', 'Turn v', 'Turn ^>', 'Turn <^', 'Turn v>', 'Turn <v', 'unexpected error']
            name_array_cfd = ['ddfl', 'udfl', 'ldfw', 'rdfw']

            for index, value in enumerate(array_cfd, start=1):
                cv2.putText(frame, f'{name_array_cfd[index-1]}={value:.3f}', (400, (index+1)*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if is_front:cv2.putText(frame, f'check = {error_text[err_code]}', (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:cv2.putText(frame, f'{error_text[err_code]}', (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            color = (0, 255, 0) if err_code == 0 else (0, 0, 255)
            cv2.drawContours(frame, [box_points], 0, color, 2)
            if err_code == 0:
                monitor_and_capture(frame, face_box)
    if start_countdown:
        current_time = int(3 - (time.time() - countdown_start_time))
        cv2.putText(frame, f'Capturing in {current_time}', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #======================================================第二個face_mesh，處理固定照
    captured_face= read_captured_face()
    results_face2 = face_mesh2.process(captured_face)
    if results_face2.multi_face_landmarks:
        for face_landmarks in results_face2.multi_face_landmarks:
            
            dis_A, _, _ = calculate_a(face_landmarks.landmark)
            dis_B, _, _ = calculate_b(face_landmarks.landmark)
            dis_C, _, _ = calculate_c(face_landmarks.landmark)
            dis_D, _, _ = calculate_d(face_landmarks.landmark)
            face_type = calculate_face_type(dis_A, dis_B, dis_C, dis_D)
            cv2.putText(frame, f'Face Type : {face_type}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'A : {dis_A:.2f}', (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'B : {dis_B:.2f}', (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'C : {dis_C:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'D : {dis_D:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    #======================================================
    cv2.imshow('Pose and Face Mesh Estimation', frame)
    if 'face_box' in locals():
        cv2.imshow('Face_box', face_box)
    
    cv2.imshow('captured_face', read_captured_face())
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Detected 'q' key press. Exiting loop.")
        break

cap.release()
cv2.destroyAllWindows()


#我的版本
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     frame = cv2.flip(frame, 1)
#     frame_height, frame_width, _ = frame.shape
#     # 将影像转换为 RGB 格式
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # 将影像输入 mediapipe
#     results_face = face_mesh.process(image)
    
#     # 计算并显示眼睛距离和脸宽
#     if results_face.multi_face_landmarks:
#         for face_landmarks in results_face.multi_face_landmarks:
#             dis_A, al, ar = calculate_a(face_landmarks.landmark)
#             dis_B, bl, br = calculate_b(face_landmarks.landmark)
#             dis_C, cl, cr = calculate_c(face_landmarks.landmark)
#             dis_D, dl, dr = calculate_d(face_landmarks.landmark)
            
#             # 获取人脸框和对齐后的人脸图像
#             face_box, left, top, right, bottom = get_face_box_and_aligned(frame.copy(), face_landmarks)
            
#             # 判斷臉型
#             face_type = calculate_face_type(dis_A, dis_B, dis_C, dis_D)
            
#             draw_point = [al, ar, bl, br, cl, cr, dl, dr]
#             # 顯示眼睛距離和臉寬
#             cv2.putText(frame, f'Face Type : {face_type}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#             cv2.putText(frame, f'A : {dis_A:.4f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#             cv2.putText(frame, f'B : {dis_B:.4f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#             cv2.putText(frame, f'C : {dis_C:.4f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#             cv2.putText(frame, f'D : {dis_D:.4f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
#             array_cfd, is_front, err_code = check_face_direction(face_landmarks.landmark)
            
#             # 更新全局 err_code
#             global_err_code = err_code
            
#             error_text = ['Good', 'Turn >', 'Turn <', 'Turn ^', 'Turn v', 'Turn ^>', 'Turn <^', 'Turn v>', 'Turn <v', 'unexpected error']
#             name_array_cfd = ['ddfl', 'udfl', 'ldfw', 'rdfw']
            
#             for index, value in enumerate(array_cfd, start=1):
#                 cv2.putText(frame, f'{name_array_cfd[index-1]}={value:.5f}', (400, (index+1)*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
#             if is_front:
#                 cv2.putText(frame, f'check = {error_text[err_code]}', (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#             else:
#                 cv2.putText(frame, f'{error_text[err_code]}', (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            
            
#             # 绘制矩形框
#             color = (0, 255, 0) if err_code == 0 else (0, 0, 255)
#             cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
#             # 检查 err_code 并启动倒计时拍照线程
#             if err_code == 0:
#                 monitor_and_capture(frame, face_box)
    
#     # 显示倒计时信息
#     if start_countdown:
#         current_time = int(3 - (time.time() - countdown_start_time))
#         cv2.putText(frame, f'Capturing in {current_time}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
#     # 显示结果影像
#     cv2.imshow('Pose and Face Mesh Estimation', frame)
#     # 显示裁剪后的人脸图像
#     if 'face_box' in locals():
#         cv2.imshow('Face Crop', face_box)
    
#     # 等待按键输入
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         print("Detected 'q' key press. Exiting loop.")
#         break

# # 释放资源
# cap.release()
# cv2.destroyAllWindows()
#=============================================================================================still work but stop now