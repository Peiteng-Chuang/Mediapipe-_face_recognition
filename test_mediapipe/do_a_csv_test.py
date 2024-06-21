import cv2
import mediapipe as mp
import math
import numpy as np
# 初始化MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# LEFT_EYE_INDEX = [33, 133]        O
# RIGHT_EYE_INDEX = [362, 263]      O
# LEFT_EYE_INNER_INDEX = 133        O
# RIGHT_EYE_INNER_INDEX = 362       O
# LEFT_EYE_OUTER_INDEX = 33         O
# RIGHT_EYE_OUTER_INDEX = 263       O

# RIGHT_BROW_INDEX = [70, 107]      O 
# LEFT_BROW_INDEX = [336, 300]      O

# RIGHT_NOSE_INDEX = 344            O
# LEFT_NOSE_INDEX = 115             O
# NOSE_TOP_INDEX=168                O
# NOSE_TIP_INDEX = 1

# UPPER_LIP_INDEX = [0,13]          O
# LOWER_LIP_INDEX = [14,16]         O

# FACE_WIDTH_INDEX = [234, 454]     O
# FACE_HEIGHT_INDEX = [10, 152]     O

# 特徵索引定義
# FEATURE_INDEXES = [
#     33,133,
#     362,263, 
#     344,115,
#     70,107,
#     336,300,
#     168,1,
#     234,454,
#     10,152,
#     0,13,14,16
#     ]


lip=[146,91,181,84,17,314,
405,321,375,291,185,40,39,
37,0,267,269,270,409,291,95,
88,178,87,14,317,402,318,
324,308,191,80,81,82,13,
312,311,310,415,308]
lip_u=[0,37,39,40,185,191,80,81,82,13,312,311,310,267,269,270]

ford_head=[54,284]
width=[234,454]
low_jaw=[172,397]
long=[175,10]
nose=[4]
all_points=[ford_head,width,low_jaw,long]
ff=long+width+low_jaw+ford_head+nose
# FEATURE_INDEXES = [16]
FEATURE_INDEXES = ff

# 讀取圖片
image_path = './test_mediapipe/test_image/max.jpg'  # 替换为你的图片路径
# image_path = './test_mediapipe/test_image/amy.jpg'  # 替换为你的图片路径
# image_path = './test_mediapipe/256img_lst/001.jpg'  # 替换为你的图片路径



def get_distance():
    return 0

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(image_rgb)
distance_array=[0,0,0,0]
WD,HG=image.shape[1],image.shape[0]

if results.multi_face_landmarks:
    
    landmarks = results.multi_face_landmarks[0].landmark
    def get_xyz(inpt):
        x,y,z=landmarks[inpt].x,landmarks[inpt].y,landmarks[inpt].z
        return x,y,z
    def get_distance(point1, point2):
        p1,p2=landmarks[point1],landmarks[point2]
        return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
    def get_imxy(number):
        imx = int(landmarks[number].x * WD)
        imy = int(landmarks[number].y * HG)
        return imx,imy
    for id,point in enumerate(all_points,start=1):
        point1 = get_imxy(point[0])
        point2 = get_imxy(point[1])
        
        for inpt in point:
            x,y=get_imxy(inpt)
            mpx,mpy,mpz=get_xyz(inpt)
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # 用绿色绘制特征点
            cv2.putText(image, f'{inpt}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            print(f"{inpt}_loc :\tx={mpx} \ty={mpy} \tz={mpz}")
        
        temp=get_distance(point[0],point[1])
        
        cv2.line(image, point1, point2, (0, 255, 0), 1)
        cv2.putText(image, f'line{id} : {temp:.3f}', (20, 35*id), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        tag_loc=(get_imxy(point[1]))
        tag_loc=tag_loc[0],tag_loc[1]-20
        cv2.putText(image, f'line{id}', tag_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        distance_array[id-1]=temp
    for id,a in enumerate(distance_array):
        print(id+1,a)
    
    
# 顯示圖片
cv2.imshow('Image with Face Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
