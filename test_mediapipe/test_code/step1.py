import cv2
import os

# 設定輸入和輸出資料夾路徑
input_folder = "test_mediapipe/image"
output_folder = "test_mediapipe/preprocessed_images"

# 如果輸出資料夾不存在，則創建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 初始化人臉檢測器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 迭代處理每張影像
for filename in os.listdir(input_folder):
    if filename.endswith(".jpeg") or filename.endswith(".png"):
        # 讀取影像
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        # 將影像轉換為灰度影像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 檢測人臉
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # 迭代處理每個檢測到的人臉
        for (x, y, w, h) in faces:
            # 裁剪人臉
            face_img = img[y:y+h, x:x+w]
            
            # 調整人臉大小（可選）
            face_img = cv2.resize(face_img, (256, 256))
            
            # 設定輸出影像的檔案路徑
            output_path = os.path.join(output_folder, filename)
            
            # 儲存處理後的影像
            cv2.imwrite(output_path, face_img)
            
            print(f"Preprocessed {filename} and saved as {output_path}")

print("Preprocessing finished.")
