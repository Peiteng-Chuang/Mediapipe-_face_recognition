import os
from PIL import Image

# 定義來源和目的地資料夾路徑
source_folder = 'C:\Users\User\Desktop\project_file\Mediapipe-_face_recognition\test_mediapipe\preprocessed_images'
destination_folder = 'path/to/destination/folder'

# 確認目的地資料夾存在，如果不存在則創建
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍歷來源資料夾中的所有文件
for filename in os.listdir(source_folder):
    # 檢查文件是否為圖片（根據文件擴展名）
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # 完整的文件路徑
        file_path = os.path.join(source_folder, filename)
        
        # 打開圖片
        with Image.open(file_path) as img:
            # 對圖片進行處理
            # 例如：調整圖片大小
            processed_img = img.resize((800, 800))

            # 保存處理後的圖片到目的地資料夾
            processed_img.save(os.path.join(destination_folder, filename))

print("圖片處理完成並保存到目的地資料夾。")
