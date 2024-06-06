from PIL import Image
import os

# 設定輸入和輸出資料夾路徑
input_folder = "./test_mediapipe/preprocessed_images"
output_folder = "./test_mediapipe/256img_lst"

def save_image_as_new_file_with_pillow(source_path, new_name, new_format):
    # 打开源图像
    with Image.open(source_path) as img:
        
        # Pillow 库中的格式标识符
        format_dict = {
            'jpg': 'JPEG',
            'jpeg': 'JPEG',
            'png': 'PNG',
            'bmp': 'BMP',
            'gif': 'GIF'
            # 可以根据需要添加更多格式
        }
        
        # 获取新的文件名和路径
        new_file_path = f"{output_folder}/{new_name}.{new_format}"
        
        # 检查并转换格式标识符
        if new_format.lower() in format_dict:
            save_format = format_dict[new_format.lower()]
        else:
            raise ValueError(f"Unsupported file format: {new_format}")

        # 保存图像到新位置，改变格式
        img.save(new_file_path, save_format)
        print(f"Image saved to: {new_file_path}")

# 如果輸出資料夾不存在，則創建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
count=1
format_img='jpg'
# 迭代處理每張影像
for filename in os.listdir(input_folder):
    if filename.endswith(".jpeg"):
        count_str=str(count)
        while len(count_str)<3:
            count_str="0"+count_str
        s_path=input_folder+"/"+filename
        
        save_image_as_new_file_with_pillow(s_path,count_str,format_img)
        count+=1

print("all done !")




# 示例使用
source_image = 'path/to/source/image.jpeg'
new_name = 'path/to/destination/001'
new_format = 'jpg'
save_image_as_new_file_with_pillow(source_image, new_name, new_format)
