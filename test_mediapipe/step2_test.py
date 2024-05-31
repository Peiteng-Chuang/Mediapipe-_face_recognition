import os
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

# 設定資料集資料夾路徑
data_folder = "test_mediapipe/preprocessed_images"

# 初始化資料列表
data = []

# 迭代處理每張影像
for filename in os.listdir(data_folder):
    if filename.endswith(".jpeg") or filename.endswith(".png"):
        # 讀取影像
        img_path = os.path.join(data_folder, filename)
        img = cv2.imread(img_path)
        
        # 將影像轉換為灰度影像並正規化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        
        # 將影像添加到資料列表中
        data.append(gray)

# 將列表轉換為NumPy數組並擴展維度以適應Keras的輸入格式
data = np.array(data).reshape(-1, 64, 64, 1)

# 定義自動編碼器模型
input_img = Input(shape=(64, 64, 1))

# 編碼器
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# 解碼器
x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# 建立模型
autoencoder = Model(input_img, decoded)

# 編譯模型
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 打印模型結構
autoencoder.summary()

# 訓練模型
autoencoder.fit(data, data, epochs=50, batch_size=32, validation_split=0.2)

# 保存模型
autoencoder.save("face_autoencoder.h5")
