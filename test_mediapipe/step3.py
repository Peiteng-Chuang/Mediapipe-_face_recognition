import cv2
import numpy as np
from keras.models import load_model

# 加載訓練好的自動編碼器模型
autoencoder = load_model('face_autoencoder.h5')

# 初始化OpenCV的人臉檢測器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 開啟攝像頭
cap = cv2.VideoCapture(0)

while True:
    # 從攝像頭讀取一幀
    ret, frame = cap.read()
    if not ret:
        break

    # 將影像轉換為灰度影像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 檢測人臉
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

    for (x, y, w, h) in faces:
        # 提取並預處理人臉區域
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))
        face_normalized = face_resized / 255.0
        face_normalized = np.reshape(face_normalized, (1, 64, 64, 1))
        
        # 使用自動編碼器進行重建
        reconstructed = autoencoder.predict(face_normalized)
        reconstructed = np.reshape(reconstructed, (64, 64))
        reconstructed = (reconstructed * 255).astype('uint8')

        # 計算重建誤差（僅用於展示，可以忽略）
        error = np.mean((face_resized - reconstructed) ** 2)

        # 在影像上畫框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'Error: {error:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 顯示結果影像
    cv2.imshow('Real-time Face Detection', frame)

    # 按下 'q' 鍵退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭並關閉所有窗口
cap.release()
cv2.destroyAllWindows()
