import cv2
import os

# 設定分類器 XML 的路徑
pictPath = r'C:\Users\USER\Desktop\opencv\fin\classifier\cascade.xml'
face_cascade = cv2.CascadeClassifier(pictPath)  # 建立分類器物件

# 設定圖片資料夾的路徑（準備辨識這個資料夾中的所有圖片）
folder_path = r'C:\Users\USER\Desktop\opencv\fin\Test photos'

# 使用 os.listdir 讀取資料夾中所有檔案名稱
for filename in os.listdir(folder_path):
    # 檢查是否為圖片副檔名（大小寫皆可）
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 取得每張圖片的完整路徑
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)  # 讀取圖片

        # 如果圖片讀取失敗，顯示錯誤訊息並跳過
        if img is None:
            print(f"無法讀取圖片: {img_path}")
            continue

        # 偵測照片（可根據模型調整參數）
        faces = face_cascade.detectMultiScale(
            img,               # 傳入圖片
            scaleFactor=2,     # 縮放比例，越大越快但不夠準
            minNeighbors=4,    # 鄰近框數，越大越嚴格
            minSize=(400, 800) # 偵測的最小臉部尺寸（寬, 高）
        )

        # 在圖片右下角畫一個黃色的背景框用來顯示文字
        cv2.rectangle(img, 
                      (img.shape[1]-140, img.shape[0]-20),  # 左上角座標
                      (img.shape[1], img.shape[0]),         # 右下角座標
                      (0, 255, 255),                        # 顏色：黃色 (BGR)
                      -1)                                   # 填滿

        # 在背景框上方寫上「Finding X face(s)」
        cv2.putText(img, 
                    "Finding " + str(len(faces)) + " face(s)",  # 顯示偵測到幾張臉
                    (img.shape[1]-135, img.shape[0]-5),          # 文字位置
                    cv2.FONT_HERSHEY_COMPLEX, 0.5,               # 字型與大小
                    (255, 0, 0), 1)                              # 顏色：紅色 + 粗細

        # 對每一張偵測到的臉畫出藍色邊框
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # 畫框 (BGR：藍)

        # 顯示這張圖片
        cv2.imshow("Face Detection", img)
        
        # 等待按鍵：按任意鍵切換下一張，按 ESC 鍵（key=27）則結束程式
        key = cv2.waitKey(0)
        if key == 27:
            break

# 關閉所有 OpenCV 開啟的視窗
cv2.destroyAllWindows()
