import cv2
import numpy as np

# 執行卷積運算
def calconv(input, m, n, kernel, bias, d):
    result = 0.0
    # loop over height and width
    for i in range(n):
        for j in range(m):
            #  Loop over depth channels
            for k in range(d):                
                result += input[i][j][k] * kernel[i][j][k]# 將kernel和input的部分，執行element-wise multiplication 
    
    # 加上bias，並將結果限制在0-255之間
    if (result + bias > 255):
        return 255
    if (result + bias < 0):
        return 0
    return int(result + bias)

# 執行pooling
def pool(input, size, stride, type):
    (h, w, d) = input.shape  # 取得圖片height 、width、depth channels
    result = np.empty(((h - size) // stride + 1, (w - size) // stride + 1), dtype="uint8")#新增pooling完成後的空白畫布
    
    # loop over height and width with stride
    for i in range(0, h - size + 1, stride):
        for j in range(0, w - size + 1, stride):
            if (type == 0):  # 平均池化
                sum = 0
                # 累加windows内的所有值
                for i2 in range(i, i + size):
                    for j2 in range(j, j + size):
                        for c in range(d):
                            sum += input[i2, j2, c]
                # 計算平均值，將給指定的位置
                result[i // stride, j // stride] = sum // (size * size * d)
            if (type == 1):  # 最大池化
                max_val = 0
                # 在windows内找到最大值
                for i2 in range(i, i + size):
                    for j2 in range(j, j + size):
                        number = 0
                        for c in range(d):
                            number += input[i2, j2, c]
                        if number > max_val:
                            max_val = number
                # 計算最大值除以depth channels，將給指定的位置
                result[i // stride, j // stride] = max_val // d
    return result

# 執行卷積操作和padding的函數
def conv(input, m, n, kernel, bias, padding):
    (h, w, d) = input.shape  # 取得圖片height 、width、depth channels
    
    if padding == -1:  # no padding
        result = np.empty((h - n + 1, w - m + 1), dtype="uint8") # 新增conv完成後的空白畫布
        for i in range(n // 2, h - n // 2):
            for j in range(m // 2, w - m // 2):
                # 計算给定區域的卷積
                result[i - n // 2][j - m // 2] = calconv(input[i - n // 2:i + n // 2 + 1, j - m // 2:j + m // 2 + 1], m, n, kernel, bias, d)
    else:  # padding
        newpicture = np.empty((h + (n - 1), w + (m - 1), d))# 新增padding完成後的空白畫布
        result = np.empty((h, w), dtype="uint8")# 新增conv完成後的空白畫布
        
        if padding == 0: # zero padding
            newpicture[:, :, :] = 0
        elif padding == 1:# one padding
            newpicture[:, :, :] = 255
        
        # 將圖片填入padding的空白畫布中心
        newpicture[(n - 1) // 2:(n - 1) // 2 + h, (m - 1) // 2:(m - 1) // 2 + w] = input
        
        # 計算给定區域的卷積(paading後)
        for i in range((n - 1) // 2, (n - 1) // 2 + h):
            for j in range((m - 1) // 2, (m - 1) // 2 + w):
                result[i - (n - 1) // 2][j - (m - 1) // 2] = calconv(newpicture[i - n // 2:i + n // 2 + 1, j - m // 2:j + m // 2 + 1], m, n, kernel, bias, d)
    
    return result

# 簽名
def sign(input, sign_img):
    (h, w) = input.shape  # 取得圖片height 、width(黑白)
    #簽名在圖片的右下角位置
    for i in range(100):
        for j in range(200):
            if sign_img[i][j] < 50:  # 如果該位置小於50(黑色)，則將該位置像素調整為全白
                input[h - 100 + i][w - 200 + j] = 255
    return input

if __name__ == "__main__":
    # 將所有要處理的圖片讀入
    img = list()
    for i in range(1, 4):
        img.append(cv2.imread(f"im{i}.jpg"))
    
    # 定義不同的kernel
    kernel1 = np.array([
        [[1/27, 1/27, 1/27], [1/27, 1/27, 1/27], [1/27, 1/27, 1/27]],
        [[1/27, 1/27, 1/27], [1/27, 1/27, 1/27], [1/27, 1/27, 1/27]],
        [[1/27, 1/27, 1/27], [1/27, 1/27, 1/27], [1/27, 1/27, 1/27]]
    ])#mean filter
    
    kernel2 = np.array([
        [[-1/3, -1/3, -1/3], [0, 0, 0], [1/3, 1/3, 1/3]],
        [[-2/3, -2/3, -2/3], [0, 0, 0], [2/3, 2/3, 2/3]],
        [[-1/3, -1/3, -1/3], [0, 0, 0], [1/3, 1/3, 1/3]]
    ])#sobel(X方向)
    
    kernel3 = np.array([
        [[1/48, 1/48, 1/48], [2/48, 2/48, 2/48], [1/48, 1/48, 1/48]],
        [[2/48, 2/48, 2/48], [4/48, 4/48, 4/48], [2/48, 2/48, 2/48]],
        [[1/48, 1/48, 1/48], [2/48, 2/48, 2/48], [1/48, 1/48, 1/48]]
    ])#gaussian filter
    
    # 讀取簽名檔，並轉成黑白
    signimg = cv2.imread("sign.jpg", cv2.IMREAD_GRAYSCALE)
    
    for i in range(1, 4):
        # 進行mean filter並簽名，保存
        result1 = conv(img[i-1], 3, 3, kernel1, 0, 0)
        result1 = sign(result1, signimg)
        cv2.imwrite(f"im{i}blur.jpg", result1)
        
        # 進行sobel並簽名，保存
        result2 = conv(img[i-1], 3, 3, kernel2, 0, 0)
        result2 = sign(result2, signimg)
        cv2.imwrite(f"im{i}sobel.jpg", result2)
        
        # 進行gaussian filter並簽名，保存
        result3 = conv(img[i-1],3,3,kernel3,0,0)
        result3 = sign(result3,signimg)
        cv2.imwrite(f"im{i}gauss.jpg",result3)

        # 進行max pool並簽名，保存
        resultmax = pool(img[i-1],2,2,1)
        resultmax = sign(resultmax,signimg)
        cv2.imwrite(f"im{i}maxpool.jpg",resultmax)

        # 進行average pool並簽名，保存
        resultaverage = pool(img[i-1],2,2,0)
        resultaverage = sign(resultaverage,signimg)
        cv2.imwrite(f"im{i}avgpool.jpg",resultaverage) 
        
    print("Processing complete.")