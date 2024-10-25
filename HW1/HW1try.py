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
    img =cv2.imread("im3.jpg")
    # 定義不同的kernel
    kernel1 = np.array([
        [[1/27, 1/27, 1/27], [1/27, 1/27, 1/27], [1/27, 1/27, 1/27]],
        [[1/27, 1/27, 1/27], [1/27, 1/27, 1/27], [1/27, 1/27, 1/27]],
        [[1/27, 1/27, 1/27], [1/27, 1/27, 1/27], [1/27, 1/27, 1/27]]
    ])#mean filter
    

    kernel15 = np.array([
        [[1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75]],
        [[1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75]],
        [[1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75]],
        [[1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75]],
        [[1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75], [1/75, 1/75, 1/75]]
    ])#mean filter5*5
    
    kernel35 = np.array([
        [[1/252, 1/252, 1/252], [2/252, 2/252, 2/252], [3/252, 3/252, 3/252],[2/252, 2/252, 2/252],[1/252, 1/252, 1/252]],
        [[2/252, 2/252, 2/252], [5/252, 5/252, 5/252], [6/252, 6/252, 6/252],[5/252, 5/252, 5/252],[2/252, 2/252, 2/252]],
        [[3/252, 3/252, 3/252], [6/252, 6/252, 6/252], [8/252, 8/252, 8/252],[6/252, 6/252, 6/252],[3/252, 3/252, 3/252]],
        [[2/252, 2/252, 2/252], [5/252, 5/252, 5/252], [6/252, 6/252, 6/252],[5/252, 5/252, 5/252],[2/252, 2/252, 2/252]],
        [[1/252, 1/252, 1/252], [2/252, 2/252, 2/252], [3/252, 3/252, 3/252],[2/252, 2/252, 2/252],[1/252, 1/252, 1/252]]
    ])#gaussian filter5*5
    
    # 讀取簽名檔，並轉成黑白
    signimg = cv2.imread("sign.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 進行mean filter onepadding並簽名，保存
    result = conv(img, 3, 3, kernel1, 0, 1)
    result = sign(result, signimg)
    cv2.imwrite(f"im3_onepadding.jpg", result)
        
    # 進行mean filter nopadding並簽名，保存
    result = conv(img,3,3,kernel1,0,-1)
    result = sign(result,signimg)
    cv2.imwrite(f"im3_nopadding.jpg",result)

    # 進行mean filter bias100並簽名，保存
    result = conv(img,3,3,kernel1,100,0)
    result = sign(result,signimg)
    cv2.imwrite(f"im3_bias100.jpg",result)

    # 進行mean filter 5*5並簽名，保存
    result = conv(img,5,5,kernel15,0,0)
    result = sign(result,signimg)
    cv2.imwrite(f"im3_blur5.jpg",result)

    # 進行gaussian filter5*5並簽名，保存
    result = conv(img,5,5,kernel35,0,0)
    result = sign(result,signimg)
    cv2.imwrite(f"im3_gauss5.jpg",result)

    # 進行max pool並簽名，保存
    resultmax = pool(img,4,4,1)
    resultmax = sign(resultmax,signimg)
    cv2.imwrite(f"im3maxpool4.jpg",resultmax)

    # 進行average pool並簽名，保存
    resultaverage = pool(img,4,4,0)
    resultaverage = sign(resultaverage,signimg)
    cv2.imwrite(f"im3avgpool4.jpg",resultaverage) 
        
    print("Processing complete.")