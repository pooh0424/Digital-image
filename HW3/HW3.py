import cv2
import numpy as np
sign_img = cv2.imread("sign.jpg", cv2.IMREAD_GRAYSCALE) #讀取簽名

def gussenfilter(img, size):
    img = cv2.GaussianBlur(img, (size, size), 0) #高斯濾波
    return img

def compute_gradient(img, size=3):
    x= cv2.Sobel(img,cv2.CV_64F, 1, 0, ksize=size) #計算x方向梯度
    y= cv2.Sobel(img,cv2.CV_64F, 0, 1, ksize=size) #計算y方向梯度
    grad = np.sqrt(np.power(x,2) + np.power(y,2)) #計算梯度
    angle = np.arctan2(y, x) #計算角度
    return grad, angle

def non_max(grad, angle):
    result = np.zeros(grad.shape,dtype="int32") #建立結果數列
    for i in range(1, grad.shape[0]-1): 
        for j in range(1, grad.shape[1]-1): #遍歷整個圖像
            if(angle[ i ,j]< 0): #調整到0-pi之間
                angle[i ,j] += np.pi 
            if (angle[i, j] >= 7/8*np.pi or angle[i, j] < 1/8*np.pi): #判斷角度
                if grad[i, j] > grad[i, j+1] and grad[i, j] > grad[i, j-1]: #判斷是否為最大值
                    result[i, j] = grad[i, j] #將最大值存入結果數列
                else:
                    result[i, j] = 0 #不是最大值存入0
            elif (angle[i, j] >= 1/8*np.pi and angle[i, j] < 3/8*np.pi): #判斷角度
                if grad[i, j] > grad[i-1, j-1] and grad[i, j] > grad[i+1, j+1]: #判斷是否為最大值
                    result[i, j] = grad[i, j] #將最大值存入結果數列
                else:
                    result[i, j] = 0 #不是最大值存入0
            elif (angle[i, j] >= 3/8*np.pi and angle[i, j] < 5/8*np.pi): 
                if grad[i, j] > grad[i-1, j] and grad[i, j] > grad[i+1, j]:
                    result[i, j] = grad[i, j]
                else:
                    result[i, j] = 0
            else:
                if grad[i, j] > grad[i-1, j+1] and grad[i, j] > grad[i+1, j-1]:
                    result[i, j] = grad[i, j]
                else:
                    result[i, j] = 0
    return result

def double_threshold(img, low, high):
    result = np.zeros(img.shape, dtype="uint8") #建立結果數列
    for i in range(1, img.shape[0]-1): 
        for j in range(1, img.shape[1]-1): #遍歷整個圖像
            if img[i, j] > high: #判斷是否大於high
                result[i, j] = 255 #強邊界
            elif img[i, j] < low: #判斷是否小於low
                result[i, j] = 0 #非邊界
            else:
                result[i, j] = 128  #弱邊界
    return result

def edge_linking_dfs(img):
    processimg = img.copy() #複製圖片
    edge = np.zeros(img.shape, dtype="uint8")  #建立結果數列
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] # 定義方向
    def dfs(x, y): # 定義深度優先搜索函數
        stack = [(x, y)] # 初始化堆疊
        while stack: # 當堆疊不為空
            cx, cy = stack.pop() # 取出堆疊頂部元素
            if edge[cx, cy] == 0:  # 尚未訪問
                edge[cx, cy] = 255  # 標記為邊緣
                for dx, dy in directions: # 遍歷所有方向
                    nx, ny = cx + dx, cy + dy # 計算下一個位置
                    if 0 <= nx < processimg.shape[0] and 0 <= ny < processimg.shape[1]: # 確保位置在圖像範圍內
                        if processimg[nx, ny] == 128 and edge[nx, ny] == 0:  # 與弱邊緣相連
                            processimg[nx, ny] = 255  # 標記為強邊緣
                            stack.append((nx, ny)) # 將位置加入堆疊
    for i in range(img.shape[0]):
        for j in range(img.shape[1]): # 遍歷整個圖像
            if processimg[i, j] == 255 and edge[i, j] == 0:  # 強邊緣且未處理
                dfs(i, j) # 進行深度優先搜索
    return edge

def hough_transform(edge_image, theta_res=np.pi/180, rho_res=1, threshold=50, sample_rate=0.5, delta_theta=1, delta_rho=1 ,picturenumber =0):
    height, width = edge_image.shape  # 取得影像尺寸
    diag_len = int(np.sqrt(height**2 + width**2))  # 影像對角線長度
    rhos = np.arange(-diag_len, diag_len , rho_res)  # ρ 的範圍 (-d 到 d)
    thetas = np.arange(0, np.pi, theta_res)  # θ 的範圍 (0 到 180 度)
    
    edge_points = np.argwhere(edge_image)  # 取得所有邊緣點
    sampled_points_idx = np.random.choice(edge_points.shape[0],size = int(len(edge_points)*sample_rate),replace= False)  # 隨機選擇部分邊緣點
    sampled_points = edge_points[sampled_points_idx]  # 取得部分邊緣點
    print("samplepoint:"+str(len(sampled_points))) #印出取樣點數量
    
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.float64) # 建立累加器

    for y, x in sampled_points:
        for theta_idx, theta in enumerate(thetas): # 遍歷所有角度
            rho = int(x * np.cos(theta) + y * np.sin(theta))  # 計算 ρ 值
            rho_idx = np.argmin(np.abs(rhos - rho))  # 取得最接近的索引
            rho_range = np.arange(max(rho_idx - delta_rho , 0), min(rho_idx + delta_rho + 1, len(rhos))) # 取得擴張範圍
            theta_range = np.arange(max(theta_idx - delta_theta,0), min(theta_idx + delta_theta + 1, len(thetas)))# 取得擴張範圍
            for r_idx in rho_range:
                for t_idx in theta_range: # 遍歷所有範圍
                    if(r_idx == rho_idx and t_idx == theta_idx): #如果是原本的點
                        accumulator[r_idx, t_idx] += 1 #累加1
                    else: #如果不是原本的點
                        accumulator[r_idx, t_idx] += 0.3 #累加0.3
    showimg(accumulator, f"accumulator_im{picturenumber+1}.jpg") #存檔
    lines = [] # 初始線段
    for rho_idx, theta_idx in np.argwhere(accumulator >= threshold): # 取得所有大於閾值的索引
        rho = rhos[rho_idx] # 取得 rho 值
        theta = thetas[theta_idx] # 取得 theta 值
        lines.append((rho, theta)) # 加入線段
    
    return lines

def drowline(img, lines):
    print("linenumber:"+str(len(lines))) #印出線段數量
    if lines is not None: # 如果有線段
        for line in lines: # 遍歷所有線段
            rho, theta = line  # 提取 rho 和 theta
            a = np.cos(theta) # 計算 cos(theta)
            b = np.sin(theta) # 計算 sin(theta)
            x0 = a * rho # 計算 x0
            y0 = b * rho # 計算 y0
            # 計算兩個端點
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)# 在圖像上繪製直線


def canny_edge(img, gussenfiltersize, compute_gradientsize,low, high, i):
    processimg = img.copy() #複製圖片
    processimg = gussenfilter(processimg, gussenfiltersize)#高斯濾波
    grad, angle = compute_gradient(processimg, compute_gradientsize)#計算梯度
    showimg(grad, f"grad_im{i+1}.jpg") #存檔
    grad = non_max(grad, angle) #非極大值抑制
    showimg(grad, f"non_max_im{i+1}.jpg") #存檔
    grad = double_threshold(grad ,low , high) #雙閾值
    gradresult = edge_linking_dfs(grad) #邊緣連接
    return gradresult, grad

def sign(input):
    (h, w) = input.shape  # 取得圖片height 、width(黑白)
    #簽名在圖片的右下角位置
    for i in range(100):
        for j in range(200):
            if sign_img[i][j] < 50:  # 如果該位置小於50(黑色)，則將該位置像素調整為全白
                input[h - 100 + i][w - 200 + j] = 255
    return input

def signcolor(input):
    (h, w , r) = input.shape  # 取得圖片height 、width(黑白)
    #簽名在圖片的右下角位置
    for i in range(100):
        for j in range(200):
            if sign_img[i][j] < 50:  # 如果該位置小於50(黑色)，則將該位置像素調整為全黑
                for k in range(r): #  遍歷所有通道
                    input[h - 100 + i][w - 200 + j][k] = 0 # 將該位置像素調整為全黑
    return input

def showimg(img, name):
    (h, w) = img.shape
    showpicture = np.zeros((h, w), dtype="uint8")
    for i in range(h):
        for j in range(w):
            if(img[i][j] > 255):
                showpicture[i][j] = 255
            elif(img[i][j] < 0):
                showpicture[i][j] = 0
            else:
                showpicture[i][j] = img[i][j]
    showpicture = sign(showpicture)
    cv2.imwrite(name, showpicture)

if __name__ == "__main__":
    parameter=[[50, 150, 70],[50, 150, 80],[50, 150, 80]]#參數(low,higt,threshold)

    for i in range(0, 3): 
        img = cv2.imread(f"im{i+1}.jpg") #讀取圖片
        imgresize = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))#縮小圖片
        imgresizegray = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)#轉換成灰階圖片
        gradresult,grad = canny_edge(imgresizegray , 5, 3, parameter[i][0], parameter[i][1],i) #canny邊緣檢測
        grad = sign(grad) #簽名
        gradresult = sign(gradresult) #簽名
        cv2.imwrite(f"double_im{i+1}.jpg", grad) #存檔
        cv2.imwrite(f"cannyedge_im{i+1}.jpg", gradresult) #存檔
         
        lines= hough_transform(gradresult,  threshold=parameter[i][2],sample_rate=0.5, delta_theta=0, delta_rho=0 ,picturenumber= i) #霍夫變換
        drowline(imgresize, lines) #畫線
        imgresize = signcolor(imgresize) #簽名
        cv2.imwrite(f"Detected_Lines_im{i+1}.jpg",imgresize) #存檔
    