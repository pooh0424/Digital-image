import cv2
import numpy as np
# def edge_linking(img):
#     edge = np.zeros(img.shape,dtype="uint8")
#     myunoin = unoin(img.shape[0], img.shape[1])
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if img[i, j] == 1 or img[i, j] == 2:
#                 if i-1 >= 0 and (img[i-1, j] == 1 or img[i-1, j] == 2):
#                     myunoin.union(i, j, i-1, j, img)
#                 if j-1 >= 0 and (img[i, j-1] == 1 or img[i, j-1] == 2):
#                     myunoin.union(i, j, i, j-1, img)
#                 if i-1 >= 0 and j-1 >= 0 and (img[i-1, j-1] == 1 or img[i-1, j-1] == 2):
#                     myunoin.union(i, j, i-1, j-1, img)
#                 if i-1 >= 0 and j+1 < img.shape[1] and (img[i-1, j+1] == 1 or img[i-1, j+1] == 2):
#                     myunoin.union(i, j, i-1, j+1, img)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if img[i, j] == 1 or img[i, j] == 2:
#                 if i-1 >= 0 and (img[i-1, j] == 1 or img[i-1, j] == 2):
#                     myunoin.union(i, j, i-1, j, img)
#                 if j-1 >= 0 and (img[i, j-1] == 1 or img[i, j-1] == 2):
#                     myunoin.union(i, j, i, j-1, img)
#                 if i-1 >= 0 and j-1 >= 0 and (img[i-1, j-1] == 1 or img[i-1, j-1] == 2):
#                     myunoin.union(i, j, i-1, j-1, img)
#                 if i-1 >= 0 and j+1 < img.shape[1] and (img[i-1, j+1] == 1 or img[i-1, j+1] == 2):
#                     myunoin.union(i, j, i-1, j+1, img)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if img[i, j] == 1:
#                 root = myunoin.find(i, j,img)
#                 if img[root[0], root[1]] == 2:
#                     edge[i, j] = 255
#                     # print("edge")
#                 else:
#                     edge[i, j] = 0
#             elif img[i, j] == 2:
#                 edge[i, j] = 255
#     return edge

# class unoin:
#     def __init__(self, M, N):
#         self.parent = np.zeros((M, N, 2), dtype=int)
#         for i in range(M):
#             for j in range(N):
#                 self.parent[i,j,0] = i
#                 self.parent[i,j,1] = j
#         self.size = np.zeros((M, N))
#     def find(self, i, j ,img):
#         if self.parent[i ,j][0] != i or self.parent[i, j][1] != j:
#             root = self.find(self.parent[i, j][0], self.parent[i, j][1],img)
#             if img[root[0],root[1]] == 2:
#                 self.parent[i, j] = root
#         return self.parent[i, j]
#     def union(self, x, y, x1, y1, img):
#         root1 = self.find(x, y, img)
#         root2 = self.find(x1, y1, img)
#         if x == 1041 and y == 2101:
#             print(root1,root2)
#             print(x1,y1)
#         if (root1 != root2).any():
#             if img[root1[0], root1[1]] == 2 and img[root2[0], root2[1]] != 2:
#                 self.parent[root2[0], root2[1]] = root1
#                 self.size[root1[0], root1[1]] += self.size[root2[0], root2[1]]
#             elif img[root2[0], root2[1]] == 2 and img[root1[0], root1[1]] != 2:
#                 self.parent[root1[0], root1[1]] = root2
#                 self.size[root2[0], root2[1]] += self.size[root1[0], root1[1]]
#             else:
#                 if self.size[root1[0], root1[1]] < self.size[root2[0], root2[1]]:
#                     self.parent[root1[0], root1[1]] = root2
#                     self.size[root2[0], root2[1]] += self.size[root1[0], root1[1]]
#                 else:
#                     self.parent[root2[0], root2[1]] = root1
#                     self.size[root1[0], root1[1]] += self.size[root2[0], root2[1]]

# def hough_transform2(edges, rho_res=1, theta_res=np.pi/180, threshold=100):
#     height, width = edges.shape
#     max_dist = int(np.sqrt(height**2 + width**2))
#     rhos = np.arange(-max_dist, max_dist+1, rho_res)
#     thetas = np.arange(0, np.pi, theta_res)
#     accumulator = np.zeros((len(rhos), len(thetas)), dtype="uint8")

#     for y in range(height):
#         for x in range(width):
#             if edges[y, x]:  # 如果是邊緣點
#                 for theta_idx in range(len(thetas)):
#                     theta = thetas[theta_idx]
#                     rho = int(x * np.cos(theta) + y * np.sin(theta))
#                     rho_idx = np.argmin(np.abs(rhos - rho))
#                     accumulator[rho_idx, theta_idx] += 1
#     cv2.imwrite("hi.jpg",accumulator)
#     lines = []
#     for rho_idx in range(len(rhos)):
#         for theta_idx in range(len(thetas)):
#             if accumulator[rho_idx, theta_idx] >= threshold:
#                 rho = rhos[rho_idx]
#                 theta = thetas[theta_idx]
#                 lines.append((rho, theta))

#     return lines
def gussenfilter(img, size):
    img = cv2.GaussianBlur(img, (size, size), 0)
    return img

def compute_gradient(img, size=3):
    x= cv2.Sobel(img, cv2.CV_64F ,1, 0, ksize=size)
    y= cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=size)
    grad = np.sqrt(x**2 +y **2)
    angle = np.arctan(y, x)
    return grad, angle

def non_max(grad, angle):
    result = np.zeros(grad.shape,dtype="int32")
    for i in range(1, grad.shape[0]-1):
        for j in range(1, grad.shape[1]-1):
            if angle[i, j] < 0:
                angle[i, j] += np.pi
            if (angle[i, j] >= 7/8*np.pi or angle[i, j] < 1/8*np.pi):
                if grad[i, j] > grad[i, j+1] and grad[i, j] > grad[i, j-1]:
                    result[i, j] = grad[i, j]
                else:
                    result[i, j] = 0
            elif (angle[i, j] >= 1/8*np.pi and angle[i, j] < 3/8*np.pi):
                if grad[i, j] > grad[i-1, j+1] and grad[i, j] > grad[i+1, j-1]:
                    result[i, j] = grad[i, j]
                else:
                    result[i, j] = 0
            elif (angle[i, j] >= 3/8*np.pi and angle[i, j] < 5/8*np.pi):
                if grad[i, j] > grad[i-1, j] and grad[i, j] > grad[i+1, j]:
                    result[i, j] = grad[i, j]
                else:
                    result[i, j] = 0
            else:
                if grad[i, j] > grad[i-1, j-1] and grad[i, j] > grad[i+1, j+1]:
                    result[i, j] = grad[i, j]
                else:
                    result[i, j] = 0
    return result

def double_threshold(img, low, high):
    result = np.zeros(img.shape, dtype="uint8")
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i, j] > high:
                result[i, j] = 255
            elif img[i, j] < low:
                result[i, j] = 0
            else:
                result[i, j] = 128
    return result

def edge_linking_dfs(img):
    processimg = img.copy() #複製圖片

    # 初始化邊緣圖
    edge = np.zeros(img.shape, dtype="uint8")
    
    # 定義方向
    directions = [(-2, 0), (-2, 1), (-2, 2), (-2, -1), (-2, -2),
                  (-1, 0), (-1, 1), (-1, 2), (-1, -1), (-1, -2),
                  (0, -2), (0, -1), (0, 1), (0, 2),
                  (1, 0), (1, 1), (1, 2), (1, -1), (1, -2),
                  (2, 0), (2, 1), (2, 2), (2, -1), (2, -2)]
    "(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)"
    def dfs(x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if edge[cx, cy] == 0:  # 尚未訪問
                edge[cx, cy] = 255  # 標記為邊緣
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < processimg.shape[0] and 0 <= ny < processimg.shape[1]:
                        if processimg[nx, ny] == 128 and edge[nx, ny] == 0:  # 與弱邊緣相連
                            processimg[nx, ny] = 255  # 標記為強邊緣
                            stack.append((nx, ny))
    
    # 遍歷整個圖像，從強邊緣開始進行 DFS
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if processimg[i, j] == 255 and edge[i, j] == 0:  # 強邊緣且未處理
                dfs(i, j)
    
    return edge

def hough_transform(edge_image, theta_res=np.pi/180, rho_res=1, threshold=50, sample_rate=0.5, delta_theta=1, delta_rho=1):

    # 1. 初始化變數
    height, width = edge_image.shape  # 影像尺寸
    diag_len = int(np.sqrt(height**2 + width**2))  # 影像對角線長度
    rhos = np.arange(-diag_len, diag_len , rho_res)  # ρ 的範圍
    thetas = np.arange(0, np.pi, theta_res)  # θ 的範圍 (0 到 180 度)
    
    # 2. 隨機選擇邊緣點 (根據 sample_rate 控制樣本數量)
    edge_points = np.argwhere(edge_image)  # 取得所有邊緣點 (y, x)
    sampled_points_idx = np.random.choice(edge_points.shape[0],size = int(len(edge_points)*sample_rate),replace= False)  # 隨機選擇部分邊緣點
    sampled_points = edge_points[sampled_points_idx] 
    # 3. 初始化累加器 (Accumulator)
    print("samplepoint:"+str(len(sampled_points)))
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    
    # 4. 投票：對每個邊緣點，計算 θ 和 ρ，並擴展範圍進行投票
    for y, x in sampled_points:
        for theta_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))  # 計算 ρ 值
            rho_idx = np.argmin(np.abs(rhos - rho))  # 取得最接近的索引
            rho_range = np.arange(max(rho_idx - delta_rho , 0), min(rho_idx + delta_rho + 1, len(rhos)))
            theta_range = np.arange(max(theta_idx - delta_theta,0), min(theta_idx + delta_theta + 1, len(thetas)))
            # print(len(rho_range), len(theta_range))
            for r_idx in rho_range:
                for t_idx in theta_range:
                    accumulator[r_idx, t_idx] += 1
                    
    # 5. 找出符合閾值的直線
    lines = []
    for rho_idx, theta_idx in np.argwhere(accumulator >= threshold):
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        lines.append((rho, theta))
    
    return lines

def drowline(img, lines):
    print("linenumber:"+str(len(lines)))
    if lines is not None:
        for line in lines:
            rho, theta = line  # 提取 rho 和 theta
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # 計算兩個端點
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            # 在圖像上繪製直線
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)


def canny_edge(img, gussenfiltersize, compute_gradientsize,low, high):
    processimg = img.copy() #複製圖片
    imgresize = gussenfilter(processimg, gussenfiltersize)#高斯濾波
    grad, angle = compute_gradient(imgresize, compute_gradientsize)#計算梯度
    grad = non_max(grad, angle)
    grad = double_threshold(grad ,low , high)
    gradresult = edge_linking_dfs(grad)
    return gradresult, grad

def sign(input, sign_img):
    (h, w) = input.shape  # 取得圖片height 、width(黑白)
    #簽名在圖片的右下角位置
    for i in range(100):
        for j in range(200):
            if sign_img[i][j] < 50:  # 如果該位置小於50(黑色)，則將該位置像素調整為全白
                input[h - 100 + i][w - 200 + j] = 255
    return input

def signcolor(input, sign_img):
    (h, w , r) = input.shape  # 取得圖片height 、width(黑白)
    #簽名在圖片的右下角位置
    for i in range(100):
        for j in range(200):
            if sign_img[i][j] < 50:  # 如果該位置小於50(黑色)，則將該位置像素調整為全白
                for k in range(r):
                    input[h - 100 + i][w - 200 + j][k] = 0
    return input
if __name__ == "__main__":
    parameter=[[50, 150, 80],[50, 150, 50],[60, 120, 60]]#參數(low,higt,threshold)
    signimg = cv2.imread("sign.jpg", cv2.IMREAD_GRAYSCALE)
    for i in range(0, 3):
        img = cv2.imread(f"im{i+1}.jpg") #讀取圖片
        imgresize = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4)))#縮小圖片
        imgresizegray = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)#轉換成灰階圖片
        gradresult,grad = canny_edge(imgresizegray , 5, 3, parameter[i][0], parameter[i][1]) 
        grad = sign(grad, signimg)
        gradresult = sign(gradresult, signimg)
        cv2.imwrite(f"double_im{i+1}.jpg", grad)
        cv2.imwrite(f"cannyedge_im{i+1}.jpg", gradresult)
        
        lines= hough_transform(gradresult,  threshold=parameter[i][2],sample_rate=0.5, delta_theta=0, delta_rho=0)
        drowline(imgresize, lines)
        imgresize = signcolor(imgresize, signimg)
        cv2.imwrite(f"Detected_Lines_im{i+1}.jpg",imgresize)
    