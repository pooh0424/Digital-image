import cv2
import numpy as np

class unoin:
    def __init__(self, M, N):
        self.parent = np.zeros((M, N, 2), dtype=int)
        for i in range(M):
            for j in range(N):
                self.parent[i,j,0] = i
                self.parent[i,j,1] = j
        self.size = np.zeros((M, N))
    def find(self, i, j ,img):
        if self.parent[i ,j][0] != i or self.parent[i, j][1] != j:
            root = self.find(self.parent[i, j][0], self.parent[i, j][1],img)
            if img[root[0],root[1]] == 2:
                self.parent[i, j] = root
        return self.parent[i, j]
    def union(self, x, y, x1, y1, img):
        root1 = self.find(x, y, img)
        root2 = self.find(x1, y1, img)
        if x == 1041 and y == 2101:
            print(root1,root2)
            print(x1,y1)
        if (root1 != root2).any():
            if img[root1[0], root1[1]] == 2 and img[root2[0], root2[1]] != 2:
                self.parent[root2[0], root2[1]] = root1
                self.size[root1[0], root1[1]] += self.size[root2[0], root2[1]]
            elif img[root2[0], root2[1]] == 2 and img[root1[0], root1[1]] != 2:
                self.parent[root1[0], root1[1]] = root2
                self.size[root2[0], root2[1]] += self.size[root1[0], root1[1]]
            else:
                if self.size[root1[0], root1[1]] < self.size[root2[0], root2[1]]:
                    self.parent[root1[0], root1[1]] = root2
                    self.size[root2[0], root2[1]] += self.size[root1[0], root1[1]]
                else:
                    self.parent[root2[0], root2[1]] = root1
                    self.size[root1[0], root1[1]] += self.size[root2[0], root2[1]]

def gussenfilter(img, size):
    img = cv2.GaussianBlur(img, (size, size), 0)
    return img

def compute_gradient(img):
    x= cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    y= cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = (x**2 +y **2)**0.5
    angle = np.arctan(y, x)
    return grad, angle

def non_max(grad, angle):
    result = np.zeros(grad.shape)
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
                result[i, j] = 2
            elif img[i, j] < low:
                result[i, j] = 0
            else:
                result[i, j] = 1
    return result

def edge_linking(img):
    edge = np.zeros(img.shape,dtype="uint8")
    myunoin = unoin(img.shape[0], img.shape[1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 1 or img[i, j] == 2:
                if i-1 >= 0 and (img[i-1, j] == 1 or img[i-1, j] == 2):
                    myunoin.union(i, j, i-1, j, img)
                if j-1 >= 0 and (img[i, j-1] == 1 or img[i, j-1] == 2):
                    myunoin.union(i, j, i, j-1, img)
                if i-1 >= 0 and j-1 >= 0 and (img[i-1, j-1] == 1 or img[i-1, j-1] == 2):
                    myunoin.union(i, j, i-1, j-1, img)
                if i-1 >= 0 and j+1 < img.shape[1] and (img[i-1, j+1] == 1 or img[i-1, j+1] == 2):
                    myunoin.union(i, j, i-1, j+1, img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 1 or img[i, j] == 2:
                if i-1 >= 0 and (img[i-1, j] == 1 or img[i-1, j] == 2):
                    myunoin.union(i, j, i-1, j, img)
                if j-1 >= 0 and (img[i, j-1] == 1 or img[i, j-1] == 2):
                    myunoin.union(i, j, i, j-1, img)
                if i-1 >= 0 and j-1 >= 0 and (img[i-1, j-1] == 1 or img[i-1, j-1] == 2):
                    myunoin.union(i, j, i-1, j-1, img)
                if i-1 >= 0 and j+1 < img.shape[1] and (img[i-1, j+1] == 1 or img[i-1, j+1] == 2):
                    myunoin.union(i, j, i-1, j+1, img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 1:
                root = myunoin.find(i, j,img)
                if img[root[0], root[1]] == 2:
                    edge[i, j] = 255
                    # print("edge")
                else:
                    edge[i, j] = 0
            elif img[i, j] == 2:
                edge[i, j] = 255
    return edge

def hough_transform(edge_image, theta_res=1, rho_res=1, threshold=50, sample_rate=0.5, delta_theta=0.5, delta_rho=1):

    # 1. 初始化變數
    height, width = edge_image.shape  # 影像尺寸
    diag_len = int(np.sqrt(height**2 + width**2))  # 影像對角線長度
    rhos = np.arange(-diag_len, diag_len + 1, rho_res)  # ρ 的範圍
    thetas = np.deg2rad(np.arange(0, 180, theta_res))  # θ 的範圍 (0 到 180 度)
    
    # 2. 隨機選擇邊緣點 (根據 sample_rate 控制樣本數量)
    edge_points = np.argwhere(edge_image)  # 取得所有邊緣點 (y, x)
    sampled_points = edge_points[np.random.rand(len(edge_points)) < sample_rate]  # 隨機選擇部分邊緣點
    
    # 3. 初始化累加器 (Accumulator)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    
    # 4. 投票：對每個邊緣點，計算 θ 和 ρ，並擴展範圍進行投票
    for y, x in sampled_points:
        for theta_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))  # 計算 ρ 值
            # 扩展范围，允许投票给附近的 (ρ, θ)
            rho_range = range(max(0, rho - delta_rho), min(len(rhos), rho + delta_rho + 1))
            theta_range = range(max(0, theta_idx - int(delta_theta / theta_res)), min(len(thetas), theta_idx + int(delta_theta / theta_res) + 1))
            # 對周圍的 (ρ, θ) 投票
            for r_idx in rho_range:
                for t_idx in theta_range:
                    accumulator[r_idx, t_idx] += 1

    # 5. 找出符合閾值的直線
    lines = []
    for rho_idx, theta_idx in np.argwhere(accumulator >= threshold):
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        lines.append((rho, theta))
    
    return lines, accumulator, rhos, thetas

if __name__ == "__main__":
    img = cv2.imread("test.jpg", 0)
    # img = gussenfilter(img, 3)
    # grad, angle = compute_gradient(img)      
    # grad2 = non_max(grad, angle)
    # grad3 = double_threshold(grad2, 150, 300)
    # grad3 = edge_linking(grad3)


    # lines, accumulator, rhos, thetas = hough_transform(grad3, theta_res=1, rho_res=1, threshold=100, sample_rate=0.5, delta_theta=0.5, delta_rho=1)
    shimg = np.zeros((img.shape[0], img.shape[1],3), dtype="uint8")
    # 繪製直線
    # for rho, theta in lines:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     print(x0, y0)
    #     # 根據圖片大小計算直線的端點
    #     length = max(img.shape[0], img.shape[1])
    #     x1 = int(x0 + length * (-b))
    #     y1 = int(y0 + length * (a))
    #     x2 = int(x0 - length * (-b))
    #     y2 = int(y0 - length * (a))

    #     print(x1, y1, x2, y2)
    #     cv2.line(shimg, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 顯示結果
    cv2.line(shimg, (-1749, 1786), (1144,-2258), (255, 0, 0), 5)
    
    cv2.imwrite("1.jpg", shimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imshow("result2", grad3)
    # cv2.waitKey(0)
    # print(grad2)
    