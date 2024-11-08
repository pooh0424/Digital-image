import cv2
import numpy as np

def calhis(img):#計算直方圖
    (w,h) = img.shape#取得圖片的寬和高
    histfrequency = np.zeros(256)#新增計算數量的陣列
    for i in range(w):
        for j in range(h):
            histfrequency[img[i,j]] += 1#在對應的位置+1
    return histfrequency

def drowhis(histfrequency,name):#畫直方圖
    hist = np.empty((512,512),dtype=np.uint8)#新增直方圖畫布
    for i in range(512):
        for j in range(512):
            if((512-j)>histfrequency[i//2]/max(histfrequency)*512):#將直方圖畫在畫布上
                hist[j,i] = 0
            else:
                hist[j,i] = 255
    cv2.imwrite(name+".jpg",hist)

def Equalization(img,histfrequency):#直方圖均衡
    (w,h) = img.shape #取得圖片的寬和高
    cor = np.empty((256),dtype=np.uint8) #新增對應的陣列
    sum =0 #新增總和
    for i in range(256):
        sum += histfrequency[i] #計算總和
        cor[i] = round(sum/(w*h)*255,0) #計算對應的值
    for i in range(w):
        for r in range(h):
            img[i,r] = cor[img[i,r]] #將對應的值填入圖片
    return img

def graypicture(img,name):#灰階圖片處理
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #轉換成灰階圖片
    histfrequency = calhis(img) #計算直方圖
    drowhis(histfrequency,name+"original") #畫直方圖
    img = Equalization(img,histfrequency)  #直方圖均衡
    histfrequency = calhis(img) #計算直方圖
    drowhis(histfrequency,name+"Equalization") #畫直方圖
    cv2.imwrite(name+"Equalizationresult"+".jpg",img)  #儲存圖片

def colorpicture(img,name):#彩色圖片處理
    (w,h,c) = img.shape #取得圖片的寬和高和通道
    color =["B","G","R"] #通道名稱
    for i in range(c): #對每個通道做處理
        histfrequency = calhis(img[:,:,i]) #計算直方圖
        drowhis(histfrequency,name+"original"+color[i]) #畫直方圖
        img[:,:,i] = Equalization(img[:,:,i],histfrequency) #直方圖均衡
        histfrequency = calhis(img[:,:,i]) #計算直方圖
        drowhis(histfrequency,name+"Equalization"+color[i]) #畫直方圖
    cv2.imwrite(name+"Equalizationresult"+".jpg",img) #儲存圖片(總和)

if __name__ == '__main__':
    for i in range(1, 3):
        img = cv2.imread(f"im{i}.jpg") #讀取圖片
        graypicture(img,f"im{i}gray") #灰階圖片處理
        colorpicture(img,f"im{i}color") #彩色圖片處理