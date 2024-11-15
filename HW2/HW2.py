import cv2
import numpy as np

def calhis(img):#計算直方圖
    (h,w) = img.shape#取得圖片的高和寬
    histfrequency = np.zeros(256)#新增計算數量的陣列
    for i in range(w):
        for j in range(h):
            histfrequency[img[j,i]] += 1#在對應的位置+1
    return histfrequency

def drowhis(histfrequency,name):#畫直方圖
    hist = np.empty((512,512),dtype=np.uint8)#新增直方圖畫布
    for i in range(512):
        for j in range(512):
            if((512-j)>histfrequency[i//2]/max(histfrequency)*512):#將直方圖畫在畫布上
                hist[j,i] = 0
            else:
                hist[j,i] = 255
    cv2.imwrite(name+".jpg",hist)#儲存直方圖

def Equalization(img,histfrequency):#直方圖均衡
    (h,w) = img.shape #取得圖片的高和寬
    cor = np.empty((256),dtype=np.uint8) #新增對應的陣列
    sum =0 #新增總和
    for i in range(256):
        sum += histfrequency[i] #計算總和
        cor[i] = round(sum/(w*h)*255,0) #計算對應的值
    for i in range(w):
        for r in range(h):
            img[r,i] = cor[img[r,i]] #將對應的值填入圖片
    return img

def graypicture(img,name):#灰階圖片處理
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #轉換成灰階圖片
    histfrequency = calhis(img) #計算直方圖
    drowhis(histfrequency,name+"original") #畫直方圖
    img = Equalization(img,histfrequency)  #直方圖均衡
    histfrequency = calhis(img) #計算直方圖
    drowhis(histfrequency,name+"Equalization") #畫直方圖
    cv2.imwrite(name+"Equalizationresult"+".jpg",img)  #儲存圖片

def colorpicture(img,name):#彩色圖片處理(RGB通道)
    processimg = img.copy() #複製圖片
    (h,w,c) = processimg.shape #取得圖片的高和寬和通道
    color =["B","G","R"] #通道名稱
    for i in range(c): #對每個通道做處理
        histfrequency = calhis(processimg[:,:,i]) #計算直方圖
        drowhis(histfrequency,name+"original"+color[i]) #畫直方圖
        processimg[:,:,i] = Equalization(processimg[:,:,i],histfrequency) #直方圖均衡
        histfrequency = calhis(processimg[:,:,i]) #計算直方圖
        drowhis(histfrequency,name+"Equalization"+color[i]) #畫直方圖
    cv2.imwrite(name+"Equalizationresult"+".jpg",processimg) #儲存圖片(總和)

def hsvpicture(img,name):#彩色圖片處理(HSV通道)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #轉換成HSV圖片
    histfrequency = calhis(img[:,:,2]) #計算直方圖(V通道)
    drowhis(histfrequency,name+"original"+"v") #畫直方圖
    img[:,:,2] = Equalization(img[:,:,2],histfrequency) #直方圖均衡(V通道)
    histfrequency = calhis(img[:,:,2]) #計算直方圖
    drowhis(histfrequency,name+"Equalization"+"v") #畫直方圖(V通道)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR) #轉換成BGR圖片
    cv2.imwrite(name+"Equalizationresult"+".jpg",img) #儲存圖片(總和)
    
if __name__ == '__main__':
    for i in range(1, 3):
        img = cv2.imread(f"im{i}.jpg") #讀取圖片
        graypicture(img,f"im{i}gray") #灰階圖片處理
        colorpicture(img,f"im{i}color") #彩色圖片處理(RGB通道)
        hsvpicture(img,f"im{i}hsv") #HSV圖片處理
        