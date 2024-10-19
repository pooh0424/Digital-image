import cv2
import numpy as np
# Function to perform convolution operation
def calconv(input,m,n,kernel,bias,d):
    result = 0.0
    for i in range(n):
        for j in range(m):
        # Loop over height and width
                for k in range(d):
                # Loop over depth channels 
                    result += input[i][j][k]*kernel[j][i][k]# Perform element-wise multiplication of input and kernel
    # Add bias to the result and clamp values between 0 and 255
    if(result+bias>255):
        return 255
    if(result+bias<0):
        return 0
    return int(result+bias)

# Pooling function with either average or max pooling
def pool(input,size,stride,type):
    (h,w,d) = input.shape# Get height and width of the input image
    result = np.empty(((h-size)//stride+1,(w-size)//stride+1),dtype="uint8")
    
    # Loop over the input image with the specified stride
    for i in range(0,h-1,stride):
        for j in range(0,w-1,stride):
            if(type == 0):
                sum = 0
                for i2 in range(i,i+size):
                    for j2 in range(j,j+stride):
                        for c in range(d):
                            sum += input[i2,j2,c]
                result[i//stride,j//stride] = sum//(size*size*d)
            if(type == 1):
                max=0
                for i2 in range(i,i+size):
                    for j2 in range(j,j+stride):
                        number = 0
                        for c in range(d):
                            number += input[i2,j2,c]
                        if number>max:
                            max = number
                result[i//stride,j//stride] = max//d
    return result
 

def conv(input,m,n,kernel,bias,padding):
    (h,w,d)=input.shape
    if padding==-1:
        result = np.empty((h-n+1,w-m+1),dtype="uint8")
        for i in range(n//2,h-n//2):
            for j in range (m//2,w-m//2):
                result[i-n//2][j-m//2]=calconv(input[i-n//2:i+n//2+1,j-m//2:j+m//2+1],m,n,kernel,bias,d)
    else:
        newpicture = np.empty((h+(n-1),w+(m-1),d))
        result = np.empty((h,w),dtype="uint8")
        if padding==0:
            newpicture[:,:,:]=0
        elif padding==1:
            newpicture[:,:,:]=255
        newpicture[(n-1)//2:(n-1)//2+h,(m-1)//2:(m-1)//2+w]=input
        for i in range((n-1)//2,(n-1)//2+h):
            for j in range ((m-1)//2,(m-1)//2+w):
                result[i-(n-1)//2][j-(m-1)//2]=calconv(newpicture[i-n//2:i+n//2+1,j-m//2:j+m//2+1],m,n,kernel,bias,d)
    return result

def sign(input,sign):
    (h,w)=input.shape
    for i in range (200):
        for j in range (400):
            if sign[i][j]<50:
                input[h-200+i][w-400+j]=255
    return input
    
if __name__ == "__main__":
    img=list()
    for i in range (1,4):
        img.append(cv2.imread(f"im{i}.jpg"))

    kernel1 = np.array([
        [[0.037037037037037,0.037037037037037,0.037037037037037],[0.037037037037037,0.037037037037037,0.037037037037037],[0.037037037037037,0.037037037037037,0.037037037037037]],
        [[0.037037037037037,0.037037037037037,0.037037037037037],[0.037037037037037,0.037037037037037,0.037037037037037],[0.037037037037037,0.037037037037037,0.037037037037037]],
        [[0.037037037037037,0.037037037037037,0.037037037037037],[0.037037037037037,0.037037037037037,0.037037037037037],[0.037037037037037,0.037037037037037,0.037037037037037]],
    ])
    kernel2 = np.array([
        [[-1/3,-1/3,-1/3],[0,0,0],[1/3,1/3,1/3]],
        [[-2/3,-2/3,-2/3],[0,0,0],[2/3,2/3,2/3]],
        [[-1/3,-1/3,-1/3],[0,0,0],[1/3,1/3,1/3]],
    ])
    kernel3 = np.array([
        [[1/48,1/48,1/48],[2/48,2/48,2/48],[1/48,1/48,1/48]],
        [[2/48,2/48,2/48],[4/48,4/48,4/48],[2/48,2/48,2/48]],
        [[1/48,1/48,1/48],[2/48,2/48,2/48],[1/48,1/48,1/48]],
    ])
    signimg = cv2.imread("sign.jpg",cv2.IMREAD_GRAYSCALE)
    for i in range (1,4):
        result1 = conv(img[i-1],3,3,kernel1,0,0)
        result1 = sign(result1,signimg)
        cv2.imwrite(f"im{i}blur.jpg",result1)

        result2 = conv(img[i-1],3,3,kernel2,0,0)
        result2 = sign(result2,signimg)
        cv2.imwrite(f"im{i}sobel.jpg",result2)

        result3 = conv(img[i-1],3,3,kernel3,0,0)
        result3 = sign(result3,signimg)
        cv2.imwrite(f"im{i}gauss.jpg",result3)

        resultmax = pool(img[i-1],2,2,1)
        resultmax = sign(resultmax,signimg)
        cv2.imwrite(f"im{i}maxpool.jpg",resultmax)

        resultaverage = pool(img[i-1],2,2,0)
        resultaverage = sign(resultaverage,signimg)
        cv2.imwrite(f"im{i}avgpool.jpg",resultaverage)
    # result2 = cv2.Sobel(result2,-1,0,1)
    # check = result[100:500,100:1000]-result2[100:500,100:1000]
    # print(check)
    # result2 = pool(result,2,2,1)
    # cv2.imshow("result",result)
    # cv2.imshow("result2",result2)
    # cv2.waitKey(0)