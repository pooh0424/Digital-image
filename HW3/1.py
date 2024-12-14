import cv2
import numpy as np
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
    result = np.zeros(img.shape)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i, j] > high:
                result[i, j] = 255
            elif img[i, j] < low:
                result[i, j] = 0
            else:
                if img[i-1, j-1] > high or img[i-1, j] > high or img[i-1, j+1] > high or img[i, j-1] > high or img[i, j+1] > high or img[i+1, j-1] > high or img[i+1, j] > high or img[i+1, j+1] > high:
                    result[i, j] = 255
                else:
                    result[i, j] = 0
    return result
if __name__ == "__main__":
    img = cv2.imread("test.png", 0)
    img = gussenfilter(img, 3)
    grad, angle = compute_gradient(img)
    grad2 = non_max(grad, angle)
    cv2.imshow("result", grad)
    cv2.imshow("result2", grad2)
    cv2.waitKey(0)
    print(grad2)
    print(angle)
    