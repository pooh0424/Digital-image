import cv2
import numpy as np

# Function to perform convolution operation
def calconv(input, m, n, kernel, bias, d):
    result = 0.0
    # Loop over the height and width of the input
    for i in range(n):
        for j in range(m):
            # Loop over depth channels of the input
            for k in range(d):
                # Perform element-wise multiplication of input and kernel
                result += input[i][j][k] * kernel[j][i][k]
    
    # Add bias to the result and clamp values between 0 and 255
    if (result + bias > 255):
        return 255
    if (result + bias < 0):
        return 0
    return int(result + bias)

# Pooling function with either average or max pooling
def pool(input, size, stride, type):
    (h, w) = input.shape  # Get height and width of the input image
    # Initialize an empty array for storing the pooled result
    result = np.empty(((h - size) // stride + 1, (w - size) // stride + 1), dtype="uint8")
    
    # Loop over the input image with the specified stride
    for i in range(0, h - 1, stride):
        for j in range(0, w - 1, stride):
            if (type == 0):  # Average pooling
                sum = 0
                # Calculate the sum of pixels in the pooling window
                for i2 in range(i, i + size):
                    for j2 in range(j, j + stride):
                        sum += input[i2, j2]
                # Store the average value in the result array
                result[i // stride, j // stride] = sum // (size * size)
            if (type == 1):  # Max pooling
                max = 0
                # Find the maximum value in the pooling window
                for i2 in range(i, i + size):
                    for j2 in range(j, j + stride):
                        if input[i2, j2] > max:
                            max = input[i2, j2]
                # Store the maximum value in the result array
                result[i // stride, j // stride] = max
    return result

# Function to perform convolution with optional padding
def conv(input, m, n, kernel, bias, padding):
    (h, w, d) = input.shape  # Get height, width, and depth of input image
    if padding == -1:  # No padding
        result = np.empty((h - n + 1, w - m + 1), dtype="uint8")
        # Perform convolution for valid pixels without padding
        for i in range(n // 2, h - n // 2):
            for j in range(m // 2, w - m // 2):
                result[i - n // 2][j - m // 2] = calconv(input[i - n // 2:i + n // 2 + 1, j - m // 2:j + m // 2 + 1], m, n, kernel, bias, d)
    else:
        # Create a new padded image with an empty array
        newpicture = np.empty((h + (n - 1), w + (m - 1), d))
        result = np.empty((h, w), dtype="uint8")
        # Set padding to 0 (black) or 255 (white)
        if padding == 0:
            newpicture[:, :, :] = 0
        elif padding == 1:
            newpicture[:, :, :] = 255
        # Place the original image in the center of the padded array
        newpicture[(n - 1) // 2:(n - 1) // 2 + h, (m - 1) // 2:(m - 1) // 2 + w] = input
        
        # Perform convolution on the padded image
        for i in range((n - 1) // 2, (n - 1) // 2 + h):
            for j in range((m - 1) // 2, (m - 1) // 2 + w):
                result[i - (n - 1) // 2][j - (m - 1) // 2] = calconv(newpicture[i - n // 2:i + n // 2 + 1, j - m // 2:j + m // 2 + 1], m, n, kernel, bias, d)
    return result

# Function to add a sign (watermark or marking) to the image
def sign(input, sign):
    (h, w) = input.shape  # Get height and width of the input image
    # Loop over part of the image to place the sign
    for i in range(200):
        for j in range(400):
            # If the sign image has dark pixels, modify the corresponding pixels in the input image
            if sign[i][j] < 50:
                input[h - 200 + i][w - 400 + j] = 255
    return input

# Main execution code
if __name__ == "__main__":
    img = list()
    # Load 3 input images
    for i in range(1, 4):
        img.append(cv2.imread(f"im{i}.jpg"))

    # Define 3 convolution kernels
    kernel1 = np.array([
        [[0.037, 0.037, 0.037], [0.037, 0.037, 0.037], [0.037, 0.037, 0.037]],
        [[0.037, 0.037, 0.037], [0.037, 0.037, 0.037], [0.037, 0.037, 0.037]],
        [[0.037, 0.037, 0.037], [0.037, 0.037, 0.037], [0.037, 0.037, 0.037]],
    ])
    
    kernel2 = np.array([
        [[-1/3, -1/3, -1/3], [0, 0, 0], [1/3, 1/3, 1/3]],
        [[-2/3, -2/3, -2/3], [0, 0, 0], [2/3, 2/3, 2/3]],
        [[-1/3, -1/3, -1/3], [0, 0, 0], [1/3, 1/3, 1/3]],
    ])
    
    kernel3 = np.array([
        [[1/48, 1/48, 1/48], [2/48, 2/48, 2/48], [1/48, 1/48, 1/48]],
        [[2/48, 2/48, 2/48], [4/48, 4/48, 4/48], [2/48, 2/48, 2/48]],
        [[1/48, 1/48, 1/48], [2/48, 2/48, 2/48], [1/48, 1/48, 1/48]],
    ])
    
    # Load a grayscale sign image to be used for marking
    signimg = cv2.imread("sign.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Apply convolution, pooling, and sign marking on each image
    for i in range(1, 4):
        # Convolution with kernel1, followed by max pooling
        result1 = conv(img[i-1], 3, 3, kernel1, 0, 0)
        resultmax = pool(result1, 2, 2, 1)
        # Apply the sign to both convolution and pooled results
        result1 = sign(result1, signimg)
        resultmax = sign(resultmax, signimg)
        # Save the results to files
        cv2.imwrite(f"im{i}blur.jpg", result1)
        cv2.imwrite(f"im{i}maxpool.jpg", resultmax)
        
        # Convolution with kernel2 (Sobel-like filter)
        result2 = conv(img[i-1], 3, 3, kernel2, 0, 0)
        result2 = sign(result2, signimg)
        cv2.imwrite(f"im{i}sobel.jpg", result2)
        
        # Convolution with kernel3 (Gaussian-like filter) and average pooling
        result3 = conv(img[i-1], 3, 3, kernel3, 0, 0)
        resultaverage = pool(result1, 2, 2, 0)
        result3 = sign(result3, signimg)
        resultaverage = sign(resultaverage, signimg)
        # Save the results
        cv2.imwrite(f"im{i}gauss.jpg", result3)
        cv2.imwrite(f"im{i}avgpool.jpg", resultaverage)
