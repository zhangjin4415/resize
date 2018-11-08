import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('img.jpg', 0)

print(img)

img = cv2.resize(img, (512,512), interpolation=cv2.INTER_CUBIC)##[np.newaxis, :, :]

print(img)