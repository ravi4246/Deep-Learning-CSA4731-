

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread(r'C:\Users\raviv\OneDrive\Pictures\zoro.jpg')  # Replace with the actual image path

# Check if image is loaded correctly
if img is None:
    print("Error: Image not found.")
else:
    # Split the image into BGR channels
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define kernel for morphological operations (a small 3x3 kernel in this case)
    kernel = np.ones((3, 3), np.uint8)

    # Perform morphological closing (dilation followed by erosion)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Compute sure background by dilating the thresholded image
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)

    # Display the results
    plt.subplot(211), plt.imshow(closing, cmap='gray')
    plt.title("MorphologyEx: Closing"), plt.xticks([]), plt.yticks([])

    plt.subplot(212), plt.imshow(sure_bg, cmap='gray')
    plt.title("Dilation (Sure Background)"), plt.xticks([]), plt.yticks([])

    # Save the sure background image
    plt.imsave(r'dilation.png', sure_bg)

    plt.tight_layout()
    plt.show()
