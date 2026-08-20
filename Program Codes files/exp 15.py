import cv2
import numpy as np
import matplotlib.pyplot as plt
# Create a sample image
image = np.zeros((300, 400, 3), dtype=np.uint8)
# Add different colored regions
image[0:150, 0:200] = [255, 0, 0]
image[0:150, 200:400] = [0, 255, 0]
image[150:300, 0:200] = [0, 0, 255]
image[150:300, 200:400] = [255, 255, 0]
# Convert image to pixel list
pixels = image.reshape((-1, 3))
pixels = np.float32(pixels)
# Define criteria
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    0.2
)
# Number of clusters
K = 4
# K-Means clustering
_, labels, centers = cv2.kmeans(
    pixels,
    K,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)
# Convert centers to integer
centers = np.uint8(centers)
# Replace pixels with cluster centers
segmented = centers[labels.flatten()]
# Reshape into image
segmented = segmented.reshape(image.shape)
# Display
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.title("K-Means Segmentation")
plt.axis("off")
plt.show()
