import cv2
import numpy as np
import matplotlib.pyplot as plt
# Create a black image
image = np.zeros((300, 400), dtype=np.uint8)
# Draw white objects
cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
cv2.circle(image, (280, 120), 60, 255, -1)
# Add small noise
noise = np.random.randint(0, 50, (300, 400), dtype=np.uint8)
noisy_image = cv2.add(image, noise)
# Thresholding
_, threshold = cv2.threshold(
    noisy_image,
    120,
    255,
    cv2.THRESH_BINARY
)
# Morphological kernel
kernel = np.ones((5, 5), np.uint8)
# Opening removes small noise
opening = cv2.morphologyEx(
    threshold,
    cv2.MORPH_OPEN,
    kernel
)
# Closing fills small gaps
closing = cv2.morphologyEx(
    opening,
    cv2.MORPH_CLOSE,
    kernel
)
# Display results
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(threshold, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(closing, cmap="gray")
plt.title("Morphological Result")
plt.axis("off")
plt.tight_layout()
plt.show()
