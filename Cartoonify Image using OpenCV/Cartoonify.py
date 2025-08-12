import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load image
def read_file(filename):
    img = cv2.imread(filename)  # Load the image
    if img is None:  # Check if the image is loaded
        raise FileNotFoundError(f"Image file '{filename}' not found. Please check the path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img


filename = "image.jpeg"
img = read_file(filename)  # Load the image

org_img = np.copy(img)  # Copy the original image for backup


# Create Edge Mask
def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        line_size,
        blur_value,
    )
    return edges


line_size, blur_value = 7, 7
edges = edge_mask(img, line_size, blur_value)


# Reducing the color palette
def color_quantization(img, k):
    # Transforming the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-means
    _, label, center = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    center = np.uint8(center)

    result = center[label.flatten()]
    result = result.reshape(img.shape)

    return result


k = 9
img_quant = color_quantization(img, k)

# Reduce noise in the quantized image
blurred = cv2.bilateralFilter(img_quant, d=9, sigmaColor=250, sigmaSpace=250)


# Combine Edge Mask with Quantized Image
def cartoonify(img, edges):
    # Ensure edges have three channels to match the image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Combine edges with the blurred image
    cartoon = cv2.bitwise_and(img, edges_colored)
    return cartoon


cartoon = cartoonify(blurred, edges)

# Display results
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(org_img)
plt.title("Original Image")
plt.axis("off")

# Cartoonified image
plt.subplot(1, 2, 2)
plt.imshow(cartoon)
plt.title("Cartoonify Image")
plt.axis("off")

plt.tight_layout()
plt.show()
