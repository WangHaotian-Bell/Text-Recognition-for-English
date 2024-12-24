#Student name: Wang Haotian
#Student ID: 3160423

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import time

# Load the image
img_path = r"C:\Users\Travis\Pictures\12.png"
img = cv2.imread(img_path)
start_time = time.time()
text = pytesseract.image_to_string(img)
print(text)

# Display the original image
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 1: Convert image to grayscale
step1_start_time = time.time()
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform OCR on the grayscale image
custom_config = r'--oem 3 --psm 6'
text_gray = pytesseract.image_to_string(gray_img, config=custom_config)
step1_end_time = time.time()
print("Text after converting to grayscale:")
print(text_gray)
print(f"Time for grayscale: {step1_end_time - step1_start_time:.4f} seconds")

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 2: Apply thresholding
step2_start_time = time.time()
_, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Perform OCR on the thresholded image
text_thresh = pytesseract.image_to_string(thresh_img, config=custom_config)
step2_end_time = time.time()
print("Text after thresholding:")
print(text_thresh)
print(f"Time for Threshold: {step2_end_time - step2_start_time:.4f} seconds")

# Display the thresholded image
cv2.imshow('Thresholded Image', thresh_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3: Apply opening morphological operation
step3_start_time = time.time()
kernel = np.ones((5, 5), np.uint8)
opening_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    
# Perform OCR on the opening image
text_opening = pytesseract.image_to_string(opening_img, config=custom_config)
step3_end_time = time.time()
print("Text after opening morphological operation:")
print(text_opening)
print(f"Time for opening: {step3_end_time - step3_start_time:.4f} seconds")

# Display the opening image
cv2.imshow('Opening Image', opening_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 4: Apply Canny edge detection
step4_start_time = time.time()
canny_img = cv2.Canny(gray_img, 100, 200)

# Perform OCR on the Canny edge image
text_canny = pytesseract.image_to_string(canny_img, config=custom_config)
step4_end_time = time.time()
print("Text after Canny edge detection:")
print(text_canny)
print(f"Time for Canny: {step4_end_time - step4_start_time:.4f} seconds")

# Display the Canny edge image
cv2.imshow('Canny Edge Image', canny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the image
img_path = r"C:\Users\Travis\Pictures\12.png"
img = cv2.imread(img_path)

# Get image dimensions
h, w, c = img.shape

# Perform OCR to get bounding boxes
d = pytesseract.image_to_data(img, output_type=Output.DICT)

# Draw bounding boxes around the detected text
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 20:
        x, y, width, height = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        img = cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

# Display the image with text boxes
cv2.imshow('Image with Text Boxes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract and print the text within the detected boxes
for i in range(n_boxes):
    if int(d['conf'][i]) > 20:
        print("Text in box {}: {}".format(i, d['text'][i]))


end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")