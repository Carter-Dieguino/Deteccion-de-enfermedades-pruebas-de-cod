import cv2
import numpy as np

def remove_blurry_background(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Dilate the edges to make them more pronounced
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)

##    dilation = cv2.dilate(edges, kernel, iterations=1)

    # Create a mask from the edges
    mask = cv2.threshold(dilation, 55, 255, cv2.THRESH_BINARY)[1]

    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Save or display the result
    cv2.imshow('Original', img)
##    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'path_to_image.jpg' with the path to your image file
remove_blurry_background('photo5.jpg')
