import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot_histogram_and_detect_disease(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found")
        return

    # Convert image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Calculate histograms for the RGB channels
    colors = ('r', 'g', 'b')
    channel_data = {}
    peaks = []
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
        peak = np.argmax(hist)
        peaks.append(peak)
        channel_data[color] = hist
        plt.plot(hist, color=color)
        plt.plot(peak, hist[peak], "o", label=f'{color} peak at {peak}')
        plt.xlim([0, 256])

    # Calculate a health index
    # This is a simplistic example, where we assume that an unusually high red peak might indicate a problem
    health_index = peaks[0] * 0.5 + peaks[1] * 0.3 + peaks[2] * 0.2
    health_threshold = 120  # Set based on empirical data or expert advice for leaf health

    # Display health assessment
    print(f"Health Index: {health_index:.2f}")
    if health_index > health_threshold:
        print("Possible disease detected")
    else:
        print("Leaves appear healthy")

    plt.title('Color Histogram with Peaks')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Replace 'path_to_your_image.jpg' with the path to your image file
plot_histogram_and_detect_disease('photo2.jpg')
