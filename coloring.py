from PIL import Image
import cv2
import numpy as np


def change_purple_to_red_ai(image_path, output_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for purple color in HSV
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([150, 255, 255])

    # Create a mask for purple color
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Change purple to red
    image[mask > 0] = [0, 0, 255]

    cv2.imwrite(output_path, image)


def change_green_to_blue(image_path, output_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Change green to blue
    image[mask > 0] = [255, 0, 0]

    cv2.imwrite(output_path, image)

# Example usage
change_green_to_blue('ex.jpg', 'ex_blue.jpg')
change_purple_to_red_ai('ex_blue.jpg', 'ex_red.jpg')

