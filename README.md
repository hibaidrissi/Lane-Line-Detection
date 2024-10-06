# Lane-Line-Detection

This project implements lane line detection using Hough Transform on road images or videos. It uses edge detection and Hough Transform to identify lane lines, useful for self-driving car applications.

#### Prerequisites

Python 3.10
OpenCV
NumPy

#### How It Works

1. Convert the input image to grayscale.
2. Apply Hough Line Transform to detect straight lines representing lane lines.
3. Overlay detected lines onto the original image.

#### Usage

Run the script on an image or video:

python lineDetection.py path_to_image_or_video

![Lane Detection Output](./home/hp/Images/lanedetection.png)
