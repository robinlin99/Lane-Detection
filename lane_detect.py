#!/usr/bin/env python3

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="file path for the video to be annotated",
    )

    return parser.parse_args()


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines:
        for line in lines:
            # Reshape from [[x1, y1, x2, y2]] to [x1, y1, x2, y2].
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image


def roi(img):
    mask = np.zeros_like(img)
    # Filling pixels inside the polygon defined by "vertices" with the fill color.
    pts1 = np.array([[200, img.shape[0]], [1100, img.shape[0]], [550, 300]])
    cv2.fillPoly(mask, np.int32([pts1]), (255, 255, 255))
    # Returning the image only where mask pixels are non-zero.
    masked = cv2.bitwise_and(img, mask)

    return masked


def main():
    args = parse_cli()
    cap = cv2.VideoCapture(args.file_path)

    while cap.isOpened():
        _, frame = cap.read()
        canny_img = canny(frame)
        cropped_region = roi(canny_img)
        detect_lines = cv2.HoughLinesP(
            cropped_region,
            2,
            np.pi / 180,
            10,
            np.array([]),
            minLineLength=2,
            maxLineGap=4,
        )
        line_image = display_lines(frame, detect_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("RoadDetect", combo_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
