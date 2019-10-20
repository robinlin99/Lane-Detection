import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # reshape from [[x1,y1,x2,y2]] to [x1,y1,x2,y2]
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0), 10)
    return line_image

def roi(img):
    # blank mask:
    mask = np.zeros_like(img)
    # filling pixels inside the polygon defined by "vertices" with the fill color
    pts1 = np.array([[200,img.shape[0]],[1100,img.shape[0]],[550,300]])
    pts2 = np.array([[200,400],[1100,400],[550,200]])
    cv2.fillPoly(mask, np.int32([pts1]), (255,255,255))
    # returning the image only where mask pixels are nonzero
    masked = cv2.bitwise_and(img, mask)
    return masked

# read test image called "road.jpg"
#image = cv2.imread("road.jpg")
#lane_image = np.copy(image)
#canny_img = canny(image)
#pts = np.array([[3,270],[460,270],[260,145]])
#cropped_region = roi(canny_img,np.int32([pts]))
# detect lines
#detect_lines = cv2.HoughLinesP(cropped_region,2,np.pi/180,10,np.array([]),minLineLength = 2,maxLineGap=4)
# draw the lines
#line_image = display_lines(lane_image,detect_lines)
#combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)
#cv2.imshow('results',combo_image)
#cv2.waitKey(0)



test1 = "test.mp4"
test2 = "test1.mov"
cap = cv2.VideoCapture(test1)
while(cap.isOpened()):
    _, frame = cap.read()
    canny_img = canny(frame)
    cropped_region = roi(canny_img)
    detect_lines = cv2.HoughLinesP(cropped_region,2,np.pi/180,10,np.array([]),minLineLength = 2,maxLineGap=4)
    line_image = display_lines(frame,detect_lines)
    combo_image = cv2.addWeighted(frame,0.8,line_image,1,1)
    cv2.imshow('RoadDetect',combo_image)
    cv2.waitKey(1)
