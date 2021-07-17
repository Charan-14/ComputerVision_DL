import cv2 as cv
import numpy as np

def parameterTuning():
    
    def nothing(x):
        return None

    cv.namedWindow('Parameter Tuning')

    cv.createTrackbar('lt', 'Parameter Tuning', 0 , 255, nothing)
    cv.createTrackbar('ut', 'Parameter Tuning', 0 , 600, nothing)
    cv.createTrackbar('lr', 'Parameter Tuning', 0 , 255, nothing)
    cv.createTrackbar('lg', 'Parameter Tuning', 115, 255, nothing)
    cv.createTrackbar('lb', 'Parameter Tuning', 0 , 255, nothing)
    cv.createTrackbar('ur', 'Parameter Tuning', 56, 255, nothing)
    cv.createTrackbar('ug', 'Parameter Tuning', 255 , 255, nothing)
    cv.createTrackbar('ub', 'Parameter Tuning', 255 , 255, nothing)
    #cv.createTrackbar('kernel', 'Parameter Tuning', 1 , 25, nothing)

def trackbarPos():
    global lt
    global ut
    global lr
    global lg
    global lb
    global ur
    global ug
    global ub
    #Position of trackbar is stored
    lt = cv.getTrackbarPos('lt', 'Parameter Tuning')
    ut = cv.getTrackbarPos('ut', 'Parameter Tuning')   
    lr = cv.getTrackbarPos('lr', 'Parameter Tuning')   
    lg = cv.getTrackbarPos('lg', 'Parameter Tuning')   
    lb = cv.getTrackbarPos('lb', 'Parameter Tuning')   
    ur = cv.getTrackbarPos('ur', 'Parameter Tuning')   
    ug = cv.getTrackbarPos('ug', 'Parameter Tuning')   
    ub = cv.getTrackbarPos('ub', 'Parameter Tuning')   
    
#ker = cv.getTrackbarPos('kernel', 'Parameter Tuning')   
parameterTuning()
img = cv.imread("lemons1.jpeg")

while True:
    trackbarPos()
    hls = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_hls = np.array([lr, lg, lb])#lower_y = np.array([8, 122, 115])
    upper_hls = np.array([ur, ug, ub])#upper_y = np.array([47, 172, 151])
    mask_hls = cv.inRange(hls, lower_hls, upper_hls)
    res_hls = cv.bitwise_and(img, img, mask=mask_hls)
    res_hls = cv.cvtColor(res_hls, cv.COLOR_BGR2GRAY)
    _, thr = cv.threshold(res_hls, 0, 255, cv.THRESH_BINARY)

    cv.imshow("hsv", thr)
    cv.imshow("lemons", img)
    cv.waitKey(0)

cv.destroyAllWindows()
