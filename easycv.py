import cv2 as cv
import numpy as np

def trackbar():
    def nothing(x):
        return None

    cv.namedWindow('tuning')

    cv.createTrackbar('lowerH', 'tuning', 0 , 255, nothing)
    cv.createTrackbar('upperH', 'tuning', 255 , 255, nothing)
    cv.createTrackbar('lowerS', 'tuning', 0 , 255, nothing)
    cv.createTrackbar('upperS', 'tuning', 255 , 255, nothing)
    cv.createTrackbar('lowerV', 'tuning', 0 , 255, nothing)
    cv.createTrackbar('upperV', 'tuning', 255 , 255, nothing)


def trackbarPos():
    lh = cv.getTrackbarPos('lowerH', "tuning")
    uh = cv.getTrackbarPos('upperH', "tuning")
    ls = cv.getTrackbarPos('lowerS', "tuning")
    us = cv.getTrackbarPos('upperS', "tuning")
    lv = cv.getTrackbarPos('lowerV', "tuning")
    uv = cv.getTrackbarPos('upperV', "tuning")
    return lh, uh, ls, us, lv, uv

def sobelGrad(image, show="ok"):
    gradx = cv.Sobel(image, ddepth=cv.CV_64F, dx = 1, dy = 0, ksize = -1)
    
    grady = cv.Sobel(image, ddepth=cv.CV_64F, dx = 0, dy = 1, ksize = -1)
    
    if show=="all":
        cv.imshow("vertical", gradx)
        cv.imshow("horizontal", grady)

    grad = cv.subtract(gradx, grady)
    grad = cv.convertScaleAbs(grad)
    cv.imshow("Grad", grad)

    return grad
    
def camTest(id):
    cap = cv.VideoCapture(id)

    while cap.isOpened:
        ret, frame = cap.read()
        
        if ret:
            cv.imshow("webcam", frame)
            if cv.waitKey(1) & ord('q')==0xFF:
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()

def auto_canny(image, sigma=0.25):
	# compute the median of the single channel pixel intensities
    v = np.median(image)
	# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
	# return the edged image
    return edged

