import cv2 as cv
import numpy as np

img = cv.imread("transition.png")

def BboxFinder(img, i):
    
    copy = img.copy()
    copy2 = img.copy()
    gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(copy, cv.COLOR_BGR2HSV)
    low = np.array([14, 145, 165]) 
    up = np.array([139, 196, 255])
    mask = cv.inRange(hsv, low, up)
    rect = cv.bitwise_and(img, img, mask=mask)
    rect = cv.cvtColor(rect, cv.COLOR_BGR2GRAY)
    rect = cv.dilate(rect, (5,5), iterations=4)
    
    _, cnts, _= cv.findContours(rect, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(cnts)>0:
        
        cnt = sorted(cnts, key = cv.contourArea, reverse=True)[i]

        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(copy2, (x,y), (x+w, y+h), (255,0,0), 2, cv.LINE_AA)
        
        return copy2, x,y,w,h

def transition(img, x,y,w,h):

    scale = 1
    delta = 0
    ddepth = cv.CV_32F
    copy3 = img.copy()
    box = img[y+5:y+h-5, x+5:x+w-5]
    boxGray = cv.cvtColor(box, cv.COLOR_BGR2GRAY)
    boxBlur = cv.bilateralFilter(boxGray, 13, 75, 75)
    grad_x = cv.Sobel(boxBlur, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    _, thr = cv.threshold(abs_grad_x, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    thr = cv.morphologyEx(thr, cv.MORPH_OPEN, (3,3), iterations=5)
    _, cn, _ = cv.findContours(thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(cn)>0:
        cont = sorted(cn, key = cv.contourArea, reverse=True)[0]
        xe,ye,we,he = cv.boundingRect(cont)
        if x<(img.shape[1]/2):
            cv.circle(box, (xe,ye), 3, (0,255,255), 4, cv.LINE_AA)
        else:
            cv.circle(box, (xe+we,ye), 3, (0,255,255), 4, cv.LINE_AA)
        cv.imshow("thr", thr)
        return xe, ye


box1, x,y,w,h = BboxFinder(img, 1)
cv.imshow("box1", box1)
box2, x2,y2,w2,h2 = BboxFinder(img, 0)
cv.imshow("box2", box2)

xe,ye = transition(img,x,y,w,h)
xe2, ye2 = transition(img, x2,y2,w2,h2)
cv.imshow("orig", img)
cv.waitKey(0)
cv.destroyAllWindows()



