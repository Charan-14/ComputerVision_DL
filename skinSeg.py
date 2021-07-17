from mtcnn.mtcnn import MTCNN
import cv2 as cv
import numpy as np
import sys

from numpy.core.defchararray import upper
import easycv

def pixelLength(knownWidth, focalLength, dist):
    return (knownWidth*focalLength)/dist 

def objectDist(knownWidth, focalLength, pixelWidth):
    return (knownWidth*focalLength)/pixelWidth

KNOWN_DISTANCE = 70
KNOWN_WIDTH = 20
PIXEL_WIDTH = 155
focalLength = (PIXEL_WIDTH*KNOWN_DISTANCE)/KNOWN_WIDTH


detector = MTCNN()
cap =  cv.VideoCapture(-1)

easycv.trackbar()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        lh, uh, ls, us, lv, uv = easycv.trackbarPos()
        low = np.array([lh, ls, lv ]) #98
        up = np.array([uh, us, 137 ]) #141
        testFrame = frame.copy()
        faces = detector.detect_faces(testFrame)
        if len(faces)>0:
            x, y, w, h = faces[0]['box']
            
            conf = faces[0]['confidence']
            if conf>0.4:
                face = testFrame[y:y+h, x:x+w]
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2, cv.LINE_AA)
                if face.size>0:
                    hsv = cv.cvtColor(face, cv.COLOR_BGR2HSV)
                    cbcr = cv.cvtColor(face, cv.COLOR_BGR2YCrCb)
                    mask = cv.inRange(cbcr, low, up)
                    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
                    mask = cv.erode(mask, kernel, iterations = 2)
                    mask = cv.dilate(mask, kernel, iterations = 2)
                    mask = cv.GaussianBlur(mask, (3, 3), 0)
                    skin = cv.bitwise_and(face, face, mask=mask)
                    skin = cv.erode(skin, (3,3), iterations=3)
                    height, width, ch = skin.shape 
                    skinGray = cv.cvtColor(skin, cv.COLOR_BGR2GRAY)

                    forehead = skin[0:int(height*0.26), 0:width]
                    leftCheek = skin[int(height*0.40):int(height*0.75), 0:int(width*0.38)]
                    rightCheek = skin[int(height*0.4):int(height*0.75), int(width*0.67):width]
                    nose = skin[int(height*0.3):int(height*0.65), int(width*0.38):int(width*0.677)]
                    
                    foreGray = cv.cvtColor(forehead, cv.COLOR_BGR2GRAY)
                    leftGray = cv.cvtColor(leftCheek, cv.COLOR_BGR2GRAY)
                    rightGray = cv.cvtColor(rightCheek, cv.COLOR_BGR2GRAY)
                    noseGray = cv.cvtColor(nose, cv.COLOR_BGR2GRAY)
                    
                    _, cntsFore, _ = cv.findContours(foreGray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    _, cntsLeft, _ = cv.findContours(leftGray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    _, cntsRight, _ = cv.findContours(rightGray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    _, cntsNose, _ = cv.findContours(noseGray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    

                    dist = objectDist(20, focalLength, w)
                    pixelWidth = pixelLength(0.5, focalLength, dist)

                    if len(cntsFore)>0:
                        cntFore = sorted(cntsFore, key=cv.contourArea, reverse=True)[0]
                        xf,yf,wf,hf = cv.boundingRect(cntFore)
                        cv.rectangle(foreGray, (xf,yf), (xf+wf, yf+hf), (255, 0,0), 1, cv.LINE_AA)
                        for i in range(0,wf,int(pixelWidth)):
                            cv.line(forehead, (xf+i,yf), (xf+i, yf+hf), (0,0,255), 1)
                    if len(cntsLeft)>0:
                        cntLeft = sorted(cntsLeft, key=cv.contourArea, reverse=True)[0]
                        xl,yl,wl,hl = cv.boundingRect(cntLeft)
                        cv.rectangle(leftGray, (xl,yl), (xl+wl, yl+hl), (255, 0,0), 1, cv.LINE_AA)
                        for i in range(0,wl+5,int(pixelWidth)):
                            cv.line(leftCheek, (xl+i,yl), (xl+i, yl+hl), (0,255,0), 1)
                    if len(cntsRight)>0:
                        cntRight = sorted(cntsRight, key=cv.contourArea, reverse=True)[0]
                        xr,yr,wr,hr = cv.boundingRect(cntRight)
                        cv.rectangle(rightGray, (xr,yr), (xr+wr, yr+hr), (255, 0,0), 1, cv.LINE_AA)
                        for i in range(5,wr,int(pixelWidth)):
                            cv.line(rightCheek, (xr+i,yr), (xr+i, yr+hr), (255,0,0), 1)
                    if len(cntsNose)>0:
                        cntNose = sorted(cntsNose, key=cv.contourArea, reverse=True)[0]
                        xn,yn,wn,hn = cv.boundingRect(cntNose)
                        cv.rectangle(noseGray, (xn,yn), (xn+wn, yn+hn), (255,0,0), 1, cv.LINE_AA)
                        for i in range(5,wn-5,int(pixelWidth)):
                            cv.line(nose, (xn+i,yn), (xn+i, yn+hn), (255,0,255), 1)

                    cv.imshow("face", face)
                    cv.imshow("Forehead", foreGray)
                    cv.imshow("LeftCheek", leftGray)
                    cv.imshow("RightCheek", rightGray)
                    cv.imshow("Nose", noseGray)
                    cv.imshow("Fore head", forehead)
                    cv.imshow("Left Cheek", leftCheek)
                    cv.imshow("Right Cheek", rightCheek)
                    cv.imshow("Nose", nose)
                    cv.imshow("Skin Only", skin)

        cv.imshow("face detect", frame)
        if cv.waitKey(1) & 0xFF==ord('q'):
            break

cap.release()
cv.destroyAllWindows()
