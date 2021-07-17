from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit
from skinSegment import Ui_MainWindow
from mtcnn.mtcnn import MTCNN
import cv2 as cv
import numpy as np
import sys
import easycv

KNOWN_DISTANCE = 70
KNOWN_WIDTH = 20
PIXEL_WIDTH = 155
SPLIT_SIZE = 0.5  # 5 mm
focalLength = (PIXEL_WIDTH*KNOWN_DISTANCE)/KNOWN_WIDTH

detector = MTCNN()



def objectDist(knownWidth, focalLength, pixelWidth):
    return (knownWidth*focalLength)/pixelWidth


def pixelLength(knownWidth, focalLength, dist):
    return (knownWidth*focalLength)/dist


class Main(QMainWindow, Ui_MainWindow):
    skinSeg = 0
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.btnCam.clicked.connect(self.camOn)
        self.btnClose.clicked.connect(self.camClose)
        self.btnSegment.clicked.connect(self.showSegment)

    def showSegment(self):
        if self.skinSeg==0:
            self.skinSeg=1

    def updateDist(self, dist):
        self.labelValue.setText(dist)
        self.labelValue.adjustSize()

    def camOn(self):
        self.cap = cv.VideoCapture(-1)
        easycv.trackbar()
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                lh, uh, ls, us, lv, uv = easycv.trackbarPos()
                low = np.array([89, 124, lv])  # 98
                up = np.array([uh, us, 133])  # 141
                testFrame = frame.copy()
                faces = detector.detect_faces(testFrame)
                if len(faces) > 0:
                    x, y, w, h = faces[0]['box']
                    
                    conf = faces[0]['confidence']
                    if conf > 0.4:
                        dist = objectDist(KNOWN_WIDTH, focalLength, w)
                        dist = str(round(dist, 1))+" cm"
                        self.updateDist(dist)
                        cv.putText(frame, dist, (20, 40),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv.rectangle(frame, (x, y), (x+w,
                                                         y+h), (0, 255, 255), 2, cv.LINE_AA)
                        self.face = testFrame[y:y+h, x:x+w]
                        if self.face.size > 0:
                            hsv = cv.cvtColor(self.face, cv.COLOR_BGR2HSV)
                            cbcr = cv.cvtColor(self.face, cv.COLOR_BGR2YCrCb)
                            mask = cv.inRange(cbcr, low, up)
                            kernel = cv.getStructuringElement(
                                cv.MORPH_ELLIPSE, (11, 11))
                            mask = cv.erode(mask, kernel, iterations=2)
                            mask = cv.dilate(mask, kernel, iterations=2)
                            mask = cv.GaussianBlur(mask, (3, 3), 0)
                            self.skin = cv.bitwise_and(
                                self.face, self.face, mask=mask)
                            self.skin = cv.erode(
                                self.skin, (3, 3), iterations=4)
                            height, width, ch = self.skin.shape

                            self.forehead = self.skin[0:int(
                                height*0.26), 0:width]
                            self.leftCheek = self.skin[int(
                                height*0.40):int(height*0.75), 0:int(width*0.38)]
                            self.rightCheek = self.skin[int(
                                height*0.4):int(height*0.75), int(width*0.67):width]
                            self.nose = self.skin[int(
                                height*0.3):int(height*0.65), int(width*0.38):int(width*0.677)]

                            foreGray = cv.cvtColor(
                                self.forehead, cv.COLOR_BGR2GRAY)
                            leftGray = cv.cvtColor(
                                self.leftCheek, cv.COLOR_BGR2GRAY)
                            rightGray = cv.cvtColor(
                                self.rightCheek, cv.COLOR_BGR2GRAY)
                            noseGray = cv.cvtColor(
                                self.nose, cv.COLOR_BGR2GRAY)

                            _, cntsFore, _ = cv.findContours(
                                foreGray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                            _, cntsLeft, _ = cv.findContours(
                                leftGray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                            _, cntsRight, _ = cv.findContours(
                                rightGray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                            _, cntsNose, _ = cv.findContours(
                                noseGray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                            dist = objectDist(20, focalLength, w)
                            pixelWidth = pixelLength(0.5, focalLength, dist)

                            if len(cntsFore) > 0:
                                cntFore = sorted(
                                    cntsFore, key=cv.contourArea, reverse=True)[0]
                                xf, yf, wf, hf = cv.boundingRect(cntFore)
                                cv.rectangle(
                                    foreGray, (xf, yf), (xf+wf, yf+hf), (255, 0, 0), 1, cv.LINE_AA)
                                for i in range(0, wf, int(pixelWidth)):
                                    cv.line(self.forehead, (xf+i, yf),
                                            (xf+i, yf+hf), (0, 0, 255), 1)
                            if len(cntsLeft) > 0:
                                cntLeft = sorted(
                                    cntsLeft, key=cv.contourArea, reverse=True)[0]
                                xl, yl, wl, hl = cv.boundingRect(cntLeft)
                                cv.rectangle(
                                    leftGray, (xl, yl), (xl+wl, yl+hl), (255, 0, 0), 1, cv.LINE_AA)
                                for i in range(0, wl+5, int(pixelWidth)):
                                    cv.line(self.leftCheek, (xl+i, yl),
                                            (xl+i, yl+hl), (0, 255, 0), 1)
                            if len(cntsRight) > 0:
                                cntRight = sorted(
                                    cntsRight, key=cv.contourArea, reverse=True)[0]
                                xr, yr, wr, hr = cv.boundingRect(cntRight)
                                cv.rectangle(
                                    rightGray, (xr, yr), (xr+wr, yr+hr), (255, 0, 0), 1, cv.LINE_AA)
                                for i in range(5, wr, int(pixelWidth)):
                                    cv.line(self.rightCheek, (xr+i, yr),
                                            (xr+i, yr+hr), (255, 0, 0), 1)
                            if len(cntsNose) > 0:
                                cntNose = sorted(
                                    cntsNose, key=cv.contourArea, reverse=True)[0]
                                xn, yn, wn, hn = cv.boundingRect(cntNose)
                                cv.rectangle(
                                    noseGray, (xn, yn), (xn+wn, yn+hn), (255, 0, 0), 1, cv.LINE_AA)
                                for i in range(5, wn-5, int(pixelWidth)):
                                    cv.line(self.nose, (xn+i, yn),
                                            (xn+i, yn+hn), (255, 0, 255), 1)
                            if self.skinSeg==1:                
                                cv.imshow("face", self.face)
                                cv.imshow("Fore head", self.forehead)
                                cv.imshow("Left Cheek", self.leftCheek)
                                cv.imshow("Right Cheek", self.rightCheek)
                                cv.imshow("Nose", self.nose)
                                cv.imshow("Skin Only", self.skin)
                                

                cv.imshow("face detect", frame)
                cv.waitKey(1)

    def camClose(self):
        self.cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())
