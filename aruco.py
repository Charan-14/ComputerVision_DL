import numpy as np
import cv2 as cv
import cv2.aruco as aruco

cap = cv.VideoCapture(-1)

while True:
    ret, frame = cap.read()
    if ret==True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        arucoParams = aruco.DetectorParameters_create()

        corners, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)

        if len(corners)>0:
            ids = ids.flatten
            for (markerCorner, markerID) in zip(corners, ids):
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
                (rvec-tvec).any()
                aruco.drawDetectedMarkers(frame, corners)
                aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

            cv2.imshow('frame', frame)
            key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()