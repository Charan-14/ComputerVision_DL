# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'skinSegment.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnCam = QtWidgets.QPushButton(self.centralwidget)
        self.btnCam.setGeometry(QtCore.QRect(90, 80, 221, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnCam.sizePolicy().hasHeightForWidth())
        self.btnCam.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Sans")
        font.setPointSize(14)
        self.btnCam.setFont(font)
        self.btnCam.setStyleSheet("QPushButton{\n"
"    \n"
"    background-color: rgb(181, 205, 77);\n"
"    border-radius:20px;\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QPushButton:hover{    \n"
"    background-color: rgb(252, 175, 62);\n"
"}\n"
"\n"
"QPushButton:hover:pressed{\n"
"    background-color: rgb(206, 92, 0);\n"
"}")
        self.btnCam.setObjectName("btnCam")
        self.btnSegment = QtWidgets.QPushButton(self.centralwidget)
        self.btnSegment.setGeometry(QtCore.QRect(90, 270, 221, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnSegment.sizePolicy().hasHeightForWidth())
        self.btnSegment.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Sans")
        font.setPointSize(14)
        self.btnSegment.setFont(font)
        self.btnSegment.setStyleSheet("QPushButton{\n"
"    \n"
"    \n"
"    background-color: rgb(233, 185, 110);\n"
"    border-radius:20px;\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QPushButton:hover{    \n"
"    \n"
"    background-color: rgb(228, 226, 168);\n"
"    \n"
"}\n"
"\n"
"QPushButton:hover:pressed{\n"
"    background-color: rgb(206, 92, 0);\n"
"    background-color: rgb(244, 243, 184);\n"
"}")
        self.btnSegment.setObjectName("btnSegment")
        self.labelDist = QtWidgets.QLabel(self.centralwidget)
        self.labelDist.setGeometry(QtCore.QRect(530, 10, 231, 51))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelDist.setFont(font)
        self.labelDist.setObjectName("labelDist")
        self.labelValue = QtWidgets.QLabel(self.centralwidget)
        self.labelValue.setGeometry(QtCore.QRect(600, 60, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(26)
        self.labelValue.setFont(font)
        self.labelValue.setObjectName("labelValue")
        self.btnClose = QtWidgets.QPushButton(self.centralwidget)
        self.btnClose.setGeometry(QtCore.QRect(120, 170, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Sans")
        font.setPointSize(14)
        self.btnClose.setFont(font)
        self.btnClose.setStyleSheet("QPushButton{\n"
"    \n"
"    \n"
"    background-color: rgb(239, 41, 41);\n"
"    border-radius:20px;\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"QPushButton:hover{    \n"
"    \n"
"    background-color: rgb(194, 105, 105);\n"
"}\n"
"\n"
"QPushButton:hover:pressed{\n"
"    \n"
"    background-color: rgb(164, 0, 0);\n"
"}")
        self.btnClose.setObjectName("btnClose")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnCam.setText(_translate("MainWindow", "Start Camera"))
        self.btnSegment.setText(_translate("MainWindow", "Segment Skin Region"))
        self.labelDist.setText(_translate("MainWindow", "Distance From Camera:"))
        self.labelValue.setText(_translate("MainWindow", "0 m "))
        self.btnClose.setText(_translate("MainWindow", "Close Camera"))
