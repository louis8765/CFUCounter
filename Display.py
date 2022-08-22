# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'portal.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(390, 335)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.instructions = QtWidgets.QLabel(Form)
        self.instructions.setAlignment(QtCore.Qt.AlignCenter)
        self.instructions.setObjectName("instructions")
        self.verticalLayout.addWidget(self.instructions)
        self.img = QtWidgets.QGraphicsView(Form)
        self.img.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.img.setMouseTracking(True)
        self.img.setObjectName("img")
        self.verticalLayout.addWidget(self.img)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.progressBar = QtWidgets.QProgressBar(Form)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_2.addWidget(self.progressBar)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.instructions.setText(_translate("Form", "Select a colony (1 of each color)"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())