#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 19:35:19 2022

@author: louiszhang
"""

import sys
from random import randint

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()