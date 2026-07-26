from PyQt5 import QtWidgets
import sys
import os

import app_framework as af

app = QtWidgets.QApplication(sys.argv)
form = af.MyWindow()
form.show()
sys.exit(app.exec_())
