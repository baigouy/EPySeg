import os
from epyseg.settings.global_settings import set_UI # set the UI to be used py qtpy
set_UI()
#import argparse # later maybe allow arguments
from epyseg.epygui import EPySeg
from qtpy.QtWidgets import QApplication
import sys

if __name__ == '__main__':
    # run the deep learning GUI
    # code can be run from the command line using 'python -m epyseg'
    app = QApplication(sys.argv)
    w = EPySeg()
    w.show()
    sys.exit(app.exec_())

