#import argparse # later maybe allow arguments
from epyseg.epygui import EPySeg
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == '__main__':
    # run the deep learning GUI
    # code can be run from the command line using 'python -m epyseg'
    app = QApplication(sys.argv)
    w = EPySeg()
    w.show()
    sys.exit(app.exec_())

