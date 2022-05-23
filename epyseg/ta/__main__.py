#import argparse # later maybe allow arguments
from epyseg.ta.GUI.pyta import TissueAnalyzer
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == '__main__':
    # run the deep learning GUI
    # code can be run from the command line using 'python -m epyseg'
    app = QApplication(sys.argv)
    w = TissueAnalyzer()
    w.show()
    sys.exit(app.exec_())

