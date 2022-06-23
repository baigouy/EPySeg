#import argparse # later maybe allow arguments
from epyseg.ta.GUI.pyta import TissueAnalyzer
from PyQt5.QtWidgets import QApplication
import sys
import multiprocessing
import platform

if __name__ == '__main__':
    # mac OSX fix to test --> this does not work at all in my hands...
    # if platform.system() == 'Darwin':
    #     multiprocessing.set_start_method('spawn')
    # multiprocessing.set_start_method('fork')
    # multiprocessing.freeze_support()
    # multiprocessing.set_start_method('spawn')

    # run the deep learning GUI
    # code can be run from the command line using 'python -m epyseg'
    app = QApplication(sys.argv)
    w = TissueAnalyzer()
    w.show()
    sys.exit(app.exec_())

