import os
from epyseg.settings.global_settings import set_UI
from epyseg.epygui import EPySeg
from qtpy.QtWidgets import QApplication
import sys

def run_epyseg():
    """
    Runs the EPySeg deep learning GUI.

    This function sets the UI and initializes the QApplication.
    The EPySeg GUI window is created and shown.
    The application event loop is started and the program exits
    when the GUI window is closed.

    # Examples:
    #     >>> run_epyseg()
    """
    set_UI()

    # Initialize the QApplication
    app = QApplication(sys.argv)

    # Create and show the EPySeg GUI window
    w = EPySeg()
    w.show()

    # Start the application event loop and exit when the GUI window is closed
    sys.exit(app.exec_())


if __name__ == '__main__':
    # Run the EPySeg GUI
    run_epyseg()
