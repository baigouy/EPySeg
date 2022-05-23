# modified from https://stackoverflow.com/questions/24469662/how-to-redirect-logger-output-into-pyqt-text-widget

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import logging

class QtHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)

    def emit(self, record):
        record = self.format(record)
        if record:
            if record.startswith('ERROR') or record.startswith('CRITICAL') or record.startswith('WARNING'):
                XStream.stderr().write('%s' % record)
            else:
                XStream.stdout().write('%s' % record)


class XStream(QtCore.QObject):
    _stdout = None
    _stderr = None
    messageWritten = QtCore.pyqtSignal(str)

    def flush(self):
        pass

    def fileno(self):
        return -1

    def write(self, msg):
        if (not self.signalsBlocked()):
            self.messageWritten.emit(msg)

    @staticmethod
    def stdout():
        if (not XStream._stdout):
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout

    @staticmethod
    def stderr():
        if (not XStream._stderr):
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr
