# modified from https://stackoverflow.com/questions/24469662/how-to-redirect-logger-output-into-pyqt-text-widget
import os
from epyseg.settings.global_settings import set_UI
set_UI()
import sys
from qtpy import QtCore
import logging


class QtHandler(logging.Handler):
    """
    Custom logging handler for Qt application.
    It emits log records to either stdout or stderr based on the log level.

    Attributes:
        messageWritten (QtCore.Signal): Signal emitted when a log message is written.

    Methods:
        emit(record): Formats and emits the log record.

    """

    def __init__(self):
        logging.Handler.__init__(self)

    def emit(self, record):
        """
        Formats and emits the log record.

        Args:
            record (logging.LogRecord): The log record to be emitted.

        """
        record = self.format(record)
        if record:
            if record.startswith('ERROR') or record.startswith('CRITICAL') or record.startswith('WARNING'):
                XStream.stderr().write('%s' % record)
            else:
                XStream.stdout().write('%s' % record)


class XStream(QtCore.QObject):
    """
    Custom QObject class for capturing stdout and stderr.

    Attributes:
        _stdout (XStream): The singleton instance for stdout.
        _stderr (XStream): The singleton instance for stderr.
        messageWritten (QtCore.Signal): Signal emitted when a log message is written.

    Methods:
        flush(): Flushes the output.
        fileno(): Returns the file number.
        write(msg): Writes the message to the output stream.
        stdout(): Returns the singleton instance for stdout.
        stderr(): Returns the singleton instance for stderr.

    """

    _stdout = None
    _stderr = None
    messageWritten = QtCore.Signal(str)

    def flush(self):
        """
        Flushes the output.

        """
        pass

    def fileno(self):
        """
        Returns the file number.

        Returns:
            int: The file number (-1).

        """
        return -1

    def write(self, msg):
        """
        Writes the message to the output stream.

        Args:
            msg (str): The message to be written.

        """
        if not self.signalsBlocked():
            self.messageWritten.emit(msg)

    @staticmethod
    def stdout():
        """
        Returns the singleton instance for stdout.

        Returns:
            XStream: The singleton instance for stdout.

        """
        if not XStream._stdout:
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout

    @staticmethod
    def stderr():
        """
        Returns the singleton instance for stderr.

        Returns:
            XStream: The singleton instance for stderr.

        """
        if not XStream._stderr:
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr
