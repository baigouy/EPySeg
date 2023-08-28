import os
from epyseg.settings.global_settings import set_UI
set_UI()

from qtpy.QtWidgets import *
from qtpy.QtCore import Qt, QFileSystemWatcher, QSettings, Signal
from qtpy.QtCore import *
import traceback
import sys


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Signals:
        finished: No data
        error: Tuple (exctype, value, traceback.format_exc())
        result: Any data returned from processing
        progress: Integer indicating % progress
    """

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class FakeWorker(QObject):
    """
    FakeWorker class.

    Args:
        fn (function): The function callback to run on this worker thread. Supplied args and kwargs will be passed through to the runner.
        args: Arguments to pass to the callback function
        kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super(FakeWorker, self).__init__()

        # Store constructor arguments (reused for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        """
        Initialize the runner function with passed args and kwargs.
        """
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

    def start(self):
        """
        Start the worker thread by running the callback function.
        """
        self.run()
