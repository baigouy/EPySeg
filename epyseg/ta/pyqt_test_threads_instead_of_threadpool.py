# ça marche enfin et c'est pas trop dur à faire --> je peux améliorer ça je pense!!!
import sys
from time import sleep

from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt
# Snip...

# Step 1: Create a worker class
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
import traceback

from epyseg.tools.early_stopper_class import early_stop
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()

class FakeWorker2(QObject):
    # loop = pyqtSignal(object)
    # progress = pyqtSignal(int)
    # finished = pyqtSignal()

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

    def __init__(self, fn, *args, **kwargs):
        # QThread.__init__(self)
        early_stop.stop = False
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        # self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.progress


    def run(self):


        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.result.emit(result)  # Return the result of the processing
        finally:
            self.finished.emit()  # Done


    def stop(self):
        # self.terminate()
        # self.finished.emit()
        # self.signals.finished.emit()
        early_stop.stop = True
        logger.info('Stopping the process... (this may take time)')


class Worker2(QThread):
    # loop = pyqtSignal(object)
    # progress = pyqtSignal(int)
    # finished = pyqtSignal()

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

    def __init__(self, fn, *args, **kwargs):
        QThread.__init__(self)
        early_stop.stop = False
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        # self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.progress


    def run(self):


        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            try:
                traceback.print_exc()
                exctype, value = sys.exc_info()[:2]
                self.error.emit((exctype, value, traceback.format_exc()))
            except:
                pass
        else:
            self.result.emit(result)  # Return the result of the processing
        finally:
            self.finished.emit()  # Done


    def stop(self):

        # global stop_threads
        # stop_threads = True
        # def stop(self):

        # self.quit()
        # self.threadactive = False





        # self.terminate()
        # self.wait()

        early_stop.stop = True # let the code finish on its own
        logger.info('Stopping the process... (this may take time)')
        # EZDeepLearning.

        try:
            self.terminate()
        except:
            pass

        # qtimer = QTimer()
        # qtimer.setSingleShot(True)
        # # self.path.textChanged.connect(lambda: qtimer.start(600))
        # self.path.textChanged.connect(
        #     lambda x: qtimer.start(600))  # somehow my weird bug is here and only in some conditions
        # # # Is there a better way TODO that ????
        # # timer.timeout.connect(self._text_changed)
        # qtimer.timeout.connect(lambda: _thread.ter)
        # timer.singleShot(1000, this, SLOT(hardTerminate()))
        #
        # // Normal termination
        # _foo->terminate();
        #
        # // Waiting thread
        # _thread->wait();
        #
        # // Thread finished, cancel timer
        # timer.stop();
        # }
        # void hardTerminate()
        # {
        # // Ooops, normal termination failed, will terminate thread
        # _thread->terminate();
        # }


        # self.finished.emit()
        # self.signals.finished.emit()

    #
# class Worker(QObject):
#     finished = pyqtSignal()
#     progress = pyqtSignal(int)
#
#     def run(self):
#         """Long-running task."""
#         for i in range(10):
#             sleep(1)
#             self.progress.emit(i + 1)
#         self.finished.emit()
#
#     def stop(self):
#         self.terminate()

class Window(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("QThreadPool + QRunnable")
        self.resize(250, 150)
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        # Create and connect widgets
        self.stepLabel = QLabel("Hello, World!")
        self.stepLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.longRunningBtn = QPushButton("Click me!")
        self.longRunningBtn.clicked.connect(self.runLongTask)

        self.stop = QPushButton("Stop me!")
        self.stop.clicked.connect(self.early_stop)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.stepLabel)
        layout.addWidget(self.longRunningBtn)
        layout.addWidget(self.stop)
        self.centralWidget.setLayout(layout)
    # Snip...
    def runLongTask(self):
        # Step 2: Create a QThread object
        # self.thread = Worker()
        # self.thread.setTerminationEnabled(True)
        # Step 3: Create a worker object
        self.thread = Worker2(self.loop)
        self.thread.setTerminationEnabled(True)
        # Step 4: Move worker to the thread
        # self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        # self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.thread.quit)
        # self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.progress.connect(self.reportProgress)
        # Step 6: Start the thread
        self.thread.start()

        # Final resets
        self.longRunningBtn.setEnabled(False)
        self.thread.finished.connect(
            lambda: self.longRunningBtn.setEnabled(True)
        )
        self.thread.finished.connect(
            lambda: self.stepLabel.setText("Long-Running Step: 0")
        )


    def early_stop(self):
        if self.thread is not None:
            print('stopping')
            # self.thread.stop()
            # self.thread.quit()
            # self.thread.terminate()
            # self.thread.wait()
            self.thread.stop()

        self.thread = None

    def reportProgress(self, value):
        # TODO --> do that better
        print('progress', value)

    def loop(self, progress_callback):
        for i in range(100):
            # self.x = i
            # self.loop.emit(i)
            # self.progress.emit(i)
            if progress_callback is not None:
                progress_callback.emit(i)
            sleep(0.05)
        # self.finished.emit()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())

