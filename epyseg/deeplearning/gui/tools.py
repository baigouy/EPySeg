import os

os.environ['SM_FRAMEWORK'] = 'tf.keras'  # set env var for changing the segmentation_model framework
import sys
from epyseg.uitools.blinker import Blinker
import logging
import traceback
from epyseg.deeplearning.deepl import EZDeepLearning
from PyQt5.QtWidgets import QListWidgetItem, QAbstractItemView, QSpinBox, QComboBox, QProgressBar, \
    QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QRadioButton, QButtonGroup, QGroupBox, \
    QTextBrowser, QToolTip, QDoubleSpinBox
from PyQt5.QtCore import Qt, QThreadPool, QPoint, QRect
from PyQt5.QtGui import QColor, QTextCharFormat, QTextCursor, QPixmap, QIcon
from PyQt5.QtWidgets import QGridLayout, QListWidget, QFrame, QTabWidget
from epyseg.gui.open import OpenFileOrFolderWidget
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets, QtCore, QtGui
from epyseg.tools.qthandler import XStream, QtHandler
from epyseg.gui.pyqtmarkdown import PyQT_markdown
from epyseg.img import Img
from PyQt5.QtWidgets import QPushButton, QWidget
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)  # high DPI fix

DEBUG = True  # set to True if GUI crashes
__MAJOR__ = 0
__MINOR__ = 0
__MICRO__ = 1
__RELEASE__ = ''  # a #b  # https://www.python.org/dev/peps/pep-0440/#public-version-identifiers --> alpha beta, ...
__VERSION__ = ''.join([str(__MAJOR__), '.', str(__MINOR__), '.',
                       str(__MICRO__)])  # if __MICRO__ != 0 else '', __RELEASE__]) # bug here fix some day
__AUTHOR__ = 'Benoit Aigouy'
__NAME__ = 'Deep Tools'
__EMAIL__ = 'baigouy@gmail.com'

# do something to play with the model and save various layers
# do something to create the training files and or maybe to check things...
# TODO
# load/visualize filters or weights ??? --> can be useful...

# make a tool to load and edit models with and without weights --> just get the previous stuff
# do something that shows all the layers in a combo or tree so that one can save or show every layer --> hope this will not crash but probably ok...
# the tree is a good idea as I can collapse or open it and can select one or several layers --> TODO
# TODO create a tree out of the model
# TODO allow for block creation visually
# tree drag and drop may also be an easy way to build a model
# could also build by input output --> and connect like that --> good idea
# see how to show all the layers and to deal with them

# TODO maye use that https://www.tensorflow.org/tensorboard/graphs --> rather than reinventing the wheel...

# The function to be traced.
# @tf.function
# def my_func(x, y):
#   # A simple hand-rolled layer.
#   return tf.nn.relu(tf.matmul(x, y))
#
# # Set up logging.
# stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir = 'logs/func/%s' % stamp
# writer = tf.summary.create_file_writer(logdir)
#
# # Sample data for your function.
# x = tf.random.uniform((3, 3))
# y = tf.random.uniform((3, 3))
#
# # Bracket the function call with
# # tf.summary.trace_on() and tf.summary.trace_export().
# tf.summary.trace_on(graph=True, profiler=True)
# # Call only one tf.function when tracing.
# z = my_func(x, y)
# with writer.as_default():
#   tf.summary.trace_export(
#       name="my_func_trace",
#       step=0,
#       profiler_outdir=logdir)

# https://stackoverflow.com/questions/43702323/how-to-load-only-specific-weights-on-keras --> load specific weights --> easy I guess
# https://www.tensorflow.org/resources/tools --> really cool too


# TODO --> something like that in fact
# model.save(filepath) method:
#
# custom_model = keras.models.Model(inputs=model.layers[0].input,
#                                   outputs=model.layers[-2].output)
#
# custom_model.save('model.h5')

class DeepTools(QWidget):
    '''a deep learning GUI

    '''

    def __init__(self, parent=None):
        '''init for gui

        Parameters
        ----------
        parent : QWidget
            parent window if any (for future dvpt such as tyssue analyzer)

        '''
        self.currently_selected_metrics = []
        super().__init__(parent)
        self.blinker = Blinker()
        self.initUI()
        self.to_blink_after_worker_execution = None
        self.deepTA = EZDeepLearning()  # init empty model

    def initUI(self):
        print('here')
        '''init ui

        '''
        self.threading_enabled = True
        # logging
        self.logger_console = QTextBrowser(self)
        self.logger_console.setReadOnly(True)
        self.logger_console.textCursor().movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, 1)
        self.logger_console.setHtml('<html>')
        self.logger_console.ensureCursorVisible()
        self.logger_console.document().setMaximumBlockCount(1000)  # limits to a 1000 entrees
        if not DEBUG:
            XStream.stdout().messageWritten.connect(self.set_html_black)
            XStream.stderr().messageWritten.connect(self.set_html_red)
        self.handler = QtHandler()
        self.handler.setFormatter(logging.Formatter(TA_logger.default_format))
        # change default handler for logging
        TA_logger.setHandler(self.handler)

        self.setWindowTitle(__NAME__ + ' v' + str(__VERSION__))

        # Initialize tab screen
        self.tabs = QTabWidget(self)
        self.model_tab = QWidget()
        self.train_tab = QWidget()
        self.ensemble_tab = QWidget()
        self.predict_tab = QWidget()
        self.advanced_tab = QWidget()
        # self.post_process = QWidget() # redundant with predict

        # Add tabs
        self.tabs.addTab(self.model_tab, 'Model') # split concatenate visualize model and or weights/ can I rename also things and allow for freeze of the layers too...
        self.tabs.addTab(self.train_tab, 'Build Train/GT Sets = Image concatenation')
        # self.tabs.addTab(self.predict_tab, 'Predict')
        self.tabs.addTab(self.ensemble_tab, 'Ensemble')  # To combine several outputs/models to improve seg quality
        self.tabs.addTab(self.advanced_tab, 'Advanced')  # mutate model, feeze, 3D, ... TODO
        self.advanced_tab.setVisible(False)
        # self.tabs.setVisible(False)
        # self.tabs.addTab(self.post_process, 'Post Process') # redundant with predict

        self.tabs.currentChanged.connect(self._onTabChange)

        # creating model tab
        self.model_tab.layout = QGridLayout()
        self.model_tab.layout.setAlignment(Qt.AlignTop)
        self.model_tab.layout.setColumnStretch(0, 25)
        self.model_tab.layout.setColumnStretch(1, 75)
        self.model_tab.layout.setHorizontalSpacing(3)
        self.model_tab.layout.setVerticalSpacing(3)


        # should I allow this to be stored in config file, probably not...
        self.input_model = OpenFileOrFolderWidget(parent_window=self, label_text='Load model',is_file=True,
                                                  extensions="Supported Files (*.h5 *.H5 *.hdf5 *.HDF5 *.json *.JSON *.model);;All Files (*)",
                                                  tip_text='Drag and drop a model file here')

        self.input_weights = OpenFileOrFolderWidget(parent_window=self, label_text='Load weights (optional)',
                                                    is_file=True,
                                                    extensions="Supported Files (*.h5 *.H5 *.hdf5 *.HDF5);;All Files (*)",
                                                    tip_text='Drag and drop a single weight file here')  # TODO shall i add *.model ???

        # parameters for the pretrained models
        self.groupBox_pretrain = QGroupBox('Model',objectName='groupBox_pretrain')
        self.groupBox_pretrain.setEnabled(True)
        # groupBox layout
        self.groupBox_pretrain_layout = QGridLayout()
        self.groupBox_pretrain_layout.setAlignment(Qt.AlignTop)
        self.groupBox_pretrain_layout.setColumnStretch(0, 25)
        self.groupBox_pretrain_layout.setColumnStretch(1, 25)
        self.groupBox_pretrain_layout.setColumnStretch(2, 50)
        self.groupBox_pretrain_layout.setColumnStretch(3, 2)
        self.groupBox_pretrain_layout.setHorizontalSpacing(3)
        self.groupBox_pretrain_layout.setVerticalSpacing(3)

        self.groupBox_pretrain_layout.addWidget(self.input_model, 1, 0, 1, 3)
        self.groupBox_pretrain_layout.addWidget(self.input_weights, 2, 0, 1, 3)

        self.groupBox_pretrain.setLayout(self.groupBox_pretrain_layout)
        self.model_tab.layout.addWidget(self.groupBox_pretrain, 1, 0, 1, 2)
        self.model_tab.setLayout(self.model_tab.layout)

        # widget global layout
        table_widget_layout = QVBoxLayout()
        table_widget_layout.setAlignment(Qt.AlignTop)
        table_widget_layout.addWidget(self.tabs)

        log_and_main_layout = QHBoxLayout()
        log_and_main_layout.setAlignment(Qt.AlignTop)

        # TODO put this in a group to get the stuff
        log_groupBox = QGroupBox('Log',objectName='log_groupBox')
        log_groupBox.setEnabled(True)

        help = PyQT_markdown()
        try:
            # this_dir, this_filename = os.path.split(__file__)
            # help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'getting_started2.md'),
            #                             title='getting started: predict using pre-trained network')
            # help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'getting_started.md'),
            #                             title='getting started: build and train a custom network')
            # help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'getting_started3.md'),
            #                             title='getting started: further train the EPySeg model on your data')
            # help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'pretrained_model.md'),
            #                             title='Load a pre-trained model')
            # help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'model.md'),
            #                             title='Build a model from scratch')
            # help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'load_model.md'),
            #                             title='Load a model')
            # help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'train.md'), title='Train')
            # help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'predict.md'), title='Predict')
            # help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'preprocessing.md'),
            #                             title='Training dataset parameters')
            # help.set_markdown_from_file(os.path.join(this_dir, 'deeplearning/docs', 'data_augmentation.md'),
            #                             title='Data augmentation')
            # TODO prevent this window to be resize upon tab changes
            pass
        except:
            traceback.print_exc()

        # Initialize tab screen
        self.help_tabs = QTabWidget(self)
        self.help_tabs.setMinimumWidth(500)
        self.log_tab = QWidget()
        self.help_html_tab = QWidget()
        self.settings_GUI = QWidget()

        # Add tabs
        self.help_tabs.addTab(self.log_tab, 'Log')
        self.help_tabs.addTab(self.help_html_tab, 'Help')
        self.help_tabs.addTab(self.settings_GUI, 'GUI settings')

        # creating model tab
        self.log_tab.layout = QVBoxLayout()
        self.log_tab.layout.setAlignment(Qt.AlignTop)

        # global system progress bar
        self.pbar = QProgressBar(self)

        self.log_tab.layout.addWidget(self.logger_console)
        # self.log_tab.layout.addWidget(self.instant_help)
        self.log_tab.layout.addWidget(self.pbar)
        self.log_tab.setLayout(self.log_tab.layout)

        self.help_html_tab.layout = QVBoxLayout()
        self.help_html_tab.layout.setAlignment(Qt.AlignTop)
        self.help_html_tab.layout.setContentsMargins(0, 0, 0, 0)
        self.help_html_tab.layout.addWidget(help)
        self.help_html_tab.setLayout(self.help_html_tab.layout)

        self.enable_threading_check = QCheckBox('Threading enable/disable',objectName='enable_threading_check')
        self.enable_threading_check.setChecked(True)
        # self.enable_threading_check.stateChanged.connect(self._set_threading)
        self.enable_debug = QCheckBox('Debug mode', objectName='enable_debug')
        self.enable_debug.setChecked(False)
        self.enable_debug.stateChanged.connect(self._enable_debug)

        self.settings_GUI.layout = QVBoxLayout()
        self.settings_GUI.layout.setAlignment(Qt.AlignTop)
        self.settings_GUI.layout.addWidget(self.enable_threading_check)
        self.settings_GUI.layout.addWidget(self.enable_debug)
        self.settings_GUI.setLayout(self.settings_GUI.layout)

        log_and_main_layout.addLayout(table_widget_layout)
        log_and_main_layout.addWidget(self.help_tabs)

        self.setLayout(log_and_main_layout)

        try:
            screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
            centerPoint = QApplication.desktop().screenGeometry(screen).center()
            self.setGeometry(QtCore.QRect(centerPoint.x() - self.width(), centerPoint.y() - self.height(), self.width(),
                                          self.height()))
        except:
            pass

        # # monitor mouse position to show helpful tips/guide the user
        # self.setMouseTracking(True) # does not work well because of contained objects capturing mouse --> maybe simplest is to have the
        self.show()

    def _onTabChange(self):
        pass

    def set_html_red(self, text):
        # quick n dirty log coloring --> improve when I have time
        textCursor = self.logger_console.textCursor()
        textCursor.movePosition(QTextCursor.End)
        self.logger_console.setTextCursor(textCursor)
        format = QTextCharFormat()
        format.setForeground(QColor(255, 0, 0))  # red
        self.logger_console.setCurrentCharFormat(format)
        self.logger_console.insertPlainText(text)
        self.logger_console.verticalScrollBar().setValue(self.logger_console.verticalScrollBar().maximum())

    def set_html_black(self, text):
        # quick n dirty log coloring --> improve when I have time
        textCursor = self.logger_console.textCursor()
        textCursor.movePosition(QTextCursor.End)
        self.logger_console.setTextCursor(textCursor)
        format = QTextCharFormat()
        format.setForeground(QColor(0, 0, 0))  # black
        self.logger_console.setCurrentCharFormat(format)
        self.logger_console.insertPlainText(text)

    def _enable_debug(self):
        if self.enable_debug.isChecked():
            # enable debug extensive log
            logger.setLevel(TA_logger.DEBUG)
            logger.debug('Debug enabled...')
        else:
            # disable debug log
            logger.setLevel(TA_logger.DEFAULT)
            logger.debug('Debug disabled...')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = DeepTools()
    w.show()
    sys.exit(app.exec_())
