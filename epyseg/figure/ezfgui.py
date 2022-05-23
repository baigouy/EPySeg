# TODO handle selection and all the fusion options and also the normal save without export
# do a graph editor embedded --> see how --> make it as simple as possible and make the macro language so simple
# find an easy way to convert any action to a simple macro code --> would be great if I could save some simple code just for the fun of it
# think about it but I guess it will be easy
# store all the macro commands

# first thing to do is to get the ctrl clicks

# TODO implement the snap --> not so hard I guess if I drop on nothing

# icons to use at some point maybe:
# mdi.format-paint
# fa5s.paint-brush
# fa.file-picture-o
# fa.newspaper-o
# mdi.pdf-box
# mdi.export
# fa.folder-open-o
# ei.graph
# fa.paragraph
# fa5s.tools
# mdi.tools
# mdi.information-outline
# mdi.format-vertical-align-center
# mdi.format-vertical-align-top
# mdi.format-underline
# mdi.format-text-rotation-up

# pas mal --> good start for a GUI but add menus too


# this is the full EZF GUI that will contain everything needed for the plot
# need a menu a main panel and a side bar
# or many side bars --> think about it and keep it as simple as possible
# create a dynamic demo with an easy possibility to launch it so that the user can easily learn things/could have a demo tab
# see how to serialize everything --> maybe in the form of a script
# --> would be very easy to do I guess
import copy
import traceback

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QColor, QTextCursor, QTextCharFormat
from PyQt5.QtWidgets import QApplication, QStackedWidget, QWidget, QTabWidget, QScrollArea, QVBoxLayout, QPushButton, \
    QGridLayout, QTextBrowser, QFrame, QProgressBar, QGroupBox
import qtawesome as qta
import logging
import os

from epyseg.dialogs.opensave import saveFileDialog
from epyseg.draw.shapes.circle2d import Circle2D
from epyseg.draw.shapes.ellipse2d import Ellipse2D
from epyseg.draw.shapes.freehand2d import Freehand2D
from epyseg.draw.shapes.image2d import Image2D
from epyseg.draw.shapes.line2d import Line2D
from epyseg.draw.shapes.point2d import Point2D
from epyseg.draw.shapes.polygon2d import Polygon2D
from epyseg.draw.shapes.polyline2d import PolyLine2D
from epyseg.draw.shapes.rect2d import Rect2D
from epyseg.draw.shapes.scalebar import ScaleBar
from epyseg.draw.shapes.square2d import Square2D
from epyseg.draw.shapes.txt2d import TAText2D
from epyseg.draw.shapes.vectorgraphics2d import VectorGraphics2D
from epyseg.figure.column import Column
from epyseg.figure.row import Row
from epyseg.figure.scrollableEZFIG import scrollable_EZFIG
from epyseg.tools.qthandler import XStream, QtHandler
from epyseg.uitools.blinker import Blinker
# from epyseg.ta.GUI.pyta import Overlay
from epyseg.ta.GUI.overlay_hints import Overlay

from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()


__MAJOR__ = 0
__MINOR__ = 1
__MICRO__ = 0
__RELEASE__ = 'b'  # https://www.python.org/dev/peps/pep-0440/#public-version-identifiers --> alpha beta, ...
__VERSION__ = ''.join(
    [str(__MAJOR__), '.', str(__MINOR__), '.'.join([str(__MICRO__)]) if __MICRO__ != 0 else '', __RELEASE__])
__AUTHOR__ = 'Benoit Aigouy'
__NAME__ = 'PyFig: A Scientific Figure Creation Assistant' # TODO think about a name and how to handle that
__EMAIL__ = 'baigouy@gmail.com'


# DEBUG = False
DEBUG = True

class EZFIG_GUI(QtWidgets.QMainWindow):
    # stop_threads = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

        # global early_stop
        # early_stop = False
        self.setAcceptDrops(True)  # we accept DND but just to transfer it to the current list handlers!!!

    def initUI(self):

        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()

        # should fit in 1024x768 (old computer screens)
        window_width = 900
        window_height = 700
        self.setGeometry(
            QtCore.QRect(centerPoint.x() - int(window_width / 2), centerPoint.y() - int(window_height / 2),
                         window_width,
                         window_height))

        # set the window icon
        # self.setWindowIcon(QtGui.QIcon('./../../IconsPA/src/main/resources/Icons/ico_packingAnalyzer2.gif'))
        # variantMap =QVariant()
        # variantMap.insert("color", QColor(10, 10, 10));



        self.setWindowIcon(qta.icon('fa.newspaper-o',
                                    color=QColor(200, 200, 200)))  # very easy to change the color of my icon

        # zoom parameters

        self.setWindowTitle(__NAME__ + ' v' + str(__VERSION__))

        # this is the default paint window of TA
        self.paint = scrollable_EZFIG()
        # this is the properties window (allows to set useful parameters that can be used by TA

        self.Stack = QStackedWidget(self)



        # Initialize tab screen
        self.tabs = QTabWidget(self)
        self.tab0 = QWidget()
        # self.tab0.setToolTip("jsdhqsjkdhkj sdqsh d")
        self.tab1 = QWidget()


        # Add tabs
        self.tabs.addTab(self.tab0, 'Pre-process')  # contains stuff to do max projs and heightmaps from images
        self.tabs.setTabToolTip(0,'Use a CARE model to perform surface extraction/create a 2D projection from a 3D epithelial stack.')
        self.tabs.addTab(self.tab1, 'Segmentation')
        self.tabs.setTabToolTip(1, 'Segment epithelial images.')
        # self.tabs.addTab(self.tab2, 'Analysis')

        self.tabs.currentChanged.connect(self.onTabChange)

        # force seg tab to appear first
        # self.tabs.setCurrentIndex(1)

        # Create first tab
        self.tab0.layout = QVBoxLayout()
        self.surf_proj = QPushButton("Surface projection")
        self.surf_proj.setToolTip(
            "Extracts, using a CARE model without the denoising module,\nthe apical fluorescent signal from a 3D image of an epithelium and creates a height map")
        # self.surf_proj.clicked.connect(self.surf_proj_run)
        self.tab0.layout.addWidget(self.surf_proj)
        # self.denoising_surf_proj = QPushButton("Denoising surface projection")
        # self.denoising_surf_proj.setToolTip(
        #     "Same as \"Surface projection\" using the complete CARE model, denoiser included.\nBe aware that the denoising module can generate unexpected/unwanted results,\nespecially, when the input data significantly differs from the data used to train the model.\nSo, altogether, denoising/image reconstruction is not necessarily improving the quality of the 2D projection.")
        # self.denoising_surf_proj.clicked.connect(self.surf_proj_run)
        # self.tab0.layout.addWidget(self.denoising_surf_proj)
        self.tab0.setLayout(self.tab0.layout)

        # Create first tab
        self.tab1.layout = QVBoxLayout()

        # self.pushButton0 = QPushButton("Deep Learning/EPySeg segmentation")
        # self.pushButton0.clicked.connect(self.epyseg_seg)
        # self.pushButton0.setToolTip("Segment an epithelium using the EPySeg model")
        # self.tab1.layout.addWidget(self.pushButton0)

        # self.pushButton1 = QPushButton("Watershed segmentation")
        # self.pushButton1.clicked.connect(self.run_watershed)
        # self.pushButton1.setToolTip("Segment an epithelium using the wahtershed algorithm")
        # self.tab1.layout.addWidget(self.pushButton1)
        self.tab1.setLayout(self.tab1.layout)
        # self.pushButton2 = QPushButton("Deep learning segmentation")
        # self.pushButton2.clicked.connect(self.deep_learning)
        # self.tab1.layout.addWidget(self.pushButton2)



        # content_widget.layout = QVBoxLayout()
        # sub_layout_preview = QGridLayout()
        # sub_layout_preview.setColumnStretch(0, 10)
        # sub_layout_preview.setColumnStretch(1, 90)
        #
        # preview_img_label = QLabel("Image/Data")
        # sub_layout_preview.addWidget(preview_img_label, 0, 0)
        #
        # # content_widget.layout.addWidget(preview_img_label)
        # self.image_preview_combo = QComboBox()
        # self.image_preview_combo.currentTextChanged.connect(self.preview_changed)
        # sub_layout_preview.addWidget(self.image_preview_combo, 0, 1)
        # content_widget.layout.addLayout(sub_layout_preview)
        #
        # self.groupBox_color_coding = QGroupBox('Color coding:')
        # self.groupBox_color_coding.setCheckable(True)
        # self.groupBox_color_coding.setChecked(False)
        # self.groupBox_color_coding.toggled.connect(self.preview_changed)
        #
        # color_coding_layout = QHBoxLayout()
        # color_coding_layout.setContentsMargins(0, 0, 0, 0)
        # color_coding_layout.setAlignment(Qt.AlignLeft)
        # # no local global
        # # color_coding_label = QLabel('Color coding: ')
        # # color_coding_layout.addWidget(color_coding_label, 0, 0)
        #
        # # self.radioButton1 = QRadioButton('No')
        # self.radioButton2 = QRadioButton('Current/Local')
        # self.radioButton2.setToolTip("Apply local (only for the current frame) color coding")
        # self.radioButton3 = QRadioButton('Global')
        # self.radioButton3.setToolTip("Apply global color coding (i.e. max and min is the max and min of all frames)")
        # # self.radioButton3.setEnabled(False)  # not supported yet --> reactivate it when it will be supported # see if I can show a scalebar somewhere in the stuff
        # self.radioButton2.setChecked(True)
        #
        # # self.radioButton1.toggled.connect(self.preview_changed)
        # self.radioButton2.toggled.connect(self.preview_changed)
        # self.radioButton3.toggled.connect(self.preview_changed)
        #
        # # color_coding_layout.addWidget(self.radioButton1, 0, 1)
        # color_coding_layout.addWidget(self.radioButton2)
        # color_coding_layout.addWidget(self.radioButton3)
        #
        # predict_output_radio_group = QButtonGroup()
        # # predict_output_radio_group.addButton(self.radioButton1)
        # predict_output_radio_group.addButton(self.radioButton2)
        # predict_output_radio_group.addButton(self.radioButton3)
        #
        # self.excluder_label = QCheckBox('Exclude % lower and upper outliers: ')
        # self.excluder_label.setEnabled(True)
        # self.excluder_label.setChecked(False)
        #
        # self.excluder_label.stateChanged.connect(self.upper_or_lower_limit_changed)
        #
        # # TODO enable or disable the stuffs along with it
        # color_coding_layout.addWidget(self.excluder_label)
        #
        # self.upper_percent_spin = QDoubleSpinBox()
        # self.upper_percent_spin.setEnabled(False)
        # self.upper_percent_spin.setRange(0., 0.49)
        # self.upper_percent_spin.setSingleStep(0.01)
        # self.upper_percent_spin.setValue(0.05)
        # # self.upper_percent_spin.valueChanged.connect(self.upper_or_lower_limit_changed)
        # self.upper_percent_spin.valueChanged.connect(lambda x: delayed_preview_update.start(600))
        #
        # self.lower_percent_spin = QDoubleSpinBox()
        # self.lower_percent_spin.setEnabled(False)  # TODO code and enable this soon but ok for now
        # self.lower_percent_spin.setRange(0., 0.49)
        # self.lower_percent_spin.setSingleStep(0.01)
        # self.lower_percent_spin.setValue(0.05)
        # # self.lower_percent_spin.valueChanged.connect(self.upper_or_lower_limit_changed)
        # self.lower_percent_spin.valueChanged.connect(lambda x: delayed_preview_update.start(600))
        #
        # color_coding_layout.addWidget(self.lower_percent_spin)
        # color_coding_layout.addWidget(self.upper_percent_spin)
        #
        # lut_label = QLabel('Lut: ')
        # color_coding_layout.addWidget(lut_label)
        #
        # # populate LUTs --> once for good at the beginning of the table
        # self.lut_combo = QComboBox()
        # available_luts = list_availbale_luts()
        # for lut in available_luts:
        #     self.lut_combo.addItem(lut)
        #
        # self.lut_combo.currentTextChanged.connect(self.lut_changed)
        # color_coding_layout.addWidget(self.lut_combo)
        #
        # self.groupBox_color_coding.setLayout(color_coding_layout)
        #
        # # content_widget.layout.addLayout(color_coding_layout)
        # content_widget.layout.addWidget(self.groupBox_color_coding)
        #
        # self.groupBox_overlay = QGroupBox('Overlay/Blend')
        # self.groupBox_overlay.setToolTip("Overlay selected image over the selected background image")
        # self.groupBox_overlay.setCheckable(True)
        # self.groupBox_overlay.setChecked(False)
        # self.groupBox_overlay.toggled.connect(self.preview_changed)
        #
        # overlay_layout = QHBoxLayout()
        # overlay_layout.setContentsMargins(0, 0, 0, 0)  # self.groupBox_overlay.
        # overlay_layout.setAlignment(Qt.AlignLeft)
        # # TODO --> maybe show and or hide the transparency settings depending on the selection
        # # self.overlay_check = QCheckBox("Enable overlay/blend")
        # # self.overlay_check.stateChanged.connect(self.preview_changed)
        # # overlay_layout.addWidget(self.overlay_check, 0, 0)
        #
        # overlay_bg_label = QLabel("Background image channel:")  # for overlay
        # overlay_layout.addWidget(overlay_bg_label)
        #
        # self.overlay_bg_channel_combo = QComboBox()
        # overlay_layout.addWidget(self.overlay_bg_channel_combo)
        #
        # overlay_fg_transparency_label = QLabel("Foreground opacity:")
        # overlay_layout.addWidget(overlay_fg_transparency_label)
        #
        # self.overlay_fg_transparency_spin = QDoubleSpinBox()
        # self.overlay_fg_transparency_spin.setSingleStep(0.05)
        # self.overlay_fg_transparency_spin.setRange(0.05, 1.)
        # # self.overlay_fg_transparency_spin.valueChanged.connect(self.preview_changed) # direct update upon value change --> not always very smart... better wait some time
        # self.overlay_fg_transparency_spin.valueChanged.connect(lambda x: delayed_preview_update.start(600))
        #
        # self.overlay_fg_transparency_spin.setValue(0.3)
        # overlay_layout.addWidget(self.overlay_fg_transparency_spin)
        #
        # self.groupBox_overlay.setLayout(overlay_layout)
        # content_widget.layout.addWidget(self.groupBox_overlay)
        # # self.tab3.layout.addLayout(overlay_layout)
        #
        # self.groupBox_export = QGroupBox('Export')
        # export_layout = QHBoxLayout()
        # export_layout.setContentsMargins(0, 0, 0, 0)
        # export_layout.setAlignment(Qt.AlignLeft)
        #
        # self.export_image_button = QPushButton("Single image")
        # self.export_image_button.setToolTip("Export as tif the current view")
        # self.export_image_button.clicked.connect(self.export_image)
        # export_layout.addWidget(self.export_image_button)
        #
        # self.export_stack_button = QPushButton("Stack")
        # self.export_stack_button.setToolTip("Export as a tif stack the current view")
        # self.export_stack_button.clicked.connect(self.export_stack)
        # export_layout.addWidget(self.export_stack_button)
        #
        # self.groupBox_export.setLayout(export_layout)
        # content_widget.layout.addWidget(self.groupBox_export)
        #
        # # TODO --> maybe offer overlays/ image composition here !!! --> in a way very easy but there will be issues sometimes maybe --> just think about it ???
        # # for method in EZDeepLearning.available_model_architectures:
        # #     self.image_preview_combo.addItem(method)
        # content_widget.setLayout(content_widget.layout)
        #
        # self.tab4.layout = QVBoxLayout()
        # self.force_CPU_check = QCheckBox(
        #     'Use CPU for deep learning (slower but often offers more memory) (MUST BE SET IMMEDIATELY AFTER LAUNCHING THE SOFTWARE)')
        # self.force_CPU_check.setToolTip(
        #     "Do not use a graphic card even when a compatible one is available on the system")
        # self.force_CPU_check.stateChanged.connect(self.force_CPU)  # need a fix here
        # self.tab4.layout.addWidget(self.force_CPU_check)
        # # TODO --> do the code to force
        # self.tab4.setLayout(self.tab4.layout)

        self.table_widget = QWidget()
        table_widget_layout = QGridLayout()

        # can maybe ask the nb of threads to use here too
        self.logger_console = QTextBrowser(self)
        self.logger_console.setReadOnly(True)
        self.logger_console.textCursor().movePosition(QTextCursor.Left, QTextCursor.KeepAnchor, 1)
        self.logger_console.setHtml('<html>')
        self.logger_console.ensureCursorVisible()
        self.logger_console.document().setMaximumBlockCount(1000) # limits to a 1000 entrees

        if not DEBUG:
            try:
                XStream.stdout().messageWritten.connect(self.set_html_black)
                XStream.stderr().messageWritten.connect(self.set_html_red)
                self.handler = QtHandler()
                self.handler.setFormatter(logging.Formatter(TA_logger.default_format))
                # change default handler for logging
                TA_logger.setHandler(self.handler)
            except:
                traceback.print_exc()

        table_widget_layout.addWidget(self.tabs, 0, 0)
        # ok just store console in a qgroup stuff

        self.groupBox_logging = QGroupBox('Log')
        self.groupBox_logging.setToolTip("Shows current status.")
        self.groupBox_logging.setMinimumWidth(250)
        self.groupBox_logging.setEnabled(True)
        # groupBox layout
        self.groupBox_pretrain_layout = QGridLayout()
        self.groupBox_pretrain_layout.addWidget(self.logger_console, 0, 0)
        self.groupBox_logging.setLayout(self.groupBox_pretrain_layout)

        table_widget_layout.addWidget(self.groupBox_logging, 0,
                                      1)  # not bad but maybe put it somewhere else --> such as on the side of the tabs in a gridlayout with all the necessary stuff!!
        self.table_widget.setLayout(table_widget_layout)

        # self.table_widget.setLayout(table_widget_layout)

        # print('bob')
        # Add tabs to widget
        # table_widget_layout.addWidget(self.tabs)
        # table_widget_layout.addWidget(self.logger_console) # not bad but maybe put it somewhere else --> such as on the side of the tabs in a gridlayout with all the necessary stuff!!
        #
        # table_widget_layout.addWidget(self.tabs, 0, 0)
        # # ok just store console in a qgroup stuff
        #
        # self.groupBox_logging = QGroupBox('Log')
        # self.groupBox_logging.setToolTip("Shows TA current status, method progress, warnings and errors")
        # self.groupBox_logging.setMinimumWidth(250)
        # self.groupBox_logging.setEnabled(True)
        # # groupBox layout
        # self.groupBox_pretrain_layout = QGridLayout()
        # self.groupBox_pretrain_layout.addWidget(self.logger_console, 0, 0)
        # self.groupBox_logging.setLayout(self.groupBox_pretrain_layout)
        #
        # table_widget_layout.addWidget(self.groupBox_logging, 0,
        #                               1)  # not bad but maybe put it somewhere else --> such as on the side of the tabs in a gridlayout with all the necessary stuff!!
        # self.table_widget.setLayout(table_widget_layout)

        # voilà le stackwidget --> et comment on ajoute des trucs dedans --> permet de ne montrer que certains widgets
        # print('bob')
        self.volumeViewer = QWidget()  # maybe get it to simply pop out ???
        # ça c'est juste la layout du stack --> on s'en fout

        # TODO replace that with napari
        # self.volumeViewerUI()
        self.Stack.addWidget(self.paint)
        # self.Stack.addWidget(self.properties_table)
        # self.Stack.addWidget(self.volumeViewer)

        # create a grid that will contain all the GUI interface
        self.grid = QGridLayout()
        # grid.setSpacing(10)

        # grid.addWidget(self.scrollArea, 0, 0)
        self.grid.addWidget(self.Stack, 0, 0)
        # self.grid.addWidget(self.Stack, 0, 0,2,1)
        # self.grid.addWidget(self.list, 0, 1)
        # self.grid.addWidget(self.logger_console, 0, 1)
        # self.grid.addWidget(self.list, 1, 1)
        # The first parameter of the rowStretch method is the row number, the second is the stretch factor. So you need two calls to rowStretch, like this: --> below the first row is occupying 80% and the second 20%
        self.grid.setRowStretch(0, 75)
        self.grid.setRowStretch(2, 25)

        # first col 75% second col 25% of total width
        self.grid.setColumnStretch(0, 75)
        self.grid.setColumnStretch(1, 25)

        # void QGridLayout::addLayout(QLayout * layout, int row, int column, int rowSpan, int columnSpan, Qt::Alignment alignment = 0)
        self.grid.addWidget(self.table_widget, 2, 0, 1, 2)  # spans over one row and 2 columns

        # self.setCentralWidget(self.scrollArea)
        self.setCentralWidget(QFrame())
        self.centralWidget().setLayout(self.grid)

        # self.statusBar().showMessage('Ready')
        statusBar = self.statusBar()  # sets an empty status bar --> then can add messages in it
        self.paint.statusBar = statusBar

        # add progress bar to status bar
        self.pbar = QProgressBar(self)
        self.pbar.setToolTip('Shows current progress')
        self.pbar.setGeometry(200, 80, 250, 20)
        statusBar.addWidget(self.pbar)

        self.stop_all_threads_button = QPushButton('Stop')
        self.stop_all_threads_button.setToolTip(
            "Stops the running function or deep learning prediction as soon as possible.")
        self.stop_all_threads_button.clicked.connect(self.stop_threads_immediately)
        statusBar.addWidget(self.stop_all_threads_button)

        self.about = QPushButton()
        self.about.setIcon(qta.icon('mdi.information-variant', options=[{'scale_factor': 1.5}]))
        self.about.clicked.connect(self.about_dialog)
        self.about.setToolTip('About...')
        statusBar.addWidget(self.about)

        # statusBar.addWidget(self.last_opened_file_name)

        # Set up menu bar
        # self.menuBar = QtWidgets.QMenuBar(self)
        # self.mainMenu = QtWidgets.QMenuBar(self)
        # self.menuBar.setGeometry(QtCore.QRect(0, 0, 664, 20))
        # self.menuBar.setObjectName("menuBar")
        # self.setMenuBar(self.mainMenu)
        # self.setMenuBar(self.menuBar)

        # the esc key removes the selection --> can be useful when the selection takes the whole screen
        escShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        escShortcut.activated.connect(self.paint.EZFIG_panel.remove_sel)
        escShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active

        # Setup hotkeys for whole system
        # Delete selected vectorial objects
        # deleteShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self)
        # deleteShortcut.activated.connect(self.down)
        # deleteShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active
        #
        # # shall I remove this useless shortcut
        # # set drawing window fullscreen
        # fullScreenShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_F), self)
        # fullScreenShortcut.activated.connect(self.fullScreen)
        # fullScreenShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active
        #
        # # set drawing window fullscreen
        # fullScreenShortcut2 = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_F12), self)
        # fullScreenShortcut2.activated.connect(self.fullScreen)
        # fullScreenShortcut2.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active
        #
        # # exit from full screen TODO add quit the app too ??
        # escapeShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        # escapeShortcut.activated.connect(self.escape)
        # escapeShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active
        #
        # spaceShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        # spaceShortcut.activated.connect(self.nextFrame)
        # spaceShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active
        #
        # backspaceShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self)
        # backspaceShortcut.activated.connect(self.prevFrame)
        # backspaceShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active

        self.blinker = Blinker()
        self.to_blink_after_worker_execution = None
        self.threading_enabled = True

        # self.threads = [] # contains all threads so that I can stop them immediately
        # self.thread = None
        # self.event_stop = threading.Event() # allows to stop a thread
        # self.threadpool = QThreadPool()
        # self.threadpool.setMaxThreadCount(self.threadpool.maxThreadCount() - 1)  # spare one core for system

        self.overlay = Overlay(self.centralWidget())  # maybe ok if draws please wait somewhere
        self.overlay.hide()

        # MainWindow.setCentralWidget(self.centralWidget)
        self.mainToolBar = QtWidgets.QToolBar(self)
        self.mainToolBar.setObjectName("mainToolBar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(self)
        self.statusBar.setObjectName("statusBar")
        self.setStatusBar(self.statusBar)
        self.menuBar = QtWidgets.QMenuBar(self)
        # self.menuBar.setGeometry(QtCore.QRect(0, 0, 664, 20))
        # self.menuBar.setObjectName("menuBar")
        self.menuMenu = QtWidgets.QMenu(self.menuBar)
        self.menuMenu.setObjectName("menuMenu")
        self.setMenuBar(self.menuBar)
        # self.menuBar.addAction(self.menuMenu.menuAction())
        file_menu = self.menuBar.addMenu("File")
        export_as_svg_action = QtWidgets.QAction("Export as svg", self)
        file_menu.addAction(export_as_svg_action)
        export_as_svg_action.triggered.connect(self.export_as_svg)

        export_as_tif_action = QtWidgets.QAction("Export as... (jpg, tif, png)", self)
        file_menu.addAction(export_as_tif_action)
        export_as_tif_action.triggered.connect(self.export_as_tif)

    def onTabChange(self):
        # TODO --> maybe select row images or columns depending on the selected tab
        # --> always store all the possibilities when a click is made --> TODO
        # python serialize a numpy array --> to serialize images or gather the images --> have sort of embed --> maybe put a warning depending on the stuff
        # maybe various options for aggregation of data and images

        selected_tab_idx = self.tabs.currentIndex()
        selected_tab_name = self.tabs.tabText(selected_tab_idx).lower()

        print(selected_tab_idx, selected_tab_name)
        print('TODO')

        # pass
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

    def stop_threads_immediately(self):
        print('TODO stop_threads_immediately')
        pass

    def about_dialog(self):
        print('TODO about_dialog')
        pass

    def export_as_svg(self):
        # just for debug
        # print('TADA')
        # pass
        # TODO launch the export of the image as a svg file
        # --> call the stuff
        # qsdqsdqsd
        output_file = saveFileDialog(parent_window=self, extensions="Supported Files (*.svg);;All Files (*)",default_ext='.svg')
        self._save(output_file)

    # could do export as raster and handle the stuff
    def export_as_tif(self):
        # just for debug
        # print('TADA')
        # pass
        # TODO launch the export of the image as a svg file
        # --> call the stuff
        # qsdqsdqsd
        # TODO can I save as pdf ???? --> maybe
        output_file = saveFileDialog(parent_window=self, extensions="Supported Files (*.tif *.jpg *.png);;All Files (*)",default_ext='.tif')
        self._save(output_file)

    def _save(self, output_file):
        if output_file is not None:
            self.paint.EZFIG_panel.save(output_file)


if __name__ == '__main__':
    # vraiment pas mal --> j'y suis presque...
    import sys
    app = QApplication(sys.argv)
    w = EZFIG_GUI()
    demo = True
    if demo:
        shapes_to_draw = []
        shapes_to_draw.append(Polygon2D(0, 0, 10, 0, 10, 20, 0, 20, 0, 0, color=0x00FF00))
        shapes_to_draw.append(
            Polygon2D(100, 100, 110, 100, 110, 120, 10, 120, 100, 100, color=0x0000FF, fill_color=0x00FFFF,
                      stroke=2))
        shapes_to_draw.append(Line2D(0, 0, 110, 100, color=0xFF0000, stroke=3))
        shapes_to_draw.append(Rect2D(200, 150, 250, 100, stroke=10, fill_color=0xFF0000))
        shapes_to_draw.append(Ellipse2D(0, 50, 600, 200, stroke=3))
        shapes_to_draw.append(Circle2D(150, 300, 30, color=0xFF0000, fill_color=0x00FFFF))
        shapes_to_draw.append(PolyLine2D(10, 10, 20, 10, 20, 30, 40, 30, color=0xFF0000, stroke=2))
        shapes_to_draw.append(PolyLine2D(10, 10, 20, 10, 20, 30, 40, 30, color=0xFF0000, stroke=2))
        shapes_to_draw.append(Point2D(128, 128, color=0xFF0000, fill_color=0x00FFFF, stroke=0.65))
        shapes_to_draw.append(Point2D(128, 128, color=0x00FF00, stroke=0.65))
        shapes_to_draw.append(Point2D(10, 10, color=0x000000, fill_color=0x00FFFF, stroke=3))
        shapes_to_draw.append(Rect2D(0, 0, 512, 512, color=0xFF00FF, stroke=6))
        img0 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/00.png')
        inset = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
        inset2 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
        inset3 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
        scale_bar = ScaleBar(30, '<font color="#FF00FF">10µm</font>')
        scale_bar.set_P1(0, 0)
        img0.add_object(scale_bar, Image2D.TOP_LEFT)
        img0.add_object(inset3, Image2D.TOP_LEFT)
        img0.add_object(inset,Image2D.BOTTOM_RIGHT)  # ça marche meme avec des insets mais faudrait controler la taille des trucs... --> TODO
        img0.add_object(TAText2D(text='<font color="#FF0000">bottom right3</font>'), Image2D.BOTTOM_RIGHT)
        img0.setLettering('<font color="red">A</font>')
        img0.annotation.append(Rect2D(88, 88, 200, 200, stroke=3, color=0xFF00FF))
        img0.annotation.append(Ellipse2D(88, 88, 200, 200, stroke=3, color=0x00FF00))
        img0.annotation.append(Circle2D(33, 33, 200, stroke=3, color=0x0000FF))
        img0.annotation.append(Line2D(33, 33, 88, 88, stroke=3, color=0x0000FF))
        img0.annotation.append(Freehand2D(10, 10, 20, 10, 20, 30, 288, 30, color=0xFFFF00, stroke=3))
        img0.annotation.append(Point2D(128, 128, color=0xFFFF00, stroke=6))
        img1 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
        test_text = '''
        </style></head><body style=" font-family:'Comic Sans MS'; font-size:22pt; font-weight:400; font-style:normal;">
        <p style="color:#00ff00;"><span style=" color:#ff0000;">toto</span><br />tu<span style=" vertical-align:super;">tu</span></p>
        '''
        img1.setLettering(TAText2D(text=test_text))
        # background-color: orange;
        # span div et p donnent la meme chose par contre c'est sur deux lignes
        # display:inline; float:left # to display as the same line .... --> does that work html to svg
        # https://stackoverflow.com/questions/10451445/two-div-blocks-on-same-line --> same line for two divs
        img2 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/02.png')
        # crop is functional again but again a packing error
        img2.crop(left=60)
        img2.crop(right=30)
        img2.crop(bottom=90)
        img2.crop(top=60)
        img3 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/03.png')
        img4 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/04.png')
        img5 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/05.png')
        img6 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/06.png')
        img7 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/07.png')
        img8 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/08.png')
        img8.annotation.append(Rect2D(60, 60, 100, 100, stroke=20, color=0xFF00FF))
        img9 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/09.png')
        img10 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/10.png')
        import numpy as np
        import matplotlib.pyplot as plt
        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)
        fig, ax = plt.subplots()
        ax.plot(t, s)
        ax.set(xlabel='time (s)', ylabel='voltage (mV)',
               title='About as simple as it gets, folks')
        ax.grid()
        # graph2d = VectorGraphics2D(fig)
        # graph2d.crop(all=20)  # not great neither
        # vectorGraphics = VectorGraphics2D('/E/Sample_images/sample_images_svg/cartman.svg')
        # vectorGraphics.crop(left=10, right=30, top=10, bottom=10)
        # animatedVectorGraphics = VectorGraphics2D('/E/Sample_images/sample_images_svg/animated.svg')
        # animatedVectorGraphics.crop(left=30)  # , top=20, bottom=20
        # row1 = Row(img0, img1, img2, graph2d,animatedVectorGraphics)  # , graph2d, animatedVectorGraphics  # , img6, #, nCols=3, nRows=2 #le animated marche mais faut dragger le bord de l'image mais pas mal qd meme
        # col1 = Column(img4, img5, img6, vectorGraphics)  # , vectorGraphics# , img6, img6, nCols=3, nRows=2,

        # all seems to  work except the vector graphics --> handle that later then

        row1 = Row(img0, img1, img2)  # , graph2d, animatedVectorGraphics  # , img6, #, nCols=3, nRows=2 #le animated marche mais faut dragger le bord de l'image mais pas mal qd meme
        col1 = Column(img4, img5, img6)  # , vectorGraphics# , img6, img6, nCols=3, nRows=2,
        col2 = Column(img3, img7, img10)
        col1 /= col2
        row2 = Row(img8, img9)
        row2.setLettering('<font color="#FFFFFF">a</font>')
        row1 /= row2
        col1.setToHeight(512)  # creates big bugs now
        row1.setToWidth(512)
        row1.set_P1(128,128)
        shapes_to_draw.append(col1)
        shapes_to_draw.append(row1)
        img4.setLettering('<font color="#0000FF">Z</font>')  # ça marche mais voir comment faire en fait
        shapes_to_draw.append(Square2D(300, 260, 250, stroke=3))


        # first implement the magic selection going inner an object
        # not bad --> it shows a copy of the rects --> that could be used to do a preview --> finalize that
        # see how complex this would be
        # could do the same with all possible shapes for source and target --> TODO
        # detect all the possibilities
        print(type(row1)) # --> a column --> clone it with rects
        # really need implement cloning anyway
        for iii,rect in enumerate(row1):
            print(rect) # -> ce truc est un rect2D --> just need to clone it
            # print(type(rect), isinstance(rect, Rect2D)) # -- > ok --> seems to work
            rect_copy = Rect2D(rect.boundingRect()) # ça marche --> ça cree une copy du rect # and update the colors when
            rect_copy.fill_color = 0xFF0000
            rect_copy.translate(25,25)
            shapes_to_draw.append(rect_copy)
            # if iii==3:
            break


        # does that work ??? --> is my error an error of scaling ???

        # I need implement the cloning from inside
        # images_bounds = []
        # for rect in col1:
        #     print(rect) # -> ce truc est un rect2D --> just need to clone it
        #     # print(type(rect), isinstance(rect, Rect2D)) # -- > ok --> seems to work
        #     # --> not ok because keeps the original rect dimension, not taking into account the scaling --> just get bounds and duplictae
        #     rect_copy = Rect2D(rect.boundingRect()) # ça marche --> ça cree une copy du rect # and update the colors when
        #     # rect_copy = rect_copy*rect.scaling # --> marche pas --> would need to clone the stuff
        #     rect_copy.fill_color = 0x00FF00
        #     rect_copy.translate(25,25)
        #     shapes_to_draw.append(rect_copy)
            # images_bounds.append(rect_copy)
            # marche pas car scaling non preservé --> voir comment faire en fait
        # cree un clone de la classe mais un peu complexe en fait --> car
        # clone_rect_type = type(col1)
        # new_clone_with_rect = clone_rect_type(*images_bounds)
        # new_clone_with_rect.setX(col1.x())
        # new_clone_with_rect.setY(col1.y())
        # new_clone_with_rect.setWidth(col1.width())
        # new_clone_with_rect.setHeight(col1.height())
        #
        # shapes_to_draw.append(new_clone_with_rect)
        # put a + and add a rect to it maybe
        # see


        w.paint.EZFIG_panel.shapes_to_draw = shapes_to_draw # TODO -> make a function for that and hide it so that it is not possible to access it directly
        w.paint.EZFIG_panel.update_size()





        # ça marche --> tt marche --> voir comment faire la suite

    # w.setWindowIcon(app_icon)

    # NB il y a des bugs d'elements qui sont à droite --> petit pb d'alignement --> probablement un pb
    # faire toujours fitter le plot à sa taille --> TODO
    #

    w.show()
    sys.exit(app.exec_())
