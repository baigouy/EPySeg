# shall I put third party in the about infos --> that may make sense too
# or best is to put a link to third party in there

# why is CUDA called in MT for 'Measure cell properties' --> there is a big MT error in MT for some reason --> why
# NB my worker2 is not very smart --> recode all of the multithreading properly some day !!!! (I use a threadpool which is useless too, but ok for now as the non killing of threads otherwise creates dramatic pbs, in a not very reproducible way !!!)

# weird bug --> if no images analyzed if I take the trash_test_mem/mini_vide images then select the channel and do 'deep learning seg', then go to properties then go to 'analysis' and press 'measure cell properties' with MT active then it freezes ...

# TODO --> ask for what model weigths to use for surface projection --> offer between the 3 models pushed --> results will depend a lot on the model!!!


'''
error track cells static tissue not showing progress even though it's working properly...

NEW BUG --> fixed

Traceback (most recent call last):
  File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/ta/pyqt_test_threads_instead_of_threadpool.py", line 89, in run
    result = self.fn(*self.args, **self.kwargs)M
  File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/ta/GUI/pyta.py", line 1812, in _track_cells_static
    warp_using_mermaid_if_map_is_available=warp_using_mermaid_if_map_is_available, pre_register=True, progress_callback=progress_callback)
  File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/ta/tracking/last_tracking_based_on_matching_or_on_translation_from_mermaid_warp.py", line 693, in match_by_max_overlap_lst
    match_by_max_overlap(original_t1, original_t0, channel_of_interest=channel_of_interest, recursive_assignment=recursive_assignment, warp_using_mermaid_if_map_is_available=warp_using_mermaid_if_map_is_available, pre_register=pre_register)
  File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/ta/tracking/last_tracking_based_on_matching_or_on_translation_from_mermaid_warp.py", line 197, in match_by_max_overlap
    track_t1 = assign_random_ID_to_missing_cells(track_t1, label_t1, regprps=rps_label_t1, assigned_ids=assigned_IDs)
  File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/ta/tracking/tools.py", line 257, in assign_random_ID_to_missing_cells
    new_col = get_unique_random_color_int24(forbidden_colors=assigned_ids, assign_new_col_to_forbidden=True)
  File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/ta/colors/colorgen.py", line 38, in get_unique_random_color_int24
    color = get_unique_random_color_int24(forbidden_colors=forbidden_colors)
  File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/ta/colors/colorgen.py", line 38, in get_unique_random_color_int24
    color = get_unique_random_color_int24(forbidden_colors=forbidden_colors)
  File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/ta/colors/colorgen.py", line 38, in get_unique_random_color_int24
    color = get_unique_random_color_int24(forbidden_colors=forbidden_colors)
  [Previous line repeated 2988 more times]
  File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/ta/colors/colorgen.py", line 30, in get_unique_random_color_int24
    r = random.getrandbits(8)
RecursionError: maximum recursion depth exceeded while calling a Python object

solutions:

https://stackoverflow.com/questions/6809402/python-maximum-recursion-depth-exceeded-while-calling-a-python-object

import sys
sys.setrecursionlimit(10000)

'''

# TODO --> do quantif --> choose angle --> TODO on first and last
# do plots of the stretch for all the cells --> TODO

# TODO ajouter les border exclusion dans les plots si possible --> si la table contient ce qu'il faut pour le faire --> verifier aussi si les plots 3D marchent et sont possibles --> à faire en fait
# check how I can handle border cells and alike ???
# see how I can get the displayed image as something I can use in EZF for example --> good idea
# my border bonds work --> shall I now add border plus one for bonds because they can be wrong too because they can be cut --> detect them by having a vertex touching the border of the image or in close proximity of the image border --> easy to implement
# also do the same for the vertices
# marche parfaitement --> juste ajouter vertices border and border plus one and stuff alike !!!

# TODO add the parameters for the surface projection --> need create the db and load the tracked cells resized file if exists

# self.blinker.blink([self.paint.channel_label, self.paint.channels]) --> put blinkers everywhere where needed
# TODO finalize the tracking correction then stop and write the MS!!!

# almost all ok now finalize the last connections and the seg and tracks correction and write the MS
# logging added just add blinker and connect everything to the logger

# TODO try the redirect to a qlabel instead of a qtextbrowser and change my logger --> in fact keep the qtextbrowser
# --> see where to put it though!!! --> in a popup window or within main gui in a small window --> maybe below the list --> good idea or above the list with a name log --> TODO

# TODO add progress bar then connect all stuff

# connect the tracking add an advanced correction, finalize wshed to make it faster...
# faire la surface proj puis faire directement passer les properties du fichiers vers la db --> TODO --> should not be too hard I think
# tip -->  to set the tab by its name tabwidget.setCurrentWidget(tabwidget.findChild(QWidget, tabname))
# add support for
# add support for scale bar in the TA viewer --> put it as a separate panel on the size that is not scrollable maybe --> good idea and to which I will have no interaction with --> and allow save it on export too --> should not be too hard
# also offer the conversion of areas using the user pixel size

# maybe could say files not supported if they have a time dimension

# faire un smart get file for every list --> so that I always have the same command and things get adjusted depending on input

# TODO see how to handle complex stacks 3D or ND --> TODO
# maybe allow popup of napari window when necessary...
# TODO test the python viewer napari maybe...
# URGENT TODO whenever I do anything to the image --> save in the db that I did something so that I know everything need be updated --> will prevent a lot of bugs I had in TA
# TODO read this https://blog.miguelgrinberg.com/post/how-to-make-python-wait to better handle wait especially for 3D stacks

# https://doc.qt.io/qt-5/macos-issues.html --> again a lot of mac specific crap...  but good thing is right click is supported
# set taskbar icon https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105

import os
import os.path
import traceback
from functools import partial
from PyQt5.QtWidgets import QComboBox, QProgressBar, \
    QVBoxLayout, QLabel, QTextBrowser, QGroupBox, QDoubleSpinBox, QCheckBox, \
    QRadioButton, QButtonGroup, QDialogButtonBox, QDialog, QHBoxLayout, QScrollArea, QMessageBox
from PyQt5.QtGui import QPalette, QPixmap, QColor, QPainter, QBrush, QPen, QFontMetrics, QTextCursor, \
    QTextCharFormat
from PyQt5.QtWidgets import QStackedWidget
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QFrame, QTabWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer, QThreadPool
from epyseg.deeplearning.deepl import EZDeepLearning
from epyseg.dialogs.opensave import saveFileDialog
from epyseg.img import Img, blend, to_stack, mask_colors
from epyseg.ta.GUI.denoise_recursion_dialog import DenoiseRecursionDialog
from epyseg.ta.GUI.finish_all_dialog import FinishAllDialog
from epyseg.ta.GUI.overlay_hints import Overlay
from epyseg.ta.GUI.tracking_static_dialog import TrackingDialog
from epyseg.ta.tracking.last_tracking_based_on_matching_or_on_translation_from_mermaid_warp import \
    match_by_max_overlap_lst
from epyseg.tools.qthandler import XStream, QtHandler
from epyseg.uitools.blinker import Blinker
from epyseg.utils.loadlist import create_list
from epyseg.ta.database.preview_and_edit_sqlite_db import Example
from epyseg.ta.database.advanced_sql_plotter import plot_as_any
from epyseg.ta.database.sql import populate_table_content, createMasterDB
from epyseg.ta.GUI.watershed_dialog import WshedDialog
from epyseg.ta.minimal_code_surf_proj_pyTA import surface_projection_pyta
# from epyseg.ta.pyqt_test_threads_instead_of_threadpool import Worker2, FakeWorker2
from epyseg.ta.GUI.stackedduallist import dualList
from epyseg.ta.GUI.tascrollablepaint import tascrollablepaint
from epyseg.ta.luts.lut_minimal_test import list_availbale_luts, apply_lut, PaletteCreator
from epyseg.ta.segmentation.neo_wshed import wshed
from epyseg.ta.measurements.TAmeasures import TAMeasurements
from epyseg.ta.tracking.local_to_track_correspondance import add_localID_to_trackID_correspondance_in_DB
from epyseg.ta.tracking.tools import smart_name_parser
import qtawesome as qta  # icons
from epyseg.tools.early_stopper_class import early_stop
import logging
from epyseg.ta.tracking.tracking_error_detector_and_fixer import help_user_correct_errors
from epyseg.ta.tracking.tracking_yet_another_approach_pyramidal_registration_n_neo_swapping_correction import \
    track_cells_dynamic_tissue
from epyseg.ta.GUI.input_text_dialog import QPlainTextInputDialog
from epyseg.ta.deep.fixed_script_yonit import run_seg
from epyseg.tools.logger import TA_logger  # logging
from epyseg.ta.deep.create_model_for_projection import \
    create_surface_projection_denoise_and_height_map_combinatorial_model
from epyseg.worker.fake import FakeWorker
from epyseg.worker.threaded import Worker

logger = TA_logger()

# allow high dpi scaling only on systems that support it it's really cool and I should have this in all main classes of PyQT stuff
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

# can I create an on the fly stuff ??? for the masterdb
# maybe just for one command --> maybe that is the smartest way of doing this
# and plot only for the table

__MAJOR__ = 1
__MINOR__ = 0
__MICRO__ = 0
__RELEASE__ = 'b'  # https://www.python.org/dev/peps/pep-0440/#public-version-identifiers --> alpha beta, ...
__VERSION__ = ''.join([str(__MAJOR__), '.', str(__MINOR__), '.'.join([str(__MICRO__)]) if __MICRO__ != 0 else '', __RELEASE__])
__AUTHOR__ = 'Benoit Aigouy'
__NAME__ = 'PyTA: Python Tissue Analyzer'
__EMAIL__ = 'baigouy@gmail.com'

DEBUG = False

class TissueAnalyzer(QtWidgets.QMainWindow):
    # stop_threads = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

        # global early_stop
        # early_stop = False
        self.setAcceptDrops(True)  # we accept DND but just to transfer it to the current list handlers!!!

    def initUI(self):
        # this is a timer to delay the preview updates upon spin valuechanged
        delayed_preview_update = QTimer()
        delayed_preview_update.setSingleShot(True)
        delayed_preview_update.timeout.connect(self.preview_changed)

        self.master_db = None  # will store the master db if exists # need close it otherwise # if fast enough --> just do it on the fly maybe --> see how I can do that
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
        self.setWindowIcon(qta.icon('mdi.hexagon-multiple-outline',
                                    color=QColor(200, 200, 200)))  # very easy to change the color of my icon
        # 'mdi.hexagon-outline'

        # zoom parameters
        self.scale = 1.0
        self.min_scaling_factor = 0.1
        self.max_scaling_factor = 20
        self.zoom_increment = 0.05

        self.setWindowTitle(__NAME__ + ' v' + str(__VERSION__))

        # this is the default paint window of TA
        self.paint = tascrollablepaint()
        # this is the properties window (allows to set useful parameters that can be used by TA

        self.Stack = QStackedWidget(self)
        self.properties_table = Example(table_name='properties')
        self.last_opened_file_name = QLabel('')

        # self.list = QListWidget(self)  # a list that contains files to read or play with
        # self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.list.selectionModel().selectionChanged.connect(self.selectionChanged)  # connect it to sel change
        self.list = dualList()
        self.list.setToolTip("Drag and drop your images over this list")
        for lst in self.list.lists:
            lst.list.selectionModel().selectionChanged.connect(self.selectionChanged)
            # print list_connected

        self.table_widget = QWidget()
        table_widget_layout = QGridLayout()
        # table_widget_layout.setColumnStretch(0, 60)
        # table_widget_layout.setColumnStretch(1, 40)

        # Initialize tab screen
        self.tabs = QTabWidget(self)
        self.tab0 = QWidget()
        # self.tab0.setToolTip("jsdhqsjkdhkj sdqsh d")
        self.tab1 = QWidget()
        # self.tab1b = QWidget()
        self.tab1c = QWidget()
        self.tab2 = QWidget()
        self.tab2a = QWidget()
        self.tab2b = QWidget()
        self.tab2c = QWidget()
        self.tab2d = QWidget()
        self.tab3 = QScrollArea()
        content_widget = QWidget()
        self.tab3.setWidget(content_widget)
        self.tab3.setWidgetResizable(True)
        self.tab4 = QWidget()
        # self.tabs.resize(300, 200)
        # self.tabs.setFixedHeight(150)

        # Add tabs
        self.tabs.addTab(self.tab0, 'Pre-process')  # contains stuff to do max projs and heightmaps from images
        self.tabs.setTabToolTip(0,
                                'Use a CARE model to perform surface extraction/create a 2D projection from a 3D epithelial stack.')
        self.tabs.addTab(self.tab1, 'Segmentation')
        self.tabs.setTabToolTip(1, 'Segment epithelial images.')
        self.tabs.addTab(self.tab1c, 'Properties')  # things like time...
        self.tabs.setTabToolTip(2,
                                'Preview image properties (e.g. time, voxel ratios, ...) extracted during the pre-process step.')
        # self.tabs.addTab(self.tab1b, 'Correction') # same as segmentation in the end --> redundant
        self.tabs.addTab(self.tab2, 'Analysis')
        self.tabs.setTabToolTip(3,
                                'Collects various cell infos (e.g. cell area, perimeter, ...) from the segmented mask.')
        # self.tabs.addTab(self.tab2a, 'Polarity') # TODO --> add at some point
        self.tabs.addTab(self.tab2b, 'Tracking')  # track cells first then bonds
        self.tabs.setTabToolTip(4, 'Track cells (Segmentation is required)')
        self.tabs.addTab(self.tab2c, 'Advanced segmentation and tracking correction')  # track cells first then bonds
        self.tabs.setTabToolTip(5,
                                'Easily find out segmentation and tracking errors (Segmentation and Tracking required).')
        # self.tabs.addTab(self.tab2d, 'Plots')  # track cells first then bonds
        self.tabs.addTab(self.tab2d, 'Export (CSV files)')  # track cells first then bonds

        self.tabs.addTab(self.tab3, 'Preview')
        self.tabs.setTabToolTip(7, 'Create nice looking images for your publications.')
        self.tabs.addTab(self.tab4, 'PyTA Settings')  # GUI/deep learning/procs MT settings --> TODO
        self.tabs.setTabToolTip(8, 'Various GUI settings.')  # GUI/deep learning/procs MT settings --> TODO

        self.tabs.setCurrentIndex(1)
        self.tabs.currentChanged.connect(self.onTabChange)

        # force seg tab to appear first

        self.list.set_list(1)

        # Create first tab
        self.tab0.layout = QVBoxLayout()
        self.surf_proj = QPushButton("Surface projection")
        self.surf_proj.setToolTip(
            "Extracts, using a CARE model without the denoising module,\nthe apical fluorescent signal from a 3D image of an epithelium and creates a height map")
        self.surf_proj.clicked.connect(self.surf_proj_run)
        self.tab0.layout.addWidget(self.surf_proj)
        self.denoising_surf_proj = QPushButton("Denoising surface projection")
        self.denoising_surf_proj.setToolTip(
            "Same as \"Surface projection\" using the complete CARE model, denoiser included.\nBe aware that the denoising module can generate unexpected/unwanted results,\nespecially, when the input data significantly differs from the data used to train the model.\nSo, altogether, denoising/image reconstruction is not necessarily improving the quality of the 2D projection.")
        self.denoising_surf_proj.clicked.connect(self.surf_proj_run)
        self.tab0.layout.addWidget(self.denoising_surf_proj)
        self.tab0.setLayout(self.tab0.layout)

        # Create first tab
        self.tab1.layout = QVBoxLayout()

        self.pushButton0 = QPushButton("Deep Learning/EPySeg segmentation")
        self.pushButton0.clicked.connect(self.epyseg_seg)
        self.pushButton0.setToolTip("Segment an epithelium using the EPySeg model")
        self.tab1.layout.addWidget(self.pushButton0)

        self.pushButton1 = QPushButton("Watershed segmentation")
        self.pushButton1.clicked.connect(self.run_watershed)
        self.pushButton1.setToolTip("Segment an epithelium using the wahtershed algorithm")
        self.tab1.layout.addWidget(self.pushButton1)
        self.tab1.setLayout(self.tab1.layout)
        # self.pushButton2 = QPushButton("Deep learning segmentation")
        # self.pushButton2.clicked.connect(self.deep_learning)
        # self.tab1.layout.addWidget(self.pushButton2)

        # self.tab1c.layout = QVBoxLayout()
        # self.add_column_to_properties_table_button = QPushButton("Add column")
        # self.add_column_to_properties_table_button.clicked.connect(self.add_column_to_properties_table)
        # self.tab1c.layout.addWidget(self.add_column_to_properties_table_button)
        # self.tab1c.setLayout(self.tab1c.layout)

        self.tab2.layout = QVBoxLayout()
        self.finish_all_button = QPushButton("Measure cell properties")
        self.finish_all_button.setToolTip(
            "Extracts numerous metrics from a segmented image.\nTo obtain 3D measures, a height map is required (see Pre-processing).")
        self.finish_all_button.clicked.connect(self.finish_all)
        self.tab2.layout.addWidget(self.finish_all_button)
        self.tab2.setLayout(self.tab2.layout)

        self.tab2b.layout = QVBoxLayout()
        self.track_cells_static_button = QPushButton("Track cells (static tissue)") # otherwise call it algo 1 and 2
        self.track_cells_static_button.setToolTip( "Track cells (static tissue).\nUse this method whenever the cells (or the field of view) don't move much between frames or if you want to track cells rapidly.")
        # self.track_cells_static_button.setEnabled(False)
        self.track_cells_static_button.clicked.connect(self.track_cells_static)
        self.tab2b.layout.addWidget(self.track_cells_static_button)
        self.track_cells_dynamic_button = QPushButton("Track cells")
        self.track_cells_dynamic_button.setToolTip(
            "Track cells (dynamic tissue).\nUse this method whenever the cells (or the field of view) move a lot between consecutive frames.")
        self.track_cells_dynamic_button.clicked.connect(self.track_cells_dynamic)
        self.tab2b.layout.addWidget(self.track_cells_dynamic_button)
        self.tab2b.setLayout(self.tab2b.layout)

        self.tab2c.layout = QVBoxLayout()
        self.fix_mask_and_tracks_button = QPushButton("Fix masks and tracks dynamically")
        self.fix_mask_and_tracks_button.setToolTip(
            "Tool to visually help you find out segmentation and tracking errors.")
        self.fix_mask_and_tracks_button.clicked.connect(self.fix_mask_and_tracks)
        self.tab2c.layout.addWidget(self.fix_mask_and_tracks_button)
        self.tab2c.setLayout(self.tab2c.layout)

        self.tab2d.layout = QVBoxLayout()
        self.export_cells_button = QPushButton("Export cell data")
        self.export_cells_button.setToolTip("Export the TA cell databases as a CSV (tab separated) file")
        self.export_cells_button.clicked.connect(self.export_cells)
        self.tab2d.layout.addWidget(self.export_cells_button)
        self.export_bonds_button = QPushButton("Export bond data")
        self.export_bonds_button.setToolTip("Export the TA bond databases as a CSV (tab separated) file")
        self.export_bonds_button.clicked.connect(self.export_bonds)
        self.tab2d.layout.addWidget(self.export_bonds_button)
        self.export_SQL_button = QPushButton("Export any SQL command (For experts only)")
        self.export_SQL_button.setToolTip("Export any SQL command as a CSV (tab separated) file")
        self.export_SQL_button.clicked.connect(self.export_SQL)
        self.tab2d.layout.addWidget(self.export_SQL_button)
        self.tab2d.setLayout(self.tab2d.layout)

        content_widget.layout = QVBoxLayout()
        sub_layout_preview = QGridLayout()
        sub_layout_preview.setColumnStretch(0, 10)
        sub_layout_preview.setColumnStretch(1, 90)

        preview_img_label = QLabel("Image/Data")
        sub_layout_preview.addWidget(preview_img_label, 0, 0)

        # content_widget.layout.addWidget(preview_img_label)
        self.image_preview_combo = QComboBox()
        self.image_preview_combo.currentTextChanged.connect(self.preview_changed)
        sub_layout_preview.addWidget(self.image_preview_combo, 0, 1)
        content_widget.layout.addLayout(sub_layout_preview)

        self.groupBox_color_coding = QGroupBox('Color coding:')
        self.groupBox_color_coding.setCheckable(True)
        self.groupBox_color_coding.setChecked(False)
        self.groupBox_color_coding.toggled.connect(self.preview_changed)

        color_coding_layout = QHBoxLayout()
        color_coding_layout.setContentsMargins(0, 0, 0, 0)
        color_coding_layout.setAlignment(Qt.AlignLeft)
        # no local global
        # color_coding_label = QLabel('Color coding: ')
        # color_coding_layout.addWidget(color_coding_label, 0, 0)

        # self.radioButton1 = QRadioButton('No')
        self.radioButton2 = QRadioButton('Current/Local')
        self.radioButton2.setToolTip("Apply local (only for the current frame) color coding")
        self.radioButton3 = QRadioButton('Global')
        self.radioButton3.setToolTip("Apply global color coding (i.e. max and min is the max and min of all frames)")
        # self.radioButton3.setEnabled(False)  # not supported yet --> reactivate it when it will be supported # see if I can show a scalebar somewhere in the stuff
        self.radioButton2.setChecked(True)

        # self.radioButton1.toggled.connect(self.preview_changed)
        self.radioButton2.toggled.connect(self.preview_changed)
        self.radioButton3.toggled.connect(self.preview_changed)

        # color_coding_layout.addWidget(self.radioButton1, 0, 1)
        color_coding_layout.addWidget(self.radioButton2)
        color_coding_layout.addWidget(self.radioButton3)

        predict_output_radio_group = QButtonGroup()
        # predict_output_radio_group.addButton(self.radioButton1)
        predict_output_radio_group.addButton(self.radioButton2)
        predict_output_radio_group.addButton(self.radioButton3)

        self.excluder_label = QCheckBox('Exclude % lower and upper outliers: ')
        self.excluder_label.setEnabled(True)
        self.excluder_label.setChecked(False)

        self.excluder_label.stateChanged.connect(self.upper_or_lower_limit_changed)

        # TODO enable or disable the stuffs along with it
        color_coding_layout.addWidget(self.excluder_label)

        self.upper_percent_spin = QDoubleSpinBox()
        self.upper_percent_spin.setEnabled(False)
        self.upper_percent_spin.setRange(0., 0.49)
        self.upper_percent_spin.setSingleStep(0.01)
        self.upper_percent_spin.setValue(0.05)
        # self.upper_percent_spin.valueChanged.connect(self.upper_or_lower_limit_changed)
        self.upper_percent_spin.valueChanged.connect(lambda x: delayed_preview_update.start(600))

        self.lower_percent_spin = QDoubleSpinBox()
        self.lower_percent_spin.setEnabled(False)  # TODO code and enable this soon but ok for now
        self.lower_percent_spin.setRange(0., 0.49)
        self.lower_percent_spin.setSingleStep(0.01)
        self.lower_percent_spin.setValue(0.05)
        # self.lower_percent_spin.valueChanged.connect(self.upper_or_lower_limit_changed)
        self.lower_percent_spin.valueChanged.connect(lambda x: delayed_preview_update.start(600))

        color_coding_layout.addWidget(self.lower_percent_spin)
        color_coding_layout.addWidget(self.upper_percent_spin)

        lut_label = QLabel('Lut: ')
        color_coding_layout.addWidget(lut_label)

        # populate LUTs --> once for good at the beginning of the table
        self.lut_combo = QComboBox()
        available_luts = list_availbale_luts()
        for lut in available_luts:
            self.lut_combo.addItem(lut)

        self.lut_combo.currentTextChanged.connect(self.lut_changed)
        color_coding_layout.addWidget(self.lut_combo)

        self.groupBox_color_coding.setLayout(color_coding_layout)

        # content_widget.layout.addLayout(color_coding_layout)
        content_widget.layout.addWidget(self.groupBox_color_coding)

        self.groupBox_overlay = QGroupBox('Overlay/Blend')
        self.groupBox_overlay.setToolTip("Overlay selected image over the selected background image")
        self.groupBox_overlay.setCheckable(True)
        self.groupBox_overlay.setChecked(False)
        self.groupBox_overlay.toggled.connect(self.preview_changed)

        overlay_layout = QHBoxLayout()
        overlay_layout.setContentsMargins(0, 0, 0, 0)  # self.groupBox_overlay.
        overlay_layout.setAlignment(Qt.AlignLeft)
        # TODO --> maybe show and or hide the transparency settings depending on the selection
        # self.overlay_check = QCheckBox("Enable overlay/blend")
        # self.overlay_check.stateChanged.connect(self.preview_changed)
        # overlay_layout.addWidget(self.overlay_check, 0, 0)

        overlay_bg_label = QLabel("Background image channel:")  # for overlay
        overlay_layout.addWidget(overlay_bg_label)

        self.overlay_bg_channel_combo = QComboBox()
        overlay_layout.addWidget(self.overlay_bg_channel_combo)

        overlay_fg_transparency_label = QLabel("Foreground opacity:")
        overlay_layout.addWidget(overlay_fg_transparency_label)

        self.overlay_fg_transparency_spin = QDoubleSpinBox()
        self.overlay_fg_transparency_spin.setSingleStep(0.05)
        self.overlay_fg_transparency_spin.setRange(0.05, 1.)
        # self.overlay_fg_transparency_spin.valueChanged.connect(self.preview_changed) # direct update upon value change --> not always very smart... better wait some time
        self.overlay_fg_transparency_spin.valueChanged.connect(lambda x: delayed_preview_update.start(600))

        self.overlay_fg_transparency_spin.setValue(0.3)
        overlay_layout.addWidget(self.overlay_fg_transparency_spin)

        self.groupBox_overlay.setLayout(overlay_layout)
        content_widget.layout.addWidget(self.groupBox_overlay)
        # self.tab3.layout.addLayout(overlay_layout)

        self.groupBox_export = QGroupBox('Export')
        export_layout = QHBoxLayout()
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setAlignment(Qt.AlignLeft)

        self.export_image_button = QPushButton("Single image")
        self.export_image_button.setToolTip("Export as tif the current view")
        self.export_image_button.clicked.connect(self.export_image)
        export_layout.addWidget(self.export_image_button)

        self.export_stack_button = QPushButton("Stack")
        self.export_stack_button.setToolTip("Export as a tif stack the current view")
        self.export_stack_button.clicked.connect(self.export_stack)
        export_layout.addWidget(self.export_stack_button)

        self.groupBox_export.setLayout(export_layout)
        content_widget.layout.addWidget(self.groupBox_export)

        # TODO --> maybe offer overlays/ image composition here !!! --> in a way very easy but there will be issues sometimes maybe --> just think about it ???
        # for method in EZDeepLearning.available_model_architectures:
        #     self.image_preview_combo.addItem(method)
        content_widget.setLayout(content_widget.layout)

        self.tab4.layout = QVBoxLayout()
        self.force_CPU_check = QCheckBox(
            'Use CPU for deep learning (slower but often offers more memory) (MUST BE SET IMMEDIATELY AFTER LAUNCHING THE SOFTWARE)')
        self.force_CPU_check.setToolTip(
            "Do not use a graphic card even when a compatible one is available on the system")
        self.force_CPU_check.stateChanged.connect(self.force_CPU)  # need a fix here
        self.tab4.layout.addWidget(self.force_CPU_check)
        # TODO --> do the code to force
        self.tab4.setLayout(self.tab4.layout)

        # can maybe ask the nb of threads to use here too
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

        # Add tabs to widget
        # table_widget_layout.addWidget(self.tabs)
        # table_widget_layout.addWidget(self.logger_console) # not bad but maybe put it somewhere else --> such as on the side of the tabs in a gridlayout with all the necessary stuff!!

        table_widget_layout.addWidget(self.tabs, 0, 0)
        # ok just store console in a qgroup stuff

        self.groupBox_logging = QGroupBox('Log')
        self.groupBox_logging.setToolTip("Shows TA current status, method progress, warnings and errors")
        self.groupBox_logging.setMinimumWidth(250)
        self.groupBox_logging.setEnabled(True)
        # groupBox layout
        self.groupBox_pretrain_layout = QGridLayout()
        self.groupBox_pretrain_layout.addWidget(self.logger_console, 0, 0)
        self.groupBox_logging.setLayout(self.groupBox_pretrain_layout)

        table_widget_layout.addWidget(self.groupBox_logging, 0,
                                      1)  # not bad but maybe put it somewhere else --> such as on the side of the tabs in a gridlayout with all the necessary stuff!!
        self.table_widget.setLayout(table_widget_layout)

        # voilà le stackwidget --> et comment on ajoute des trucs dedans --> permet de ne montrer que certains widgets

        self.volumeViewer = QWidget()  # maybe get it to simply pop out ???
        # ça c'est juste la layout du stack --> on s'en fout

        # TODO replace that with napari
        # self.volumeViewerUI()
        self.Stack.addWidget(self.paint)
        self.Stack.addWidget(self.properties_table)
        # self.Stack.addWidget(self.volumeViewer)

        # create a grid that will contain all the GUI interface
        self.grid = QGridLayout()
        # grid.setSpacing(10)

        # grid.addWidget(self.scrollArea, 0, 0)
        self.grid.addWidget(self.Stack, 0, 0)
        # self.grid.addWidget(self.Stack, 0, 0,2,1)
        self.grid.addWidget(self.list, 0, 1)
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

        statusBar.addWidget(self.last_opened_file_name)

        # Set up menu bar
        self.mainMenu = self.menuBar()
        self.setMenuBar(self.mainMenu)

        # Setup hotkeys for whole system
        # Delete selected vectorial objects
        deleteShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self)
        deleteShortcut.activated.connect(self.down)
        deleteShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active

        # shall I remove this useless shortcut
        # set drawing window fullscreen
        fullScreenShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_F), self)
        fullScreenShortcut.activated.connect(self.fullScreen)
        fullScreenShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active

        # set drawing window fullscreen
        fullScreenShortcut2 = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_F12), self)
        fullScreenShortcut2.activated.connect(self.fullScreen)
        fullScreenShortcut2.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active

        # exit from full screen TODO add quit the app too ??
        escapeShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        escapeShortcut.activated.connect(self.escape)
        escapeShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active

        spaceShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        spaceShortcut.activated.connect(self.nextFrame)
        spaceShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active

        backspaceShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self)
        backspaceShortcut.activated.connect(self.prevFrame)
        backspaceShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active

        self.blinker = Blinker()
        self.to_blink_after_worker_execution = None
        self.threading_enabled = True

        # self.threads = [] # contains all threads so that I can stop them immediately
        # self.thread = None
        # self.event_stop = threading.Event() # allows to stop a thread
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(self.threadpool.maxThreadCount() - 1)  # spare one core for system

        self.overlay = Overlay(self.centralWidget())  # maybe ok if draws please wait somewhere
        self.overlay.hide()

        try:
            self.oldvalue_CUDA_DEVICE_ORDER = os.environ["CUDA_DEVICE_ORDER"]
            # print('ok', self.oldvalue_CUDA_DEVICE_ORDER)
        except:
            self.oldvalue_CUDA_DEVICE_ORDER = ''

        try:
            self.oldvalue_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
            # print('ok', self.oldvalue_CUDA_VISIBLE_DEVICES)
        except:
            self.oldvalue_CUDA_VISIBLE_DEVICES = ''

        # button.clicked.connect(self.overlay.show)

        # self.setAcceptDrops(True)  # KEEP IMPORTANT # --> nb can I delegate the dnd of self to a progeny

    def dragEnterEvent(self, event):
        # we transfer all the DNDs to the list
        selected_tab_idx, _ = self.get_cur_tab_index_and_name()
        self.list.get_list(selected_tab_idx).dragEnterEvent(event)

    def dragMoveEvent(self, event):
        # we transfer all the DNDs to the list
        selected_tab_idx, _ = self.get_cur_tab_index_and_name()
        self.list.get_list(selected_tab_idx).dragMoveEvent(event)

    def dropEvent(self, event):
        # we transfer all the DNDs to the list
        selected_tab_idx, _ = self.get_cur_tab_index_and_name()
        self.list.get_list(selected_tab_idx).dropEvent(event)

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

    def resizeEvent(self, event):
        self.overlay.resize(event.size())
        event.accept()

    def progress_fn(self, current_progress):
        '''basic progress function

        '''
        print("%d%% done" % current_progress)
        self.pbar.setValue(current_progress)

    def _update_channels(self, img):

        # if not       self.overlay_check.isChecked():
        #     self.overlay_bg_channel_combo.disconnect()
        #     self.overlay_bg_channel_combo.clear()
        #     self.overlay_bg_channel_combo.currentIndexChanged.connect(self.preview_changed)
        #     return

        if isinstance(img, str):
            img = Img(img)

        selection = self.overlay_bg_channel_combo.currentIndex()
        self.overlay_bg_channel_combo.disconnect()
        self.overlay_bg_channel_combo.clear()
        comboData = ['merge']
        if img is not None:
            try:
                if img.has_c():
                    for i in range(img.get_dimension('c')):
                        comboData.append(str(i))
            except:
                # assume image has no channel and return None
                # or assume last stuff is channel if image has more than 2
                pass

        self.overlay_bg_channel_combo.addItems(comboData)

        if selection != -1 and selection < self.overlay_bg_channel_combo.count():
            self.overlay_bg_channel_combo.setCurrentIndex(selection)
        else:
            self.overlay_bg_channel_combo.setCurrentIndex(0)
        # check if it should be here or not
        self.overlay_bg_channel_combo.currentIndexChanged.connect(self.preview_changed)

    # maybe reset only if image changed otherwise update --> see hwo to do that
    # code below can be simplified
    def update_preview_depending_on_selected_tab(self):
        # selected_tab_name = self.tabs.currentWidget()
        selected_tab_idx = self.tabs.currentIndex()

        self.list.set_list(selected_tab_idx)
        list = self.list.get_list(
            selected_tab_idx).list  # NB CHANGE THIS --> NOT SMART TO USE LIST AS A NAME --> MAY CAUSE TROUBLE WITH REAL LIST TYPE OF PYTHON --> call it othrewise !!!!
        # TODO replace by self.get_full_list or something alike
        selected_items = list.selectedItems()
        selected_tab_name = self.tabs.tabText(selected_tab_idx).lower()

        # store file within the tab --> and no relaod except if forced
        # debug
        logger.debug('selected_tab_name "' + str(selected_tab_name) + '" ' + str(selected_tab_idx))

        if not 'review' in selected_tab_name:
            try:
                # close master db and set it to null
                if self.master_db is not None:
                    print('closing master db')
                    self.master_db.close()
            except:
                pass
            finally:
                self.master_db = None
        # print('selected_tab_name "',selected_tab_name,'"', selected_tab_idx)

        if 'roperties' in selected_tab_name:
            self.list.setDisabled(True)  # useful to diable this to avoid having trouble such as people deleting stuff
            # TODO need find a way to reinject stuff if tab changed and
            # maybe if something changed --> reinject everything
            # if anything was changed in the table --> force update

            # show properties and let the user edit them if needed
            self.Stack.setCurrentIndex(1)
            # I need create and set the master db here
            lst = self.get_full_list(warn_on_empty_list=False)
            # if lst is None or not lst:
            #     db = None
            # else:
            #     db = get_properties_master_db(lst)
            # print('db',db)
            # print(db.get_tables())
            # print(db.print_query('SELECT * FROM properties'))
            # print(db.)
            self.properties_table.set_db(lst)
            return
        else:

            # TODO also need update if user quits the soft or I should have a save button to force save ??? --> think about that

            # by default use the paint widget
            self.list.setDisabled(False)
            self.Stack.setCurrentIndex(0)
            self.properties_table.close()

            # not very smart to autosave --> put a save button instead
            # if self.properties_table.update_required:
            #     lst = self.get_full_list(warn_on_empty_list=False)
            #     if lst is not None and lst:
            #         self.properties_table.saveToDb('properties')
            #         self.properties_table.update_required = False
            #         # I need get its db and reinject it
            #         reinject_properties_to_TA_files(lst, TAsql(filename_or_connection=self.properties_table.db_connect))
            #         self.properties_table.close()

        if 'review' in selected_tab_name:

            # could create a masterdb if needed --> need create it once for all
            # TODO I also need to populate the channels of the parent image --> TODO probably not so hard I guess

            # qsdqsdsqqsdsdq

            self.list.freeze(True)
            self.paint.freeze(True, level=2)

            if selected_items:
                selected_file = selected_items[0].toolTip()
                self._update_channels(selected_file)
            else:
                self._update_channels(None)

            self.populate_preview_combo()
            self.preview_changed()
            return
        else:
            self.list.freeze(False)

        # pr settings faudrait desactiver save et alike
        # or 'ettings'

        # ça marche mais faudrait
        # self.paint.freeze(False)

        # faurdrait plusieurs freeze zt faudrait la plupart du temps garder le channel

        if selected_tab_name.startswith('seg'):
            self.paint.freeze(False)
        else:
            self.paint.freeze(True, level=1)

        if selected_items:
            selected_file = selected_items[0].toolTip()

            # if preview tab is selected need run the update

            # whatever happens I need to draw this image except if already loaded

            # try:
            #     # print
            #       # nb this creates a bug and prevents from real channel selection to use for tracking --> do not show this here or find a trick
            #     if selected_tab_name.endswith('racking'):
            #         # dirty trick to block channels, a better solution is required at some point
            #         self.paint.set_image(None)
            #         self.paint.channels.setCurrentIndex(0)
            #         self.paint.set_image(Img(smart_name_parser(selected_file, 'tracked_cells_resized.tif')))
            #         # self.paint.channels.clear()
            #         # self.paint.channels.addItem('merge')
            #
            #         return
            # except:
            #     # probably tracking image does not exist --> just continue
            #     pass

            # start = timer()
            self.paint.set_image(selected_file)
            # self.statusBar().showMessage('Loading ' + selected_items[0].toolTip()) # hides the bar sadly --> not what i want --> need do better --> add a label to the bar
            self.last_opened_file_name.setText(selected_items[0].toolTip())

            # fixed a seg fault in image --> due to inversion of height and witdh for indexed images
            try:
                # update icon if it does not exist yet #TODO recode that in a better way some day
                if list.currentItem() and list.currentItem().icon().isNull():
                    logger.debug('Updating icon')
                    icon = QIcon(QPixmap.fromImage(self.paint.paint.image))
                    pixmap = icon.pixmap(24, 24)
                    icon = QIcon(pixmap)
                    list.currentItem().setIcon(icon)
            except:
                # no big deal if can't create a thumb out of the selected image
                traceback.print_exc()
                logger.warning('failed creating an icon for the selected file')

            # update content depending on tabs in fact
            if selected_tab_name.startswith('seg'):

                # enable mask and drawing
                self.paint.maskVisible = True
                self.paint.enableMouseTracking()
                TA_path_alternative, TA_path = smart_name_parser(
                    selected_file,
                    ordered_output=['handCorrection.png', 'handCorrection.tif'])
                mask_name = None
                if os.path.isfile(TA_path):
                    mask_name = TA_path
                else:
                    if os.path.isfile(TA_path_alternative):
                        mask_name = TA_path_alternative
                if mask_name is not None:
                    self.paint.set_mask(mask_name)
            else:
                # disable mask and drawing

                self.paint.maskVisible = False
                self.paint.disableMouseTracking()
        else:
            self.paint.set_image(None)
            self.last_opened_file_name.setText('')

    def onTabChange(self):
        # tab changed --> therefore need update the image --> TODO
        # tab changed --> therefore need update the image --> TODO
        # maybe should have different behaviours depending on image change or tab change --> TODO
        self.update_preview_depending_on_selected_tab()

    def selectionChanged(self):
        try:
            self.update_preview_depending_on_selected_tab()
        except:
            traceback.print_exc()

    # gros bug --> need hide or show slider rather than delete them
    def clearlayout(self, layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)

    # def replaceWidget(self):
    #     if self.Stack.currentIndex() == 0:
    #         self.Stack.setCurrentIndex(1)
    #         self.selectionChanged()
    #     else:
    #         self.Stack.setCurrentIndex(0)
    #         self.selectionChanged()

    def escape(self):
        if self.Stack.isFullScreen():
            self.fullScreen()

    # brings stuff full screen then restore it back after
    # maybe best is to keep the paint version of full screen ??? --> think about it but ok for now
    def fullScreen(self):
        if not self.Stack.isFullScreen():
            self.Stack.setWindowFlags(
                QtCore.Qt.Window |
                QtCore.Qt.CustomizeWindowHint |
                # QtCore.Qt.WindowTitleHint |
                # QtCore.Qt.WindowCloseButtonHint |
                QtCore.Qt.WindowStaysOnTopHint
            )
            '''
                  self.setWindowFlags(
                    QtCore.Qt.Window |
                    QtCore.Qt.CustomizeWindowHint |
                    QtCore.Qt.WindowTitleHint |
                    QtCore.Qt.WindowCloseButtonHint |
                    QtCore.Qt.WindowStaysOnTopHint
                  )
            '''
            self.Stack.showFullScreen()
        else:
            # settings = QSettings()
            # self.Stack.restoreGeometry(settings.value("geometry")) #.toByteArray()
            self.Stack.setWindowFlags(QtCore.Qt.Widget)
            # self.Stack.setWindowFlags(self.flags)
            self.grid.addWidget(self.Stack, 0, 0)  # pas trop mal mais j'arrive pas à le remettre dans le truc principal
            # dirty hack to make it repaint properly --> obviously not all lines below are required but some are --> need test, the last line is key though
            self.grid.update()
            self.Stack.update()
            self.Stack.show()
            self.centralWidget().setLayout(self.grid)
            self.centralWidget().update()
            self.update()
            self.show()
            self.repaint()
            self.Stack.update()
            self.Stack.repaint()
            self.centralWidget().repaint()

    def about_dialog(self):
        msg = QMessageBox(parent=self)
        # msg.setTextFormat(Qt.RichText)
        msg.setIcon(QMessageBox.Information)
        msg.setText("Python Tissue Analyzer")
        # msg.setInformativeText("License BSD-3\n\nCopyright 2021-2022\n\nBy Benoit Aigouy\n\nbaigouy@gmail.com\n\nPlease check the third party licenses (press 'show details')") # TODO make it dynamic until the current system year
        msg.setInformativeText("License BSD-3\n\nCopyright 2021-2022\n\nBy Benoit Aigouy\n\n<a href='baigouy@gmail.com'>baigouy@gmail.com</a>\n\nPlease check the third party licenses (press 'show details')") # TODO make it dynamic until the current system year
        msg.setWindowTitle("About...")
        # i could also put here the licence and link to the link to something else
        # msg.setDetailedText("pyTA uses numerous third party libraries that come with their own licenses, details can be found here: https://... If you disagree with any of these licenses please uninstall EPySeg")
        msg.setStandardButtons(QMessageBox.Ok)
        # TODO also add the link to the web page on github for the tuto
        retval = msg.exec_()

    def down(self):
        print('down')

    def nextFrame(self):
        print('next frame pressed')

    def prevFrame(self):
        print('prev frame pressed')

    def simulate_progress_in_progressbar(self):
        self.completed = 0

        while self.completed < 100:
            self.completed += 0.0001
            self.pbar.setValue(self.completed)

        self.pbar.setValue(0)

    def get_current_TA_path(self):
        selection = self.get_selection()
        if selection is None:
            return None
        return smart_name_parser(selection, ordered_output='TA')

    def get_selection(self):
        selected_tab_idx, _ = self.get_cur_tab_index_and_name()
        self.list.set_list(selected_tab_idx)
        list = self.list.get_list(selected_tab_idx)
        selection = list.get_selection()
        return selection

    def get_selection_index(self):
        selected_tab_idx, _ = self.get_cur_tab_index_and_name()
        self.list.set_list(selected_tab_idx)
        list = self.list.get_list(selected_tab_idx)
        selection = list.get_selection_index()
        return selection

    def get_full_list(self, warn_on_empty_list=True):
        lst = self.list.get_list(self.tabs.currentIndex()).get_full_list()
        if warn_on_empty_list:
            if lst is None or not lst:
                logger.error('Empty list, please load files first')
                # self.to_blink_after_worker_execution = [self.list]
                self.blinker.blink(self.list)
                return
        return lst

    def check_channel_is_selected(self):
        # print( self.paint.get_selected_channel(), self.paint.channels.count(), self.paint.paint.raw_image is None)
        if self.paint.paint.raw_image is None:
            logger.warning("Please select an image first")
            self.blinker.blink(self.list)
            return False
        if self.paint.get_selected_channel() is None and self.paint.channels.count() > 1:
            logger.warning("Please select a channel first")
            self.blinker.blink([self.paint.channel_label, self.paint.channels])
            return False
        return True

    def _get_worker(self, func, *args, **kwargs):
        # returns the worker to proceed with building, training or running the model
        if self.threading_enabled:
            # threaded worker
            return Worker(func, *args, **kwargs)
        else:
            # non threaded worker
            return FakeWorker(func, *args, **kwargs)

    # def _get_worker_neo(self, func, *args, **kwargs):
    #     # returns the worker to proceed with building, training or running the model
    #     if self.threading_enabled:
    #         # threaded worker
    #         return Worker2(func, *args, **kwargs)
    #     else:
    #         # non threaded worker
    #         return FakeWorker2(func, *args, **kwargs)

    def thread_complete(self):
        '''Called every time a thread completed

        I use it to blink things in case there are errors

        '''
        # print(self.sender())
        # self._set_model_inputs_and_outputs()

        # reset progress upon thread complete
        self.pbar.setValue(0)

        self.overlay.hide()

        self.update_preview_depending_on_selected_tab()

        if self.to_blink_after_worker_execution is not None:
            self.blinker.blink(self.to_blink_after_worker_execution)
            self.to_blink_after_worker_execution = None

    # def print_output(self, s):
    #     print(s)

    # def launch_in_a_tread_bckp2(self, func):
    #     self.pbar.setValue(0)
    #     self.overlay.show()
    #     worker = self._get_worker(func)
    #
    #     self.thread = worker
    #
    #     # worker.signals.result.connect(self.print_output)
    #     worker.signals.finished.connect(self.thread_complete)
    #     # this is specific of this method I must update the nb of inputs and outputs of the model # be CAREFUL IF COPYING THIS CODE THE FOLLOWING MUST BE REMOVED
    #     # worker.signals.finished.connect(self._set_model_inputs_and_outputs)
    #     worker.signals.progress.connect(self.progress_fn)
    #     self.thread.setTerminationEnabled(True)
    #     # Execute
    #     if isinstance(worker, FakeWorker2):
    #         # no threading
    #         worker.run()
    #     else:
    #         # threading
    #         self.threadpool.start(worker)
    #         # self.thread.start()
    #         # self.thread.join()
    #         # self.threads.append(worker)

    def launch_in_a_tread(self, func):
        early_stop.stop = False
        self.pbar.setValue(0)
        self.overlay.show()
        worker = self._get_worker(func)
        # self.thread = worker

        # worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        # this is specific of this method I must update the nb of inputs and outputs of the model # be CAREFUL IF COPYING THIS CODE THE FOLLOWING MUST BE REMOVED
        # worker.finished.connect(self._set_model_inputs_and_outputs)
        worker.signals.progress.connect(self.progress_fn)
        # self.thread.setTerminationEnabled(True)
        # Execute
        if isinstance(worker, FakeWorker):
            # no threading
            worker.run()
        else:
            # threading
            self.threadpool.start(worker)
            # self.thread.start()
            # self.thread.join()
            # self.threads.append(worker)

    # def launch_in_a_tread_neo_buggy(self, func):
    #     # print('launching in a thread')
    #
    #     # stop_threads = False
    #     self.pbar.setValue(0)
    #
    #     # self.thread = QThread()
    #
    #     self.overlay.show()
    #     self.thread = self._get_worker(func)
    #
    #     # self.thread.result.connect(self.print_output)
    #     self.thread.finished.connect(self.thread_complete)
    #     # this is specific of this method I must update the nb of inputs and outputs of the model # be CAREFUL IF COPYING THIS CODE THE FOLLOWING MUST BE REMOVED
    #     # worker.signals.finished.connect(self._set_model_inputs_and_outputs)
    #     self.thread.progress.connect(self.progress_fn)
    #     self.thread.setTerminationEnabled(True)
    #
    #     # Execute
    #     if isinstance(self.thread, FakeWorker2):
    #         # no threading
    #         self.thread.run()
    #     else:
    #         # threading
    #         # self.threadpool.start(worker)
    #         # self.threads.append(worker)
    #         # worker.moveToThread(self.thread)
    #
    #         # self.thread.started.connect(self.thread.run)
    #
    #         # print('inside')
    #
    #         self.thread.start()
    #     # print('end launching in a thread')

    def surf_proj_run(self):
        lst = self.get_full_list(warn_on_empty_list=True)
        # print('lst', lst)
        if lst is None:
            # print('problem --> stopping')
            return

        if not self.check_channel_is_selected():
            return

        recursion = 1
        sender = self.sender()
        if sender == self.denoising_surf_proj:
            recursion, ok = DenoiseRecursionDialog.get_value(parent=self, title="Denoiser options")
            if not ok:
                return

        tmp = partial(self._surf_proj_run, sender=sender, recursion=recursion)

        # self.launch_in_a_tread(self._surf_proj_run)
        self.launch_in_a_tread(tmp)
        # worker = self._get_worker(self._surf_proj_run, model_parameters=None)
        # worker.signals.result.connect(self.print_output)
        # worker.signals.finished.connect(self.thread_complete)
        # # this is specific of this method I must update the nb of inputs and outputs of the model # be CAREFUL IF COPYING THIS CODE THE FOLLOWING MUST BE REMOVED
        # # worker.signals.finished.connect(self._set_model_inputs_and_outputs)
        # worker.signals.progress.connect(self.progress_fn)
        #
        # # Execute
        # if isinstance(worker, FakeWorker):
        #     # no threading
        #     worker.run()
        # else:
        #     # threading
        #     self.threadpool.start(worker)

    def _surf_proj_run(self, sender=None, recursion=1, progress_callback=None):
        # self._enable_training(False)
        # self._enable_predict(False)

        # import time
        # time.sleep(3)
        # Do surface proj and denoising here --> TODO
        # print('NOT IMPLEMENTED YET', self.sender()) # do also save height map

        # pretty good but need to add a progress bar and add controls that the list is not empty
        # print('in here')
        lst = self.get_full_list(warn_on_empty_list=True)
        # print('lst', lst)
        if lst is None:
            # print('problem --> stopping')
            return

        if not self.check_channel_is_selected():
            return

        # TODO need add checks to make sure the stuff is loading properly
        # print('in here2')
        # deepTA = EZDeepLearning(use_cpu=self.force_CPU_check.isChecked())
        deepTA = EZDeepLearning()
        # deepTA.load_or_build(            model='/E/Sample_images/sample_images_pyta/test_merged_model.h5')  # TODO --> replace this by online model --> TODO

        # TODO change this with smthg relevant --> TODO and maybe offer some choice some day
        # surface_proj_model = 'SURFACE_PROJECTION'
        # surface_proj_model = 'SURFACE_PROJECTION_2'
        # surface_proj_model = 'SURFACE_PROJECTION_3' # in some case this is better --> should offer a choice --> would be good to offer the two then  --> is that on the gitlab too --> this one was not pushed --> needs be done
        surface_proj_model = 'SURFACE_PROJECTION_4' # this one was not pushed neither --> need do it too
        # surface_proj_model = 'SURFACE_PROJECTION_5' # this one was not pushed neither --> need do it too
        # surface_proj_model = 'SURFACE_PROJECTION_6' # this one was not pushed neither --> need do it too --> sometimes better with folds, otherwise the 3 or 4 are better
        # denoiser_model = '2D_DENOISER'
        denoiser_model = '2D_DENOISEG'

        # surface_proj_model = '/E/models/my_model/bckup/CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/surface_projection_model.h5'
        # denoiser_model = '/E/models/my_model/bckup/CARESEG_masked_data_and_masked_loss_to_prevent_identity_v1_210107_good/2D_denoiser_model.h5'  # not great on CARE great on others --> really need different denoisers depending on DATA --> ok --> though I doubt people will use CARE like data as input...
        # denoiser_model = '/E/models/my_model/bckup/CARESEG_another_normal_training_201216_colab_but_gave_crap/2D_denoiser_model.h5'# great for CARE not great for the others
        # denoiser_model = '/E/models/my_model/bckup/CARESEG_retrained_normally_201216/2D_denoiser_model.h5'# great for CARE not great for the others
        # denoiser_model = '/E/models/my_model/bckup/CARESEG_trained_on_CARE_data_with_dilation_good/2D_denoiser_model.h5'# great for CARE not great for the others

        # pb is that
        model = create_surface_projection_denoise_and_height_map_combinatorial_model(surface_proj_model, denoiser_model)

        # shall I create a submodel ????
        print(model.summary(line_length=250))
        deepTA.model = model

        # print('sender',sender) # always None

        if sender == self.denoising_surf_proj:
            save_raw_image = False
        else:
            save_raw_image = True
        # --> also need

        # why always true
        # print('save_raw_image', save_raw_image)
        # return

        processed_images = surface_projection_pyta(deepTA, lst, progress_callback=progress_callback,
                                                   save_raw_image=save_raw_image,
                                                   channel=self.paint.get_selected_channel(),
                                                   recursion_for_denoising=recursion)

        # print('save_raw_image',save_raw_image)

        # add all of these files to the next list
        self.list.get_list(1).add_to_list(processed_images, check_if_supported=False)

        del deepTA
        # shall I add helps that guide the user through it
        # pass
        # TODO --> try implement that --> maybe it's a good idea
        # TODO --> do that and send the files to the other list
        # save in a folder named
        # maybe warn if can't write to the folder

        # test if I have this and how to make it in a simple way
        # probably not so hard
        # maybe just retrain a pretrained model so that I get the best of what I want
        # do the fix for adding black images top and bottom automatically
        # TODO --> do a try with the pre process --> should be easy to use in fact !!!
        progress_callback.emit(100)

    # shall I make it irreversible
    def force_CPU(self):
        if self.force_CPU_check.isChecked():
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self.force_CPU_check.setEnabled(
                False)  # somehow irreversible and must be done immediately after launching the software
        else:
            # self.force_CPU_check.setEnabled(False)
            if hasattr(os,
                       'unsetenv') and self.oldvalue_CUDA_DEVICE_ORDER == '' and self.oldvalue_CUDA_VISIBLE_DEVICES == '':
                os.unsetenv("CUDA_DEVICE_ORDER")
                os.unsetenv("CUDA_VISIBLE_DEVICES")
            else:
                os.environ["CUDA_DEVICE_ORDER"] = self.oldvalue_CUDA_DEVICE_ORDER  # see issue #152
                os.environ["CUDA_VISIBLE_DEVICES"] = self.oldvalue_CUDA_VISIBLE_DEVICES

        #     self.oldvalue_CUDA_DEVICE_ORDER = os.environ["CUDA_DEVICE_ORDER"]
        #     except:
        #     self.oldvalue_CUDA_DEVICE_ORDER = ''
        #
        # try:
        #     self.oldvalue_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
        # except:

        # print('TODO implement force_CPU')
        # pass

    def export_stack(self):
        # print('TODO implement export_stack')
        # loop over all images in the list and try append the image and if not the same size then see
        list_of_files = self.get_full_list(warn_on_empty_list=True)
        preview_selected = self.image_preview_combo.currentText()
        if list_of_files is not None:
            images = []
            for file in list_of_files:
                # do the color code and save it as a stack
                # maybe do a stack saver in my image stuff that is smart enough to handle several images
                try:
                    image = self.create_preview(preview_selected, file,
                                                TA_path=smart_name_parser(file, ordered_output='TA'))
                    images.append(image)
                except:
                    logger.warning('could not create file for image"' + str(file) + '"')
                    images.append(None)
            images = [Img(image) if isinstance(image, str) else image for image in images]
            stack = to_stack(images)
            if stack is not None:
                output_file = saveFileDialog(parent_window=self, extensions="Supported Files (*.tif);;All Files (*)",
                                             default_ext='.tif')
                if output_file is not None:
                    # print(stack.shape)# huge bug --> fix it
                    dimensions = 'hw'
                    if stack.shape[-1] == 3:
                        dimensions += 'c'
                        if len(stack.shape) >= 3:
                            dimensions = 'd' + dimensions
                    else:
                        if len(stack.shape) >= 2:
                            dimensions = 'd' + dimensions
                    Img(stack, dimensions=dimensions).save(output_file)

    def export_image(self):
        # TODO export scale bar if there is a LUT
        # print('TODO implement export_image')
        # need get the image  and export it --> TODO
        img_to_save = self.paint.paint.get_raw_image()
        if img_to_save is None:
            logger.error('Nothing to save')
        else:
            # save the image
            output_file = saveFileDialog(parent_window=self, extensions="Supported Files (*.tif);;All Files (*)",
                                         default_ext='.tif')
            if output_file is not None:
                Img(img_to_save).save(output_file, mode='raw')

    def upper_or_lower_limit_changed(self):
        self.lower_percent_spin.setEnabled(self.excluder_label.isChecked())
        self.upper_percent_spin.setEnabled(self.excluder_label.isChecked())
        self.preview_changed()

    def lut_changed(self):
        # if no color code --> do nothing
        if not self.groupBox_color_coding.isChecked():
            return
        # print('TODO implement LUT changed')
        # then apply LUT to image --> not that hard to do I guess
        self.preview_changed()

    def preview_changed(self):
        # print('TODO implement preview_changed')
        # pass
        # TODO do a color code of the image

        # here I would need to load the image and to display it
        # can it be smart to have two or three paints --> like for the list --> maybe ???  --> think about it
        # or need define a bg and a blend mode for the image
        # shall I allow corrections in there ???
        # how can I edit the db ??? if user does changes there

        # print(self.image_preview_combo.currentText())

        preview_selected = self.image_preview_combo.currentText()

        out = self.create_preview(preview_selected, self.get_selection(), self.get_current_TA_path())

        if out is not None:
            self.paint.set_image(out)

    # TODO for the master db --> maybe base myself on the query for all the stuff
    # or create the masterdb and run the query on it--> ultimately faster
    # need pass in a db if db is passed then ignore the current frame stuff
    def create_preview(self, preview_selected, file, TA_path=None):
        if preview_selected == '':
            # nothing selected or no image --> nothing TODO
            return

        # TODO -->  do a generic code for both
        # offer dilation also as a step after the color coding --> last step --> do it on the RGB int24 image maybe --> think about it

        # TODO a lot of code duplication in here --> try as much as possible to reduce that but ok for now

        if preview_selected.startswith('#'):

            # check if master db is required and open it

            if self.groupBox_color_coding.isChecked():
                if self.radioButton3.isChecked():
                    if self.master_db is None:
                        print('creating master db')
                        self.master_db = createMasterDB(self.get_full_list())
                # pass

            # TODO --> allow blend images and also allow to handle and exclude borders --> maybe also do border vertices
            # check the presence of a column with name vertices
            # do autoscale for polarity --> can I do so based on magnitude or by reading max and min of the values --> most likely possible

            try:
                # load data from the databse
                # assume plot as cells by default and just need to get the db and corresponding data --> from a table
                # print('TODO implement preview changed, load and plot a db')

                # TODO call an advanced SQL plotter here on the current image
                # and check whether or not I should color code it

                # file = self.get_selection()
                # SQL_command = "PLOT AS CELLS SELECT local_id, area FROM cells_2D + LUT DNA + OPACITY 35% + DILATATION 1"
                # SQL_command = "SELECT local_id, area FROM cells_2D"
                # SQL_command = "SELECT local_id, area FROM cells_2D WHERE is_border == FALSE"  # a bit more complex just to see which method to choose --> very easy
                # SQL_command = "SELECT local_id, area FROM cells_2D WHERE is_border == FALSE"  # a bit more complex just to see which method to choose --> very easy
                # print('preview_selected',preview_selected) #preview_selected #cells_2D.area

                # TODO --> add this as an option --> TODO +' WHERE is_border==False' maybe also implement this for vertices
                # rename tracked_cells_resized.tif to tracked cells
                # store global displacement too!!!
                table, column = preview_selected[1:].split('.')

                if table == 'properties':
                    # properties tables can't be plotted --> ignore
                    # alternatively could show the table in the future
                    return

                # TODO --> do a plot as packing

                if 'bonds' in table:
                    plot_type = 'bonds'
                elif 'vertices' in table:
                    plot_type = 'vertices'
                else:
                    # if 'cell' in table:
                    plot_type = 'cells'
                    # print('column', column)
                    if ('Q1' in column or 'Q2' in column) and 'ch' in column:
                        # force plot as nematic
                        # if plot is nematic plot then plot it as a nematic
                        nematic_root, channel_nb = column.split('ch')
                        # print('nematic_root', nematic_root, 'channel_nb', channel_nb)
                        nematic_default_command = nematic_root.replace('Q1', 'Q#').replace('Q2', 'Q#')
                        default_nematic_command = nematic_default_command.replace('Q#',
                                                                                  'Q1') + 'ch' + channel_nb + ', ' + nematic_default_command.replace(
                            'Q#', 'Q2') + 'ch' + channel_nb
                        column = default_nematic_command
                        plot_type = 'nematics'
                        # TODO at some point put some better code for that --> e.g. some custom scaling for the user --> can even be done directly in the options
                        if 'normalized' in column:
                            column += ', 60 AS SCALING_FACTOR'
                        else:
                            column += ', 0.06 AS SCALING_FACTOR'
                        # print('column',column)
                    # do the same for the stretch
                    if ('s1'==column.lower() or 's2'==column.lower()):
                        column = ' S1, S2, 10 AS SCALING_FACTOR'
                        plot_type = 'nematics'



                if column in ['nb_of_vertices_or_neighbours', 'nb_of_vertices_or_neighbours_cut_off']:
                    plot_type = 'packing'

                SQL_command = 'SELECT local_id,' + column + ' FROM ' + table  # +' WHERE is_border==False' # --> need change this

                # print('SQL_command',SQL_command)

                # extras = {} # see how I can apply LUTS
                # extras = {'DILATION':1} # ça marche
                # extras = {'EROSION': 1}  # ça marche pas avec methode2 # seems to give me a bug --> why ??? --> shifts ids --> yes makes sense because some cells may be lost
                # extras = {'EROSION': 3}  # ça marche # seems to give me a bug --> why ??? --> shifts ids --> yes makes sense because some cells may be lost
                # extras = {'EROSION': 5}  # ça marche meme avec une strong erosion # seems to give me a bug --> why ??? --> shifts ids --> yes makes sense because some cells may be lost
                # extras = {'EROSION': 10}  # ça marche meme avec une strong erosion # seems to give me a bug --> why ??? --> shifts ids --> yes makes sense because some cells may be lost
                # extras = {'DILATION':22} # ça marche # dilation gives crappy results --> dialtion should be 1 not really more, just to fill the holes

                # otherwise no lut
                extras = {}
                if self.groupBox_color_coding.isChecked():
                    extras[
                        'LUT'] = self.lut_combo.currentText()  # ça marche # dilation gives crappy results --> dialtion should be 1 not really more, just to fill the holes

                if self.excluder_label.isChecked():
                    extras['freq'] = (self.lower_percent_spin.value(), self.upper_percent_spin.value())

                # extras = {'LUT':'copper'} # a cmap lut just to try # ça marche # dilation gives crappy results --> dialtion should be 1 not really more, just to fill the holes

                # print(SQL_command, plot_type)
                # TODO handle dilation and also handle things alike

                selected_tab_idx, _ = self.get_cur_tab_index_and_name()
                current_idx_in_list = self.list.get_list(selected_tab_idx).get_selection_index()
                # print('current_idx_in_list',current_idx_in_list)

                # print('current_idx_in_list',current_idx_in_list)
                mask, SQL_plot = plot_as_any(file, SQL_command, plot_type=plot_type, return_mask=True,
                                             db=self.master_db if self.radioButton3.isChecked() else None,
                                             current_frame=current_idx_in_list, **extras)

                # need check how to plot stuff from the master db
                # in fact just need get the max and min from the masterdb, the rest we don't care!!!!
                # how can I plot just from the master db --> need add a where frame_nb == current_frame for the plot part but the  search for a max should be done everywhere...

                if self.groupBox_overlay.isChecked():
                    # ok but need implemnt the channel
                    # right
                    img = Img(file)
                    if len(img.shape) > 2:
                        # get channel if image has channel
                        channel = self.overlay_bg_channel_combo.currentIndex() - 1
                        if channel != -1:
                            img = img[..., channel]
                    composite = blend(img, SQL_plot, alpha=self.overlay_fg_transparency_spin.value(),
                                      mask_or_forbidden_colors=mask)
                else:
                    composite = SQL_plot
                return composite
            except:
                traceback.print_exc()
                logger.error(
                    'failed to plot from the databse, is that a custom created table ??? ' + str(preview_selected))
        else:
            try:
                # load file directly
                # TA_path = self.get_current_TA_path()
                if TA_path is None:
                    # no image selected --> quit
                    return
                full_path = os.path.join(TA_path, preview_selected)

                palette = None
                if self.groupBox_color_coding.isChecked():
                    lut = self.lut_combo.currentText()
                    lutcreator = PaletteCreator()
                    luts = lutcreator.list
                    # lut = lutcreator.create3(luts[lut])
                    try:
                        palette = lutcreator.create3(luts[lut])
                    except:
                        if lut is not None:
                            logger.error('could not load the specified lut (' + str(
                                lut) + ') a gray lut is loaded instead')  # --> ignoring or shall I default to gray
                        # palette = None
                        # default to grey palette
                        palette = lutcreator.create3(luts['GRAY'])

                # mask, SQL_plot = plot_as_any(file, SQL_command, plot_type=plot_type, return_mask=True, **extras)
                SQL_plot = Img(full_path)

                mask = None
                if len(SQL_plot.shape) != 3:
                    if palette is not None:
                        # by default convert all black pixels of the image to no signal for color coding maybe
                        # TODO maybe apply a few special rules for files I know such as tracking images --> remove pure white for example !!!

                        # TODO --> should I really make this by default ??? --> maybe yes
                        mask = mask_colors(SQL_plot, 0)
                        # plt.imshow(mask)
                        # plt.show()

                        # print(mask.shape)

                        SQL_plot = apply_lut(SQL_plot, palette, True)
                else:
                    if preview_selected == 'tracked_cells_resized.tif':
                        mask = mask_colors(SQL_plot, [(255, 255, 255), (0, 0, 0)])

                    if self.groupBox_color_coding.isChecked():
                        logger.warning('Lut cannot be applied to RGB images --> ignoring')

                # pas mal mais manque aussi le color code qui doit etre fait dans le plotter malheureusement

                if self.groupBox_overlay.isChecked():
                    # ok but need implemnt the channel
                    # right
                    img = Img(file)
                    if len(img.shape) > 2:
                        # get channel if image has channel
                        channel = self.overlay_bg_channel_combo.currentIndex() - 1
                        if channel != -1:
                            img = img[..., channel]
                    composite = blend(img, SQL_plot, alpha=self.overlay_fg_transparency_spin.value(),
                                      mask_or_forbidden_colors=mask)
                else:
                    composite = SQL_plot

                # self.paint.set_image(full_path) # why that does not work
                # return full_path
                return composite
            except:
                traceback.print_exc()
                logger.error(
                    'failed to create composite image for ' + str(preview_selected))
        return

    # def get_cur_tab_index(self):
    #     return self.tabs.currentIndex()

    def get_cur_tab_index_and_name(self):
        selected_tab_idx = self.tabs.currentIndex()
        selected_tab_name = self.tabs.tabText(selected_tab_idx).lower()
        return selected_tab_idx, selected_tab_name

    def populate_preview_combo(self):
        # get current image and see how to populate things for it
        # --> load all images in the list and also load all the database columns
        # then offer color code
        # if not the right tab --> ignore
        _, cur_tab_name = self.get_cur_tab_index_and_name()

        if not 'review' in cur_tab_name:
            # nothing todo
            return

        # do the heavy stuff there --> list TA files in the TA folder for current image
        # read the db(s) and populate it, just start with the pyta db only
        # TODO implement that!!!

        # print('TODO implement populate combo')
        # self.image_preview_combo
        # pass

        cur_sel_value = self.image_preview_combo.currentText()

        selection = self.get_selection()

        self.image_preview_combo.disconnect()

        # empty combo
        self.image_preview_combo.clear()
        if not selection:
            # self.image_preview_combo.connec
            self.image_preview_combo.currentTextChanged.connect(self.preview_changed)
            return
        # populate combo with images contained in the folder and with table entries in a smart fashion
        # maybe add a hashtag when it is a table entry --> smart and I really like it --> and it means dynamic plotting
        # list all images in the TA folder
        TA_path = smart_name_parser(selection,
                                    ordered_output='TA')  # if ordered output is just one string and not an array --> do not return an array but return a string --> that way i don't break anything
        # print('TA_path',TA_path) # populate files in there !!!
        list_of_images_in_TA_folder = create_list(TA_path)  #

        # add them to the combo box --> TODO

        # TODO if it has a selection --> try preserve it --> TODO

        for img in list_of_images_in_TA_folder:
            short_name = smart_name_parser(img, ordered_output='short')
            self.image_preview_combo.addItem(
                short_name)  # how can I store the full path --> in fact easy because I don't need it
        # qsdqqsdqsqsd

        database_entries_for_cur_image = populate_table_content(os.path.join(TA_path, 'pyTA.db'))
        if database_entries_for_cur_image is not None:
            for col in database_entries_for_cur_image:
                self.image_preview_combo.addItem(col)

        # try reapply selection and disconnect it before updating it
        if cur_sel_value:
            index = self.image_preview_combo.findText(cur_sel_value, QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.image_preview_combo.setCurrentIndex(index)

        # reconnect
        self.image_preview_combo.currentTextChanged.connect(self.preview_changed)

        # load also the content of the tables maybe with a dot to know the table and the columns --> it is a very good idea --> and start with a hash
        # print(list_of_images_in_TA_folder)


    # nb it's also possible to have several filters --> I could easily generate a filter a posteriori for the clone even if this is not the most efficient way of doing things
    # SELECT * FROM (SELECT * FROM cells_2D) WHERE local_id IN(120, 130) AND cytoplasmic_area IN (802)
    def export_cells(self):

        # TODO would be great to be able to handle clones here
        # maybe further filter the output or filter the initial stuff

        sql_command = 'SELECT * FROM cells_2D'
        # need create the master db for all
        lst = self.get_full_list(warn_on_empty_list=True)
        if lst is not None and lst:
            master_db = createMasterDB(lst, force_track_cells_db_update=True)
            # first create a master db
            # can i do two ins at the same time --> then filter by clone and name at the end --> this would be doable in fact
            try:
                available_tables = master_db.get_tables(force_lower_case=True)
                if 'cells_3d' in available_tables:
                    sql_command += ' NATURAL JOIN cells_3D'
                if 'cell_tracks' in available_tables:
                    sql_command += ' NATURAL JOIN cell_tracks'
                if 'properties' in available_tables:
                    sql_command += ' NATURAL JOIN properties'
            except:
                pass
            try:
                path = smart_name_parser(lst[0], ordered_output='parent')
                path = os.path.join(path, 'cells.csv')
                output_file_name = saveFileDialog(parent_window=self, path=path,
                                                  extensions="CSV (*.csv);;Supported Files (*.csv *.txt);;All Files (*)",
                                                  default_ext='.csv')

                # master_db.print_query(sql_command)
                # print(sql_command)

                if output_file_name is not None:
                    master_db.save_query_to_csv_file(sql_command, output_file_name)
            except:
                traceback.print_exc()
            finally:
                master_db.close()

    def export_bonds(self):
        sql_command = 'SELECT * FROM bonds_2D'
        # need create the master db for all
        lst = self.get_full_list(warn_on_empty_list=True)
        if lst is not None and lst:
            master_db = createMasterDB(lst)
            try:
                available_tables = master_db.get_tables(force_lower_case=True)
                if 'bonds_3d' in available_tables:
                    sql_command += ' NATURAL JOIN bonds_3D'
                if 'bond_tracks' in available_tables:
                    sql_command += ' NATURAL JOIN bond_tracks'
                if 'properties' in available_tables:
                    sql_command += ' NATURAL JOIN properties'
            except:
                pass
            try:
                path = smart_name_parser(lst[0], ordered_output='parent')
                path = os.path.join(path, 'bonds.csv')
                output_file_name = saveFileDialog(parent_window=self, path=path,
                                                  extensions="CSV (*.csv);;Supported Files (*.csv *.txt);;All Files (*)",
                                                  default_ext='.csv')
                if output_file_name is not None:
                    master_db.save_query_to_csv_file(sql_command, output_file_name)
            except:
                traceback.print_exc()
            finally:
                master_db.close()

    def export_SQL(self):
        lst = self.get_full_list(warn_on_empty_list=True)
        if lst is None or not lst:
            return
        text, ok = QPlainTextInputDialog.get_text(default_text="SELECT * FROM vertices_2D",
                                                  title="Please enter an SQL command")
        if ok:
            sql_command = text
            try:
                if sql_command.strip() == '':
                    logger.error('Invalid SQL command: "' + str(sql_command) + '" Ignoring')
                    return
            except:
                logger.error('Invalid SQL command: "' + str(sql_command) + '" Ignoring')
                return
        else:
            return

        # print('in here')
        # TODO try implement that
        # sql_command = SQL_command
        # need create the master db for all
        lst = self.get_full_list(warn_on_empty_list=True)
        if lst is not None and lst:
            master_db = createMasterDB(lst)
            try:
                path = smart_name_parser(lst[0], ordered_output='parent')
                path = os.path.join(path, 'custom_command.csv')
                output_file_name = saveFileDialog(parent_window=self, path=path,
                                                  extensions="CSV (*.csv);;Supported Files (*.csv *.txt);;All Files (*)",
                                                  default_ext='.csv')
                if output_file_name is not None:
                    # shall I add this --> maybe not ...
                    # if 'properties' in available_tables:
                    #     sql_command += ' NATURAL JOIN properties'
                    master_db.save_query_to_csv_file(sql_command, output_file_name)
            except:
                traceback.print_exc()
            finally:
                master_db.close()

    def fix_mask_and_tracks(self):
        # avoid launching this in a thread to avoid issues with paint

        # self.launch_in_a_tread(self._fix_mask_and_tracks)

        # def _fix_mask_and_tracks(self, progress_callback):
        # do the classical stuff
        # print('TODO implement epyseg')
        lst = self.get_full_list(warn_on_empty_list=True)
        # print('lst', lst)
        if lst is None:
            # print('problem --> stopping')
            return

        if not self.check_channel_is_selected():
            return

        # track_cells_dynamic_tissue(lst, channel=self.paint.get_selected_channel(), progress_callback=progress_callback)
        # print('TODO implement fix_mask_and_tracks')
        # pass
        # probably don't need the call back ???? maybe I do ...
        # TODO need allow show mouse, also need to to allow early stop with a button, need cmap for images
        # need better labels, need support for channels, need connect progress bar also!!!!
        # need allow track correction
        # maybe need a few text lines (that can be closed to gain space) to guide the user rapidly
        # ideally try remove images if possible, probably not possible!!!
        # need force enable mouse always because it does make things much simpler in this extreme case

        # TODO add early stop, progress bar, color code

        # noyt bad --> need change LUT and connect tracks also --> TODO
        # help_user_correct_errors(lst, channel=self.paint.get_selected_channel(), progress_callback=progress_callback)
        help_user_correct_errors(lst, channel=self.paint.get_selected_channel(),
                                 progress_callback=None)  # do not set the progress callback or its paint method will cause a crash

    def track_cells_static(self):
        lst = self.get_full_list(warn_on_empty_list=True)
        # print('lst', lst)
        if lst is None:
            # print('problem --> stopping')
            return
        if not self.check_channel_is_selected():
            return
        # TODO also need a GUI for the parameters...

        values, ok = TrackingDialog.getValues(parent=self)

        if ok:
            tmp2 = partial(self._track_cells_static, lst=lst,channel_of_interest=self.paint.get_selected_channel(), recursive_assignment=values[0], warp_using_mermaid_if_map_is_available=values[1])
            self.launch_in_a_tread(tmp2)

            # self.launch_in_a_tread(self._track_cells_static)
        # maybe do that using a progess bar --> would be smart

    def _track_cells_static(self, progress_callback, lst =None, channel_of_interest=None, recursive_assignment=False, warp_using_mermaid_if_map_is_available=True):
        # do the classical stuff
        # print('TODO implement epyseg')
        # lst = self.get_full_list(warn_on_empty_list=False)
        # # print('lst', lst)
        # if lst is None:
        #     # print('problem --> stopping')
        #     return
        # if not self.check_channel_is_selected():
        #     return
        match_by_max_overlap_lst(lst, channel_of_interest=channel_of_interest, recursive_assignment=recursive_assignment,
                                 warp_using_mermaid_if_map_is_available=warp_using_mermaid_if_map_is_available, pre_register=True, progress_callback=progress_callback)
        # track_cells_dynamic_tissue(lst, channel=self.paint.get_selected_channel(), progress_callback=progress_callback)
        # TODO --> hack the stuff below ...
        logger.info('Creating correspondence between local cell id and track/global cell id')
        add_localID_to_trackID_correspondance_in_DB(lst, progress_callback)

    # need add this at the very end of the track file --> much smarter in fact
    # def assign_trackID_to_localID(self):
    #     self.launch_in_a_tread(self._assign_trackID_to_localID)
    #
    # def _assign_trackID_to_localID(self, progress_callback):
    #     lst = self.get_full_list()
    #     add_localID_to_trackID_correspondance_in_DB(lst, progress_callback)

    def track_cells_dynamic(self):
        lst = self.get_full_list(warn_on_empty_list=True)
        # print('lst', lst)
        if lst is None:
            # print('problem --> stopping')
            return
        if not self.check_channel_is_selected():
            return
        self.launch_in_a_tread(self._track_cells_dynamic)
        # self.assign_trackID_to_localID()

    def _track_cells_dynamic(self, progress_callback):
        # do the classical stuff
        # print('TODO implement epyseg')
        lst = self.get_full_list(warn_on_empty_list=False)
        # print('lst', lst)
        if lst is None:
            # print('problem --> stopping')
            return
        if not self.check_channel_is_selected():
            return
        track_cells_dynamic_tissue(lst, channel=self.paint.get_selected_channel(), progress_callback=progress_callback)
        logger.info('Creating correspondence between local cell id and track/global cell id')
        add_localID_to_trackID_correspondance_in_DB(lst, progress_callback)
        # print('TODO implement track_cells_dynamic')
        # pass

    def finish_all(self):
        lst = self.get_full_list(warn_on_empty_list=True)
        if lst is None:
            logger.error('Please load files first')
            return

        parameters, ok = FinishAllDialog.get_values(parent=self, title="Options")
        # form.show()
        # text, ok = app.exec_()
        # print(form.get_value())

        # text, ok = QInputDialog.getText(self, 'Text Input Dialog', 'Enter your SQl command:')
        if not ok:
            return

        # cell_area_cutoff, four_way_cutoff]
        tmp = partial(self._finish_all, cell_area_cutoff=parameters[0], four_way_cut_off=parameters[1], measure_polarity=parameters[2], measure_3D=parameters[3],multi_threading= parameters[4])
        self.launch_in_a_tread(tmp)

    def _finish_all(self, progress_callback=None, cell_area_cutoff=10, four_way_cut_off=2, measure_polarity=False, measure_3D=False, multi_threading=True):
        # print('TODO implement finish_all')
        # ask which measures should be computed and check how to best handle 3D parameters
        # pass

        lst = self.get_full_list(warn_on_empty_list=True)
        if lst is None:
            logger.error('Please load files first')
            return
        # TODO --> try MT that --> give it a try -->
        TAMeasurements(lst,
                       measure_polarity=measure_polarity,
                       measure_3D=measure_3D,
                       progress_callback=progress_callback, min_cell_size=cell_area_cutoff, bond_cut_off=four_way_cut_off, multi_threading=multi_threading)  # --> almost there just need create one db per file now!!!

    # def add_column_to_properties_table(self):
    #     pass
    #     # add a column to the table --> TODO --> so that I can add custom parameters maybe --> can be very useful I guess
    #     self.properties_table.
    #     qsdsqdsqdqsd


    def run_watershed(self):
        lst = self.get_full_list(warn_on_empty_list=True)
        if lst is None:
            return

        if not self.check_channel_is_selected():
            return

        input_channel_of_interest = self.paint.get_selected_channel()
        values, ok = WshedDialog.getValues(parent=self)

        if ok == QDialogButtonBox.Apply:
            # just run it once and set it as the current mask
            mask = wshed(self.paint.paint.get_raw_image(), channel=input_channel_of_interest, weak_blur=values[0],
                         strong_blur=values[1], min_seed_area=values[2], is_white_bg=False)
            self.paint.set_mask(mask)
        elif ok == QDialog.Accepted:
            # , *[lst, values, input_channel_of_interest]
            tmp = partial(wshed, channel=input_channel_of_interest, weak_blur=values[0], strong_blur=values[1],
                          min_seed_area=values[2], is_white_bg=False)
            tmp2 = partial(self._run_watershed, wshed=tmp, lst=lst)
            self.launch_in_a_tread(tmp2)

    # TODO do multithread that some day but ok for now!!! --> anyway wshed is out of date...
    def _run_watershed(self, progress_callback, wshed=None, lst=None):
        if lst is not None:
            for iii, file in enumerate(lst):
                try:
                    if early_stop.stop:
                        return
                    if progress_callback is not None:
                        progress_callback.emit((iii / len(lst)) * 100)
                    else:
                        print(str((iii / len(lst)) * 100) + '%')
                except:
                    pass
                try:
                    mask_path = smart_name_parser(file, ordered_output='handCorrection.tif')
                    mask = wshed(Img(file))
                    Img(mask, dimensions='hw').save(mask_path)
                except:
                    traceback.print_exc()
                    logger.error('could not create watershed mask for image ' + str(file))
        # now loop for all
        # print(*args)
        # TODO code all
        # lst = self.get_full_list(warn_on_empty_list=True)
        # input_channel_of_interest = self.paint.get_selected_channel()
        # if lst is None:
        #     return
        # TODO finalize that but almost all ok
        # pass

    def epyseg_seg(self):
        lst = self.get_full_list(warn_on_empty_list=True)
        # print('lst', lst)
        if lst is None:
            # print('problem --> stopping')
            return

        if not self.check_channel_is_selected():
            return

        self.launch_in_a_tread(self._epyseg_seg)

    def _epyseg_seg(self, progress_callback):
        # do the classical stuff
        # print('TODO implement epyseg')
        lst = self.get_full_list(warn_on_empty_list=True)
        # print('lst', lst)
        if lst is None:
            # print('problem --> stopping')
            return

        if not self.check_channel_is_selected():
            return

        # TODO need add checks to make sure the stuff is loading properly
        # print('in here2')
        # need get the stuff

        # INPUT_FOLDER = '/path/to/files_to_segment/'
        IS_TA_OUTPUT_MODE = True  # stores as handCorrection.tif in the folder with the same name as the parent file without ext
        input_channel_of_interest = self.paint.get_selected_channel()  # assumes image is single channel or multichannel nut channel of interest is ch0, needs be changed otherwise, e.g. 1 for channel 1
        TILE_WIDTH = 256  # 128 # 64
        TILE_HEIGHT = 256  # 128 # 64
        TILE_OVERLAP = 32
        EPYSEG_PRETRAINING = 'Linknet-vgg16-sigmoid-v2'  # or 'Linknet-vgg16-sigmoid' for v1
        SIZE_FILTER = None  # 100 # set to 100 to get rid of cells having pixel area < 100 pixels

        # deepTA = EZDeepLearning(use_cpu=self.force_CPU_check.isChecked())
        deepTA = EZDeepLearning()

        # check whether that would work or not

        run_seg(deepTA=deepTA, INPUT_FOLDER=lst, IS_TA_OUTPUT_MODE=IS_TA_OUTPUT_MODE,
                progress_callback=progress_callback,
                input_channel_of_interest=input_channel_of_interest, TILE_WIDTH=TILE_WIDTH, TILE_HEIGHT=TILE_HEIGHT,
                TILE_OVERLAP=TILE_OVERLAP, EPYSEG_PRETRAINING=EPYSEG_PRETRAINING, SIZE_FILTER=SIZE_FILTER)

        # deepTA.load_or_build(
        #     model='/E/Sample_images/sample_images_pyta/test_merged_model.h5')  # TODO --> replace this by online model --> TODO
        # if self.sender() == self.denoising_surf_proj:
        #     save_raw_image = False
        # else:
        #     save_raw_image = True
        # --> also need
        # surface_projection_pyta(deepTA, lst,
        #                         save_raw_image=save_raw_image, channel=self.paint.get_selected_channel())

        del deepTA
        # run update to load mask if exists

        # pas mal mais vraiment besoin d'une progress bar...

    # en fait c'est impossible de stopper un Thread --> faut un Truc qui sert de sentinelle et bloque qd besoin
    def stop_threads_immediately(self):
        # set a global variable that is always checked for at the beginning of every loop that can be used as a stopping point
        # pass
        # print('how to stop?')
        # self.event_stop.set()
        # global early_stop
        # print(early_stop)

        # global stop_threads
        # stop_threads = True

        # self.overlay.hide()
        # if self.thread is not None:
            # print('stopping thread')
            # self.thread.stop()
            # print('thread stopped')
        early_stop.stop = True
        #

        # self.thread = None

        # stop_threads = False

        # def stop_threads_immediately(self):
    #
    #     # QThreadPool::clear()
    #     # stop thread
    #
    #     # procs = []  # this is not a Pool, it is just a way to handle the
    #     # # processes instead of calling them p1, p2, p3, p4...
    #     # for _ in range(4):
    #     #     p = mp.Process(target=some_long_task_from_library, args=(1000,))
    #     #     p.start()
    #     #     procs.append(p)
    #     # mp.active_children()
    #     # for p in procs:
    #     #     p.terminate()
    #
    #     # threads = []
    #     # for rn in range(4):
    #     #     r = pool.apply_async(a.foo_pulse, (nPulse, 'loop ' + str(rn)))
    #     #     threads.append(r)
    #     # self.threadpool.
    #
    #     for p in self.threads:
    #         try:
    #             p.terminate()
    #         except:
    #             traceback.print_exc()
    #
    #     self.threads.clear()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    # set app icon
    # app_icon = QtGui.QIcon('./../../IconsPA/src/main/resources/Icons/ico_packingAnalyzer2.gif')  # add taskbar icon
    # app_icon = qta.icon('mdi.hexagon-multiple-outline', color=QColor(200, 200, 200))
    # app_icon.addFile('gui/icons/16x16.png', QtCore.QSize(16, 16))
    # app_icon.addFile('gui/icons/24x24.png', QtCore.QSize(24, 24))
    # app_icon.addFile('gui/icons/32x32.png', QtCore.QSize(32, 32))
    # app_icon.addFile('gui/icons/48x48.png', QtCore.QSize(48, 48))
    # app_icon.addFile('gui/icons/256x256.png', QtCore.QSize(256, 256))
    # app.setWindowIcon(app_icon)
    # QApplication.setWindowIcon(app_icon)
    w = TissueAnalyzer()
    # w.setWindowIcon(app_icon)
    w.show()
    sys.exit(app.exec_())
