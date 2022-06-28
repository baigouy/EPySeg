# TODO --> add a shift enter to remove small blobs because there are a lot --> can be done as a a post-processing !!!
# pb some channels are kept 0 or 1 when others are attributed 255 --> big pb --> need a fix or need do a post process to recover all the images
# maybe do a check to normalize all the images in the same way ... when I save
# need a small thingy remover and need change the default values
# make it a blob remover
# ok for now all of these things can be done as post process ...


# shortcuts --> Enter --> apply --> needed to ensure edit is taken into account
# C --> increase contrast of displayed image


# that seems to work but I need create the GT in the TA shape --> TODO --> then test it for real

# TODO --> maybe do a version with two lists so that the user can also load the GT even if the GT is not having a generic name ???
# or shall I look in more folder sur as GT ou ground truth ou predict
import os
import os.path
import traceback
from functools import partial
from PyQt5.QtWidgets import QProgressBar, \
    QLabel, QMessageBox
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import QStackedWidget
from PyQt5.QtWidgets import QGridLayout, QPushButton, QFrame
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets, QtCore, QtGui

from epyseg.draw.shapes.freehand2d import Freehand2D
from epyseg.draw.shapes.image2d import Image2D
from epyseg.img import Img, RGB_to_int24
from epyseg.ta.GUI.paint2 import Createpaintwidget
from epyseg.ta.GUI.scrollablepaint_multichannel_or_GT_masks_editor import scrollablepaint_multichannel_edit
from epyseg.ta.tracking.tools import smart_name_parser
from epyseg.ta.GUI.stackedduallist import dualList
import qtawesome as qta  # icons
from epyseg.ta.GUI.scrollablepaint import scrollable_paint
import numpy as np

# main parameter !!!
nb_of_GT_channels = 10


__MAJOR__ = 0
__MINOR__ = 0
__MICRO__ = 1
__RELEASE__ = 'b'  # https://www.python.org/dev/peps/pep-0440/#public-version-identifiers --> alpha beta, ...
__VERSION__ = ''.join([str(__MAJOR__), '.', str(__MINOR__), '.'.join([str(__MICRO__)]) if __MICRO__ != 0 else '', __RELEASE__])
__AUTHOR__ = 'Benoit Aigouy'
__NAME__ = 'GT editor'
__EMAIL__ = 'baigouy@gmail.com'

DEBUG = False

class GT_editor(QtWidgets.QMainWindow):
    # stop_threads = False

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

        # global early_stop
        # early_stop = False
        self.setAcceptDrops(True)  # we accept DND but just to transfer it to the current list handlers!!!

    def initUI(self):
        # this is a timer to delay the preview updates upon spin valuechanged
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()

        # should fit in 1024x768 (old computer screens)
        window_width = 1024
        window_height = 700
        self.setGeometry(
            QtCore.QRect(centerPoint.x() - int(window_width / 2), centerPoint.y() - int(window_height / 2),
                         window_width,
                         window_height))

        # set the window icon
        # self.setWindowIcon(QtGui.QIcon('./../../IconsPA/src/main/resources/Icons/ico_packingAnalyzer2.gif'))
        # variantMap =QVariant()
        # variantMap.insert("color", QColor(10, 10, 10));
        self.setWindowIcon(qta.icon('mdi.brain')) # , color=QColor(200, 200, 200)  # very easy to change the color of my icon
        # 'mdi.hexagon-outline'

        # zoom parameters
        self.scale = 1.0
        self.min_scaling_factor = 0.1
        self.max_scaling_factor = 20
        self.zoom_increment = 0.05

        self.setWindowTitle(__NAME__ + ' v' + str(__VERSION__))

        # this is the default paint window of TA
        self.paint = self.create_paint(number_of_GT_channels=nb_of_GT_channels)
        # print(self.paint.paint.vdp.shapes)

        # this is the properties window (allows to set useful parameters that can be used by TA

        self.Stack = QStackedWidget(self)
        self.last_opened_file_name = QLabel('')

        # self.list = QListWidget(self)  # a list that contains files to read or play with
        # self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.list.selectionModel().selectionChanged.connect(self.selectionChanged)  # connect it to sel change
        self.list = dualList()
        self.list.setToolTip("Drag and drop your images over this list")
        for lst in self.list.lists:
            lst.list.selectionModel().selectionChanged.connect(self.selectionChanged)
            # print list_connected


        # TODO replace that with napari
        # self.volumeViewerUI()
        self.Stack.addWidget(self.paint)
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
        # self.grid.setRowStretch(0, 75)
        # self.grid.setRowStretch(2, 25)

        # first col 75% second col 25% of total width
        self.grid.setColumnStretch(0, 75)
        self.grid.setColumnStretch(1, 25)

        # void QGridLayout::addLayout(QLayout * layout, int row, int column, int rowSpan, int columnSpan, Qt::Alignment alignment = 0)
        # self.grid.addWidget(self.table_widget, 2, 0, 1, 2)  # spans over one row and 2 columns

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


        self.about = QPushButton()
        self.about.setIcon(qta.icon('mdi.information-variant', options=[{'scale_factor': 1.5}]))
        self.about.clicked.connect(self.about_dialog)
        self.about.setToolTip('About...')
        statusBar.addWidget(self.about)

        statusBar.addWidget(self.last_opened_file_name)

        # Set up menu bar
        self.mainMenu = self.menuBar()
        self.setMenuBar(self.mainMenu)

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
        # backspaceShortcut.activated.connect(self.down)
        backspaceShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active

        #!!!!!!!!!! MEGA IMPORTANT KEEP no clue why not activable --> IN FACT IT WAS NOT ACTIVABLE BECAUSE THE SCROLLABLE PAINT WAS ACTIVATING THIS --> NEED DEACTIVATE IT IN THE PARENT OR HERE IT MUST NOT BE IN THE TWO OF THEM

        # supr = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self)
        # # backspaceShortcut.activated.connect(self.prevFrame)
        # supr.activated.connect(self.paint.paint.suppr_pressed)
        # supr.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active

        # Delete does not work but no clue why ????? mais backspace marche -->
        # supr = QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Delete), self)
        # # supr.activated.connect(self.paint.paint.suppr_pressed)
        # supr.activated.connect(self.down)
        # supr.setContext(QtCore.Qt.ApplicationShortcut)

        # self.blinker = Blinker()
        # self.to_blink_after_worker_execution = None
        # self.threading_enabled = True
        #
        # # self.threads = [] # contains all threads so that I can stop them immediately
        # self.thread = None
        # # self.event_stop = threading.Event() # allows to stop a thread
        # # self.threadpool = QThreadPool()
        # # self.threadpool.setMaxThreadCount(self.threadpool.maxThreadCount() - 1)  # spare one core for system
        #
        # self.overlay = Overlay(self.centralWidget())  # maybe ok if draws please wait somewhere
        # self.overlay.hide()
        #
        # try:
        #     self.oldvalue_CUDA_DEVICE_ORDER = os.environ["CUDA_DEVICE_ORDER"]
        #     # print('ok', self.oldvalue_CUDA_DEVICE_ORDER)
        # except:
        #     self.oldvalue_CUDA_DEVICE_ORDER = ''
        #
        # try:
        #     self.oldvalue_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
        #     # print('ok', self.oldvalue_CUDA_VISIBLE_DEVICES)
        # except:
        #     self.oldvalue_CUDA_VISIBLE_DEVICES = ''
        #
        # # button.clicked.connect(self.overlay.show)
        #
        # # self.setAcceptDrops(True)  # KEEP IMPORTANT # --> nb can I delegate the dnd of self to a progeny


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
                print('Empty list, please load files first')
                # self.to_blink_after_worker_execution = [self.list]
                self.blinker.blink(self.list)
                return
        return lst

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

    def down(self):
        print('down')

    def nextFrame(self):
        print('next frame pressed')

    def prevFrame(self):
        print('prev frame pressed')

    def about_dialog(self):
        msg = QMessageBox(parent=self)
        msg.setIcon(QMessageBox.Information)
        msg.setText("TEM segmenter")
        msg.setInformativeText("Copyright 2021-2022\n\nBy Benoit Aigouy\n\nbaigouy@gmail.com") # TODO make it dynamic until the current system year
        msg.setWindowTitle("About...")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok)
        # TODO also add the link to the web page on github for the tuto
        retval = msg.exec_()

    def update_preview_depending_on_selected_tab(self):
        # selected_tab_name = self.tabs.currentWidget()
        # selected_tab_idx = self.tabs.currentIndex()

        # self.list.set_list(0)
        list = self.list.get_list(
            0).list  # NB CHANGE THIS --> NOT SMART TO USE LIST AS A NAME --> MAY CAUSE TROUBLE WITH REAL LIST TYPE OF PYTHON --> call it othrewise !!!!
        # TODO replace by self.get_full_list or something alike
        selected_items = list.selectedItems()
        # selected_tab_name = self.tabs.tabText(selected_tab_idx).lower()

        # store file within the tab --> and no relaod except if forced
        # debug
        # print('selection '+ str(0))

        self.paint.freeze(False)

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
                    # logger.debug('Updating icon')
                    icon = QIcon(QPixmap.fromImage(self.paint.paint.image))
                    pixmap = icon.pixmap(24, 24)
                    icon = QIcon(pixmap)
                    list.currentItem().setIcon(icon)
            except:
                # no big deal if can't create a thumb out of the selected image
                traceback.print_exc()
                print('failed creating an icon for the selected file')


            # specific GT magic here for loading masks --> maybe some day rather offer two lists and match them
            # will crash if no file exists --> cannot create a GT...
            if os.path.exists(os.path.join(smart_name_parser(selected_file,'parent'),'predict',smart_name_parser(selected_file,'short'))):
                # seach for typical GT stuff produced by epyseg...
                self.paint.set_mask(os.path.join(smart_name_parser(selected_file,'parent'),'predict',smart_name_parser(selected_file,'short')))
            elif os.path.exists(os.path.join(smart_name_parser(selected_file,'parent'),'GT',smart_name_parser(selected_file,'short'))):
                # search in GT folder...
                self.paint.set_mask(os.path.join(smart_name_parser(selected_file,'parent'),'GT',smart_name_parser(selected_file,'short')))
            elif os.path.exists(os.path.join(smart_name_parser(selected_file,'parent'),'ground_truth',smart_name_parser(selected_file,'short'))):
                self.paint.set_mask(os.path.join(smart_name_parser(selected_file,'parent'),'ground_truth',smart_name_parser(selected_file,'short')))
            elif os.path.exists(os.path.join(smart_name_parser(smart_name_parser(selected_file,'parent'),'parent'),'GT',smart_name_parser(selected_file,'short'))):
                # search in GT folder... in parent of parent orga org GT
                self.paint.set_mask(os.path.join(smart_name_parser(smart_name_parser(selected_file,'parent'),'parent'),'GT',smart_name_parser(selected_file,'short')))
            elif os.path.exists(os.path.join(smart_name_parser(smart_name_parser(selected_file,'parent'),'parent'),'ground_truth',smart_name_parser(selected_file,'short'))):
                self.paint.set_mask(os.path.join(smart_name_parser(smart_name_parser(selected_file,'parent'),'parent'),'ground_truth',smart_name_parser(selected_file,'short')))
            elif os.path.exists(smart_name_parser(selected_file,'training_3_classes.tif')):
                self.paint.set_mask(smart_name_parser(selected_file,'training_3_classes.tif'))
            elif os.path.exists(smart_name_parser(selected_file, 'handCorrection.tif')):
                self.paint.set_mask(smart_name_parser(selected_file, 'handCorrection.tif'))
            else:
                # image not found --> need create a fake empty image --> not sure what I do below is the smartest but maybe ok still
                try:
                    # if len(self.paint.paint.raw_image.shape)<=2:
                    #     self.paint.set_mask(np.zeros(shape=(*self.paint.paint.raw_image.shape,),dtype=np.uint8))

                        # print(self.paint.paint.raw_image.shape)
                        self.paint.set_mask(np.zeros(shape=(self.paint.paint.raw_image.shape[0],self.paint.paint.raw_image.shape[1]),dtype=np.uint8))
                    # else:
                    #     self.paint.set_mask(np.zeros(shape=(*self.paint.paint.raw_image.shape[0:-1], 1), dtype=np.uint8))
                except:
                    traceback.print_exc()
                # self.paint.set_mask(None)

            # # try load the axon mask
            # try:
            #     self.paint.paint.vdp.shapes = []
            #     path_to_axons = smart_name_parser(selected_file,'axons.tif')
            #     if os.path.exists(path_to_axons):
            #         axons = Img(path_to_axons)
            #
            #         contours = get_contours(axons)
            #         if contours:
            #             for contour in contours:
            #                 # color = 0xFF0000, stroke = 2.5)
            #                 contour.color = 0xFF0000
            #                 contour.stroke = 2.5
            #             self.paint.paint.vdp.shapes.extend(contours)
            # except:
            #     traceback.print_exc()
            #     # self.paint.paint.vdp.shapes = []

            # update content depending on tabs in fact
            # if selected_tab_name.startswith('seg'):
            #
            #     # enable mask and drawing
            #     self.paint.maskVisible = True
            #     self.paint.enableMouseTracking()
            #     TA_path_alternative, TA_path = smart_name_parser(
            #         selected_file,
            #         ordered_output=['handCorrection.png', 'handCorrection.tif'])
            #     mask_name = None
            #     if os.path.isfile(TA_path):
            #         mask_name = TA_path
            #     else:
            #         if os.path.isfile(TA_path_alternative):
            #             mask_name = TA_path_alternative
            #     if mask_name is not None:
            #         self.paint.set_mask(mask_name)
            # else:
            #     # disable mask and drawing
            #
            #     self.paint.maskVisible = False
            #     self.paint.disableMouseTracking()
        else:
            self.paint.set_image(None)
            self.last_opened_file_name.setText('')

    def selectionChanged(self):
        try:
            self.update_preview_depending_on_selected_tab()
            # print('TODO --> DO CODE')
        except:
            traceback.print_exc()


    def create_paint(self, number_of_GT_channels=10):

        w = scrollablepaint_multichannel_edit(force_nb_of_channels=number_of_GT_channels)
        w.paint.multichannel_mode = True
        # w.paint.drawing_enabled = True
        # w.enableMouseTracking()
        # w.paint.setMouseTracking(True)

        # w = scrollable_paint(custom_paint_panel=test)

        # test.vdp.active = True
        # test.vdp.drawing_mode = True

        # w.set_image(
        #     '/E/Sample_images/sample_image_neurons_laurence_had/data_test/image.png')

        return w

    def dragEnterEvent(self, event):
        # we transfer all the DNDs to the list
        # selected_tab_idx, _ = self.get_cur_tab_index_and_name()
        self.list.get_list(0).dragEnterEvent(event)

    def dragMoveEvent(self, event):
        # we transfer all the DNDs to the list
        # selected_tab_idx, _ = self.get_cur_tab_index_and_name()
        self.list.get_list(0).dragMoveEvent(event)

    def dropEvent(self, event):
        # we transfer all the DNDs to the list
        # selected_tab_idx, _ = self.get_cur_tab_index_and_name()
        self.list.get_list(0).dropEvent(event)


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
    w = GT_editor()
    # w.setWindowIcon(app_icon)
    w.show()
    sys.exit(app.exec_())