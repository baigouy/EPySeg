# channels seem be fixed --> cool
# finalize whsed and corrections --> make sure they are channel dependent
# check which channels are being restored by the new wshed within the paint arena
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPalette, QKeySequence, QPainter
from PyQt5.QtWidgets import QScrollArea, QVBoxLayout, QWidget, QSpinBox, QComboBox, QToolBar, QStatusBar, QLabel, QHBoxLayout
import qtawesome as qta

from epyseg.draw.shapes.freehand2d import Freehand2D
from epyseg.draw.shapes.image2d import Image2D
from epyseg.draw.shapes.rect2d import Rect2D
from epyseg.img import Img, int24_to_RGB
from epyseg.ta.GUI.paint2 import Createpaintwidget

# TODO --> could maybe add icons just below scroll if needed to add actions
# j'arrive pas à creer un truc dejà self embedded
# make it also handle other stuff

# --> nb replace the Createpaintwidget in multi_image_display.py by this --> better in fact
# shall I do full screen ???
# TODO handle channels too if needed
# could store mask and memory of drawing here too --> maybe on mouse release ???
# avec des ctrl+Z et des ctrl+R


# need zooms
#

class scrollable_paint(QWidget):

    def __init__(self, custom_paint_panel=None):
        super().__init__()
        if custom_paint_panel is None:
            self.paint = Createpaintwidget()
        else:
            self.paint = custom_paint_panel
        layout = QVBoxLayout()
        self.is_width_for_alternating = True
        # same as the other with embed in scroll area
        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.paint)
        self.paint.scrollArea = self.scrollArea
        layout.addWidget(self.scrollArea)
        self.setLayout(layout)
        status_bar = QStatusBar()
        layout.addWidget(status_bar)
        self.paint.statusBar = status_bar
        self.add_shortcuts()
        self.drawing_commands()

    # this guy should handle the shortcuts it can and pass the rest to the other layer
    def add_shortcuts(self):
        # print('SHORTCUTS ADDED')

        # if True:
        #     return
        zoomPlus = QtWidgets.QShortcut("Ctrl+Shift+=", self)
        zoomPlus.activated.connect(self.zoomIn)
        zoomPlus.setContext(QtCore.Qt.ApplicationShortcut)



        # that works and disconnects all
        # zoomPlus.disconnect()



        zoomPlus2 = QtWidgets.QShortcut("Ctrl++", self)
        zoomPlus2.activated.connect(self.zoomIn)
        zoomPlus2.setContext(QtCore.Qt.ApplicationShortcut)

        zoomMinus = QtWidgets.QShortcut("Ctrl+Shift+-", self)
        zoomMinus.activated.connect(self.zoomOut)
        zoomMinus.setContext(QtCore.Qt.ApplicationShortcut)

        zoomMinus2 = QtWidgets.QShortcut("Ctrl+-", self)
        zoomMinus2.activated.connect(self.zoomOut)
        zoomMinus2.setContext(QtCore.Qt.ApplicationShortcut)

        ctrl0 = QtWidgets.QShortcut("Ctrl+0", self)
        ctrl0.activated.connect(self.zoom_reset)
        ctrl0.setContext(QtCore.Qt.ApplicationShortcut)

        self.ctrlS = QtWidgets.QShortcut("Ctrl+S", self)
        self.ctrlS.activated.connect(self.paint.save)
        self.ctrlS.setContext(QtCore.Qt.ApplicationShortcut)


        self.ctrlM = QtWidgets.QShortcut("Ctrl+M", self)
        self.ctrlM.activated.connect(self.paint.ctrl_m_apply)
        self.ctrlM.setContext(QtCore.Qt.ApplicationShortcut)

        self.shrtM = QtWidgets.QShortcut("M", self)
        self.shrtM.activated.connect(self.paint.m_apply)
        self.shrtM.setContext(QtCore.Qt.ApplicationShortcut)

        # I can connect to progeny --> makes sense and really easy --> cool
        self.enterShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self)
        self.enterShortcut.activated.connect(self.paint.apply)
        self.enterShortcut.setContext(QtCore.Qt.ApplicationShortcut)

        self.enterShortcut2 = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self)
        self.enterShortcut2.activated.connect(self.paint.apply)
        self.enterShortcut2.setContext(QtCore.Qt.ApplicationShortcut)

        # en fait je peux aussi connecter des trucs au descendant --> pas bete et tres facile à faire!!!

        # pkoi ça ne marche pas !!!
        self.shiftEnterShortcut = QtWidgets.QShortcut("Shift+Enter", self)
        self.shiftEnterShortcut.activated.connect(self.paint.shift_apply)
        self.shiftEnterShortcut.setContext(QtCore.Qt.ApplicationShortcut)
        #
        self.shiftEnterShortcut2 = QtWidgets.QShortcut("Shift+Return", self)
        self.shiftEnterShortcut2.activated.connect(self.paint.shift_apply)
        self.shiftEnterShortcut2.setContext(QtCore.Qt.ApplicationShortcut)

        self.ctrl_shift_S_grab_screen_shot = QtWidgets.QShortcut('Ctrl+Shift+S', self)
        self.ctrl_shift_S_grab_screen_shot.activated.connect(self.paint.grab_screen_shot)
        self.ctrl_shift_S_grab_screen_shot.setContext(QtCore.Qt.ApplicationShortcut)

        self.supr = QtWidgets.QShortcut(QKeySequence(Qt.Key_Delete), self)
        self.supr.activated.connect(self.paint.suppr_pressed)
        self.supr.setContext(QtCore.Qt.ApplicationShortcut)



        # peut etre faire un fullscreen shortcut --> can be useful

    def enable_shortcuts(self):
        self.disable_shortcuts()
        self.ctrlS.activated.connect(self.paint.save)
        self.ctrlM.activated.connect(self.paint.ctrl_m_apply)
        self.shrtM.activated.connect(self.paint.m_apply)
        self.enterShortcut.activated.connect(self.paint.apply)
        self.enterShortcut2.activated.connect(self.paint.apply)
        self.shiftEnterShortcut.activated.connect(self.paint.shift_apply)
        self.shiftEnterShortcut2.activated.connect(self.paint.shift_apply)

    def disable_shortcuts(self):
        # TODO
        try:
            self.ctrlS.disconnect()
        except:
            pass
        # TODO
        try:
            self.ctrlM.disconnect()
        except:
            pass
        try:
            self.shrtM.disconnect()
        except:
            pass
        try:
            self.enterShortcut.disconnect()
        except:
            pass
        try:
            self.enterShortcut2.disconnect()
        except:
            pass
        try:
            self.shiftEnterShortcut.disconnect()
        except:
            pass
        try:
            self.shiftEnterShortcut2.disconnect()
        except:
            pass

    def enableMouseTracking(self):
        self.paint.drawing_enabled = True

    def disableMouseTracking(self):
        self.paint.drawing_enabled = False



    # ça marche --> ajouter fit to width ou fit to height
    def fit_to_width_or_height(self):
        if self.is_width_for_alternating:
            self.fit_to_width()
        else:
            self.fit_to_height()
        self.is_width_for_alternating = not self.is_width_for_alternating

    # almost there but small bugs
    def fit_to_width(self):
        if self.paint.image is None:
            return
        # self.scrollArea = self.scroll_areas[nb]
        width = self.scrollArea.width() - 2
        width -= self.scrollArea.verticalScrollBar().sizeHint().width()
        # height-=self.scrollArea.horizontalScrollBar().sizeHint().height()
        width_im = self.paint.image.width()
        scale = width / width_im
        self.scaleImage(scale)

    def fit_to_height(self, nb=0):
        # paint = self.paints[nb]
        if self.paint.image is None:
            return
        # scrollArea = self.scroll_areas[nb]
        height = self.scrollArea.height() - 2
        height -= self.scrollArea.horizontalScrollBar().sizeHint().height()
        height_im = self.paint.image.height()
        scale = height / height_im
        self.scaleImage(scale)

    def fit_to_window(self):
        # compute best fit
        # just get the size of the stuff and best fit it
        # paint = self.paints[nb]
        if self.paint.image is None:
            return

        # QScrollArea.ensureWidgetVisible(
        # QScrollArea.ensureVisible
        # QScrollArea.setWidgetResizable
        # scrollArea = self.scroll_areas[nb]
        width = self.scrollArea.width() - 2  # required to make sure bars not visible
        height =self. scrollArea.height() - 2

        # scale image that it fits in --> change scale

        # width-=self.scrollArea.verticalScrollBar().sizeHint().width()
        # height-=self.scrollArea.horizontalScrollBar().sizeHint().height()

        height_im = self.paint.image.height()
        width_im = self.paint.image.width()
        scale = height / height_im
        if width / width_im < scale:
            scale = width / width_im
        self.scaleImage(scale)




    def zoomIn(self):
        # 10% more each time
        self.scaleImage(self.paint.scale + (self.paint.scale*10.0/100.0))# une hausse de 10% à chaque fois

    def zoomOut(self):
        # 10% less each time
        self.scaleImage(self.paint.scale - (self.paint.scale*10.0/100.0))

    def zoom_reset(self):
        # resets zoom
        self.scaleImage(1.0)

    # pb this is redundant with the main GUI and not in sync --> can cause a lot of trouble --> dactivate for now
    def full_screen(self):
            if not self.isFullScreen():
                self.fullscreen.setIcon(qta.icon('mdi.fullscreen-exit'))
                self.fullscreen.setToolTip('Exit full screen')
                self.showFullScreen()
                # self.setWindowFlags(
                #     QtCore.Qt.Window |
                #     QtCore.Qt.CustomizeWindowHint |
                #     # QtCore.Qt.WindowTitleHint |
                #     # QtCore.Qt.WindowCloseButtonHint |
                #     QtCore.Qt.WindowStaysOnTopHint
                # )
            else:
                # settings = QSettings()
                # self.Stack.restoreGeometry(settings.value("geometry")) #.toByteArray()
                self.fullscreen.setIcon(qta.icon('mdi.fullscreen'))
                self.fullscreen.setToolTip('Enter full screen')
                self.setWindowFlags(QtCore.Qt.Widget)
                # # self.Stack.setWindowFlags(self.flags)
                # self.grid.addWidget(self.Stack, 0,
                #                     0)  # pas trop mal mais j'arrive pas à le remettre dans le truc principal
                # # dirty hack to make it repaint properly --> obviously not all lines below are required but some are --> need test, the last line is key though
                # self.grid.update()
                # self.Stack.update()
                # self.Stack.show()
                # self.centralWidget().setLayout(self.grid)
                # self.centralWidget().update()
                # self.update()
                self.show()
                self.repaint()
                # self.Stack.update()
                # self.Stack.repaint()
                # self.centralWidget().repaint()

    def scaleImage(self, scale):
        self.paint.set_scale(scale)
        if self.paint.image is not None:
            self.paint.resize(self.paint.scale * self.paint.image.size())
        else:
            self.paint.resize(QSize(0, 0))
        self.paint.update()

    # can maybe add pen stuff here --> simpler and easier
    # need zoom buttons here too
    '''
    buttons/icons to add at some point
    
        
        "Cell smaller than this value will be destroyed"
        "small_cell_remover"
        
        
        "/Icons/refresh.png"
        "run 2 seeds watershed (click twice in a poorly segmented cell to have a the bond be created using the watershed algorithm)"
        
        "/Icons/Knob Snapback.png"
        "Undo"
        
        "/Icons/Knob Valid Blue.png"
        "remove small cells and apply correction"
        
        "/Icons/Knob Valid Green.png"
        "apply correction"
        
        "/Icons/channels.png"
        "Channel Selection"
        "Channel to use for Watershed, polarity, recentering,...")
        
        "/Icons/onebit_11.png"
        "Save"
        
        
        "/Icons/eye.png"
        "View below the mask (lasts 2s)"
        
        
        "/Icons/zoom in.png"
        "Zoom +"
        
        "/Icons/zoom out.png"
        "Zoom -"
        
        "/Icons/1in1.png"
        "1:1 aspect ratio"
        
        "/Icons/fit_2_screen.gif"
        "Fit to panel"
        
        
        "/Icons/double_diagonal_arrow.gif"
        "Alternate between best fit in height or in width"
        
        
        "/Icons/Import Picture Document.png"
        "Send Current View To The System Clipboard"
        
        "/Icons/pipet_small.png"
        "Color pipet"
        
        "Mask col"
        "mask color"
        
        "Correct Drift" 
    
    '''

    # add shortcut for show hide mask and maybe also full screen --> not a good idea

    def drawing_commands(self):
        self.penSize = QSpinBox(objectName='penSize')
        self.penSize.setToolTip('Change this value to change pen size')
        self.penSize.setSingleStep(1)
        self.penSize.setRange(1, 256)
        self.penSize.setValue(3)
        self.penSize.valueChanged.connect(self.penSizechange)

        self.channels = QComboBox(objectName='channels')
        self.channels.setToolTip('Select the channel of interest')
        self.channels.addItem("merge")
        # self.channels.addItem("0")
        # self.channels.addItems(["1", "2", "3"])
        self.channels.currentIndexChanged.connect(self.channelChange)


        hlayout_of_toolbars = QHBoxLayout()
        hlayout_of_toolbars.setAlignment(Qt.AlignLeft)
        self.tb1 = QToolBar()
        self.tb2 = QToolBar()
        self.tb3 = QToolBar()


        pencil_action = QtWidgets.QAction(qta.icon('ei.pencil'), 'Pen Size', self)
        pencil_action.setEnabled(False)
        self.tb2.addAction(pencil_action)
        # tb.addAction("sq...")
        self.tb2.addWidget(self.penSize)

        # toolButton = QToolButton()
        # toolButton.setText("Draw")
        # toolButton.setIcon(self.style().standardIcon(QStyle.SP_DriveFDIcon))
        # tb.addWidget(toolButton)

        # fa5.save
        save_action = QtWidgets.QAction(qta.icon('fa5.save'), 'Save mask', self)
        save_action.triggered.connect(self.paint.save)
        self.tb2.addAction(save_action)


        # toolButton2 = QToolButton()
        # toolButton2.setText("Draw")
        # toolButton2.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        # tb.addWidget(toolButton2)


        show_hide_mask =QtWidgets.QAction(qta.icon('fa.bullseye'), 'Show/hide mask (same as pressing "M")', self)
        show_hide_mask.triggered.connect(self.paint.m_apply)
        self.tb2.addAction(show_hide_mask)

        apply_drawing = QtWidgets.QAction(qta.icon('fa.check'), 'Apply drawing (same as pressing the "Enter" key)', self)
        apply_drawing.triggered.connect(self.paint.apply)
        self.tb2.addAction(apply_drawing)

        # toolButton3 = QToolButton()
        # toolButton3.setText("Draw")
        # toolButton3.setIcon(self.style().standardIcon(QStyle.SP_DialogCancelButton))
        # tb.addWidget(toolButton3)

        # toolButton4 = QToolButton()
        # toolButton4.setText("Draw")
        # toolButton4.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        # tb.addWidget(toolButton4)

        # toolButton5 = QToolButton()
        # toolButton5.setText("Draw")
        # toolButton5.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        # tb.addWidget(toolButton5)

        # toolButton6 = QToolButton()
        # toolButton6.setText("Draw")
        # toolButton6.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        # tb.addWidget(toolButton6)

        # toolButton7 = QToolButton()
        # toolButton7.setText("Draw")
        # toolButton7.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        # tb.addWidget(toolButton7)

        self.small_cell_size = QSpinBox()
        self.small_cell_size.setToolTip('Select the minimal pixel area for a cell to be considered as a cell.\nAny cell having an area less than this value will be removed.')
        self.small_cell_size.setSingleStep(1)
        self.small_cell_size.setRange(1, 1000000)
        self.small_cell_size.setValue(10)
        self.small_cell_size.valueChanged.connect(self.small_cell_size_changed)

        self.tb2.addWidget(self.small_cell_size)
        apply_rm_small_cells = QtWidgets.QAction(qta.icon('mdi.check-underline'), 'Apply drawing and remove small cells (same as pressing "Shift+Enter")', self)
        apply_rm_small_cells.triggered.connect(self.paint.shift_apply)# self.paint.shift_apply
        self.tb2.addAction(apply_rm_small_cells)





        # marche pas --> comment faire pr que ça marche...
        local_reshed = QtWidgets.QAction(qta.icon('mdi.reload'), 'Locally seeded watershed (same as pressing "Ctrl/Cmd + M")', self)
        local_reshed.triggered.connect(self.paint.ctrl_m_apply)# self.paint.shift_apply
        self.tb2.addAction(local_reshed)
        # hgjhghj

        # toolButton8 = QToolButton()
        # toolButton8.setText("Zoom+")
        # toolButton8.setIcon(qta.icon('ei.zoom-in')) # mdi.magnify-plus-outline --> marche aussi
        # tb.addWidget(toolButton8)

        # toolButton9 = QToolButton()
        # toolButton9.setText("Zoom-")
        # toolButton9.setIcon(qta.icon('ei.zoom-out'))
        # tb.addWidget(toolButton9)

        zoom_plus = QtWidgets.QAction(qta.icon('ei.zoom-in'), 'Zoom+', self)
        zoom_plus.triggered.connect(self.zoomIn)
        self.tb1.addAction(zoom_plus)

        zoom_minus = QtWidgets.QAction(qta.icon('ei.zoom-out'), 'Zoom-', self)
        zoom_minus.triggered.connect(self.zoomOut)
        self.tb1.addAction(zoom_minus)


        zoom_width_or_height = QtWidgets.QAction(qta.icon('mdi.fit-to-page-outline'), 'Alternates between best fit in width and height', self)
        zoom_width_or_height.triggered.connect(self.fit_to_width_or_height)
        self.tb1.addAction(zoom_width_or_height)

        zoom_0 = QtWidgets.QAction(qta.icon('ei.resize-full'), 'Reset zoom', self)
        zoom_0.triggered.connect(self.zoom_reset)
        self.tb1.addAction(zoom_0)

        # redundant with TA full screen and not in sync --> disable for now
        # self.fullscreen = QtWidgets.QAction(qta.icon('mdi.fullscreen'), 'Full screen', self)
        # self.fullscreen.setToolTip('Enter full screen')
        # self.fullscreen.triggered.connect(self.full_screen)
        # self.tb1.addAction(self.fullscreen)




        # exit_fullscreen = QtWidgets.QAction(qta.icon('mdi.fullscreen'), 'Full screen', self)
        # exit_fullscreen.triggered.connect(self.showFullScreen)
        # self.tb1.addAction(fullscreen)



        self.channel_label = QLabel('Channels:')
        self.channel_label.setToolTip('Select the channel of interest')
        self.tb3.addWidget(self.channel_label)
        self.tb3.addWidget(self.channels)

        # mdi.fullscreen
        # mdi.fullscreen-exit
        # ei.fullscreen --> a bit less good I find
        # fa.bullseye

        # could have show and hide mask here --> see how I can do that --> but do not delete it in fact

        # self.layout().addWidget(self.tb1)
        hlayout_of_toolbars.addWidget(self.tb2)


        # self.tb2.addAction('toto')
        # self.tb2.addAction('ttutu')
        # self.tb2.addAction('tata')
        hlayout_of_toolbars.addWidget(self.tb3)


        # one for the zooms
        # one for drawing and apply different things
        # TODO do several toolbars to deactivate different levels of stuff
        # and only allow save in the appropriate stuff

        # self.tb3.addAction('toto')
        # self.tb3.addAction('ttutu')
        # self.tb3.addAction('tata')
        hlayout_of_toolbars.addWidget(self.tb1)

        self.layout().addLayout(hlayout_of_toolbars)

    def get_selected_channel(self):
        idx = self.channels.currentIndex()
        if idx == 0:
            return None
        else:
            return idx-1

    # TODO also implement channel change directly within the display tool
    # if merge is applied --> apply on average of all channels --> maybe not so smart an idea but ok to start with and better than what I do in TA
    def channelChange(self, i):
        # update displayed image depending on channel
        # dqqsdqsdqsd
        # pass
        # try change channel if
        self.paint.channelChange(i)
        # print('in channel change !!!')
        # if self.img is not None:
        #     # print('in', self.img.metadata)
        #     if self.Stack.currentIndex() == 0:
        #         # need copy the image --> implement that
        #         # print(self.img[..., i].copy())
        #         # print(self.img[..., i])
        #         if i == 0:
        #             self.paint.setImage(self.img)
        #             # print('original', self.img.metadata)
        #         else:
        #             # print('modified0', self.img.metadata)
        #             channel_img = self.img.imCopy(c=i - 1)  # it's here that it is affected
        #             # print('modified1', self.img.metadata)
        #             # print('modified2', channel_img.metadata)
        #             self.paint.setImage(channel_img)
        #         self.paint.update()
        # else:
        #     # logger.error("Not implemented yet TODO add support for channels in 3D viewer")
        #     # sdqdqsdsqdqsd
        #     self.loadVolume()


    # TODO maybe handle pen size change directly within the display --> can be useful by the way
    def small_cell_size_changed(self):
        self.paint.minimal_cell_size = self.small_cell_size.value()

    # TODO maybe handle pen size change directly within the display --> can be useful by the way
    def penSizechange(self):
        self.paint.brushSize = self.penSize.value()
        # self.update() # too slow if I update --> avoid
        # self.l1.setFont(QFont("Arial", size))

    def set_image(self, img):

        # probably I need store raw image to avoid issues --> TODO ???
        self.paint.set_image(img)# bug seems to be here

        img = self.paint.raw_image
        # print(type(img))

        # else I need also reset channels

        # if img is not None:
        #     if img.has_c():
        self._update_channels(img=img)
        # print('out')
        # channels = self.paint.get_nb_channels()
        # self._update_channels(channels)
        # make it update also the channels

    def set_mask(self, mask):
        self.paint.set_mask(mask)

    def _update_channels(self, img):
        selection = self.channels.currentIndex()
        self.channels.disconnect()
        self.channels.clear()
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
        # logger.debug('channels found ' + str(comboData))
        self.channels.addItems(comboData)
        # index = self.channels.findData(selection)

        self.channels.currentIndexChanged.connect(self.channelChange)
        # print("data", index)
        if selection != -1 and selection < self.channels.count():
            self.channels.setCurrentIndex(selection)
            # self.channelChange(selection)
        else:
            self.channels.setCurrentIndex(0)
            # self.channelChange(0)


    def freeze(self, bool, level=1):

        self.tb1.setEnabled(True)
        self.tb2.setEnabled(True)
        self.tb3.setEnabled(True)

        if level == 1:
            self.tb2.setEnabled(not bool)
        elif level == 2:
            self.tb2.setEnabled(not bool)
            self.tb3.setEnabled(not bool)

        if bool:
            self.disable_shortcuts()
        else:
            self.enable_shortcuts()
        # remove draw mode maybe and freeze shortcuts

# pas mal faut juste voir ce que je peux faire
# peut etre permetre de mettre des custom panels

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication


    # hand drawing panel
    if False:

        # TODO add a main method so it can be called directly
        # maybe just show a canvas and give it interesting props --> TODO --> really need fix that too!!!


        # should probably have his own scroll bar embedded somewhere

        app = QApplication(sys.argv)


        # quite easy todo in fact

        # no clue why it always works just once


        # maybe make a scrollable TA paint that will handle and behave like that
        # maybe handle also stacks at some point ---> see how
        # --> almost all is ok now
        class overriding_apply(Createpaintwidget):
            # all seems ok now and functions as in TA
            # just do the shift enter to get rid of small cells

            # avec ça ça marche 100% à la TA ... --> cool --> maybe make it a TA drawing pad
            def apply(self):
               self.apply_drawing(minimal_cell_size=0)

            def shift_apply(self):
                # MEGA TODO IMPLEMENT SIZE within this stuff!!!
                self.apply_drawing(minimal_cell_size=10)

            def ctrl_m_apply(self):
                self.manually_reseeded_wshed()

            def save(self):
                self.save_mask()

        w = scrollable_paint(
            custom_paint_panel=overriding_apply())  # ça marche --> permet de mettre des paint panels avec des proprietes particulieres --> assez facile en fait
        # w.set_image('/E/Sample_images/sample_images_PA/trash_test_mem/mini (copie)/focused_Series012.png')
        # w.set_mask('/E/Sample_images/sample_images_PA/trash_test_mem/mini (copie)/focused_Series012/handCorrection.png')
        # w.set_image('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/Optimized_projection_018.png')
        # w.set_mask('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif')

        w.set_image('/D/VirtualBox/book_chapter_image_segmentation_final_2021/new_figs/dataset_1/minimal_demo_sample_simplified_bckup_before_messing_with_two_seg_errors/1.tif')
        w.set_mask('/D/VirtualBox/book_chapter_image_segmentation_final_2021/new_figs/dataset_1/minimal_demo_sample_simplified_bckup_before_messing_with_two_seg_errors/1/handCorrection.tif')


        # ça marche mais tt finalizer
        # that seems to work

        # w.freeze(True)
        # w.freeze(False)
        # w.freeze(True)

        # all is so easy to do this way

        # ça marche --> tres facile a tester
        # --> how can I do a zoom
        # ask whether self scroll or not --> if scrollable

        # mask = w.get_mask()
        #
        # plt.imshow(mask)
        # plt.show()

        w.show()
        sys.exit(app.exec_())

    # vectorial panel for drawing ROIs --> that works --> keep this code cause very useful and many bug fix
    # TODO try convert any ROIs to something like a contour using the pavlidis algo --> put this in rps tools

    if True:
        # TODO add a main method so it can be called directly
        # maybe just show a canvas and give it interesting props --> TODO --> really need fix that too!!!

        # should probably have his own scroll bar embedded somewhere

        app = QApplication(sys.argv)


        # quite easy todo in fact

        # no clue why it always works just once

        # maybe make a scrollable TA paint that will handle and behave like that
        # maybe handle also stacks at some point ---> see how
        # --> almost all is ok now
        class overriding_apply(Createpaintwidget):
            # all seems ok now and functions as in TA
            # just do the shift enter to get rid of small cells
            def m_apply(self):
                self.vdp.active = not self.vdp.active
                self.update()

            # avec ça ça marche 100% à la TA ... --> cool --> maybe make it a TA drawing pad
            def apply(self):
                # self.apply_drawing(minimal_cell_size=0)
                pass

            def shift_apply(self):
                # MEGA TODO IMPLEMENT SIZE within this stuff!!!
                # self.apply_drawing(minimal_cell_size=10)
                pass

            def ctrl_m_apply(self):
                # print('seed')
                # self.manually_reseeded_wshed()
                pass

            def save(self):
                # self.save_mask()
                print('saving ROIs')
                print(self.vdp.shapes)
                # create an image the save size as the displayed one and open it
                if self.raw_image is None:
                    return
                if len(self.raw_image.shape)==2:
                    black_image = np.zeros_like(self.raw_image, dtype=np.uint64)
                else:
                    black_image = np.zeros(shape=(*self.raw_image[:-1],), dtype=np.uint64)
                # plt.imshow(black_image)
                # plt.show()
                # make it fill all the ROIs inside that
                im2d = Image2D(black_image)

                for iii,shape in enumerate(self.vdp.shapes):
                    # ideally need clone it and make it the enumerate color +1

                    shape.fill_color = iii+1
                    shape.color = None
                    # print(shape.fill_color)
                    # print(type(shape), shape.fill_color)
                    im2d.annotation.append(shape)



                # im2d.annotation.extend(self.vdp.shapes)

                # need clone the shape

                # I need remove antialiasing
                im = im2d.save(None,quality=QPainter.NonCosmeticDefaultPen)
                im = im2d.convert_qimage_to_numpy(im)

                # restore colors
                for shape in self.vdp.shapes:
                    shape.fill_color = None
                    shape.color = 0xFFFF00

                # faudrait tester en grandeur reelle pr voir si ça marche --> à faire !!!
                if True:
                    # TODO just save the raw file in one channel not in many --> check the  output by the way

                    # from skimage.measure import label
                    # lab = label(im, connectivity=2, background=0)
                    # im=lab

                    # I need remove antialiasing
                    Img(im.astype(np.float64), dimensions='hwc').save('/E/Sample_images/sample_image_neurons_laurence_had/data_test/test_raw.tif')

                    plt.imshow(im)
                    plt.show()

                # what I need to do here now is to save in the folder of the parent à la TA --> can save it as neurons.tif maybe




            # def suppr_pressed(self):
            #     # self.save_mask()
            #     # print('suppr')
            #     self.pa


        test = overriding_apply()
        test.vdp.shape_to_draw = Freehand2D

        # test.vdp.shape_to_draw = Rect2D # ça ça marche --> bug is in the other then...

        # there is a bug in the vectorial but no big deal probably

        w = scrollable_paint(
            custom_paint_panel=test)  # ça marche --> permet de mettre des paint panels avec des proprietes particulieres --> assez facile en fait
        # w.set_image('/E/Sample_images/sample_images_PA/trash_test_mem/mini (copie)/focused_Series012.png')
        # w.set_mask('/E/Sample_images/sample_images_PA/trash_test_mem/mini (copie)/focused_Series012/handCorrection.png')
        # w.set_image('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/Optimized_projection_018.png')
        # w.set_mask('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif')

        test.vdp.active = True
        test.vdp.drawing_mode = True

        # QApplication.setOverrideCursor(Qt.WaitCursor)
        # do lengthy process
        # QApplication.restoreOverrideCursor()

        w.set_image(
            '/E/Sample_images/sample_image_neurons_laurence_had/data_test/image.png')
        # w.set_mask('/D/VirtualBox/book_chapter_image_segmentation_final_2021/new_figs/dataset_1/minimal_demo_sample_simplified_bckup_before_messing_with_two_seg_errors/1/handCorrection.tif')

        # w.setCursor(Qt.BlankCursor)

        # ça marche mais tt finalizer
        # that seems to work

        # w.freeze(True)
        # w.freeze(False)
        # w.freeze(True)

        # we force the cursor to be visible --> set it by default
        # cursor = Qt.ArrowCursor
        # w.paint.setCursor(cursor)

        # all is so easy to do this way

        # ça marche --> tres facile a tester
        # --> how can I do a zoom
        # ask whether self scroll or not --> if scrollable

        # mask = w.get_mask()
        #
        # plt.imshow(mask)
        # plt.show()

        w.show()
        # w.setCursor(Qt.BlankCursor)
        sys.exit(app.exec_())
