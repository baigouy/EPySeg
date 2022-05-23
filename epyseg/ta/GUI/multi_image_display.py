# TODO need handle tracking and need break buttons to stop the loop

# none of the label drawings is great --> ignore t for now or maybe add it to the vectorial draw pane but do not allow edit and save of this and move it with the image!!!
import traceback
from functools import partial
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QRect, Qt, QSize
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication, QGridLayout, QScrollArea, QLabel, QVBoxLayout, QDialog, QStyle, \
    QStyleOptionTitleBar, QMessageBox, QDialogButtonBox
# from deprecated_demos.ta.wshed import Wshed
from epyseg.ta.segmentation.neo_wshed import wshed
# from epyseg.draw.widgets.paint import Createpaintwidget
from epyseg.img import Img, RGB_to_int24
from epyseg.draw.shapes.rect2d import Rect2D

# TODO add shortcuts to zoom and so on so that I can handle it better
# TODO maybe store the raw image in it

# in fact that is maybe already what I want!!!
# but I may also want to draw on it with a pen --> should have everything

# draw_mode= None #'pen' #'rect'

# fit to window or not
# maybe whatever happens the scroll should fit to window size

# NB in TA that works exactly as I want it !!!


# ok I finally have it
# implement all the zooms, crops, etc...
# --> TODO


# TODO --> maybe add a title
# TODO maybe offer to set a title in setimage --> can be a good idea
# faire un get image also
# faire un get mask
# ...
# faire un get shapes --> pas mal
# peut etre aussi permettre le right click pr supprimer des images


# TODO allow scroll all or zoom all at the same time or not!!!
# TODO allow support for shortcuts
# if sync shortcuts are for all, if not they are for the in focus stuff --> TODO

# vraiment tres bon mais à finaliser!!!

# TODO add labels --> would be really cool in a way
# maybe also add labels on the image ???

# TODO allow add labels on the sides --> like a table --> can be really useful!!!
from epyseg.ta.GUI.paint2 import Createpaintwidget
from epyseg.ta.tracking.local_to_track_correspondance import add_localID_to_trackID_correspondance_in_DB
from epyseg.ta.tracking.tools import smart_name_parser
from epyseg.ta.tracking.track_correction import swap_tracks, connect_tracks
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()

class ImgDisplayWindow(QDialog):

    # replace preview only by a drawing mode --> TODO
    # TODO try several displays at once then sync all
    def __init__(self, parent_window=None, draw_mode=None, default_width=512, default_height=512, nb_rows=1, nb_cols=1,
                 synced=True , lst=None, cur_frame_idx=None):
        super().__init__(parent=parent_window)
        self.default_width = default_width
        self.default_height = default_height
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.draw_mode = draw_mode
        self.synced = synced
        self.initUI()
        self.lst = lst
        self.cur_frame_idx = cur_frame_idx

        # self.last_propagated_mouse_event = None

    # override the QDialoog keyPressEvent to prevent it from stealing the return key from the drawing panels
    def keyPressEvent(self, event):

        # print('key intercepted ') # can I pass it to the paint in focus ??? next ????

        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.apply()
            return


        #
        # else:
        #
        #     print('in there')
        #     idx = self.get_idx_of_widget_under_mouse()
        #     # dirty hack to transfer shortcut to child
        #     if idx != -1:
        #         print('passing')
        #         # print('applying in', idx)
        #         if idx in [0,1,2]:
        #             print('forwarding')
        #             self.paints[idx].keyPressEvent(event)
        super().keyPressEvent(event)

    def initUI(self):
        self.is_width_for_alternating = True
        self.scale = 1.0
        self.scroll_areas = []
        self.paints = []
        self.labels = []
        # TODO should I also store the labels ???

        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # maybe set one master that can propagate the event to all others

        # TODO draw a label directly on the scrollpane --> need override its paint method
        # if self.label is not None:
        #     # formato
        #     canvasPainter.setPen(QColor("Green"))
        #     canvasPainter.setFont(QFont('Helvetica', 48))
        #     print('drawing text')
        #     canvasPainter.drawText(50, 50, self.label)

        for jjj in range(self.nb_rows):
            for iii in range(self.nb_cols):

                # ça marche pas mal est ce plus simple outside ????
                # or add a label directly on the image at various position just by drawing it in the paint window --> can also be very interesting!!!
                main_container = QVBoxLayout()
                # need store this in arrays
                # scrollArea = QScrollArea()
                scrollArea = QScrollArea()

                # maybe add a vertical stuff above

                # try add a Qlabel on top of it to label things up

                label = QLabel('')
                self.labels.append(label)

                scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

                # ui->scrollArea->verticalScrollBar()->isVisible();
                # self.setMouseTracking(not self.preview_only) # I was doing crazy stuff here --> rather keep the draing mode in the inner object --> much simpler to handle

                paint = Createpaintwidget()


                paint.vdp.active = False
                paint.vdp.drawing_mode = True

                paint.force_cursor_visible = True # required to force activate cursor to be visible
                if self.synced:
                    paint.propagate_mouse_move = True
                    paint.propagate_mouse_event = self.mouse_move_multihandler

                # code to override apply
                if True:
                    if jjj == 0:
                        # save a function with its partial
                        paint.apply = partial(self.custom_apply, paint)
                    elif jjj == 1:
                        paint.apply = partial(self.get_colors_drawn_over, paint)
                    else:
                        paint.vdp.drawing_mode = False


                if self.draw_mode == 'rect':
                    paint.vdp.active = True
                    paint.vdp.shape_to_draw = Rect2D
                # removing this makes it paint drawable --> really easy in fact
                # print(self.draw_mode is not None)
                paint.setMouseTracking(self.draw_mode is not None)  # KEEP IMPORTANT
                paint.drawing = False  # is that useful, I don't like it --> make it better

                # print(self.paint.hasMouseTracking())
                # self.paint.drawing = False # why always true
                # self.paint.mouseMoveEvent = self.mouseMoveEvent
                # self.paint.mousePressEvent = self.mousePressEvent
                # self.paint.mouseReleaseEvent = self.mouseReleaseEvent

                # self.prev_width = 192
                # self.prev_height = 192

                scrollArea.setGeometry(QRect(0, 0, self.default_width, self.default_height))
                scrollArea.setBackgroundRole(QPalette.Dark)
                scrollArea.setWidgetResizable(
                    False)  # allows it to resize when the parent window is resized --> force a new repaint

                if self.synced:
                    scrollArea.verticalScrollBar().valueChanged.connect(self.scroll_vertically_together)
                    scrollArea.horizontalScrollBar().valueChanged.connect(self.scroll_horizontally_together)

                scrollArea.setWidget(paint)
                paint.scrollArea = scrollArea
                # self.paint.setBaseSize(512,512)

                self.scroll_areas.append(scrollArea)
                self.paints.append(paint)

                self.setGeometry(QRect(0, 0, self.default_width, self.default_height))

                main_container.addWidget(label)
                main_container.addWidget(scrollArea)

                # layout.addWidget(scrollArea,jjj,iii)
                layout.addLayout(main_container, jjj, iii)
        # self.setFixedSize(self.size()) # I actually want it resizable and upon resize change image size accordingly

        # see how to define scrollarea size in the window --> that is really key though --> I think I have a bug maybe check in top TA
        # add shortcuts --> needs love
        # if self.synced:
        self.add_shortcuts()  # ça marche --> mais à améliorer if in sync remove self stuff
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        # dirty way of doing this but ok!!!
        self.buttonBox.button(QDialogButtonBox.Ok).setText("Next")
        self.buttonBox.button(QDialogButtonBox.Cancel).setText("Stop")

        self.buttonBox.accepted.connect(self.accept_if_clicked)
        self.buttonBox.rejected.connect(self.reject)



        # self.buttonBox.button(QDialogButtonBox.Ok).setAutoDefault(False)
        # self.buttonBox.button(QDialogButtonBox.Cancel).setAutoDefault(False)
        # self.buttonBox.button(QDialogButtonBox.Ok).setDefault(False)
        # self.buttonBox.button(QDialogButtonBox.Cancel).setDefault(False)



        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    # dirty hack to ignore enter pressed
    def accept_if_clicked(self):
        if  not  self.buttonBox.button(QDialogButtonBox.Ok).hasFocus():
            return
        return self.accept()


    def get_colors_drawn_over(self, paint):
        mask = paint.get_user_drawing()
        # print('I was called')
        if mask is None:
            return
        # print(mask)


        # need get all the colors of the raw image that correspond to user selection
        selected_colors =paint.get_raw_image()[mask!=0]
        selected_colors = RGB_to_int24(selected_colors)

        selected_colors =set(selected_colors.ravel().tolist())
        if 0xFFFFFF in selected_colors:
            selected_colors.remove(0xFFFFFF)
        selected_colors = list(selected_colors)

        # print('colors', selected_colors) #.ravel()
        #neee=d get the colors as RGB single values --> convert the output to RGB



        # ned convert to


        # plt.imshow(mask)
        # plt.show()

        # paint.set_mask(None)# reset the mask
        # paint.set_mask(None)
        # paint.reset_user_drawing()
        # paint.raw_user_drawing.fill(QtCore.Qt.transparent)

        # this is the only way I found to reset the user drawing but this is dirty and I must find a better  way
        paint.set_mask(np.zeros_like(mask))


         # marche pas...Why ?
        # paint.raw_user_drawing = QtGui.QImage(paint.imageDraw.size(), QtGui.QImage.Format_ARGB32)
        # paint.raw_user_drawing.fill(QtCore.Qt.transparent)  # somehow this is really required to have an empty image otherwise it is not --> ??? why

        # test = paint.get_user_drawing()
        # plt.imshow(test)
        # plt.show()
        # update drawing after user input has been reset

        paint.update()
        return selected_colors

    def custom_apply(self, paint):
        # print('you called me', paint, filename)
        # print(QtGui.QImageWriter.supportedImageFormats())
        # save as tiff ???

        # drawn_img = paint.imageDraw

        # ça marche mais c'est vraiment juste le dessin --> faudrait donc que je passe le masque entier en mask et pas seulement ça --> voir ds TA comment faire en fait et aussi faire que le mask et le wshed tourne là dedans en fait car smarter

        # drawn_img.save("/E/Sample_images/sample_images_PA/trash/image.png", "PNG", -1)
        # drawn_img.save("/E/Sample_images/sample_images_PA/trash/image.tif", "TIF", -1)
        # print('saving',"/E/Sample_images/sample_images_PA/trash/image.tif")
        # print(drawn_img)

        # nb will only work for red pen !!!! --> do not allow to change pen color
        mask = paint.get_mask()
        if mask is None:
            return
        # TODO would need to run the wshed on it --> finally fix or recode my wshed can be done with numba

        # ça marche en fait maintenant --> utiliser ça du coup!!!
        # d'ailleurs verif ce code et le nettoyer une bonne fois pr toute et le mettre dans paint et dans mask d'ailleurs...
        # if nothing is saved image become compeletely full of white
        mask = wshed(mask, seeds='mask')

        # print('saving filename',filename)
        # plt.imshow(mask[...,2])
        # plt.show()
        # Img(mask, dimensions='hw').save(filename)

        # update the mask
        # nb maybe only save on demand each ??? --> think what is the best way to do that
        paint.set_mask(mask)


        # Img(img, dimensions='hwc').save(filename, mode='raw')

    def mouse_move_multihandler(self, event):
        # pb is I need get the sender not to send it it all
        # very dirty and infinite loop --> see how to do that for all
        # print('toto', self, event)
        # just get it to draw the cursor --> no infinite propagation then
        # if event == self.last_propagated_mouse_event:
        #     return
        # idx = self.get_idx_of_widget_under_mouse()
        # print(idx)
        # copied_event = event
        # if event.type() == QEvent.MouseMove:
        #     copied_event = event.pos()
        # copied_event = event.copy()

        # copied_event = QMouseEvent()
        # self.receiver.move(event.pos())
        for iii, paint in enumerate(self.paints):
            try:
                #     if iii != idx:
                # MEGA TODO: SUPER ULTIMATE DIRTY HACK FOR MOUSE EVENT PROPAGATION THROUGH PAINT WIDGETS THINK ABOUT A SMARTER WAY TO DO THAT IN THE FUTURE yet ok for now...
                paint.propagate_mouse_move = False
                # paint.mouseMove(copied_event) #Event( event)
                # paint.mouseMoveEvent(copied_event) #Event( event)
                paint.mouseMoveEvent(event)  # Event( event)
                paint.propagate_mouse_move = True
            # self.last_propagated_mouse_event = event
            except:
                traceback.print_exc()
                pass


    # could add the same shortcuts within individual panels
    def add_shortcuts(self):
        # if True:
        #     return
        zoomPlus = QtWidgets.QShortcut("Ctrl+Shift+=", self)
        zoomPlus.activated.connect(self.zoomIn)
        zoomPlus.setContext(QtCore.Qt.ApplicationShortcut)  

        zoomPlus2 = QtWidgets.QShortcut("Ctrl++", self)
        zoomPlus2.activated.connect(self.zoomIn)
        zoomPlus2.setContext(QtCore.Qt.ApplicationShortcut)  

        zoomMinus = QtWidgets.QShortcut("Ctrl+Shift+-", self)
        zoomMinus.activated.connect(self.zoomOut)
        zoomMinus.setContext(QtCore.Qt.ApplicationShortcut)  

        zoomMinus2 = QtWidgets.QShortcut("Ctrl+-", self)
        zoomMinus2.activated.connect(self.zoomOut)
        zoomMinus2.setContext(QtCore.Qt.ApplicationShortcut)  

        # dirty reimplementation of the shortcuts --> TODO recode that better some day --> but ok for now
        enterShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self)  
        enterShortcut.activated.connect(self.apply)
        enterShortcut.setContext(QtCore.Qt.ApplicationShortcut)  

        enterShortcut2 = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self)  
        enterShortcut2.activated.connect(self.apply)
        enterShortcut2.setContext(QtCore.Qt.ApplicationShortcut)  

        shrtM = QtWidgets.QShortcut("M", self)
        shrtM.activated.connect(self.m_apply)
        shrtM.setContext(QtCore.Qt.ApplicationShortcut)

        ctrlM = QtWidgets.QShortcut("Ctrl+M", self)
        ctrlM.activated.connect(self.ctrl_m_apply)
        ctrlM.setContext(QtCore.Qt.ApplicationShortcut)

        # pkoi ça ne marche pas !!!
        shiftEnterShortcut = QtWidgets.QShortcut("Shift+Enter", self)
        shiftEnterShortcut.activated.connect(self.shift_apply)
        shiftEnterShortcut.setContext(QtCore.Qt.ApplicationShortcut)
        #
        shiftEnterShortcut2 = QtWidgets.QShortcut("Shift+Return", self)
        shiftEnterShortcut2.activated.connect(self.shift_apply)
        shiftEnterShortcut2.setContext(QtCore.Qt.ApplicationShortcut)



    def take_intelligent_decision_for_track_editing_based_on_user_selection(self, selected_cells_prev, selected_cells_cur, selected_cells_next):
        # if selected_cells_prev is not None and selected_cells_prev:
        #     # check how many cells have been selected and take action depending in the stuff
        #     pass
        # if selected_cells_cur is not None and selected_cells_cur:
        #     pass
        # if selected_cells_next is not None and selected_cells_next:
        #     pass
        if selected_cells_next is None:
            selected_cells_next = []
        if selected_cells_prev is None:
            selected_cells_prev = []
        if selected_cells_cur is None:
            selected_cells_cur = []

        if len(selected_cells_prev) + len(selected_cells_cur) + len(selected_cells_next) != 2:
            # print('Error: ')
            if len(selected_cells_prev) + len(selected_cells_cur) + len(selected_cells_next) < 2:
                # not enough cells --> need add more
                self._show_warning_msage(text='Not enough cells drawn over, please click on exactly two cells')
                return

            # print(len(selected_cells_prev) + len(selected_cells_cur) + len(selected_cells_next))
            # print(selected_cells_next, selected_cells_prev, selected_cells_cur)
            self._show_warning_msage()
            return

        # the user has drawn properly over the image --> some decision can be taken then
        # --> do the stuff

        # sqsqdsqsqdqsdqsdqsd

        # if both are in the same then do swap otherwise try connect cells
        # I  also need to know the current frame --> TODO

        # I need have the list and the current frame

        try:
            path_to_cur_track = smart_name_parser(self.lst[self.cur_frame_idx], 'tracked_cells_resized.tif')
        except:
            path_to_cur_track = None
        try:
            path_to_next_track = smart_name_parser(self.lst[self.cur_frame_idx+1], 'tracked_cells_resized.tif')
        except:
            path_to_next_track = None
        try:
            path_to_prev_track = smart_name_parser(self.lst[self.cur_frame_idx-1], 'tracked_cells_resized.tif')
        except:
            path_to_prev_track = None

        something_changed = False
        if len(selected_cells_cur) == 2:
            # launch a swap starting from the current image onwards
            # start swapping fropm current idx onwards
            something_changed = True
            swap_tracks(self.lst, self.cur_frame_idx, selected_cells_cur[0],selected_cells_cur[1])
            # check the images that need be reloaded and update them
            # I need update cur and next image
            if path_to_cur_track is not None:
                img = Img(path_to_cur_track)
                self.paints[4].set_image(img)
            if path_to_next_track is not None:
                img = Img(path_to_next_track)
                self.paints[5].set_image(img)

            # need reload two images
        if len(selected_cells_next) == 2:
            something_changed = True
            swap_tracks(self.lst, self.cur_frame_idx+1, selected_cells_next[0], selected_cells_next[1])

            if path_to_next_track is not None:
                img = Img(path_to_next_track)
                self.paints[5].set_image(img)

        if len(selected_cells_prev) == 2:
            something_changed = True
            swap_tracks(self.lst, self.cur_frame_idx-1, selected_cells_prev[0], selected_cells_prev[1])

            if path_to_prev_track is not None:
                img = Img(path_to_prev_track)
                self.paints[3].set_image(img)
            if path_to_cur_track is not None:
                img = Img(path_to_cur_track)
                self.paints[4].set_image(img)
            if path_to_next_track is not None:
                img = Img(path_to_next_track)
                self.paints[5].set_image(img)


        if len(selected_cells_cur)==1:
            if len(selected_cells_prev)==1:
                something_changed = True
                connect_tracks(self.lst, self.cur_frame_idx, selected_cells_cur[0],selected_cells_prev[0])
                if path_to_cur_track is not None:
                    img = Img(path_to_cur_track)
                    self.paints[4].set_image(img)
                if path_to_next_track is not None:
                    img = Img(path_to_next_track)
                    self.paints[5].set_image(img)

            if len(selected_cells_next)==1:
                something_changed = True
                connect_tracks(self.lst, self.cur_frame_idx+1, selected_cells_cur[0],selected_cells_next[0])

                if path_to_next_track is not None:
                    img = Img(path_to_next_track)
                    self.paints[5].set_image(img)
        else:
            logger.error('Please try to connect tracks across two consecutive frames, not between images distant from one another')

        # TODO better check the something changed to avoid unnecessary db updates that may be lengthy
        if something_changed:
            # need update the track to local ID correpsondace
            add_localID_to_trackID_correspondance_in_DB(self.lst)



    def shift_apply(self):
        idx = self.get_idx_of_widget_under_mouse()
        # dirty hack to transfer shortcut to child
        if idx != -1:
            # print('applying in', idx)
            if idx in [0,1,2]:
                self.paints[idx].apply_drawing(minimal_cell_size=10)


    # full of bugs --> deactivate it here ????
    def ctrl_m_apply(self):
        idx = self.get_idx_of_widget_under_mouse()
        # dirty hack to transfer shortcut to child
        if idx != -1:
            # print('applying in', idx)
            if idx in [0, 1, 2]:
                try:
                    self.paints[idx].manually_reseeded_wshed()
                except:
                    traceback.print_exc()


    def m_apply(self):
        idx = self.get_idx_of_widget_under_mouse()
        # dirty hack to transfer shortcut to child
        if idx != -1:
            # print('applying in', idx)
            if idx in [0,1,2]:
                self.paints[idx].m_apply()




    # ça marche mais c'est super dirty je pense ???? en plus pr je ne sais quelle raison marche pas avec enter
    # ça fera
    def apply(self):
        # print('apply method to override in main')
        idx = self.get_idx_of_widget_under_mouse()
        # dirty hack to transfer shortcut to child
        if idx != -1:
            # print('applying in', idx)
            if idx not in [3,4,5]:
                self.paints[idx].apply()
            else:
                # get swaps from all the images and take action if makes sense
                # or warn for errors
                selected_cells_prev = []
                selected_cells_cur = []
                selected_cells_next = []
                if len(self.paints) >= 4:
                    selected_cells_prev = self.paints[3].apply()
                if len(self.paints) >= 5:
                    selected_cells_cur = self.paints[4].apply()
                if len(self.paints) >= 6:
                    selected_cells_next = self.paints[5].apply()

                self.take_intelligent_decision_for_track_editing_based_on_user_selection(selected_cells_prev, selected_cells_cur, selected_cells_next)

    def _show_warning_msage(self, text='Too many cells for track correction please just draw over two cells, no more...'):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(text)
        msg.setWindowTitle("Error")
        # msg.setDetailedText("Detailed explanation of the error: .......") # TODO maybe add more some day
        retval = msg.exec_()
        # in case I would need to do something with the retval but I gss I do


        # print()
    # allows the bars to scroll together --> quite good in fact
    # do it also for horiz
    def scroll_vertically_together(self, event):
        # print(event)
        for scrollbar in self.scroll_areas:
            scrollbar.verticalScrollBar().setValue(event)

    def scroll_horizontally_together(self, event):
        # print(event)
        for scrollbar in self.scroll_areas:
            scrollbar.horizontalScrollBar().setValue(event)

    def set_labels(self, *args):
        for iii, label in enumerate(args):
            self.set_label(label, nb=iii)

    def set_images(self, *args):
        for iii, img in enumerate(args):
            self.set_image(img, nb=iii)

    def set_label(self, label, nb=0):
        if label is not None:
            self.labels[nb].setText(label)
        else:
            self.labels[nb].setText('')
        # paint = self.paints[nb]
        # paint.vdp.shapes.clear()
        # paint.set_image(img)  # bug is here
        # if img is None:
        #     paint.scale = self.scale = paint.vdp.scale = 1.  # will that work ??? remove global scale ???
        # else:
        #     max_size = min(self.prev_width / img.get_width(), self.prev_height / img.get_height())
        #     self.paint.scale = self.scale = self.paint.vdp.scale = max_size

        # TODO handle scale nicely here
        # also allow drawing over the image

        # if paint.image is not None:
        #     paint.resize(self.scale * paint.image.size())
            # self.scrollArea.resize(self.scale * self.paint.image.size())
            # self.scrollArea.resize(self.size())

        # print('self.paint.hasMouseTracking()', self.paint.hasMouseTracking())# how come it can track if no mouse tracking ???



    # should I offer a crop of the image --> maybe also make it to store the relevant raw data --> can be very useful
    # do a clear also for the image
    def set_image(self, img, nb=0):

        paint = self.paints[nb]
        paint.vdp.shapes.clear()
        paint.set_image(img)  # bug is here
        if img is None:
            paint.scale = self.scale = paint.vdp.scale = 1.  # will that work ??? remove global scale ???
        # else:
        #     max_size = min(self.prev_width / img.get_width(), self.prev_height / img.get_height())
        #     self.paint.scale = self.scale = self.paint.vdp.scale = max_size

        # TODO handle scale nicely here
        # also allow drawing over the image

        if paint.image is not None:
            paint.resize(self.scale * paint.image.size())
            # self.scrollArea.resize(self.scale * self.paint.image.size())
            # self.scrollArea.resize(self.size())

        # print('self.paint.hasMouseTracking()', self.paint.hasMouseTracking())# how come it can track if no mouse tracking ???



    # clear image
    def clear(self, nb=0):
        self.set_image(None, nb=nb)
        # print(self.size())
        # self.scrollArea.resize(self.size())
        size = self.size()
        scrollArea = self.scroll_areas[nb]
        scrollArea.setGeometry(QRect(0, 0, size.width(), size.height()))
        paint = self.paints[nb]
        paint.setGeometry(
            QRect(0, 0, self.default_width - 2, self.default_height - 2))  # force all black and no scrollbars ---> ok
        # self.scale = 1.

    # ça marche --> ajouter fit to width ou fit to height
    def fit_to_width_or_height(self):
        if self.is_width_for_alternating:
            self.fit_to_width()
        else:
            self.fit_to_height()
        self.is_width_for_alternating = not self.is_width_for_alternating

    # almost there but small bugs
    def fit_to_width(self, nb=0):
        paint = self.paints[nb]
        if paint.image is None:
            return
        scrollArea = self.scroll_areas[nb]
        width = scrollArea.width() - 2
        width -= scrollArea.verticalScrollBar().sizeHint().width()
        # height-=self.scrollArea.horizontalScrollBar().sizeHint().height()
        width_im = paint.image.width()
        scale = width / width_im
        self.scaleImage(scale, nb=nb)

    def fit_to_height(self, nb=0):
        paint = self.paints[nb]
        if paint.image is None:
            return
        scrollArea = self.scroll_areas[nb]
        height = scrollArea.height() - 2
        height -= scrollArea.horizontalScrollBar().sizeHint().height()
        height_im = paint.image.height()
        scale = height / height_im
        self.scaleImage(scale, nb=nb)

    def fit_to_window(self, nb=0):
        # compute best fit
        # just get the size of the stuff and best fit it
        paint = self.paints[nb]
        if paint.image is None:
            return

        # QScrollArea.ensureWidgetVisible(
        # QScrollArea.ensureVisible
        # QScrollArea.setWidgetResizable
        scrollArea = self.scroll_areas[nb]
        width = scrollArea.width() - 2  # required to make sure bars not visible
        height = scrollArea.height() - 2

        # scale image that it fits in --> change scale

        # width-=self.scrollArea.verticalScrollBar().sizeHint().width()
        # height-=self.scrollArea.horizontalScrollBar().sizeHint().height()

        height_im = paint.image.height()
        width_im = paint.image.width()
        scale = height / height_im
        if width / width_im < scale:
            scale = width / width_im
        self.scaleImage(scale, nb=nb)

    # don't want a factor
    def scaleImage(self, scale, nb=0):
        paint = self.paints[nb]
        self.scale = scale
        if paint.image is not None:
            paint.resize(self.scale * paint.image.size())
        else:
            # no image set size to 0, 0 --> scroll pane will auto adjust
            paint.resize(QSize(0, 0))
            # self.scale -= factor  # reset zoom

        paint.scale = self.scale
        paint.vdp.scale = self.scale
        paint.update()

        # self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        # self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        # self.zoomInAct.setEnabled(self.scale < self.max_scaling_factor)
        # self.zoomOutAct.setEnabled(self.scale > self.min_scaling_factor)

    # synced version of scale
    def scale_image_all(self, scale):
        for img_nb in range(len(self.paints)):
            self.scaleImage(scale, nb=img_nb)

    #
    # def fitToWindow(self):
    #     # TODO most likely a bug there because calls self.defaultSize that resets the scale --> MAKE NO SENSE
    #     fitToWindow = self.fitToWindowAct.isChecked()
    #     self.scrollArea.setWidgetResizable(fitToWindow)
    #     if not fitToWindow:
    #         self.defaultSize()

    def zoomIn(self):
        if self.synced:
            self.scale_image_all(scale=self.scale + 0.1)
        else:
            idx = self.get_idx_of_widget_under_mouse()
            if idx != -1:
                self.scaleImage(self.scale + 0.1, nb=idx)
            # get index

    def zoomOut(self):
        if self.synced:
            self.scale_image_all(scale=self.scale - 0.1)
        else:
            idx = self.get_idx_of_widget_under_mouse()
            if idx != -1:
                self.scaleImage(self.scale - 0.1, nb=idx)

    # get the index of the image under the mouse!!
    def get_idx_of_widget_under_mouse(self):
        pos = QtGui.QCursor.pos()
        widget = QApplication.widgetAt(pos)
        for iii, paint in enumerate(self.paints):
            if widget == paint:
                return iii
        for iii, scrollarea in enumerate(self.scroll_areas):
            if widget == scrollarea:
                return iii
        return -1

    def get_paints(self):
        return self.paints

    @staticmethod
    def display(parent=None, draw_mode=None, default_width=512, default_height=512, nb_rows=1, nb_cols=1, synced=True,
                images=None, labels=None, lst=None, cur_frame_idx=None):


        dialog = ImgDisplayWindow(parent, draw_mode=draw_mode, default_width=default_width,
                                  default_height=default_height, nb_cols=nb_cols, nb_rows=nb_rows, synced=synced, lst=lst, cur_frame_idx=cur_frame_idx)
        # dialog.showFullScreen()  # pas mal mais vraiùment full screen --> can cause trouble --> maybe change this some day


        # show dialog full screen without maximizing, see # https://stackoverflow.com/questions/15702470/how-to-make-a-window-that-occupies-the-full-screen-without-maximising
        titleBarHeight = dialog.style().pixelMetric(
            QStyle.PM_TitleBarHeight,
            QStyleOptionTitleBar(),
            dialog
        )

        # geometry = app.desktop().availableGeometry()
        geometry = QtWidgets.QDesktopWidget().availableGeometry()
        geometry.setHeight(geometry.height() - (titleBarHeight*2))

        dialog.setGeometry(geometry)
        # end show full screen without maximizing

        if images is not None:
            dialog.set_images(*images)

        if labels is not None:
            dialog.set_labels(*labels)

        result = dialog.exec_()
        augment = dialog.get_paints()
        return (augment, result == QDialog.Accepted)


# does not work
# class labelled_scrollarea(QScrollArea):
#
#     def __init__(self):
#         super().__init__()
#
#     # def paintEvent(self, event):
#     #     # ne marche pas car l'autre truc ecrit par dessus
#     #     # super().paintEvent(event)
#     #     super(labelled_scrollarea, self).paintEvent(event)
#     #
#     #     # self.viewport().paintEvent(event)
#     #     # pb the widget overrides that --> need first paint it then call that
#     #     # print('inside')
#     #     painter = QtGui.QPainter(self.viewport())
#     #     # this will never work here --> really need be done in the other painter
#     #
#     #     # painter = QPainter()
#     #     # painter.begin(self.viewport())
#     #     # self.drawRectangles(qp)
#     #     # painter.end()
#     #
#     #     # painter = QPainter(self.viewport())  # If a widget has a viewport, you have to pass that to the QPainter constructor: https://stackoverflow.com/questions/12226930/overriding-qpaintevents-in-pyqt
#     #     # painter = QPainter(self)  # If a widget has a viewport, you have to pass that to the QPainter constructor: https://stackoverflow.com/questions/12226930/overriding-qpaintevents-in-pyqt
#     #     # painter.setRenderHint(QPainter.Antialiasing)
#     #     painter.setPen(Qt.darkRed)
#     #     rectangle = QRectF(10.0, 20.0, 80.0, 60.0)
#     #     painter.drawRect(rectangle)
#     #
#     #     painter.setPen(QColor("Green"))
#     #     painter.setFont(QFont('Helvetica', 48))
#     #     #     print('drawing text')
#     #     painter.drawText(50, 50, 'this is a test of your system')
#     #
#     #     painter.end()
#     #
#     #     # scrollArea.paintEvent = paintEvent


if __name__ == '__main__':
    # TODO --> last thing todo is to apply several 'apply' methods for each of the paint methods --> maybe simply override them ???

    # TODO allow unknown number of files, sort of on demand creation --> clear and restore all paints and scrolls
    # move most of the commands to the image panel itself
    # allow crops
    # add shortcuts
    # offer synchronous pointer maybe ???
    # offer drift maybe ???
    # store the image directly in the object --> much smarter and easier to retrieve (maybe handle Z and or t stacks also from within)
    # would be perfect to see corrections
    # for corrections maybe first do the human one fully then try to automate things with automated solutions with accept of reject à la tinder --> TODO
    # already showing the three original images with masks, the three tracks and the scores with prev and next for the current as well as new and lost cells should do most of the job
    # later I can try automated corrections but quite ok already
    # NB IDEALLY THE SCALE SHOULD ALSO BE SAVED DIRECTLY IN THE THING THAT DISPLAYS IMAGES

    if True:
        # try open this as a dialog
        import sys

        app = QApplication(sys.argv)
        # TODO find a way to synchronize the mouse for all

        # find a way to propagate mouse events from one to all
        # ok = ImgDisplayWindow.display(draw_mode='pen', nb_cols=5, images=[
        #     Img('/E/Sample_images/sample_images_PA/egg_chambers/120404_division_EcadKI.lif - Series006_RGB_manual_proj.png'),
        #     Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini10_fake_swaps/focused_Series012.png'),
        #     Img('/E/Sample_images/sample_images_PA/egg_chambers/120404_division_EcadKI.lif - Series006_RGB_manual_proj.png'),
        #     Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini10_fake_swaps/focused_Series012.png'), None])

        lst = ['/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012.png', '/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series014.png', '/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series016.png']

        augment, ok = ImgDisplayWindow.display(draw_mode='pen', nb_rows=3, nb_cols=3, images=[
            [Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012.png'),Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012/handCorrection.tif')],
            Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series014.png'),
            Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series016.png'),
            Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012/tracked_cells_resized.tif'),
            Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series014/tracked_cells_resized.tif'),
            Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series016/tracked_cells_resized.tif'), None],
                                      labels=['test1', 'test2', 'test3', None], lst=lst, cur_frame_idx=1)
        print(augment, ok)
        sys.exit(0)

    if False:
        # pas

        # TODO essayer le multidisplay
        # avec un set pr bcp de choses

        # ok in  fact that is already a great popup window --> can I further improve it ???

        # just for a test
        app = QApplication(sys.argv)
        ex = ImgDisplayWindow(nb_rows=2, nb_cols=2)
        # ex = ImgDisplayWindow(preview_only=True)
        # img = Img('/home/aigouy/mon_prog/Python/Deep_learning/unet/data/membrane/test/11.png')
        # img = Img('/home/aigouy/mon_prog/Python/Deep_learning/unet/data/membrane/test/122.png')
        # img = Img('/home/aigouy/mon_prog/Python/data/3D_bicolor_ovipo.tif')
        # img = Img('/home/aigouy/mon_prog/Python/data/Image11.lsm')
        # img = Img('/home/aigouy/mon_prog/Python/data/lion.jpeg')
        # img = Img('/home/aigouy/mon_prog/Python/data/epi_test.png')

        img = Img(
            '/E/Sample_images/sample_images_PA/egg_chambers/120404_division_EcadKI.lif - Series006_RGB_manual_proj.png')
        ex.set_image(img)

        # test = QRectF(None, None, 128, 128)
        ex.scaleImage(
            22)  # ça marche faudra finalizer des trucs mais pas mal déjà --> tt mettre en interne dans ma classe d'intéret --> I'm almost done
        # store image inside and also give it more options
        ex.fit_to_window()  # ça marche mais pb avec la taille des barres --> voir comment faire et tricher
        ex.fit_to_height()
        ex.fit_to_width()

        # ça marche pas mal et en plus c'est simple!!!
        ex.fit_to_width_or_height()
        ex.fit_to_width_or_height()
        ex.fit_to_width_or_height()
        ex.fit_to_width_or_height()

        # img = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini10_fake_swaps/focused_Series012.png')
        ex.set_image(img)
        ex.clear()

        ex.set_image(img, nb=0)
        img = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini10_fake_swaps/focused_Series012.png')
        ex.set_image(img, nb=3)

        ex.scale_image_all(0.3)
        ex.scaleImage(1, nb=3)  # tt marche et en plus vraiment pas mal --> je vais adorer ce truc à la fin!!!

        # ça a l'air de marcher --> just allow or disallow sync
        # ex.set_image(None)
        ex.show()
        app.exec_()

        # all of the things seem ok !!!

        # all is ok for now
        # maybe add ROIs to zoom on a region of the image
        # maybe stor all in the image because it is easier to handle !!!
