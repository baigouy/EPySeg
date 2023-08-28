# TODO mauybe hide green pointer if mouse is out of the image --> dangerous
# NB this will ultimately become the one and only paint widget in the app --> all others or most of them should disappear and be replaced by that
# TODO need add the save but only for the mask edit item!!!
import os
from epyseg.settings.global_settings import set_UI # set the UI to be used py qtpy
set_UI()
import traceback
from qtpy.QtGui import QKeySequence
from skimage.measure import label, regionprops
from qtpy.QtCore import QRect, QTimer, Qt
from qtpy.QtWidgets import QWidget, QApplication, QMessageBox
from epyseg.dialogs.opensave import saveFileDialog
from epyseg.draw.widgets.vectorial import VectorialDrawPane
from qtpy.QtWidgets import QMenu
from qtpy import QtCore, QtGui, QtWidgets
from epyseg.img import toQimage, Img, get_white_bounds, RGB_to_int24, is_binary, auto_scale, save_as_tiff
from epyseg.ta.measurements.measurements3D.get_point_on_surface_if_centroid_is_bad import point_on_surface
from epyseg.ta.segmentation.neo_wshed import wshed
from epyseg.ta.selections.selection import get_colors_drawn_over, convert_selection_color_to_coords
from epyseg.ta.tracking.tools import smart_name_parser
import numpy as np
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()

class Createpaintwidget(QWidget):
    # auto_convert_float_to_binary_threshold = 0.4

    default_mask_name = 'handCorrection.tif'

    # maybe store
    def __init__(self, enable_shortcuts=False): #, methods_overriding=None

        super().__init__()
        #
        # if methods_overriding is not None:
        #     for k,v in methods_overriding.items():
        #         # replace internal method by external one
        #         k = v
        # self.image_path = None
        self.save_path = None
        self.vdp = VectorialDrawPane(active=False) # the vectorial drawing panel
        self.raw_image = None # the original unprocessed image
        self.image = None # the displayed image (a qimage representation of raw_image can be a subset of raw_image_also)
        self.raw_mask = None  # the original unprocessed mask
        self.imageDraw = None # the mask image drawn over the image
        self.raw_user_drawing = None # contains just the user drawing without image mask
        self.cursor = None
        self.maskVisible = True
        self.scale = 1.0
        self.drawing = False
        self.brushSize = 3
        self.minimal_cell_size = 10
        self._clear_size = 30
        self.drawColor = QtGui.QColor(QtCore.Qt.red) # blue green cyan
        self.eraseColor = QtGui.QColor(QtCore.Qt.black)
        self.eraseColor_visualizer = QtGui.QColor(QtCore.Qt.green)
        self.cursorColor = QtGui.QColor(QtCore.Qt.green)
        self.lastPoint = QtCore.QPoint()
        self.change = False
        self.propagate_mouse_move = False
        # KEEP IMPORTANT required to track mouse even when not clicked
        self.setMouseTracking(True)  # KEEP IMPORTANT
        self.scrollArea = None
        self.statusBar = None
        self.drawing_enabled = True
        self.channel = None
        self.force_cursor_visible = False # required to force activate cursor to be always visible
        self.save_file_name=None
        self.multichannel_mode = False

        self.auto_convert_float_to_binary_threshold = 0.4

        # self.multi_channel_editor_n_save = False
        # TODO add sortcuts

        if enable_shortcuts:
            self.add_shortcuts() # TODO make this optional as it can cause trouble

    def force_cursor_to_be_visible(self, boolean):
        self.force_cursor_visible = boolean

        # can be used for saving
    def set_save_path(self, path):
        self.save_path = path

    def get_save_path(self):
        return self.save_path

    def set_scale(self, scale):
        # print(self.scale) # 0.10000000000000014 --> this is still ok --> this value crashes 1.3877787807814457e-16

        # make sure scale is bounded to avoid issues and non sense scalings
        if scale<0.01:
            scale = 0.01
        if scale>50:
            scale=50

        self.scale = scale
        self.vdp.scale =scale
        # DO I NEED ADJUST MORE SCALES ???
        # self.update()

    def get_scale(self):
        return self.scale

    # comprend pas ça marche pas si parent
    def add_shortcuts(self):
        # zoomPlus = QtWidgets.QShortcut("Ctrl+Shift+=", self)
        # zoomPlus.activated.connect(self.zoomIn)
        # zoomPlus.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active
        #
        # zoomPlus2 = QtWidgets.QShortcut("Ctrl++", self)
        # zoomPlus2.activated.connect(self.zoomIn)
        # zoomPlus2.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active
        #
        # zoomMinus = QtWidgets.QShortcut("Ctrl+Shift+-", self)
        # zoomMinus.activated.connect(self.zoomOut)
        # zoomMinus.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active
        #
        # zoomMinus2 = QtWidgets.QShortcut("Ctrl+-", self)
        # zoomMinus2.activated.connect(self.zoomOut)
        # zoomMinus2.setContext(QtCore.Qt.ApplicationShortcut)

        # enter_pressed = QtWidgets.QShortcut("Ctrl+Shift+=", self)
        # zoomPlus.activated.connect(self.zoomIn)
        # zoomPlus.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active


        padEnterShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self)  # voila
        padEnterShortcut.activated.connect(self.apply)
        padEnterShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make s

        enterShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self)  # voila
        enterShortcut.activated.connect(self.apply)
        enterShortcut.setContext(QtCore.Qt.ApplicationShortcut)  # make s

        enterShortcut2 = QtWidgets.QShortcut('Shift+Return', self)  # voila
        enterShortcut2.activated.connect(self.apply)
        enterShortcut2.setContext(QtCore.Qt.ApplicationShortcut)  # make s

        # pb if parent has same shortcut then ignored
        # the pb is really with enter --> maybe stolen focus by dialog ????
        enterShortcut3 = QtWidgets.QShortcut('Shift+Enter', self)
        enterShortcut3.activated.connect(self.apply)
        enterShortcut3.setContext(QtCore.Qt.ApplicationShortcut)  # make sure the shorctut always remain active


        self.shrtM = QtWidgets.QShortcut("M", self)
        self.shrtM.activated.connect(self.m_apply)
        self.shrtM.setContext(QtCore.Qt.ApplicationShortcut)


        self.ctrl_shift_S_grab_screen_shot = QtWidgets.QShortcut('Ctrl+Shift+S', self)
        self.ctrl_shift_S_grab_screen_shot.activated.connect(self.grab_screen_shot)
        self.ctrl_shift_S_grab_screen_shot.setContext(QtCore.Qt.ApplicationShortcut)

        self.increase_contrastC = QtWidgets.QShortcut('C', self)
        self.increase_contrastC.activated.connect(self.increase_contrast)
        self.increase_contrastC.setContext(QtCore.Qt.ApplicationShortcut)

        # QtGui.QShortcut(
        # self.supr = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self)

        # QKeySequence(tr("Ctrl+X, Ctrl+C"))
        # QKeySequence(Qt.CTRL + Qt.Key_X, Qt.CTRL + Qt.Key_C)

        # self.supr = QtWidgets.QShortcut(QKeySequence(Qt.Key_Delete), self)
        # self.supr.activated.connect(self.suppr_pressed)
        # self.supr.setContext(QtCore.Qt.ApplicationShortcut)


        # connect the m shortcut to show and hide mask

    # this returns the coords of the cells drawn over --> very useful for clones, adding/removing cells such as dividing cells and dying cells !!!
    # that seems to work very well --> TODO finalize that now or soon
    def get_selection_coords_based_on_current_mask(self, bg_color=None):
        coords_of_selection = []

        # pb --> here I need do it on the labeled image -->

        img_to_analyze = self.get_raw_image()
        # si l'image est juste un mask binaire alors -->  faut la convertir en label --> TODO
        if is_binary(img_to_analyze):
            if len(img_to_analyze.shape)==3:
                # assume it's a handCorrection mask and the mask is 3channel
                img_to_analyze = img_to_analyze[...,0]
            img_to_analyze = label(img_to_analyze, connectivity=1, background=255) # will that always work
            bg_color = 0

        # why is this shit empty
        mask=self.get_user_drawing()

        # --> seems to be ok --> then why 0
        # import matplotlib.pyplot as plt
        # plt.imshow(img_to_analyze)
        # plt.show()

        selected_cells =  get_colors_drawn_over(mask,  img_to_analyze)
        if not selected_cells:
            logger.warning('No selection found --> nothing to do')
            return coords_of_selection

        self.set_mask(np.zeros_like(mask))
        self.update()

        tmp = img_to_analyze
        if len(tmp.shape) == 3:
            tmp = RGB_to_int24(tmp)
        #
        if bg_color is None:
            if 0xFFFFFF in tmp:
                bg_color = 0xFFFFFF
            else:
                bg_color = 0
        #
        # cell_label = label(tmp, connectivity=1, background=bg_color)
        #
        # # for all the selected cells --> return the coords --> probably a quite good idea...
        # for region in regionprops(cell_label):
        #     color = tmp[region.coords[0][0], region.coords[0][1]]
        #     if color in selected_cells:
        #         point_awlays_inside_the_cell = point_on_surface(region, cell_label)
        #         coords_of_selection.append(point_awlays_inside_the_cell)
        #
        # return coords_of_selection

        return convert_selection_color_to_coords(tmp, selected_cells=selected_cells, bg_color=bg_color)

    # added this so that I can easily get user selection using a cell ID
    def get_colors_drawn_over(self, forbidden_colors=[0,0xFFFFFF]):
        mask = self.get_user_drawing()
        # print('I was called')
        if mask is None:
            return

        selected_colors = get_colors_drawn_over(mask,  self.get_raw_image(),forbidden_colors=forbidden_colors)
        # # print(mask)
        # # need get all the colors of the raw image that correspond to user selection
        # selected_colors = self.get_raw_image()[mask != 0]
        # selected_colors = RGB_to_int24(selected_colors)
        # selected_colors = set(selected_colors.ravel().tolist())
        # # if 0xFFFFFF in selected_colors:
        # #     selected_colors.remove(0xFFFFFF)
        # if forbidden_colors is not None:
        #     if not isinstance(forbidden_colors, list):
        #         forbidden_colors = [forbidden_colors]
        #     for color in forbidden_colors:
        #         if color in selected_colors:
        #             selected_colors.remove(color)
        # selected_colors = list(selected_colors)
        self.set_mask(np.zeros_like(mask))
        self.update()
        return selected_colors

    # def suppr_pressed(self):

    def increase_contrast(self):
        # print('increasing contrast')
        try:
            # metadata = self.raw_image.metadata
            # sqqsdq
            # self.raw_image = Img(auto_threshold(self.raw_image),metadata=metadata)
            # self.set_image(self.raw_image)
            # shall I do it for the displayed image only ???? --> probably smarter porbably shoudl do a get display image that is independent of all other functions and handles all dims
            # very dumb --> needs be done just for the dipslayed image and nothing else and --> maybe get the image back from the screen directly --> would be smarted and better

            # maybe do this only on the displayed image --> using convert_qimage_to_numpy or do something that gets the image to be displayed including the channels
            # or do a tool that will recover the image of interest with channels and also the rest of the dimensions --> MEGA TODO !!!
            print('auto-increase contrast')
            # hack to preserve luts --> TODO
            copy = auto_scale(np.copy(self.raw_image))
            meta = None
            try:
                meta=self.raw_image.metadata
            except:
                pass
            # print(self.raw_image.metadata)
            self.set_display(copy, metadata=meta)
        except:
            traceback.print_exc()

    def grab_screen_shot(self):
            # screenshot = QPixmap.grabWindow(self.winId())
            output_file = saveFileDialog(parent_window=self, extensions="Supported Files (*.png);;All Files (*)",
                                         default_ext='.png')
            if output_file is not None:

                try:
                    # we hide the cursor
                    self.cursor = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
                    self.cursor.fill(QtCore.Qt.transparent)
                    self.update()
                    screenshot=self.grab()
                    screenshot.save(output_file, 'png')
                    # restore cursor
                    # or paint without cursor --> TODO do that better some day
                except:
                    logger.error('Could not grab a screenshot...')

    def save(self):
        # save_img
        print('ctrl S save method to override')

    def apply(self):
        print('apply method to override')

    def shift_apply(self):
        print('shift apply method to override')

    def ctrl_m_apply(self):
        print('ctrl m apply method to override')

    def suppr_pressed(self):
        # print('Suppr method to override')
        if self.vdp.active:
            self.vdp.removeCurShape()
            self.update()
        else:
            print('Suppr method to override')


    # def m_apply(self):
    #     print('m apply method to override')
    #     qsdsqqsdsq
    def m_apply(self):
        # print('m apply called')
        self.maskVisible = not self.maskVisible
        self.update()

    def set_display(self, display, metadata=None):
        # this actually sets the displayed image --> different from set imlage, can be used to show channels instead or stuff like that instead of the actula imge
        if isinstance(display, np.ndarray):
            display = toQimage(display, metadata=metadata)
        self.image = display
        self.update()

    def set_image(self, img):
        if img is None:
            self.save_file_name = None
            self.raw_image = None
            self.image = None
            self.imageDraw = None
            self.raw_mask = None
            self.raw_user_drawing = None
            self.unsetCursor()  # restore cursor
            self.update()
            return
        else:
                # remove cursor only if image is shown and not vectorial drawing mode!
                if self.vdp.active:
                    self.unsetCursor()
                else:
                    self.setCursor(Qt.BlankCursor)
                # try:
                #     if isinstance(img, np.ndarray):
                #         self.image = img.getQimage() # bug is here
                #         # set an empty mask
                #         self.imageDraw = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
                #         self.imageDraw.fill(QtCore.Qt.transparent)
                #     else:
                #         self.image = img[0].getQimage()
                #         # set the mask
                #         self.imageDraw = Img(self.createRGBA(img[1].getQimage()), dimensions='hwc').getQimage()
                # except:
                # NB THIS IS VERY REDUNDNAT ---> STICK TO ONE METHOD
                if isinstance(img, str):
                    # self.image_path = img# maybe this image path can be used for saving, could also be provided serparately
                    self.save_file_name = smart_name_parser(img, ordered_output=self.default_mask_name)
                    img = Img(img)

                # self.image_path=None # maybe this image path can be used for saving
                if isinstance(img, np.ndarray):

                    # bug is exactly here --> why and where

                    # dimensions = None
                    # if isinstance(img, Img):
                    #     dimensions = img.get_dimensions_as_string()

                    # self.image_path = None
                    self.raw_image = img

                    # print('tests',img.shape, isinstance(img,Img))

                    self.image = toQimage(img)# maybe this image path can be used for saving
                    if self.image is None:
                        logger.error('Image could not be displayed...')
                        self.unsetCursor()
                        self.update()
                        return
                    self.imageDraw = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
                    self.imageDraw.fill(QtCore.Qt.transparent)
                    self.raw_user_drawing =QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
                    self.raw_user_drawing.fill(QtCore.Qt.transparent)
                else:
                    # self.image_path = None
                    self.image = toQimage(img[0])
                    self.raw_image = img[0]  #
                    # same as set mask --> how to do that
                    self.imageDraw = toQimage(Img(self.createRGBA(img[1]), dimensions='hwc'),preserve_alpha=True)#.getQimage()
                    self.raw_user_drawing = QtGui.QImage(self.imageDraw.size(), QtGui.QImage.Format_ARGB32)
                    self.raw_user_drawing.fill(QtCore.Qt.transparent)

        if self.force_cursor_visible:
            self.unsetCursor()

        # self.image = QPixmap(100,200).toImage()
        width = self.image.size().width()
        height = self.image.size().height()
        top = self.geometry().x()
        left = self.geometry().y()
        self.setGeometry(top, left, width*self.scale, height*self.scale)

        self.cursor = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
        self.cursor.fill(QtCore.Qt.transparent)
        self.update()

    def get_raw_image(self):
        return self.raw_image

    def binarize(self, mask, auto_convert_float_to_binary=None, force=False):
        if auto_convert_float_to_binary is None:
            auto_convert_float_to_binary = self.auto_convert_float_to_binary_threshold
        if auto_convert_float_to_binary and (mask.max() <= 1 or force):
            # print('autoconvert')
            mask = mask > auto_convert_float_to_binary
        return mask

    # do I need that ??? maybe yes if I just wanna change the mask
    def set_mask(self, mask, auto_convert_float_to_binary=None):
        if auto_convert_float_to_binary is None:
            auto_convert_float_to_binary = self.auto_convert_float_to_binary_threshold
        if isinstance(mask, str):
            self.save_file_name = mask
            # self.image_path = img# maybe this image path can be used for saving, could also be provided serparately
            mask = Img(mask)
            self.raw_mask = mask
            # for TA compat
            if mask.has_c():
                # but should also save the raw mask as for the raw_image --> TODO --> maybe
                if self.channel is not None:
                    mask=mask[...,self.channel]
                else:
                    mask = mask[..., 0]

        # convert mask to image draw
        if mask is None:
            self.imageDraw = None
            self.raw_user_drawing = None
            self.raw_mask = None
        else:
            # auto convert float deep learning masks to binary --> maybe this should be a parameter ...
            # print('tada', auto_convert_float_to_binary, mask.max())

            mask = self.binarize(mask, auto_convert_float_to_binary=auto_convert_float_to_binary)


            # print('mask.shape',mask.shape)
            # convert mask to a mask
            # self.imageDraw = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
            # self.imageDraw.fill(QtCore.Qt.transparent)

            # print('in here tmp ', mask.shape)


            # if image has z channels and is more than just 2D / is 3D I could make this smarter by loading just the right image --> TODO --> sync the channel here with the channel of the currently dispalyer image
            # SHALL I MAKE A FULL GUI FOR THAT ON RELY ON WHAT I HAVE ALREADY !!!!
            self.imageDraw = toQimage(Img(self.createRGBA(mask), dimensions='hwc'),preserve_alpha=True) #.getQimage()  # marche pas car besoin d'une ARGB
            self.raw_user_drawing = QtGui.QImage(self.imageDraw.size(), QtGui.QImage.Format_ARGB32)
            self.raw_user_drawing.fill(QtCore.Qt.transparent) # somehow this is really required to have an empty image otherwise it is not --> ??? why
        self.update()


    # do I have a way to just get what was drawn by the user and the bounds of it --> maybe by substratction --> or can I get the difference
    # I need bounds of drawn stuff --> think about how to do that !!!
    # we have used the user drawing for something --> let's reset it
    # shall I also save the raw mask unperturbed by user drawing --> can be useful too in fact to do things with it
    def reset_user_drawing(self):
        if self.imageDraw is None:
            self.raw_user_drawing = None
        else:
            self.raw_user_drawing = QtGui.QImage(self.imageDraw.size(), QtGui.QImage.Format_ARGB32)
            self.raw_user_drawing.fill(QtCore.Qt.transparent)

            # shall I draw any user interaction also in a separate file so that I can get just what was drawn without having the original mask ??? --> maybe a good idea --> call it get user drawing ???
    def get_mask(self):
        # nb will only work for red pen !!!! --> do not allow to change pen color
        # pas mal mais image est RGBA
        if self.imageDraw is not None:
            # return Img.qimageToNumpy(self.imageDraw, mode='bnw8')
            mask = self.convert_qimage_to_numpy(self.imageDraw)[..., 2]
            # for some reason there are some 1 in the image --> needs a fix
            mask[mask!=255]=0
            return mask
        else:
            return None

    def get_user_drawing(self, show_erased=False):
        if self.raw_user_drawing is not None:



            # bug here the user draing contains the original mask --> bug

            all = self.convert_qimage_to_numpy(self.raw_user_drawing)

            # plt.imshow(all)
            # plt.show()

            # no clue why this stuff is empty
            # import matplotlib.pyplot as plt
            # plt.imshow(all)
            # plt.show()

            drawn = all[..., 2]


            # plt.imshow(erased) #--> ok
            # plt.show()

            # plt.imshow(drawn)
            # plt.show()


            drawn[drawn != 255] = 0

            if show_erased:
                erased = all[..., 1]
                erased[erased != 255] = 0
                return [erased, drawn]

            return drawn
        else:
            return None

    # from https://stackoverflow.com/questions/19902183/qimage-to-numpy-array-using-pyside
    # nb there seems to be a bug in the order of the channels
    def convert_qimage_to_numpy(self, qimage):
        qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        width = qimage.width()
        height = qimage.height()
        image_pointer = qimage.bits() # creates a deep copy --> this is what I want
        try:
            image_pointer.setsize(qimage.sizeInBytes()) # qt6 version of the stuff
        except:
            image_pointer.setsize(qimage.byteCount())


        # arr = np.array(image_pointer,copy=True).reshape(height, width, 4)
        arr = np.array(image_pointer).reshape(height, width, 4)
        return arr

    def createRGBA(self, handCorrection):





        # print('Im called')
        # use pen color to display the mask
        # in fact I need to put the real color
        # nb this will definitely not work if the user wants to have full 3D stuff

        # shall I also offer the zooming on the objects --> ????




        RGBA = np.zeros((handCorrection.shape[0], handCorrection.shape[1], 4), dtype=np.uint8)

        red = self.drawColor.red()
        green = self.drawColor.green()
        blue = self.drawColor.blue()

        #(handCorrection.shape[0], handCorrection.shape[1], 4))
        # mask = handCorrection[handCorrection==255]

        # ce truc est bleu --> pkoi
        # RGBA[..., 0] = 255#handCorrection --> totalement transp --> transparence color --> GREEN --> BLUE
        # RGBA[..., 1] = 0  # this is the red channel --> RED --> GREEN
        # RGBA[..., 2] = 0 # rien ici --> tt transparent --> alpha ??? --> ALPHA -->RED
        # RGBA[..., 3] = 255 # --> BLUE --> ALPHA --> ok

        # pkoi je swappe les channels avant

        # 255 partout --> blanc
        # 0 255 255 0 --> rouge --> 1 = red channel and 2 = alpha --> weird

        # RGBA[RGBA[..., 0] > 0, 0] = blue
        # RGBA[RGBA[..., 1] > 0, 1] = green
        # RGBA[RGBA[..., 2] > 0, 2] = red
        # RGBA[..., 3] = 255 # ne s'affiche pas et je comprend pas pkoi

        #BGR in fact --> need fix here or the other --> the one in


        # how can I do that in a simpler way ???
        # bug somewhere in qimage --> fix it some day --> due to bgra instead of RGBA
        # assumes image has just the first
        RGBA[handCorrection != 0, 0] = blue # b
        # RGBA[..., 1] = handCorrection
        RGBA[handCorrection != 0, 1] = green # g
        RGBA[handCorrection != 0, 2] = red # r
        RGBA[..., 3] = 255 # alpha --> indeed alpha
        RGBA[handCorrection == 0, 3] = 0 # ça marche maintenant --> super complexe qd meme je trouve

        # can I do it in one go --> I guess yes!!!
        # RGBA[..., 3] = 255
        # RGBA[handCorrection != 0] = blue,green,red, 0
        # RGBA[handCorrection==0, 3] = 255

        # ok mais plus transparent

        #
        # print(red, green, blue)

        # RGBA[RGBA[..., 0] == 255] = red
        # RGBA[RGBA[..., 1] == 255] = green
        # RGBA[RGBA[..., 2] == 255] = blue

        return RGBA

        # TODO also implement channel change directly within the display tool
        # if merge is applied --> apply on average of all channels --> maybe not so smart an idea but ok to start with and better than what I do in TA

    # dunno if I put it here or not ???
    # def get_nb_channels(self):
    #     if self.

    # depreacted --> ideally should not be used
    # TODO--> maybe implement a multi channel save -> TODO
    # skip
    def channelChange(self, i, skip_update_display=False):
        # update displayed image depending on channel
        # dqqsdqsdqsd
        # pass
        # try change channel if

        # print('in channel change !!!')
        # if image_to_display is None:
        #     image_to_display = self.raw_image
        if self.raw_image is not None:
            # print('in', self.img.metadata)
            # if self.Stack.currentIndex() == 0:
                # need copy the image --> implement that
                # print(self.img[..., i].copy())
                # print(self.img[..., i])
                if i == 0 and not skip_update_display:
                    self.set_display(self.raw_image)
                    self.channel = None
                    # print('original', self.img.metadata)
                else:
                    if not skip_update_display:
                        # print('modified0', self.img.metadata)
                        # I need a hack when the image is single channel yet I need several masks for it !!!
                        meta = None
                        try:
                            meta = self.raw_image
                        except:
                            pass
                        if self.multichannel_mode and i-1>=self.raw_image.shape[-1]:
                            channel_img = self.raw_image.imCopy(c=0) # if out of bonds load the first channel
                        else:
                            channel_img = self.raw_image.imCopy(c=i - 1)  # it's here that it is affected
                        self.set_display(
                                channel_img, metadata=meta)  # maybe do a set display instead rather --> easier to handle --> does a subest of the other
                    self.channel = i-1
                    # print('modified1', self.img.metadata)
                    # print('modified2', channel_img.metadata)

                    if self.multichannel_mode:
                        if self.raw_mask is not None:
                            # print('multichannel_mode', self.multichannel_mode)
                            try:
                                # print('self.raw_mask.shape',self.raw_mask.shape)
                                if len(self.raw_mask.shape)<=2:
                                    self.raw_mask = self.raw_mask[..., np.newaxis]
                                self.set_mask(self.raw_mask[..., i-1])
                            except:
                                # print(i-1,  self.raw_mask.shape[-1] , len(self.raw_mask.shape))
                                if i-1 >= self.raw_mask.shape[-1] and len(self.raw_mask.shape)>2:
                                    # extend mask


                                    if len(self.raw_mask.shape) > 2:
                                        raw_mask2 = np.zeros_like(self.raw_mask, shape=(*self.raw_mask.shape[0:-1], i))
                                    else:
                                        raw_mask2 = np.zeros_like(self.raw_mask, shape=(*self.raw_mask.shape, i))

                                    # copy existing mask to it
                                    # print(raw_mask2.shape, self.raw_mask.shape)
                                    for ch in range(self.raw_mask.shape[-1]):
                                        raw_mask2[...,ch]=self.raw_mask[...,ch]
                                    self.raw_mask = raw_mask2
                                    self.set_mask(self.raw_mask[..., i - 1])
                                else:
                                    traceback.print_exc()
                        else:
                            # create empty mask
                            if len(self.raw_image.shape)>2:
                                self.raw_mask = np.zeros(shape=(*self.raw_image.shape[0:-1],i))
                            else:
                                self.raw_mask = np.zeros(shape=(*self.raw_image.shape, i))

                            # print('doubs', self.raw_mask.shape, self.raw_image.shape)
                            self.set_mask(self.raw_mask[..., i - 1])
                self.update()
        else:
            self.channel = None
            # else:
            #     # logger.error("Not implemented yet TODO add support for channels in 3D viewer")
            #     # sdqdqsdsqdqsd
            #     self.loadVolume()


    def get_colors_drawn_over(self):
        mask = self.get_user_drawing()
        # print('I was called')
        if mask is None:
            return []
        # print(mask)

        # need get all the colors of the raw image that correspond to user selection
        selected_colors =self.get_raw_image()[mask!=0]
        selected_colors = RGB_to_int24(selected_colors)

        selected_colors =set(selected_colors.ravel().tolist())
        if 0xFFFFFF in selected_colors:
            selected_colors.remove(0xFFFFFF)
        selected_colors = list(selected_colors)
        # this is the only way I found to reset the user drawing but this is dirty and I must find a better  way
        self.set_mask(np.zeros_like(mask))

        self.update()
        return selected_colors

    def edit_drawing(self):
        # if self.multichannel_mode and self.channel is None:
        #     logger.error('please select a channel first')
        #     return

        # basic mask drawing edit
        drawn_mask = self.get_mask()
        if drawn_mask is None:
            return
        drawn_mask = np.copy(drawn_mask)
        erased, drawn = self.get_user_drawing(show_erased=True)
        # bounds = get_white_bounds([drawn, erased])
        # if bounds is None:
        # pass
        # print(drawn_mask.shape)
        # print(drawn_mask.max(), drawn_mask.min())
        # print(drawn_mask.dtype)

        drawn_mask[erased!=0]=0
        drawn_mask[drawn!=0]=drawn[drawn!=0]
        if self.multichannel_mode:
            if self.raw_mask is not None:
                if self.channel is not None:
                    # print('overwriting mask in image')
                    self.raw_mask[..., self.channel]=drawn_mask
                    # big bug here this stuff contains the bg image --> really huge bug
                    # Img(self.raw_mask, dimensions='hwc').save('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/tst2.tif')
                else:
                    # if nothing is specified assume ch 0
                    if len(self.raw_mask.shape)>2:
                        self.raw_mask[..., 0] = drawn_mask
                    else:
                        self.raw_mask = drawn_mask

        # Img(drawn_mask, dimensions='hw').save('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/tst.tif')
        self.set_mask(Img(drawn_mask, dimensions='hw'))


    # draws/removes bonds locally à la TA
    def apply_drawing(self, minimal_cell_size=0):
        # print('minimal_cell_size', minimal_cell_size)

        drawn_mask = self.get_mask()
        # print(drawn_mask)
        # no image or nothing drawn --> return
        if drawn_mask is None:
            return
        erased, drawn = self.get_user_drawing(show_erased=True)
        bounds = get_white_bounds([drawn, erased])
        if bounds is None:
            # in case the guy presses shift apply with no mask then run it over the whole image --> can make sense to have this
            if minimal_cell_size is not None and minimal_cell_size !=0:
                # dirty way of changing size --> needs be improved
                minimal_cell_size = self.minimal_cell_size


                # print('minimal_cell_size2', minimal_cell_size)
                drawn_mask = wshed(drawn_mask, seeds='mask', min_seed_area=minimal_cell_size)
                self.set_mask(drawn_mask)
            return

        if np.count_nonzero(erased) != 0:
            lab_erased = label(erased, connectivity=2, background=0)
            rps_lab_erased = regionprops(lab_erased)

            lab_drawn_mask = label(drawn_mask, connectivity=1, background=255)
            rps_lab_drawn_mask = regionprops(lab_drawn_mask)

            for erased_bond in rps_lab_erased:
                ids = np.unique(lab_drawn_mask[erased_bond.coords[:, 0], erased_bond.coords[:, 1]])
                for id in ids:
                    if id != 0:
                        cur_bounds = rps_lab_drawn_mask[id - 1].bbox
                        bounds[0] = min(bounds[0], cur_bounds[0])
                        bounds[1] = max(bounds[1], cur_bounds[2])
                        bounds[2] = min(bounds[2], cur_bounds[1])
                        bounds[3] = max(bounds[3], cur_bounds[3])

        min_y, max_y, min_x, max_x = bounds

        first_x = 128
        first_y = 128
        min_x = min_x - 128
        max_x = max_x + 128
        min_y = min_y - 128
        max_y = max_y + 128

        # TODO fix upper bond too

        if min_x < 0:
            first_x = 128 + min_x
            min_x = 0
        if min_y < 0:
            first_y = 128 + min_y
            min_y = 0
        if max_y > drawn_mask.shape[0]:
            max_y = drawn_mask.shape[0]
        if max_x > drawn_mask.shape[1]:
            max_x = drawn_mask.shape[1]

        minished = wshed(drawn_mask[min_y:max_y, min_x:max_x], seeds='mask', min_seed_area=minimal_cell_size)
        drawn_mask[min_y:max_y, min_x:max_x][first_y:min(first_y + (bounds[1] - bounds[0]) + 1, minished.shape[0]),first_x:min(first_x + (bounds[3] - bounds[2]) + 1, minished.shape[1])] = minished[first_y:min(first_y + (bounds[1] - bounds[0]) + 1, minished.shape[0]), first_x:min(first_x + (bounds[3] - bounds[2]) + 1, minished.shape[1])]
        self.set_mask(drawn_mask)

    def force_consistent_range(self, img, auto_convert_float_to_binary=None):
        if auto_convert_float_to_binary is None:
            auto_convert_float_to_binary = self.auto_convert_float_to_binary_threshold
        # for ch in range(img.shape[-1]):
        #     tmp = img[...,ch]
        #     min = tmp.min()
        #     max = tmp.max()
        #     # if min!=max:
        #         # change range of the stuff
        #         # print('error in range')
        #     if max<=1.0:
        #         print('error in range')
        #     print(min, max)
        #     print(min, max)
        # img = 
        return self.binarize(img, auto_convert_float_to_binary=auto_convert_float_to_binary, force=True)

    def save_mask(self, multichannel_save=False, forced_nb_of_channels=None):
        # print('saving to ....' + self.save_file_name)
        if self.save_file_name is not None:
            # print('really saving/...')
            mask = self.get_mask()
            if mask is not None:
                logger.debug('saving mask to '+str(self.save_file_name))
                try:
                    single_channel_image = len(self.raw_image.shape)==2
                except:
                    single_channel_image = False
                if multichannel_save == False or single_channel_image  and (forced_nb_of_channels is None or forced_nb_of_channels ==0):
                    Img(mask, dimensions='hw').save(self.save_file_name)
                else:
                    if self.raw_mask is not None:
                        # print('quick saving edited mask')
                        # Img(self.raw_mask, dimensions='hwc').save(self.save_file_name)

                        if len(self.raw_mask.shape)>2:
                            # I need check all channels have the same normalization
                            Img(self.force_consistent_range(self.raw_mask).astype(np.uint8)*255, dimensions='hwc').save(self.save_file_name)
                        else:
                            Img(self.raw_mask, dimensions='hw').save(self.save_file_name)
                        return

                    # print('test', self.channel )
                    # print('orig img channels', self.raw_image.shape[-1])
                    # if self.channel is None:
                    #     logger.error('Please select a channel first')
                    #     return
                    # BELOW IS CRAPPY CODE --> BEST is if multichannel_mode to create a black empty image
                    # if os.path.exists(self.save_file_name):
                    #     try:
                    #         mask_out = Img(self.save_file_name)
                    #         if self.channel is not None:
                    #             mask_out[...,self.channel]=mask
                    #         else:
                    #             # assume ch0
                    #             mask_out[..., 0] = mask
                    #         Img(mask_out, dimensions='hwc').save(self.save_file_name)
                    #         return
                    #     except:
                    #         traceback.print_exc()
                    #     print('le fichier existe')
                    #
                    #     # on le charge
                    # # else:
                    # print('le fichier existe pas')
                    # mask_out = np.zeros_like(mask, shape=(*mask.shape, self.raw_image.shape[-1]))
                    # if self.channel is not None:
                    #     mask_out[..., self.channel] = mask
                    # else:
                    #     # assume ch0
                    #     mask_out[..., 0] = mask
                    # Img(mask_out, dimensions='hwc').save(self.save_file_name)
                    # sinon on recup le nb de canaux de l'image finale


    # def _update_channel(self, output, mask, channel):
    #     if channel is not None:
    #         mask_out = np.zeros_like(mask, shape=(*mask.shape, self.raw_image.shape[-1]))
    #         mask_out[..., self.channel] = mask
    #         Img(mask_out, dimensions='hwc').save(self.save_file_name)





    # this is the local seeded watershed à la TA
    def manually_reseeded_wshed(self):

        if self.get_raw_image() is None:
            return

        try:
            if self.channel is None and self.get_raw_image().has_c():
                logger.error('Please select a channel fisrt')
                return
        except:
            print(self.get_raw_image().shape, self.channel)
            if self.channel is None and len(self.get_raw_image().shape)>=3 and self.get_raw_image().shape[-1]>1:
                logger.error('Please select a channel fisrt')
                return

        usr_drawing = self.get_user_drawing()
        labs = label(usr_drawing, connectivity=2, background=0)

        # pb here because the user drawing also contains the mask and I don't want it
        # plt.imshow(labs)
        # plt.show()

        rps_user_drawing = regionprops(labs)
        if len(rps_user_drawing) == 0:
            return

        drawn_mask = self.get_mask()

        # plt.imshow(drawn_mask)
        # plt.show()

        drawn_mask[labs != 0] = 0
        lab_cells_user_drawing = label(drawn_mask, connectivity=1, background=255)
        rps_cells_user_drawing = regionprops(lab_cells_user_drawing)

        min_y = 100000000
        min_x = 100000000
        max_y = 0
        max_x = 0
        for rps_user in rps_user_drawing:
            label_id = lab_cells_user_drawing[rps_user.coords[0][0], rps_user.coords[0][1]]
            bbox = rps_cells_user_drawing[label_id - 1].bbox
            min_y = min(bbox[0], min_y)
            min_x = min(bbox[1], min_x)
            max_y = max(bbox[2], max_y)
            max_x = max(bbox[3], max_x)




        # here I need get the channel in fact
        img = self.get_raw_image()[min_y:max_y + 1, min_x:max_x + 1]
        if self.channel is not None:
            img = img[..., self.channel]
        minished = wshed(img,
                         seeds=labs[min_y:max_y + 1, min_x:max_x + 1], weak_blur=1.)
        drawn_mask[min_y:max_y + 1, min_x:max_x + 1][minished != 0] = minished[minished != 0]
        # plt.imshow(drawn_mask)
        # plt.show()
        self.set_mask(drawn_mask)


    def mousePressEvent(self, event):
        if not self.hasMouseTracking() or not self.drawing_enabled:
            return
        self.clickCount = 1
        if self.vdp.active:
            self.vdp.mousePressEvent(event)
            self.update()
            return

        if event.buttons() == QtCore.Qt.LeftButton or event.buttons() == QtCore.Qt.RightButton:
            self.drawing = True
            zoom_corrected_pos = event.pos() / self.scale
            self.lastPoint = zoom_corrected_pos
            self.drawOnImage(event)
        else:
            self.drawing = False


    # can I pass this event to something else to synchronize the mouse moves
    def mouseMoveEvent(self, event):
        if not self.hasMouseTracking() or not self.drawing_enabled:
            return
        else:
            if self.propagate_mouse_move == True:
                self.propagate_mouse_event(event)

        # print('in mouse move', self.hasMouseTracking(), self.drawing, self.vdp.active)
        if self.statusBar:
            zoom_corrected_pos = event.pos() / self.scale
            self.statusBar.showMessage('x=' + str(zoom_corrected_pos.x()) + ' y=' + str(
                zoom_corrected_pos.y()))
        if self.vdp.active:
            self.vdp.mouseMoveEvent(event)
            region = self.scrollArea.widget().visibleRegion()
            self.update(region)
            return
        self.drawOnImage(event)

    def mouseReleaseEvent(self, event):
        self.drawing = False
        if not self.hasMouseTracking() or not self.drawing_enabled:
            return
        if self.vdp.active:
            self.vdp.mouseReleaseEvent(event)
            self.update()  # required to update drawing
            return
        # if event.button == QtCore.Qt.LeftButton:
        #     self.drawing = False
        if self.clickCount == 1:
            QTimer.singleShot(QApplication.instance().doubleClickInterval(),
                              self.updateButtonCount)

    def propagate_mouse_event(self, event):
        print(self,event)
        pass

    def drawOnImage(self, event):
        # bug fix for image drawing when image is none
        if self.imageDraw is None:
            return
        zoom_corrected_pos = event.pos() / self.scale
        if self.drawing and (event.buttons() == QtCore.Qt.LeftButton or event.buttons() == QtCore.Qt.RightButton):
            # now drawing or erasing over the image
            self._draw_on_image(self.imageDraw, event, zoom_corrected_pos)
            # this contains only the user drawing and nothing else --> can be useful to have too!!!
            self._draw_on_image(self.raw_user_drawing, event, zoom_corrected_pos, erase_color=self.eraseColor_visualizer)



        # Drawing the cursor TODO add boolean to ask if drawing cursor should be shown
        painter = QtGui.QPainter(self.cursor)
        try:
            # We erase previous pointer
            r = QtCore.QRect(QtCore.QPoint(), self._clear_size * QtCore.QSize() * self.brushSize)
            painter.save()
            r.moveCenter(self.lastPoint)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            painter.eraseRect(r)
            painter.restore()
            # draw the new one

            stroke_size = 2
            if self.brushSize<6:
                stroke_size = 1

            # if is 1 --> I should really draw a point instead
            # I should also hide the cursor --> that would help
            painter.setPen(QtGui.QPen(self.cursorColor, stroke_size, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                      QtCore.Qt.RoundJoin))
            if self.brushSize >1:
                painter.drawEllipse(zoom_corrected_pos, self.brushSize / 2.,
                                self.brushSize / 2.)
            else:
                painter.drawPoint(zoom_corrected_pos)
        except:
            traceback.print_exc()
        finally:
            painter.end()
        try:
            region = self.scrollArea.widget().visibleRegion()
        except:
            # assume no scroll area --> use self visible region
            region =self.visibleRegion()

        self.update(region)

        # required to erase mouse pointer
        self.lastPoint = zoom_corrected_pos


    def _draw_on_image(self, image_to_be_drawn, event, zoom_corrected_pos, erase_color=None):
        painter = QtGui.QPainter(image_to_be_drawn)
        try:
            # small hack to allow use pen or fingers on touch screens to erase when Ctrl or shift is pressed along with the click --> MAYBE PUT THIS AS AN OPTION SOME DAY BECAUSE THIS MAY PERTURB PEOPLE
            if event.buttons() == QtCore.Qt.LeftButton and not (event.modifiers() == QtCore.Qt.ControlModifier or event.modifiers() ==QtCore.Qt.ShiftModifier):
                painter.setPen(QtGui.QPen(self.drawColor, self.brushSize, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                          QtCore.Qt.RoundJoin))
            else:
                if erase_color is None:
                    erase_color = self.eraseColor
                painter.setPen(QtGui.QPen(erase_color, self.brushSize, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                          QtCore.Qt.RoundJoin))
            if self.lastPoint != zoom_corrected_pos:
                painter.drawLine(self.lastPoint, zoom_corrected_pos)
            else:
                # if zero length line then draw point instead
                painter.drawPoint(zoom_corrected_pos)
        except:
            traceback.print_exc()
        finally:
            painter.end()

    # adds context/right click menu but only in vectorial mode
    def contextMenuEvent(self, event):
        if not self.vdp.active:
            return
        cmenu = QMenu(self)
        newAct = cmenu.addAction("New")
        opnAct = cmenu.addAction("Open")
        quitAct = cmenu.addAction("Quit")
        action = cmenu.exec_(self.mapToGlobal(event.pos()))
        if action == quitAct:
            sys.exit(0)

    def updateButtonCount(self):
        self.clickCount = 1

    def mouseDoubleClickEvent(self, event):
        self.clickCount = 2

        # TODO maybe add controls to be sure we are in drawing mode with a pen ???
        # double click fill or delete
        if not self.vdp.active:
            zoom_corrected_pos = event.pos() / self.scale
            if event.buttons() == QtCore.Qt.LeftButton:
                # double click fill
                if self.dialog_fill_erase('fill'):
                    self.double_click_fill(zoom_corrected_pos)
            elif event.buttons() == QtCore.Qt.RightButton:
                # double click delete shape
                if self.dialog_fill_erase('erase'):
                    self.double_click_erase(zoom_corrected_pos)

        self.vdp.mouseDoubleClickEvent(event)

    def dialog_fill_erase(self, action='fill'):
        choice = QMessageBox.question(self, 'Double click detected!',
                                      'Do you want to '+ action +' the shape?',
                                      QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            return True
        else:
            return False


    def double_click_fill(self, zoom_corrected_pos):
        try:
            # erased, drawn = self.get_user_drawing(show_erased=True)
            usr_drawing = self.get_user_drawing()
            labs = label(usr_drawing, connectivity=2, background=0)
            # do I really need that can I rather do np.unique or alike and will it be faster
            rps_user_drawing = regionprops(labs)
            if len(rps_user_drawing) == 0:
                return
            drawn_mask = self.get_mask()
            # cell clicked over
            cell_id_clicked_over = labs[int(zoom_corrected_pos.y()), int(zoom_corrected_pos.x())]
            if cell_id_clicked_over == 0:
                return
            # we remove the cell drawn over
            drawn_mask[labs==cell_id_clicked_over] = 0
            # maintenant j'inverse et je trouve la cellule à filler sur le negatif
            lab_cells_user_drawing = label(drawn_mask, connectivity=1, background=255)
            cell_id_clicked_over2 = lab_cells_user_drawing[int(zoom_corrected_pos.y()), int(zoom_corrected_pos.x())]
            drawn_mask[lab_cells_user_drawing==cell_id_clicked_over2]=255
            self.set_mask(drawn_mask)
        except:
            logger.error('Double click fill failed')
            traceback.print_exc()


    def double_click_erase(self, zoom_corrected_pos):
        try:
            # erased, drawn = self.get_user_drawing(show_erased=True)
            erased, drawn = self.get_user_drawing(show_erased=True)
            labs = label(erased, connectivity=2, background=0)

            # do I really need that can I rather do np.unique or alike and will it be faster
            rps_user_drawing = regionprops(labs)
            if len(rps_user_drawing) == 0:
                return
            drawn_mask = self.get_mask()
            # cell clicked over
            cell_id_clicked_over = labs[int(zoom_corrected_pos.y()), int(zoom_corrected_pos.x())]
            if cell_id_clicked_over==0:
                return
            # we refill the removed cell
            drawn_mask[labs==cell_id_clicked_over] = 255
            # maintenant j'inverse et je trouve la cellule à filler sur le negatif
            lab_cells_user_drawing = label(drawn_mask, connectivity=1, background=0)
            cell_id_clicked_over2 = lab_cells_user_drawing[int(zoom_corrected_pos.y()), int(zoom_corrected_pos.x())]
            drawn_mask[lab_cells_user_drawing==cell_id_clicked_over2]=0


            self.set_mask(drawn_mask)
        except:
            logger.error('Double click delete failed')
            traceback.print_exc()


    def paintEvent(self, event):
        # super(Createpaintwidget, self).paintEvent(event) # KEEP somehow tjis causes bugs --> do not uncomment it
        canvasPainter = QtGui.QPainter(self)
        try:

            # the scrollpane visible region
            try:
                visibleRegion = self.scrollArea.widget().visibleRegion()
            except:
                #assume no scroll region --> visible region is self visible region
                visibleRegion = self.visibleRegion()
            # the corresponding rect
            visibleRect = visibleRegion.boundingRect()
            # the visibleRect taking zoom into account
            scaledVisibleRect = QRect(visibleRect.x() / self.scale, visibleRect.y() / self.scale,
                                      visibleRect.width() / self.scale, visibleRect.height() / self.scale)

            if self.image is None:
                canvasPainter.eraseRect(visibleRect)
                # canvasPainter.end()
                return

            canvasPainter.drawImage(visibleRect, self.image, scaledVisibleRect)
            if not self.vdp.active and self.maskVisible and self.imageDraw is not None:
                canvasPainter.drawImage(visibleRect, self.imageDraw, scaledVisibleRect)
                # should draw the cursor
            canvasPainter.drawImage(visibleRect, self.cursor, scaledVisibleRect)

            # ça ajoute un texte --> pas mal mais le refresh on scroll est pas top --> dois je le dessiner plutot dans vdp --> another set of objects that are non interactible and also cannot be saved and local???

            # if True:
            #     print('adding text')
            #     myQPen = QPen(QtCore.Qt.blue, 3)
            #     canvasPainter.setPen(myQPen)
            #     canvasPainter.setPen(QColor("Green"))
            #     canvasPainter.setFont(QFont('Helvetica', 48))
            #     #     print('drawing text')
            #     canvasPainter.drawText(visibleRect.x(), visibleRect.y() +10, 'this is a test of your system')
            #     # pb here is that the images is not cleared --> creates bugs

            if self.vdp.active:
                self.vdp.paintEvent(canvasPainter, scaledVisibleRect)
        except:
            traceback.print_exc()
        finally:
            canvasPainter.end()

if __name__ == '__main__':
    # TODO add a main method so it can be called directly
    # maybe just show a canvas and give it interesting props --> TODO --> really need fix that too!!!
    import sys
    from qtpy.QtWidgets import QApplication
    from matplotlib import pyplot as plt

    # should probably have his own scroll bar embedded somewhere

    app = QApplication(sys.argv)


    # in fact that is easy to override methods before construction --> can easily do that in fact

    # quite easy todo in fact
    class overriding_apply(Createpaintwidget):
        def apply(self):
            plt.imshow(self.get_mask())
            plt.show()

    #Do not use show() in a GUI application.
    # If you have a FigureCanvas instance embedded in your app, call its
    # draw() method.
    # If you use pyplot.figure() to create a matplotlib window from your
    # app, call pyplot.draw().

    # maybe add a figure canvas the to my code and plot to it and draw it --> quite easy and nice plotting capabilities then in python



    # w = Createpaintwidget(enable_shortcuts=True)
    w = overriding_apply(enable_shortcuts=True)
    w.set_image('/E/Sample_images/sample_images_PA/trash_test_mem/mini (copie)/focused_Series012.png')
    # w.set_image('/E/Sample_images/fluorescent_wings_spots_charroux/909dsRed/0.tif')
    # w.set_image('/E/Sample_images/fluorescent_wings_spots_charroux/contoles_test_X1VK06/0.tif')
    # w.set_mask('/E/Sample_images/sample_images_PA/trash_test_mem/mini (copie)/focused_Series012/handCorrection.png')
    # w.set_image(Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini (copie)/focused_Series012.png'))

    # print(tst.max())
    # tst = tst[..., np.newaxis]




    # tst = Img('/E/Sample_images/fluorescent_wings_spots_charroux/909dsRed/0.tif')
    # tst = Img('/E/Sample_images/fluorescent_wings_spots_charroux/contoles_test_X1VK06/0.tif')
    # tst = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini (copie)/focused_Series012.png')
    # tst = normalization(tst, method='Rescaling (min-max normalization)', range=[0,1], individual_channels=len(tst.shape)>2, clip=True)
    # tst = normalization(tst, method='Rescaling (min-max normalization)', range=[0,1], individual_channels=len(tst.shape)>2, clip=True)
    # tst = normalization(tst, method='Standardization', range=[0,1], individual_channels=True, clip=True)
    # tst = normalization(tst, method='Percentile', range=[0,1], individual_channels=len(tst.shape)>2, clip=True, normalization_minima_and_maxima=[1.,50.0])
    # tst = np.squeeze(tst)

    # tst = auto_threshold(tst)
    # save_as_tiff(tst, output_name='/E/Sample_images/sample_images_PA/trash_test_mem/mini (copie)/tst_norm.tif')

    # do I have a standar
    # tst = normalization(tst, method='standardization', range=[0,1], individual_channels=True)
    # print(tst.max())
    # w.set_image(tst)

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

