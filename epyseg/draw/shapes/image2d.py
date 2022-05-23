# I guess there is a big bug in the scales of the components --> hack this

# faire un draw_at_scale and merge draw fill etc by just putting an option --> much simpler and that way you can keep original size which is the best idea
# set to width doit juste calculer le scaling factor in fact.... --> TODO do that and same for all objects --> no cloning and all will be simpler

# check all the things that need be packed top left/top right/bottom left/bottom right
# letter should always be packed first and top left
# make insets, etc more flexible
# should I remove the letter of insets ???
# the rest can be any position and any nb of instances of text/scale bar or insets
# see how to edit that easily
# TODO
# objects can be scale bars and or insets and or text labels as many as needed --> pack them in x or y and align them

# https://docs.python.org/2/library/operator.html
# maths fig in inkscape --> cool using latex https://castel.dev/post/lecture-notes-2/

# TODO may also contain svg or graphs or ???
# TODO handle extra labels for images directly and also for rows or cols --> think how to do that but must be doable

# TODO Add the crops --> see how though and see how to warn when width or height reaches 0...
# should I offer max projections too ??? maybe or not --> see how to do that...

# the only advantage I see for rect2D vs qrectf is that they allow for rotation ????
# see how to best handle that ???

import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtSvg import QSvgGenerator
import os
from epyseg.draw.shapes.line2d import Line2D
from epyseg.draw.shapes.point2d import Point2D
from epyseg.figure.alignment import alignRight, alignLeft, alignTop, alignBottom, alignCenterH, alignCenterV, packY, \
    packX, packYreverse
from epyseg.draw.shapes.rect2d import Rect2D
from epyseg.draw.shapes.scalebar import ScaleBar
from epyseg.draw.shapes.txt2d import TAText2D
from epyseg.figure.fig_tools import preview
# from epyseg.figure import fig_tools
from epyseg.img import Img, toQimage
from PyQt5.QtCore import QRectF, QPointF, QSize, QRect
# from sympy import nsolve, exp, Symbol
# logger
from epyseg.tools.logger import TA_logger

logger = TA_logger()

class Image2D(Rect2D):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    CENTERED = 4

    def __init__(self, *args, x=None, y=None, width=None, height=None, data=None, dimensions=None, opacity=1., **kwargs):
        self.isSet = False
        self.scale = 1
        self.translation = QPointF()

        # crops
        self.__crop_left = 0
        self.__crop_right = 0
        self.__crop_top = 0
        self.__crop_bottom = 0
        self.img = None
        self.annotation = []  # should contain the objects for annotating imaging --> shapes and texts
        self.letter = None  # when objects are swapped need change the letter
        self.top_left_objects = []
        self.top_right_objects = []
        self.bottom_right_objects = []
        self.bottom_left_objects = []
        self.centered_objects = []

        # if the image is inserted as an inset then draw it as a fraction of parent width
        # inset parameters
        self.fraction_of_parent_image_width_if_image_is_inset = 0.25
        self.border_size = None  # no border by default
        self.border_color = 0xFFFFFF  # white border by default

        if args:
            if len(args) == 1:
                if isinstance(args[0], str):
                    self.filename = args[0]
                elif isinstance(args[0], Img):
                    self.filename = None
                    self.img = args[0]
                    self.qimage =  toQimage(self.img)
                    if x is None:
                        x = 0
                    if y is None:
                        y = 0
                    try:
                        super(Image2D, self).__init__(x, y, self.img.get_width(), self.img.get_height())
                    except:
                        super(Image2D, self).__init__(x, y, self.img.shape[1], self.img.shape[0])
                    self.isSet = True
        else:
            self.filename = None

        if x is None and y is None and width is not None and height is not None:
            super(Image2D, self).__init__(0, 0, width, height)
            self.isSet = True
        elif x is None and y is None and width is None and height is None and self.filename is not None:
            # print('in 0')
            try:
                self.img = Img(self.filename)
            except:
                logger.error('could not load image '+str(self.filename))
                return
            self.qimage = toQimage(self.img)
            width = self.img.get_width()
            height = self.img.get_height()
            super(Image2D, self).__init__(0, 0, width, height)
            self.isSet = True
        elif x is not None and y is not None and width is not None and height is not None and self.img is None:
            self.img = None
            super(Image2D, self).__init__(x, y, width, height)
            self.isSet = True
        elif data is None:
            if self.filename is not None:
                self.img = Img(self.filename)
                self.qimage =  toQimage(self.img)
                if x is None:
                    x = 0
                if y is None:
                    y = 0
                super(Image2D, self).__init__(x, y, self.img.get_width(), self.img.get_height())
                self.isSet = True
        elif data is not None:
            self.img = Img(data,
                           dimensions=dimensions)  # need width and height so cannot really be only a numpy stuff --> cause no width or height by default --> or need tags such as image type for dimensions
            self.qimage =  toQimage(self.img)
            # need Image dimensions id data is not of type IMG --> could check that
            if x is None:
                x = 0
            if y is None:
                y = 0
            super(Image2D, self).__init__(x, y, self.img.get_width(), self.img.get_height())
            self.isSet = True
        self.opacity = opacity

    # @return the block incompressible width
    def getIncompressibleWidth(self):
        extra_space = 0  # can add some if boxes around to add text
        return extra_space

    # @return the block incompressible height
    def getIncompressibleHeight(self):
        extra_space = 0  # can add some if boxes around to add text
        # do not even do that --> just ignore things
        # add box around ???
        return extra_space

    def add_object(self, object, position):
        if isinstance(object, list):
            for obj in object:
                self.add_object(obj, position=position)
            return
        if position == Image2D.TOP_LEFT:
            self.top_left_objects.append(object)
        elif position == Image2D.BOTTOM_RIGHT:
            self.bottom_right_objects.append(object)
        elif position == Image2D.BOTTOM_LEFT:
            self.bottom_left_objects.append(object)
        elif position == Image2D.CENTERED:
            self.centered_objects.append(object)
        else:
            self.top_right_objects.append(object)

    # TODO --> check if contains it
    def remove_object(self, object, position):
        if position == Image2D.TOP_LEFT:
            self.top_left_objects.remove(object)
        elif position == Image2D.BOTTOM_RIGHT:
            self.bottom_right_objects.remove(object)
        elif position == Image2D.BOTTOM_LEFT:
            self.bottom_left_objects.remove(object)
        elif position == Image2D.CENTERED:
            self.centered_objects.remove(object)
        else:
            self.top_right_objects.remove(object)

    def remove_all_objects(self, position):
        if position == Image2D.TOP_LEFT:
            del self.top_left_objects
            self.top_left_objects = []
        elif position == Image2D.BOTTOM_RIGHT:
            del self.bottom_right_objects
            self.bottom_right_objects = []
        elif position == Image2D.BOTTOM_LEFT:
            del self.bottom_left_objects
            self.bottom_left_objects = []
        elif position == Image2D.CENTERED:
            del self.centered_objects
            self.centered_objects = []
        else:
            del self.top_right_objects
            self.top_right_objects = []

    def setLettering(self, letter):
        if isinstance(letter, TAText2D):
            self.letter = letter
        elif isinstance(letter, str):
            if letter.strip() == '':
                self.letter = None
            else:
                self.letter = TAText2D(letter)

    # def getRect2D(self):
    #     # self.__class__ = Rect2D
    #     # return super()
    #     # TODO ideally I'd like to get the Rect2D parent but I should think what the best way is to get it...
    #     return self

    def draw(self, painter, draw=True):
        if draw:
            painter.save()
            painter.setOpacity(self.opacity)
            # painter.setClipRect(self)  # only draw in self --> very useful for inset borders # pb clip rect does not work for svg --> remove for now users can add it manually if desired or I can add it if people really want it and then I should draw relevant lines or shifted rects --> do that later
        # prevents drawing outside from the image
            rect_to_plot = self.boundingRect(scaled=True) #scaled=True #self.adjusted(self.__crop_left, self.__crop_top, self.__crop_right, self.__crop_bottom) # need remove the crops with that
            # self.scale = 1
            # if self.scale is not None and self.scale != 1:
            # #     # TODO KEEP THE ORDER THIS MUST BE DONE THIS WAY OR IT WILL GENERATE PLENTY OF BUGS...
            #     new_width = rect_to_plot.width() * self.scale
            #     new_height = rect_to_plot.height() * self.scale
            # #     # print(rect_to_plot.width(), rect_to_plot.height())  # here ok
            # #     # setX changes width --> why is that
            # #
            # #     # TODO BE EXTREMELY CAREFUL AS SETX AND SETY CAN CHANGE WIDTH AND HEIGHT --> ALWAYS TAKE SIZE BEFORE OTHERWISE THERE WILL BE A PB AND ALWAYS RESET THE SIZE WHEN SETX IS CALLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # #     # Sets the left edge of the rectangle to the given x coordinate. May change the width, but will never change the right edge of the rectangle. --> NO CLUE WHY SHOULD CHANGE WIDTH THOUGH BUT BE CAREFUL!!!
            # #     rect_to_plot.setX(rect_to_plot.x() * self.scale)
            # #     rect_to_plot.setY(rect_to_plot.y() * self.scale)
            # #     # maybe to avoid bugs I should use translate instead rather that set x but ok anyways
            # #     # print(rect_to_plot.width(), rect_to_plot.height())# bug here --> too big
            # #
            # #     # print(new_height, new_height, self.width(), self.scale, self.scale* self.width())
            #     rect_to_plot.setWidth(new_width)
            #     rect_to_plot.setHeight(new_height)

            if self.img is not None:
                x = 0
                y = 0
                try:
                    w = self.img.get_width()
                    h = self.img.get_height()
                except:
                    w = self.img.shape[1]
                    h = self.img.shape[0]

                if self.__crop_top is not None:
                    y = self.__crop_top
                    h -= self.__crop_top
                if self.__crop_left is not None:
                    x = self.__crop_left
                    w -= self.__crop_left
                if self.__crop_right is not None:
                    w -= self.__crop_right
                if self.__crop_bottom is not None:
                    h -= self.__crop_bottom
                # pb here --> see how to really crop
                qsource = QRectF(x, y, w, h)
                painter.drawImage(rect_to_plot, self.qimage, qsource)  # , flags=QtCore.Qt.AutoColor
            else:
                painter.drawRect(rect_to_plot)



            # letter is good
            extra_space = 3

            # draw annotations first
            if self.annotation is not None and self.annotation:
                # need clone the object then set its P1 with respect to position or need a trick to keep original ref and have an updated one just for display but then need renew it all the time --> see how I can do that...
                # maybe clone is not smart as it duplicates resources without a need for it
                # but then need clone the original rect and draw with respect to that
                # and I indeed need scale the shape --> TODO too
                # indeed thanks to cloning I always preserve original info --> not bad

                # annot position is good
                # TODO see how to do that cause not so easy --> think carefully and take inspiration from EZF and improve it
                for annot in self.annotation:
                    # always empty --> why is that
                    # print('init',annot.get_P1())
                    # always assume everything is done at 0,0 then do translation
                    # annot.set_P1(self.get_P1().x() + annot.get_P1().x(),                    self.get_P1().y() + annot.get_P1().y())  # always relative to the parent image
                    # annot.set_P1(self.get_P1())  # always relative to the parent image
                    # print(annot.get_P1())

                    # print('init', self.get_P1(), 'scale', self.get_scale())
                    annot.set_to_translation(rect_to_plot.topLeft())

                    annot.set_to_scale(self.scale)  # will fuck the stuff but ok for a test
                    # print('scaled',annot.get_P1())
                    annot.draw(painter=painter)
                    # print('tranbs', annot.translation)




            # and indeed I need also to take crop into account in order not to misposition things...

            if self.letter is not None:
                self.letter.set_P1(rect_to_plot.topLeft().x() + extra_space, rect_to_plot.topLeft().y() + extra_space)

            # then draw text and insets --> on top of annotations
            # TODO need align insets differently than others and need align its bounding box also differently --> TODO but almost there
            if len(self.top_right_objects) != 0 or len(self.top_left_objects) != 0 or len(
                    self.bottom_left_objects) != 0 or len(self.bottom_right_objects) != 0 or len(
                    self.centered_objects) != 0:
                # align a scale bar to various positions
                # maybe if there is a letter first point should be place below stuff
                # top_left = Point2D(self.get_P1())
                top_left_shifted = Point2D(rect_to_plot.topLeft())
                # top_left_shifted.setX(top_left_shifted.x() )# + extra_space
                # top_left_shifted.setY(top_left_shifted.y() )#+ extra_space

                # print('before', top_left)
                # if self.letter is not None:
                #     packY(extra_space, self.letter, top_left_shifted)
                # print('after', top_left)

                # insets should be aligned to unshifted values
                # whereas texts should be aligned to shifted ones
                # what if I try all unshifted
                # cause in a way it's simpler

                # top_right = Point2D(self.get_P1())
                top_right_shifted = Point2D(rect_to_plot.topLeft())
                top_right_shifted.setX(top_right_shifted.x() + rect_to_plot.width())#- extra_space
                top_right_shifted.setY(top_right_shifted.y() )#+ extra_space

                # bottom_left = Point2D(self.get_P1())
                bottom_left_shifted = Point2D(rect_to_plot.topLeft())
                bottom_left_shifted.setX(bottom_left_shifted.x() ) #+ extra_space
                bottom_left_shifted.setY(
                    bottom_left_shifted.y() + rect_to_plot.height() )#- extra_space  # should align right then pack on top of that --> may need a direction in packing--> TODO

                bottom_right = Point2D(rect_to_plot.topLeft())
                bottom_right_shifted = Point2D(rect_to_plot.topLeft())
                bottom_right_shifted.setX(bottom_right_shifted.x() + rect_to_plot.width())# - extra_space
                bottom_right_shifted.setY(bottom_right_shifted.y() + rect_to_plot.height())#- extra_space

                center = Point2D(rect_to_plot.topLeft())
                center.setX(center.x() + rect_to_plot.width() / 2)
                center.setY(center.y() + rect_to_plot.height() / 2)

                if len(self.top_left_objects) != 0:
                    # change inset size first
                    for obj in self.top_left_objects:
                        if isinstance(obj, Image2D):
                            obj.setToWidth(rect_to_plot.width() * obj.fraction_of_parent_image_width_if_image_is_inset)

                    # if letter exists align with respect to it

                    alignTop(top_left_shifted, *self.top_left_objects)
                    alignLeft(top_left_shifted, *self.top_left_objects)

                    if self.letter is not None:
                        # packY(extra_space, self.letter, top_left_shifted)
                        top_left_shifted = self.letter

                    # in fact images really need be aligned left of the image but the others need be aligned with the letter that has an extra space --> TODO --> change some day

                    packY(extra_space, top_left_shifted, *self.top_left_objects)

                    # all images need be shifted back??? to be aligned left

                    for obj in self.top_left_objects:
                        # for drawing of inset borders
                        # if isinstance(obj, Image2D):
                        #     # make it draw a border and align it
                        #     # painter.save()
                        #     img_bounds = Rect2D(obj)
                        #     img_bounds.stroke = 3
                        #     # img_bounds.translate(-img_bounds.stroke / 2, -img_bounds.stroke / 2)
                        #     img_bounds.color = 0xFFFF00
                        #     img_bounds.fill_color = 0xFFFF00
                        #     img_bounds.draw(painter=painter)
                        #     # print(img_bounds)
                        #     # painter.restore()
                        obj.draw(painter=painter)

                if len(self.top_right_objects) != 0:
                    # change inset size first
                    for obj in self.top_right_objects:
                        if isinstance(obj, Image2D):
                            obj.setToWidth(rect_to_plot.width() * obj.fraction_of_parent_image_width_if_image_is_inset)
                    alignRight(top_right_shifted, *self.top_right_objects)
                    alignTop(top_right_shifted, *self.top_right_objects)
                    packY(extra_space, top_right_shifted, *self.top_right_objects)
                    for obj in self.top_right_objects:
                        # # for drawing of inset borders
                        # if isinstance(obj, Image2D):
                        #     # make it draw a border and align it
                        #     # painter.save()
                        #     img_bounds = Rect2D(obj)
                        #     img_bounds.stroke = 3
                        #     # img_bounds.translate(img_bounds.stroke / 2, -img_bounds.stroke / 2)
                        #     img_bounds.color = 0xFFFF00
                        #     img_bounds.fill_color = 0xFFFF00
                        #     img_bounds.draw(painter=painter)
                        #     # print(img_bounds)
                        #     # painter.restore()
                        obj.draw(painter=painter)

                if len(self.bottom_right_objects) != 0:
                    # change inset size first
                    for obj in self.bottom_right_objects:
                        if isinstance(obj, Image2D):
                            obj.setToWidth(rect_to_plot.width() * obj.fraction_of_parent_image_width_if_image_is_inset)
                    alignRight(bottom_right_shifted, *self.bottom_right_objects)
                    alignBottom(bottom_right_shifted, *self.bottom_right_objects)
                    packYreverse(extra_space, bottom_right_shifted, *self.bottom_right_objects)
                    # packY(3, top_right, *self.top_right_objects) # I do need to invert packing order
                    for obj in self.bottom_right_objects:
                        # # for drawing of inset borders
                        # if isinstance(obj, Image2D):
                        #     # make it draw a border and align it
                        #     # painter.save()
                        #     img_bounds = Rect2D(obj)
                        #     img_bounds.stroke = 3
                        #     # img_bounds.translate(-img_bounds.stroke / 2, img_bounds.stroke / 2)
                        #     # should I clip it to the image size --> maybe it's the best
                        #     img_bounds.color = 0xFFFF00
                        #     img_bounds.fill_color = 0xFFFF00
                        #     img_bounds.draw(painter=painter)
                        #     # print(img_bounds)
                        #     # painter.restore()
                        obj.draw(painter=painter)

                if len(self.bottom_left_objects) != 0:
                    # change inset size first
                    for obj in self.bottom_left_objects:
                        if isinstance(obj, Image2D):
                            obj.setToWidth(rect_to_plot.width() * obj.fraction_of_parent_image_width_if_image_is_inset)
                    alignLeft(bottom_left_shifted, *self.bottom_left_objects)
                    alignBottom(bottom_left_shifted, *self.bottom_left_objects)
                    packYreverse(extra_space, bottom_left_shifted, *self.bottom_left_objects)
                    for obj in self.bottom_left_objects:
                        # # for drawing of inset borders
                        # if isinstance(obj, Image2D):
                        #     # make it draw a border and align it
                        #     # painter.save()
                        #     img_bounds = Rect2D(obj)
                        #     img_bounds.stroke = 3
                        #     # img_bounds.translate(-img_bounds.stroke/2, img_bounds.stroke/2)
                        #     img_bounds.color = 0xFFFF00
                        #     img_bounds.fill_color = 0xFFFF00
                        #     img_bounds.draw(painter=painter)
                        #     # print(img_bounds)
                        #     # painter.restore()
                        obj.draw(painter=painter)

                if len(self.centered_objects) != 0:
                    # change inset size first
                    for obj in self.centered_objects:
                        if isinstance(obj, Image2D):
                            obj.setToWidth(rect_to_plot.width() * obj.fraction_of_parent_image_width_if_image_is_inset)
                    alignCenterH(center, *self.centered_objects)
                    alignCenterV(center, *self.centered_objects)
                    for obj in self.centered_objects:
                        # # for drawing of inset borders
                        # if isinstance(obj, Image2D):
                        #     # make it draw a border and align it
                        #     # painter.save()
                        #     img_bounds = Rect2D(obj)
                        #     img_bounds.stroke = 3
                        #     img_bounds.color = 0xFFFF00
                        #     img_bounds.fill_color = 0xFFFF00
                        #     img_bounds.draw(painter=painter)
                        #     # print(img_bounds)
                        #     # painter.restore()
                        obj.draw(painter=painter)

            # then need to draw the letter at last so that it is always on top
            if self.letter is not None:
                self.letter.draw(painter)

            painter.restore()
                # # TOP left 2
                # scale_bar = ScaleBar(30, '<font color="#FF00FF">10µm</font>')
                # scale_bar.set_scale(self.get_scale())
                # # scale_bar.set_P1(self.get_P1().x()+extra_space, self.get_P1().y()+extra_space)
                # scale_bar.set_P1(self.get_P1())
                # alignLeft(top_left, scale_bar)
                # alignTop(top_left, scale_bar)
                # # scale_bar.set_P1(scale_bar.get_P1().x()-extra_space, scale_bar.get_P1().x()+extra_space)
                # scale_bar.drawAndFill(painter=painter)

                # TOP right 2
                # scale_bar = ScaleBar(30, '<font color="#FF00FF">10µm</font>')
                # scale_bar.set_scale(self.get_scale())
                # # scale_bar.set_P1(self.get_P1().x()+extra_space, self.get_P1().y()+extra_space)
                # scale_bar.set_P1(self.get_P1())
                # alignRight(top_right, scale_bar)
                # alignTop(top_right, scale_bar)
                # # scale_bar.set_P1(scale_bar.get_P1().x()-extra_space, scale_bar.get_P1().x()+extra_space)
                # scale_bar.drawAndFill(painter=painter)

                # bottom left 2

                # # big bug in scale --> the size of the stuff isn't respected
                # # 288 is the size of image 0
                # scale_bar = ScaleBar(288, '<font color="#FF00FF">10µm</font>')
                # scale_bar.set_scale(self.get_scale())
                # # scale_bar.set_P1(self.get_P1().x()+extra_space, self.get_P1().y()+extra_space)
                # scale_bar.set_P1(self.get_P1())
                # alignLeft(bottom_left, scale_bar)
                # alignBottom(bottom_left, scale_bar)
                # # scale_bar.set_P1(scale_bar.get_P1().x()-extra_space, scale_bar.get_P1().x()+extra_space)
                # scale_bar.drawAndFill(painter=painter)

                # # bottom right 2
                # scale_bar = ScaleBar(30, '<font color="#FF00FF">10µm</font>')
                # scale_bar.set_scale(self.get_scale())
                # # scale_bar.set_P1(self.get_P1().x()+extra_space, self.get_P1().y()+extra_space)
                # scale_bar.set_P1(self.get_P1())
                # alignRight(bottom_right, scale_bar)
                # alignBottom(bottom_right, scale_bar)
                # # scale_bar.set_P1(scale_bar.get_P1().x()-extra_space, scale_bar.get_P1().x()+extra_space)
                # scale_bar.drawAndFill(painter=painter)

                # add a bunch of inner objects that should be packed left if they exist, and some right and some top, etc
                # so that these objects are packed
                # maybe loop over them
                # could have as many text labels as desired
                # only one letter
                # as many insets as needed

                # # center 2
                # scale_bar = ScaleBar(411/2, '<font color="#FFFFFF">10µm</font>')
                # scale_bar.set_scale(self.get_scale())
                # # scale_bar.set_P1(self.get_P1().x()+extra_space, self.get_P1().y()+extra_space)
                # scale_bar.set_P1(self.get_P1())
                # alignCenterH(center, scale_bar)
                # alignCenterV(center, scale_bar)
                # # scale_bar.set_P1(scale_bar.get_P1().x()-extra_space, scale_bar.get_P1().x()+extra_space)
                # scale_bar.drawAndFill(painter=painter)

                # TODO create 5 reference points for each object and align to those
                # loop over all extra objects that need be added

    # def fill(self, painter, draw=True):
        # if self.fill_color is None:
        #     return
        # if draw:
        #     painter.save()
        # painter.setOpacity(self.opacity)
        # if draw:
        #     if self.img is not None:
        #         qsource = QRectF(0, 0, self.img.get_width(), self.img.get_height())
        #         painter.drawImage(self, self.qimage , qsource)
        #     else:
        #         painter.drawRect(self)
        #     painter.restore()
        # self.draw(painter=painter, draw=draw)

    # def drawAndFill(self, painter):
        # painter.save()
        # if self.img is not None:
        #     qsource = QRectF(0, 0, self.img.get_width(), self.img.get_height())
        #     painter.drawImage(self, self.qimage , qsource)
        # else:
        #     painter.drawRect(self)
        # painter.restore()
        # self.draw(painter=painter)

    # def __add__(self, other):
    def __or__(self, other):
        from epyseg.figure.row import Row  # KEEP Really required to avoid circular imports
        return Row(self, other)

    # create a Fig with divide
    # def __truediv__(self, other):
    def __truediv__(self, other):
        from epyseg.figure.column import Column  # KEEP Really required to avoid circular imports
        return Column(self, other)

    def __floordiv__(self, other):
        return self.__truediv__(other=other)

    # ai je vraiment besoin de ça ? en fait le ratio suffit et faut aussi que j'intègre le crop sinon va y avoir des erreurs
    # Force the montage width to equal 'width_in_px'
    def setToWidth(self, width_in_px):
        # pure_image_width = self.width()
        # ratio = width_in_px / pure_image_width
        # self.setWidth(width_in_px)
        # self.setHeight(self.height() * ratio)
        # # we recompute the scale for the scale bar # TODO BE CAREFUL IF EXTRAS ARE ADDED TO THE OBJECT AS THIS WOULD PERTURB THE COMPUTATIONS
        # self.update_scale()
        pure_image_width = self.width(scaled=False)# need original height and with in fact
        # if self.__crop_left is not None:
        #     pure_image_width -= self.__crop_left
        # if self.__crop_right is not None:
        #     pure_image_width -= self.__crop_right
        scale = width_in_px / pure_image_width
        self.scale = scale


    def setToHeight(self, height_in_px):
        # pure_image_height = self.height()
        # self.setHeight(height_in_px)
        # ratio = height_in_px / pure_image_height
        # self.setWidth(self.width() * ratio)
        # # we recompute the scale for the scale bar # TODO BE CAREFUL IF EXTRAS ARE ADDED TO THE OBJECT AS THIS WOULD PERTURB THE COMPUTATIONS
        # self.update_scale()
        pure_image_height = self.height(scaled=False)
        # if self.__crop_top is not None:
        #     pure_image_height-=self.__crop_top
        # if self.__crop_bottom is not None:
        #     pure_image_height-=self.__crop_bottom
        scale = height_in_px/pure_image_height
        self.scale = scale
        # need update bounds
        # scale is ok

    # def update_scale(self):
    #     # we recompute the scale for the scale bar # TODO BE CAREFUL IF EXTRAS ARE ADDED TO THE OBJECT AS THIS WOULD PERTURB THE COMPUTATIONS
    #     self.scale = self.get_scale()

    # def get_scale(self):
    #     # we recompute the scale for the scale bar # TODO BE CAREFUL IF EXTRAS ARE ADDED TO THE OBJECT AS THIS WOULD PERTURB THE COMPUTATIONS
    #     return self.width() / self.img.get_width()

    def crop(self, left=None, right=None, top=None, bottom=None, all=None):
        # print(self.boundingRect())
        if left is not None:
            self.__crop_left = left
            # self.setWidth(self.img.get_width() - self.__crop_left)
        if right is not None:
            self.__crop_right = right
            # self.setWidth(self.img.get_width() - self.__crop_right)
        if top is not None:
            self.__crop_top = top
            # self.setHeight(self.img.get_height() - self.__crop_top)
        if bottom is not None:
            self.__crop_bottom = bottom
            # self.setHeight(self.img.get_height() - self.__crop_bottom)
        if all is not None:
            self.__crop_left = all
            self.__crop_right = all
            self.__crop_top = all
            self.__crop_bottom = all
            # self.setWidth(self.img.get_width() - self.__crop_left)
            # self.setWidth(self.img.get_width() - self.__crop_right)
            # self.setHeight(self.img.get_height() - self.__crop_top)
            # self.setHeight(self.img.get_height() - self.__crop_bottom)

        # see how to crop actually because I need to create a qimage
        # self.qimage = self.img.crop()
        # print(self.boundingRect())
    # def set_to_scale(self, factor):
    #     self.scale = factor

    def set_to_translation(self, translation):
        self.translation = translation

    # WHY 0,0 FOR BOUNDS ??? --> sounds not smart to me
    def boundingRect(self, scaled=True):
 # en fait pas good besoin de prendre les crops et le scale en compte
        # is
        # rect_to_plot = self.adjusted(self.__crop_left, self.__crop_top, -self.__crop_right, -self.__crop_bottom)
        rect_to_plot = self.adjusted(0, 0, -self.__crop_right-self.__crop_left, -self.__crop_bottom-self.__crop_top)
        # rect_to_plot = self.adjusted(-self.__crop_left, -self.__crop_top, -self.__crop_right, -self.__crop_bottom)
        # rect_to_plot = self.adjusted(0,0,0,0)
        # print('begin rect_to_plot', rect_to_plot, self.scale)
        # if kwargs['draw']==True or kwargs['fill']==True:
        # if self.scale is None or self.scale==1:
        #     painter.drawRect(self)
        # else:
        # on clone le rect
        if self.scale is not None and self.scale != 1 and scaled:
            # TODO KEEP THE ORDER THIS MUST BE DONE THIS WAY OR IT WILL GENERATE PLENTY OF BUGS...
            new_width = rect_to_plot.width() * self.scale
            new_height = rect_to_plot.height() * self.scale
            # print(rect_to_plot.width(), rect_to_plot.height())  # here ok
            # setX changes width --> why is that

            # TODO BE EXTREMELY CAREFUL AS SETX AND SETY CAN CHANGE WIDTH AND HEIGHT --> ALWAYS TAKE SIZE BEFORE OTHERWISE THERE WILL BE A PB AND ALWAYS RESET THE SIZE WHEN SETX IS CALLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Sets the left edge of the rectangle to the given x coordinate. May change the width, but will never change the right edge of the rectangle. --> NO CLUE WHY SHOULD CHANGE WIDTH THOUGH BUT BE CAREFUL!!!
            # rect_to_plot.setX(rect_to_plot.x() * self.scale)
            # rect_to_plot.setY(rect_to_plot.y() * self.scale)
            # maybe to avoid bugs I should use translate instead rather that set x but ok anyways
            # print(rect_to_plot.width(), rect_to_plot.height())# bug here --> too big

            # print(new_height, new_height, self.width(), self.scale, self.scale* self.width())
            rect_to_plot.setWidth(new_width)
            rect_to_plot.setHeight(new_height)
        return rect_to_plot

    # def set_P1(self, *args):
    #     if not args:
    #         logger.error("no coordinate set...")
    #         return
    #     if len(args) == 1:
    #         self.moveTo(args[0].x(), args[0].y())
    #     else:
    #         self.moveTo(QPointF(args[0], args[1]))

    def get_P1(self):
        return self.boundingRect().topLeft()

    def width(self, scaled=True):
        return self.boundingRect(scaled=scaled).width()

    def height(self, scaled=True):
        return self.boundingRect(scaled=scaled).height()

    SVG_INKSCAPE = 96
    SVG_ILLUSTRATOR = 72

    # NB THIS CODE IS BASED ON THE EZFIG SAVE CODE --> ANY CHANGE MADE HERE MAY ALSO BE MADE TO THE OTHER
    # best here would be for image to keep original size if nothing is specified
    # could also make it return a qimage for display in pyTA
    qualities = [QPainter.NonCosmeticDefaultPen,  QPainter.SmoothPixmapTransform, QPainter.TextAntialiasing,QPainter.Antialiasing,QPainter.HighQualityAntialiasing]

    def save(self, path, filetype=None, title=None, description=None, svg_dpi=SVG_INKSCAPE, quality=qualities[-1]):
        # if path is None or not isinstance(path, str):
        #     logger.error('please provide a valide path to save the image "' + str(path) + '"')
        #     return
        if path is None:
            filetype = '.tif'

        if filetype is None:
            if path.lower().endswith('.svg'):
                filetype = 'svg'
            else:
                filetype = os.path.splitext(path)[1]
        dpi = 72  # 300 # inkscape 96 ? check for illustrator --> check

        if filetype == 'svg':
            generator = QSvgGenerator()
            generator.setFileName(path)
            if svg_dpi == self.SVG_ILLUSTRATOR:
                generator.setSize(QSize(595, 842))
                generator.setViewBox(QRect(0, 0, 595, 842))
            else:
                generator.setSize(QSize(794, 1123))
                generator.setViewBox(QRect(0, 0, 794, 1123))

            if title is not None and isinstance(title, str):
                generator.setTitle(title)
            if description is not None and isinstance(description, str):
                generator.setDescription(description)
            generator.setResolution(
                svg_dpi)  # fixes issues in inkscape of pt size --> 72 pr illustrator and 96 pr inkscape but need change size

            painter = QPainter(generator)

            # print(generator.title(), generator.heightMM(), generator.height(), generator.widthMM(),
            #       generator.resolution(), generator.description(), generator.logicalDpiX())
        else:
            scaling_factor_dpi = 1
            # scaling_factor_dpi = self.scaling_factor_to_achieve_DPI(300)

            # in fact take actual page size ??? multiplied by factor
            # just take real image size instead


            # image = QtGui.QImage(QSize(self.cm_to_inch(21) * dpi * scaling_factor_dpi, self.cm_to_inch(29.7) * dpi * scaling_factor_dpi), QtGui.QImage.Format_RGBA8888) # minor change to support alpha # QtGui.QImage.Format_RGB32)

            # NB THE FOLLOWING LINES CREATE A WEIRD ERROR WITH WEIRD PIXELS DRAWN some sort of lines NO CLUE WHY

            img_bounds = self.boundingRect()
            image = QtGui.QImage(QSize(img_bounds.width() * scaling_factor_dpi, img_bounds.height()* scaling_factor_dpi),  QtGui.QImage.Format_RGBA8888)  # minor change to support alpha # QtGui.QImage.Format_RGB32)
            # print('size at dpi',QSize(img_bounds.width() * scaling_factor_dpi, img_bounds.height()* scaling_factor_dpi))
            # QSize(self.cm_to_inch(0.02646 * img_bounds.width())
            # self.cm_to_inch(0.02646 * img_bounds.height())
            # need convert pixels to inches
            # is there a rounding error

            # force white bg for non jpg
            try:
                # print(filetype.lower())
                # the tif and png file formats support alpha
                if not filetype.lower() == '.png' and not filetype.lower() == '.tif' and not filetype.lower() == '.tiff':
                    image.fill(QColor.fromRgbF(1,1,1))
                else:
                    # image.fill(QColor.fromRgbF(1, 1, 1, alpha=1))
                    # image.fill(QColor.fromRgbF(1, 1, 1, alpha=1))
                    # TODO KEEP in fact image need BE FILLED WITH TRANSPARENT OTHERWISE GETS WEIRD DRAWING ERRORS
                    # TODO KEEP SEE https://stackoverflow.com/questions/13464627/qt-empty-transparent-qimage-has-noise
                    # image.fill(qRgba(0, 0, 0, 0))
                    image.fill(QColor.fromRgbF(0,0,0,0))
            except:
                pass
            painter = QPainter(image)  # see what happens in case of rounding of pixels
            # painter.begin()
            painter.scale(scaling_factor_dpi, scaling_factor_dpi)
        painter.setRenderHint(quality)  # to improve rendering quality
        self.draw(painter)
        painter.end()
        if path is None:
            return image
        if filetype != 'svg':
            image.save(path)
            return image

    #based on https://stackoverflow.com/questions/19902183/qimage-to-numpy-array-using-pyside
    def convert_qimage_to_numpy(self, qimage):
        qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        width = qimage.width()
        height = qimage.height()
        image_pointer = qimage.bits() # creates a deep copy --> this is what I want
        image_pointer.setsize(qimage.byteCount())
        # arr = np.array(image_pointer,copy=True).reshape(height, width, 4)
        arr = np.array(image_pointer).reshape(height, width, 4)
        arr = arr[..., 0:3]
        arr = Img.RGB_to_BGR(arr)  # that seems to do the job
        return arr

    # simple equation for an image
    # or maybe ask for starting expression
    # autoincrement math to get the right value
    # def _get_width_equation(self):
    #     # incr = fig_tools.common_value
    #     # AR = self.width(False)/self.height(False)
    #     # variable =
    #     equa_w = Symbol('a'+str(id(self)))*self.width(False)+0
    #     # equa_h = variable*AR*self.width(False)+0
    #     # incr+=1
    #     # fig_tools.common_value = incr
    #     return equa_w #, equa_h

    # def _get_height_equation(self):
    #     # incr = fig_tools.common_value
    #     AR = self.width(False)/self.height(False)
    #     # equa_w = Symbol('a'+str(id(self)))*self.width(False)+0
    #     equa_h = Symbol('a'+str(id(self)))*AR*self.width(False)+0
    #     # incr+=1
    #     # fig_tools.common_value = incr
    #     return equa_h #, equa_h

    # use that so that I can expand
    # if unique --> do nothing if many see how to handle that???
    # see how
    def get_equation(self):
        from sympy import nsolve, exp, Symbol
        AR = self.width(False)/self.height(False)
        equa_w = Symbol('a'+str(id(self)))*self.width(False)+0
        equa_h = Symbol('a'+str(id(self)))*AR*self.width(False)+0
        return [equa_w] , [equa_h], None # equa in width, equa in height, equas to solve, not useful here though

    # allow to deepcopy and copy
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

if __name__ == '__main__':
    # ça marche --> voici deux examples de shapes
    test = Image2D(x=12, y=0, width=100, height=100)  # could also be used to create empty image with

    print(test.img)
    print(test.boundingRect())
    print(test.get_P1().x())

    # bug qd on definit une image comme param
    # test = Image2D('./../data/counter/06.png')
    test = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
    print(test.boundingRect())  # --> it is ok there so why not below # not callable --> why -->
    print(test.get_P1())  # ça marche donc où est le bug
    print(test.get_P1().y())  # ça marche donc où est le bug
    # print(test.getP1().width())
    # ça marche

    # try draw on the image the quivers
    # img0.setLettering('<font color="red">A</font>')
    # # letter
    # img0.annotation.append(Rect2D(88, 88, 200, 200, stroke=3, color=0xFF00FF))
    # img0.annotation.append(Ellipse2D(88, 88, 200, 200, stroke=3, color=0x00FF00))
    # img0.annotation.append(Circle2D(33, 33, 200, stroke=3, color=0x0000FF))
    test.annotation.append(Line2D(33, 33, 88, 88, stroke=3, color=0x0000FF))

    test.annotation.append(Line2D(128, 33, 88, 88, stroke=0.65, color=0xFFFF00))
    # img0.annotation.append(Freehand2D(10, 10, 20, 10, 20, 30, 288, 30, color=0xFFFF00, stroke=3))
    # # img0.annotation.append(PolyLine2D(10, 10, 20, 10, 20, 30, 288, 30, color=0xFFFF00, stroke=3))
    # img0.annotation.append(Point2D(128, 128, color=0xFFFF00, stroke=6))

    # add a save to the image --> so that it exports as a raster --> TODO


    # painter =
    # img = test.draw() # if no painter --> create one as not to lose any data and or allow to save as vectorial


    img = test.save('/E/Sample_images/sample_images_PA/trash_test_mem/mini_vide/analyzed/trash/test_line2D.tif')

    test.save('/E/Sample_images/sample_images_PA/trash_test_mem/mini_vide/analyzed/trash/test_line2D.svg')

    #trop facile --> just hack it so that it can return a single qimage # or return a numpy image that is then plotted -> should not be too hard !!! I think --> TODO
    img = test.convert_qimage_to_numpy(img) # --> ok but I just need to swap the channels then I'll be done --> try that maybe with plenty of input images just to see


    # shall I save
    # try with non RGB images just to see

    # img = Img.RGB_to_BGR(img)

    # almost there --> just need to check that the size of the image is ok and that everything is fine
    plt.imshow(img)
    plt.show()


    print('here')
    img2 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/02.png')
    print('here3')
    preview(img2)

    # test replace the plot of pyTA of the polarity by this one --> should be quite easy TODO I think --> TODO

    # ok --> it all seems to work --> see how I can handle that



    # --> all seems ok now
    # --> put this in the advanced sql plotter

