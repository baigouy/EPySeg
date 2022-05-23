# TODO recode cause too complex

# TODO --> clean and reimplement this properly...
# alternatively offer commands to be added to the figure so that they fit the requirements --> is another way of doing

# check if that is interesting for me https://matplotlib.org/stable/devel/MEP/MEP25.html
# CROP IS NOT OPTIMAL BUT OK FOR NOW...

# this file must be a valid svg file and/or a matplotlib figure --> TODO handle both and check that everything correct

# see also pyQT_svg_drawing_test.py --> for loading and reading svg files --> maybe this is what I should do ???

# TODO make it draw as svg rather than as raster directly --> see how
# in fact only required for save

# le retaillage des graphes a l'air de marcher mais à verif qd meme...
# nb the drawing as svg is only required for saving and not for displaying

# maybe add behaviour should be different if images are missing --> then in that case the images should be added to the missing positions, this is important for panels
# maybe instead of panel I can build complex layouts and then fill them by adding images to them
# maybe also support for human language (not sure how easy that would be), maybe return a list of commands

# TODO see my plots_python_test.py

# a good start but I really need to have handle the size better and the resize too --> think how to smartly do that ???

# maybe ask if keep AR should be set to true ??? or not think about it...
# https://docs.python.org/2/library/operator.html

# TODO may also contain svg or graphs or ???
# TODO handle extra labels for images directly and also for rows or cols --> think how to do that but must be doable

# I probably need python fig and axes
# https://matplotlib.org/3.2.2/gallery/subplots_axes_and_figures/gridspec_multicolumn.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-multicolumn-py
# https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.pyplot.figure.html
# https://matplotlib.org/3.2.2/api/matplotlib_configuration_api.html#matplotlib.rcParams
import traceback

from PyQt5.QtSvg import QSvgRenderer

from epyseg.draw.shapes.rect2d import Rect2D
from epyseg.draw.shapes.txt2d import TAText2D
from epyseg.img import Img
from PyQt5.QtCore import QRectF
import base64
from PIL import Image
import matplotlib.pyplot as plt
import io
import numpy as np
# logger
from epyseg.tools.logger import TA_logger

# should this stuff rather implement Image2D --> maybe or not think about it...

logger = TA_logger()


# do I really need crop for that and shouldn't it be done from outside to prevent pbs and it will also definitely mess with the font which is not a good thing ???
# try it

# I don't manage to get the crops to work...
# can I add a hashtag to texts so that all texts with the same hashtag are treated the same...

class VectorGraphics2D(Rect2D):  # Image2D # Rect2D

    def __init__(self, figure, *args, x=None, y=None, width=None, height=None, opacity=1., stroke=0.65, **kwargs):
        self.isSet = False

        # TODO can only force size of a graph but not of an svg --> modify code accordingly, if graph and modify need apply set_size_inches to the figure... --> TODO rapidly
        tmp_size = QRectF()
        if x is not None and str(x).isnumeric():
            tmp_size.setX(x)
        if y is not None and str(x).isnumeric():
            tmp_size.setY(y)
        if width is not None and str(width).isnumeric():
            tmp_size.setWidth(width)
        if height is not None and str(height).isnumeric():
            tmp_size.setHeight(height)
        super(VectorGraphics2D, self).__init__(tmp_size)

        # self.annotation = [] # should contain the objects for annotating imaging --> shapes and texts
        self.letter = None  # when objects are swapped need change the letter
        self.figure = None
        self.annotation = []  # should contain the objects for annotating imaging --> shapes and texts
        # self.img = None
        # self.qimage = None
        self.renderer = None
        self.filename = None
        # crops
        self.__crop_left = None
        self.__crop_right = None
        self.__crop_top = None
        self.__crop_bottom = None
        self.must_update_figure_on_first_paint = False

        # first argument should be fig
        # if args:
        #     if len(args) == 1:
        #         self.filename = args[0]
        # else:
        #     self.filename = None

        # if x is None and y is None and width is not None and height is not None:
        #     super(Graph2D, self).__init__(0, 0, width, height)
        #     self.isSet = True
        # elif x is None and y is None and width is None and height is None:
        #     # print('in 0')
        #     self.img = Img(self.filename)
        #     self.qimage = self.img.getQimage()
        #     width = self.img.get_width()
        #     height = self.img.get_height()
        #     super(Graph2D, self).__init__(0, 0, width, height)
        #     self.isSet = True
        # elif x is not None and y is not None and width is not None and height is not None:
        #     self.img = None
        #     super(Graph2D, self).__init__(x, y, width, height)
        #     self.isSet = True
        # elif data is None:
        #     if self.filename is not None:
        #         self.img = Img(self.filename)
        #         self.qimage = self.img.getQimage()
        #         if x is None:
        #             x = 0
        #         if y is None:
        #             y = 0
        #         super(Graph2D, self).__init__(x, y, self.img.get_width(), self.img.get_height())
        #         self.isSet = True
        # elif data is not None:
        #     self.img = Img(data,
        #                    dimensions=dimensions)  # need width and height so cannot really be only a numpy stuff --> cause no width or height by default --> or need tags such as image type for dimensions
        #     self.qimage = self.img.getQimage()
        #     # need Image dimensions id data is not of type IMG --> could check that
        #     if x is None:
        #         x = 0
        #     if y is None:
        #         y = 0
        #     super(Graph2D, self).__init__(x, y, self.img.get_width(), self.img.get_height())
        # TODO should I also allow aspect ratio or keep aspect ratio or fixed width or height or auto --> think about it and simply try

        # TODO should I allow to create a fig from some code using eval (i.e. a script that can be serialized, given the dangers of the eval function I dunno)
        # if self.figure is not None and isinstance(self.figure, plt.figure):
        #     self.isSet = True
        # else:
        #     self.isSet = False
        self.setFigure(figure)
        # super(Graph2D, self).__init__(0, 0, self.img.get_width(), self.img.get_height())

        # store also the image for that

        self.stroke = stroke  # DO I REALLY NEED STROKE
        self.opacity = opacity

    def getFigure(self):
        return self.figure

    def setFigure(self, figure):

        self.must_update_figure_on_first_paint = False

        # TODO load the raw image there so that it can be drawn easily
        # self.figure = figure
        if figure is not None and isinstance(figure, plt.Figure):

            # self.img = self.toImg()
            # print(self.img.get_width())
            # self.qimage = self.img.getQimage()
            # make a renderer out of it and display it ...

            # print("size inches before rendering", figure.get_size_inches())
            self.figure = figure
            buffer = self._toBuffer(bufferType='svg')
            self.renderer = QSvgRenderer(buffer.read())
            buffer.close()
            # self.setSize(QSizeF(self.renderer.defaultSize()))
            # upon init we do set the width --> should this be done here or at other position ??? think about it
            if self.width() == 0:
                size = self.renderer.defaultSize()
                # print('default size', size, 'vs', self.renderer.viewBox())
                self.setWidth(size.width())
                self.setHeight(size.height())
            self.isSet = True
        elif figure is not None and isinstance(figure, str):  # path to an svg file
            # just try load it
            # do I ever need the buffer for this too cause if yes then I would need to get it --> think how ???

            # if buffer is needed --> this is how I should do it
            # in_file = open(figure)  # opening for [r]eading as [b]inary
            # data = in_file.read()  # if you only wanted to read 512 bytes, do .read(512)
            # in_file.close()
            # TODO add tries to see if that works and if opened properly
            # print(data)
            self.renderer = QSvgRenderer(figure)  # data --> need convert data to be able to read it

            # self.renderer.render(painter, self)# the stuff is a qrectf so that should work

            # self.renderer.setViewBox(viewbox)

            self.filename = figure
            if self.width() == 0:
                size = self.renderer.defaultSize()
                # print('default size', size, 'vs', self.renderer.viewBox())
                self.setWidth(size.width())
                self.setHeight(size.height())
                if size.width() <= 0:
                    logger.error('image "' + str(self.filename) + '" could not be loaded')
                    self.isSet = False
                    return
                # j'arrive pas à faire des crop avec les viewbox
                # viewbox = self.renderer.viewBoxF()
                #
                # # does not really work
                # neo = QRectF()
                # neo.setX(30)
                # neo.setY(30)
                # neo.setHeight(self.height()-30)
                # neo.setWidth(self.width()-30)
                # self.renderer.setViewBox(neo)

            self.figure = None
            self.isSet = True
        else:
            logger.error(
                'The provided figure is not a valid matplotlib figure nor a valid svg file! Nothing can be done with it... Sorry...')
            self.figure = None
            self.isSet = False


    def getAxes(self):
        if self.figure is not None:
            return self.figure.axes
        return None


# somehow I need it --> keep it for now !!!
# WHY should I have that ???
#     @return the block incompressible width
    def getIncompressibleWidth(self):
        extra_space = 0  # can add some if boxes around to add text
        return extra_space

    # @return the block incompressible height
    def getIncompressibleHeight(self):
        extra_space = 0  # can add some if boxes around to add text
        return extra_space

    # if bg color is set then need add it --> see how to do that
    def setLettering(self, letter):
        if isinstance(letter, TAText2D):
            self.letter = letter
        elif isinstance(letter, str):
            if letter.strip() == '':
                self.letter = None
            else:
                self.letter = TAText2D(letter)

    def _toBuffer(self, bufferType='raster'):
        if self.figure is not None:
            buf = io.BytesIO()
            if bufferType == 'raster':
                self.figure.savefig(buf, format='png', bbox_inches='tight')
            else:
                self.figure.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            return buf
        return None

    # will only work for figures and is that even needed in fact --> think about it
    def _toBase64(self):
        buf = self._toBuffer()
        if buf is None:
            return None
        buf.seek(0)
        figdata_png = base64.b64encode(buf.getvalue())
        buf.close()
        return figdata_png

    # will only work for figures and is that even needed in fact --> think about it
    def toSVG(self):
        buf = self._toBuffer()
        if buf is None:
            return None
        buf.seek(0)
        text = buf.read()
        buf.close()
        return text

    # will only work for figures and is that even needed in fact --> think about it
    def _toImg(self):
        # print(self.toBase64()) # this is ok
        if self.figure is not None:
            buf = self._toBuffer()
            if buf is None:
                return None
            buf.seek(0)
            im = Image.open(buf)

            # im.show()
            pix = np.array(im)

            # print(pix.shape)
            img = Img(pix, dimensions='hwc')
            # print(img.shape, pix.shape)

            buf.close()
            # should I get image width and height there ???
            # im.show()
            return img
        return None

    # NEED CONVERSION FROM AND TO PX AND CM AND INCHES... --> TODO
    # TODO do a class to convert stuff and do it smart enough to handle any type of data and return the same type e.g. tuples as tuples, numbers as numbers and lists as lists
    # https://stackoverflow.com/questions/14708695/specify-figure-size-in-centimeter-in-matplotlib
    def cm2inch(self, *tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i / inch for i in tupl[0])
        else:
            # print(tupl) # why does it contain a qrectf ????
            return tuple(i / inch for i in tupl)

    def draw(self, painter, draw=True):
        # TODO check if this is really what I want at least it seems better than what I was doing before!!!
        if self.must_update_figure_on_first_paint:
            try:
                if self.figure is not None:
                    cm_width = 0.02646 * self.width()
                    cm_height = 0.02646 * self.height()
                    self.figure.set_size_inches(self.cm2inch(cm_width * self.scale, cm_height * self.scale))  # need do a px to
                    self.setFigure(self.figure)
            except:
                traceback.print_exc()
            self.must_update_figure_on_first_paint = False

        if draw:
            painter.save()
        painter.setOpacity(self.opacity)

        # may also work --> is that the best solution ????
        # painter.scale(0.3, 0.3)

        # new addition but probably not ok
        painter.scale(self.scale, self.scale)  # size is ok but position sucks --> how can I fix that
        # --> ok except for the coords --> fix that



        if draw and self.renderer is not None:
            # if self.img is not None:
            #     qsource = QRectF(0,0,self.img.get_width(), self.img.get_height())
            #     painter.drawImage(self, self.qimage , qsource) # , flags=QtCore.Qt.AutoColor
            # else:
            #     painter.drawRect(self)

            # print('size of drawing', self)

            # print('view box',self.renderer.viewBoxF())

            # viewbox = self.renderer.viewBoxF()

            # --> vraiment presque ça
            # does not really work
            neo = QRectF(self)

            # neo*=self.scale
            # neo.setX(self.x()-10)
            # neo.setY(self.y()-10)
            # neo.setHeight(self.height()+30)
            # neo.setWidth(self.width()+30)

            # print(neo)

            # nb this will create a shear if one dim is not cropped as the other --> really not great in fact maybe deactivate for now ???? or compute AR and adapt it
            # TODO just warn that it's buggy and should not be used for SVG files only, ok for other stuff though

            # can I preserve AR ???

            if self.__crop_left is not None:
                neo.setX(neo.x() - self.__crop_left)
                neo.setWidth(neo.width() + self.__crop_left)
            if self.__crop_top is not None:
                neo.setY(neo.y() - self.__crop_top)
                neo.setHeight(neo.height() + self.__crop_top)
            if self.__crop_bottom is not None:
                neo.setHeight(neo.height() + self.__crop_bottom)
            if self.__crop_right is not None:
                neo.setWidth(neo.width() + self.__crop_right)

            # maintenant ça a l'air bon...

            # --> ça c'est ok --> c'est le clip rect qui merde du coup

            # print('view box neo', neo)
            # self.renderer.setViewBox(neo)
            # le clipping marche mais faut le combiner avec autre chose
            # painter.setClipRect(self.x()+10, self.y()+10, self.width()-30,self.height()-30)#, Qt::ClipOperation operation = Qt::ReplaceClip

            # TODO KEEP UNFORTUNATELY  unfortunately cropping does not work when saved as svg but works when saved as raster... see https://bugreports.qt.io/browse/QTBUG-28636
            # maybe do masque d'ecretage in illustrator or inkscape https://linuxgraphic.org/forums/viewtopic.php?f=6&t=6437
            # TODO KEEP IT PROBABLY ALSO CREATES A ZOOM THAT WILL MESS WITH THE FONTS AND LINE SIZE...
            painter.setClipRect(self)  # , Qt::ClipOperation operation = Qt::ReplaceClip , operation=Qt.ReplaceClip
            # neo.setX(self.x())
            # neo.setY(self.y())


            # print('neo vs self',neo, self) # probably a bug here too

            self.renderer.render(painter, neo)  # the stuff is a qrectf so that should work
            # self.sequence_scene.render(painter, target=neo, source=self)

            painter.restore()
            # self.renderer.setViewBox(viewbox)

        # then need to draw the letter
        if self.letter is not None:
            self.letter.set_P1(self.get_P1())
            self.letter.draw(painter)

        if self.annotation is not None and self.annotation:
            for annot in self.annotation:
                annot.draw(draw=draw)

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
    # return self.draw(painter=painter, draw=draw)

    # def drawAndFill(self, painter):
    # painter.save()
    # if self.img is not None:
    #     qsource = QRectF(0, 0, self.img.get_width(), self.img.get_height())
    #     painter.drawImage(self, self.qimage , qsource)
    # else:
    #     painter.drawRect(self)
    # painter.restore()
    # return self.draw(painter=painter)

    def __add__(self, other):
        from epyseg.figure.row import Row  # KEEP Really required to avoid circular imports
        return Row(self, other)

    # create a Fig with divide
    def __truediv__(self, other):
        from epyseg.draw.widgets.col import col  # KEEP Really required to avoid circular imports
        return col(self, other)

    # Force the montage width to equal 'width_in_px'

    # ne marche qu'avec les graphes en fait et faudrait demander si on peut changer aspect ratio... avant
    def setToWidth(self, width_in_px):
        pure_image_width = self.width()
        ratio = width_in_px / pure_image_width
        self.setWidth(width_in_px)
        self.setHeight(self.height() * ratio)

        self.scale = width_in_px / pure_image_width

        # nb this is what makes everythong so slow --> skip that and do it only when necessary
        if self.figure is not None:
            # need recompute the graph and update it with the new size --> convert in pixels
            cm_width = 0.02646 * width_in_px
            cm_height = 0.02646 * self.height()

            self.must_update_figure_on_first_paint = True
            # MEGA TODO maybe set a boolean saying that the fig needs be updated on plot, then set it to false when this is done... --> set the figure so that it fits the image
            # see how I can do that ???


            # print('cm', cm_width, cm_height)
            # print('cm', cm_width, cm_height)
            # print(self.figure)
            # print('abs')
            # print('inches', self.cm2inch(cm_width,cm_height)) # bug is here
            # print('ibs')
            # self.figure.set_size_inches(self.cm2inch(cm_width, cm_height))  # need do a px to inch or to cm --> TODO

            # MEGA ULTIMATE TODO freshly inactivated because too slow --> need do it in a smart way only in the end --> this is really the slow part and need only be done on save or export --> see how to do that anyways it does not make sense to have this here
            # self.figure.set_size_inches(self.cm2inch(cm_width*self.scale, cm_height*self.scale))  # need do a px to inch or to cm --> TODO


            # print(self.figure.get_size_inches())
            # print('size before', self) # --> size ok after not ok --> bug in conversion somewhere

            # MEGA ULTIMATE TODO freshly inactivated because too slow --> need do it in a smart way only in the end --> this is really the slow part and need only be done on save or export --> see how to do that anyways it does not make sense to have this here
            # self.setFigure(self.figure)


            # print('changing size')
            # print('inches after', self.figure.get_size_inches())
            # print('size after', self)

    def setToHeight(self, height_in_px):
        pure_image_height = self.height()
        self.setHeight(height_in_px)
        ratio = height_in_px / pure_image_height
        self.setWidth(self.width() * ratio)
        # TODO implement that
        self.scale = height_in_px / pure_image_height

        if self.figure is not None:

            # MEGA TODO maybe set a boolean saying that the fig needs be updated on plot, then set it to false when this is done...

            # the lines below cause a bug I guess it's because it's an image object and not a qrect object anymore and so width has pb --> calls a function
            cm_width = 0.02646 * self.width()
            cm_height = 0.02646 * height_in_px
            self.must_update_figure_on_first_paint = True

            # print('cm', cm_width, cm_height)
            # print(self.figure)
            # print('oubs')
            # print(self.cm2inch(cm_width, cm_height))
            # print('ebs')
            # self.figure.set_size_inches(self.cm2inch(cm_width, cm_height))  # need do a px to inch or to cm --> TODO

            # MEGA ULTIMATE TODO freshly inactivated because too slow --> need do it in a smart way only in the end --> this is really the slow part and need only be done on save or export --> see how to do that anyways it does not make sense to have this here
            # self.figure.set_size_inches(self.cm2inch(cm_width*self.scale, cm_height*self.scale))  # need do a px to inch or to cm --> TODO

            # print(self.figure.get_size_inches())

            # MEGA ULTIMATE TODO freshly inactivated because too slow --> need do it in a smart way only in the end --> this is really the slow part and need only be done on save or export --> see how to do that anyways it does not make sense to have this here
            # self.setFigure(self.figure)

    # No clue how to do that --> ignore...
    def crop(self, left=None, right=None, top=None, bottom=None, all=None):
        logger.warning(
            'Crop of svg files is very buggy and should not be used, especially for scientific publications as it may distort the image...')
        # print(self.boundingRect())
        if left is not None:
            self.__crop_left = left
            self.setWidth(self.width() - self.__crop_left)
            self.setX(self.x() + self.__crop_left)
        if right is not None:
            self.__crop_right = right
            self.setWidth(self.width() - self.__crop_right)
        if top is not None:
            self.__crop_top = top
            self.setHeight(self.height() - self.__crop_top)
            self.setY(self.y() + self.__crop_top)
        if bottom is not None:
            self.__crop_bottom = bottom
            self.setHeight(self.height() - self.__crop_bottom)
        if all is not None:
            self.__crop_left = all
            self.__crop_right = all
            self.__crop_top = all
            self.__crop_bottom = all
            self.setX(self.x() + self.__crop_left)
            self.setY(self.y() + self.__crop_top)
            self.setWidth(self.width() - self.__crop_left)
            self.setWidth(self.width() - self.__crop_right)
            self.setHeight(self.height() - self.__crop_top)
            self.setHeight(self.height() - self.__crop_bottom)

        # see how to crop actually because I need to create a qimage
        # self.qimage = self.img.crop()
        # print(self.boundingRect())


if __name__ == '__main__':
    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks')
    ax.grid()

    # plt.show()
    # fig.savefig("test.png")
    # plt.show()
    # ça marche --> voici deux examples de shapes
    test = VectorGraphics2D(fig, x=12, y=0, width=100, height=100)  # could also be used to create empty image with

    # print(test.img)
    print(test.boundingRect())
    print(test.get_P1().x())

    # bug qd on definit une image comme param
    # test = Image2D('./../data/counter/06.png')
    # test = Graph2D('D:/dataset1/unseen/100708_png06.png')
    print(test.boundingRect())  # --> it is ok there so why not below # not callable --> why -->
    print(test.get_P1())  # ça marche donc où est le bug
    print(test.get_P1().y())  # ça marche donc où est le bug
    # print(test.getP1().width())
    # ça marche

    test.setToWidth(512)
    print(test.boundingRect())
    # --> ok
    # how can I think the scale to that ???

    test2 = VectorGraphics2D('/E/Sample_images/sample_images_svg/cartman.svg', x=12, y=0, width=100, height=100)
