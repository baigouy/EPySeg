# le retaillage des graphes a l'air de marcher mais à verif qd meme...

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

import matplotlib.pyplot as plt

from epyseg.draw.shapes.image2d import Image2D
from epyseg.draw.shapes.rect2d import *
from epyseg.img import Img
from PyQt5.QtCore import QRectF
import matplotlib as mpl
import base64
from PIL import Image
import matplotlib.pyplot as plt
import io
import numpy as np
# logger
from epyseg.tools.logger import TA_logger


# should this stuff rather implement Image2D --> maybe or not think about it...

logger = TA_logger()

class Graph2D(Image2D):#Image2D # Rect2D

    def __init__(self, figure, *args, x=None, y=None, width=None, height=None, data=None, dimensions=None, opacity=1.,
                 stroke=0.65, **kwargs):
        self.isSet = False

        # self.annotation = [] # should contain the objects for annotating imaging --> shapes and texts
        self.letter = None # when objects are swapped need change the letter
        self.figure = None
        self.annotation = []  # should contain the objects for annotating imaging --> shapes and texts
        self.img = None
        self.qimage = None
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
        super(Graph2D, self).__init__(self.img)

        # store also the image for that

        self.stroke = stroke  # DO I REALLY NEED STROKE
        self.opacity = opacity

    def getFigure(self):
        return self.figure

    def setFigure(self, figure):
        # TODO load the raw image there so that it can be drawn easily
        self.figure = figure
        if self.figure is not None and isinstance(self.figure, plt.Figure):
            self.isSet = True
            self.img = self.toImg()
            print(self.img.get_width())
            self.qimage = self.img.getQimage()
        else:
            logger.error('The provided figure not a valid matplotlib figure! Nothing can be done with it... Sorry...')
            self.figure = None
            self.isSet = False

    def getAxes(self):
        if self.figure is not None:
            return self.figure.axes
        return None

    # @return the block incompressible width
    def getIncompressibleWidth(self):
        extra_space = 0 # can add some if boxes around to add text
        return extra_space

    # @return the block incompressible height
    def getIncompressibleHeight(self):
        extra_space = 0  # can add some if boxes around to add text
        return extra_space

    def setLetter(self, letter):
        self.letter = letter

    def toBuffer(self, bufferType='raster'):
        if self.figure is not None:
            buf = io.BytesIO()
            if bufferType == 'raster':
                self.figure.savefig(buf, format='png', bbox_inches='tight')
            else:
                self.figure.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            return buf
        return None

    def toBase64(self):
        buf = self.toBuffer()
        if buf is None:
            return None
        buf.seek(0)
        figdata_png = base64.b64encode(buf.getvalue())
        buf.close()
        return figdata_png

    def toSVG(self):
        buf = self.toBuffer()
        if buf is None:
            return None
        buf.seek(0)
        text = buf.read()
        buf.close()
        return text

    def toImg(self):

        # print(self.toBase64()) # this is ok

        if self.figure is not None:
            buf = self.toBuffer()
            if buf is None:
                return None
            buf.seek(0)
            im = Image.open(buf)

            # im.show()
            pix = np.array(im)

            print(pix.shape)
            img = Img(pix, dimensions='hwc')
            print(img.shape, pix.shape)

            buf.close()
            # should I get image width and height there ???
            # im.show()
            return img
        return None

    # https://stackoverflow.com/questions/14708695/specify-figure-size-in-centimeter-in-matplotlib
    def cm2inch(self, *tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i / inch for i in tupl[0])
        else:
            print(tupl) # why does it contain a qrectf ????
            return tuple(i / inch for i in tupl)

    def draw(self, painter, draw=True):
        if draw:
            painter.save()
        painter.setOpacity(self.opacity)
        if draw:
            if self.img is not None:
                qsource = QRectF(0,0,self.img.get_width(), self.img.get_height())
                painter.drawImage(self, self.qimage , qsource) # , flags=QtCore.Qt.AutoColor
            else:
                painter.drawRect(self)
            painter.restore()

        # then need to draw the letter
        if self.letter is not None:
            self.letter.set_P1(self.get_P1())
            self.letter.draw(painter)

        if self.annotation is not None and self.annotation:
            for annot in self.annotation:
                annot.drawAndFill(draw=draw)

    def fill(self, painter, draw=True):
        if self.fill_color is None:
            return
        if draw:
            painter.save()
        painter.setOpacity(self.opacity)
        if draw:
            if self.img is not None:
                qsource = QRectF(0, 0, self.img.get_width(), self.img.get_height())
                painter.drawImage(self, self.qimage , qsource)
            else:
                painter.drawRect(self)
            painter.restore()

    def drawAndFill(self, painter):
        painter.save()
        if self.img is not None:
            qsource = QRectF(0, 0, self.img.get_width(), self.img.get_height())
            painter.drawImage(self, self.qimage , qsource)
        else:
            painter.drawRect(self)
        painter.restore()

    def __add__(self, other):
        from epyseg.figure.row import Row # KEEP Really required to avoid circular imports
        return Row(self, other)

    # create a Fig with divide
    def __truediv__(self, other):
        from deprecated_demos.ezfig_tests.col import col # KEEP Really required to avoid circular imports
        return col(self, other)

    #Force the montage width to equal 'width_in_px'
    def setToWidth(self, width_in_px):
        pure_image_width = self.width()
        ratio = width_in_px / pure_image_width
        self.setWidth(width_in_px)
        self.setHeight(self.height()*ratio)
        # need recompute the graph and update it with the new size --> convert in pixels
        cm_width = 0.02646*width_in_px
        cm_height = 0.02646*self.height()
        # print('cm', cm_width, cm_height)
        # print(self.figure)
        # print('abs')
        # print(self.cm2inch(cm_width,cm_height)) # bug is here
        # print('ibs')
        self.figure.set_size_inches(self.cm2inch(cm_width,cm_height)) # need do a px to inch or to cm --> TODO
        # print(self.figure.get_size_inches())
        self.setFigure(self.figure)

    def setToHeight(self, height_in_px):
        pure_image_height = self.height()
        self.setHeight(height_in_px)
        ratio = height_in_px / pure_image_height
        self.setWidth(self.width()*ratio)

        # the lines below cause a bug I guess it's because it's an image object and not a qrect object anymore and so width sucks --> calls a function
        cm_width = 0.02646*self.width()
        cm_height = 0.02646*height_in_px
        # print('cm', cm_width, cm_height)
        # print(self.figure)
        # print('oubs')
        # print(self.cm2inch(cm_width, cm_height))
        # print('ebs')
        self.figure.set_size_inches(self.cm2inch(cm_width,cm_height)) # need do a px to inch or to cm --> TODO
        # print(self.figure.get_size_inches())
        self.setFigure(self.figure)

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
    test = Graph2D(fig, x=12, y=0, width=100, height=100)  # could also be used to create empty image with

    print(test.img)
    print(test.boundingRect())
    print(test.get_P1().x())

    # bug qd on definit une image comme param
    # test = Image2D('./../data/counter/06.png')
    # test = Graph2D('D:/dataset1/unseen/100708_png06.png')
    print(test.boundingRect()) # --> it is ok there so why not below # not callable --> why -->
    print(test.get_P1()) # ça marche donc où est le bug
    print(test.get_P1().y())  # ça marche donc où est le bug
    # print(test.getP1().width())
    # ça marche

