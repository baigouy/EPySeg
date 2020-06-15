#https://docs.python.org/2/library/operator.html

# TODO may also contain svg or graphs or ???
# TODO handle extra labels for images directly and also for rows or cols --> think how to do that but must be doable

from epyseg.draw.shapes.rect2d import *
from epyseg.img import Img
from PyQt5.QtCore import QRectF
# logger
from epyseg.tools.logger import TA_logger
logger = TA_logger()

class Image2D(Rect2D):

    def __init__(self, *args, x=None, y=None, width=None, height=None, data=None, dimensions=None, opacity=1.,
                 stroke=0.65, **kwargs):
        self.isSet = False

        self.annotation = [] # should contain the objects for annotating imaging --> shapes and texts
        self.letter = None # when objects are swapped need change the letter
        if args:
            if len(args) == 1:
                self.filename = args[0]
        else:
            self.filename = None

        if x is None and y is None and width is not None and height is not None:
            super(Image2D, self).__init__(0, 0, width, height)
            self.isSet = True
        elif x is None and y is None and width is None and height is None:
            # print('in 0')
            self.img = Img(self.filename)
            self.qimage = self.img.getQimage()
            width = self.img.get_width()
            height = self.img.get_height()
            super(Image2D, self).__init__(0, 0, width, height)
            self.isSet = True
        elif x is not None and y is not None and width is not None and height is not None:
            self.img = None
            super(Image2D, self).__init__(x, y, width, height)
            self.isSet = True
        elif data is None:
            if self.filename is not None:
                self.img = Img(self.filename)
                self.qimage = self.img.getQimage()
                if x is None:
                    x = 0
                if y is None:
                    y = 0
                super(Image2D, self).__init__(x, y, self.img.get_width(), self.img.get_height())
                self.isSet = True
        elif data is not None:
            self.img = Img(data,
                           dimensions=dimensions)  # need width and height so cannot really be only a numpy stuff --> cause no width or height by default --> or need tags such as image type for dimensions
            self.qimage = self.img.getQimage()
            # need Image dimensions id data is not of type IMG --> could check that
            if x is None:
                x = 0
            if y is None:
                y = 0
            super(Image2D, self).__init__(x, y, self.img.get_width(), self.img.get_height())
            self.isSet = True
        self.stroke = stroke  # DO I REALLY NEED STROKE
        self.opacity = opacity

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

    def setToHeight(self, height_in_px):
        pure_image_height = self.height()
        self.setHeight(height_in_px)
        ratio = height_in_px / pure_image_height
        self.setWidth(self.width()*ratio)

if __name__ == '__main__':
    # ça marche --> voici deux examples de shapes
    test = Image2D(x=12, y=0, width=100, height=100)  # could also be used to create empty image with

    print(test.img)
    print(test.boundingRect())
    print(test.get_P1().x())

    # bug qd on definit une image comme param
    # test = Image2D('./../data/counter/06.png')
    test = Image2D('D:/dataset1/unseen/100708_png06.png')
    print(test.boundingRect()) # --> it is ok there so why not below # not callable --> why -->
    print(test.get_P1()) # ça marche donc où est le bug
    print(test.get_P1().y())  # ça marche donc où est le bug
    # print(test.getP1().width())
    # ça marche
