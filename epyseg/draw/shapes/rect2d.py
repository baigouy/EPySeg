from PyQt5.QtCore import QPointF, QRectF
from PyQt5.QtGui import QBrush, QPen, QColor

from epyseg.tools.logger import TA_logger

logger = TA_logger()

class Rect2D(QRectF):

    def __init__(self, *args, color=0xFFFF00, fill_color=None, opacity=1., stroke=0.65, **kwargs):
        super(Rect2D, self).__init__(*args)
        if not args:
            self.isSet = False
        else:
            self.isSet = True
        self.color = color
        self.fill_color = fill_color
        self.stroke = stroke
        self.opacity = opacity
        self.isSet = False

    def draw(self, painter, draw=True):
        if draw:
            painter.save()
        pen = QPen(QColor(self.color))
        if self.stroke is not None:
            pen.setWidthF(self.stroke)
        painter.setPen(pen)
        painter.setOpacity(self.opacity)
        if draw:
            painter.drawRect(self)
            painter.restore()

    def fill(self, painter, draw=True):
        if self.fill_color is None:
            return
        if draw:
            painter.save()
        painter.setBrush(QBrush(QColor(self.fill_color)))
        painter.setOpacity(self.opacity)
        if draw:
            painter.drawRect(self)
            painter.restore()

    def drawAndFill(self, painter):
        painter.save()
        self.draw(painter, draw=False)
        self.fill(painter, draw=False)
        painter.drawRect(self)
        painter.restore()

    def boundingRect(self):
        return self

    def add(self, *args):
        point = args[1]
        self.setWidth(point.x()-self.x())
        self.setHeight(point.y()-self.y())
        self.isSet = True

    def set_P1(self, *args):
        if not args:
            logger.error("no coordinate set...")
            return
        if len(args) == 1:
            self.moveTo(args[0].x(), args[0].y())
        else:
            self.moveTo(QPointF(args[0], args[1]))

    def get_P1(self):
        return QPointF(self.x(), self.y())

if __name__ == '__main__':
    # ça marche --> voici deux examples de shapes
    test = Rect2D(0, 0, 100, 100)

    rect = QRectF(0,0, 125,256)
    print(rect.x())
    print(rect.y())
    print(rect.width())
    print(rect.height())


    rect.translate(10,20) # ça marche
    print(rect)



    (test.x(), test.y(), test.width(), test.height())
    print(test.contains(QPointF(50, 50)))
    print(test.contains(QPointF(-1, -1)))
    print(test.contains(QPointF(0, 0)))
    print(test.contains(QPointF(100, 100)))
    print(test.contains(QPointF(100, 100.1)))

    point = QPointF(50, 50)
    point.x()

    # p1 = test.p1()
    # print(p1.x(), p1.y())
    # p2 = test.p2()
    # print(p2.x(), p2.y())
    # print(test.arrow)
    # print(test.length()) # sqrt 2 --> 141
    # # if it's an arrow I can add easily all the stuff I need
    #
    # test = Rect2D(0, 0, 1, 1)
    # p1 = test.p1()
    # print(p1.x(), p1.y())
    # p2 = test.p2()
    # print(p2.x(), p2.y())
    # print(test.arrow)
    # import math
    # print(test.length() == math.sqrt(2))  # sqrt 2
    #
    # test2 = Rect2D()
    # p1 = test2.p1()
    # print(p1.x(), p1.y())
    # p2 = test2.p2()
    # print(p2.x(), p2.y())
    # print(test2.arrow)
