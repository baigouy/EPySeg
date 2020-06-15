from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint, QPointF
from PyQt5.QtGui import QBrush, QPen, QColor

from epyseg.tools.logger import TA_logger

logger = TA_logger()

class Ellipse2D(QtWidgets.QGraphicsEllipseItem):

    isSet = False

    def __init__(self, *args, color=0xFFFF00, fill_color=None, opacity=1., stroke=0.65, **kwargs):
        super(Ellipse2D, self).__init__(*args)
        if not args:
            self.isSet = False
        else:
            self.isSet = True
        self.setRect(self.rect())
        self.color = color
        self.fill_color = fill_color
        self.stroke = stroke
        self.opacity = opacity

    def draw(self, painter, draw=True):
        if draw:
            painter.save()
        pen = QPen(QColor(self.color))
        if self.stroke is not None:
            pen.setWidthF(self.stroke)
        painter.setPen(pen)
        painter.setOpacity(self.opacity)
        if draw:
            painter.drawEllipse(self.rect())
            painter.restore()

    def fill(self, painter, draw=True):
        if self.fill_color is None:
            return
        if draw:
            painter.save()
        painter.setBrush(QBrush(QColor(self.fill_color)))
        painter.setOpacity(self.opacity)
        if draw:
            painter.drawEllipse(self.rect())
            painter.restore()

        # TODO pb will draw the shape twice.... ---> because painter drawpolygon is called twice

    def drawAndFill(self, painter):
        painter.save()
        self.draw(painter, draw=False)
        self.fill(painter, draw=False)
        painter.drawEllipse(self.rect())
        painter.restore()

    def translate(self, translation):
        self.moveBy(translation.x(), translation.y())
        rect = self.rect()
        rect.translate(translation.x(), translation.y())
        self.setRect(rect)

    def add(self, *args):
        p1 = args[0]
        p2 = args[1]
        rect = self.rect()
        rect.setWidth(abs(p1.x()-p2.x()))
        rect.setHeight(abs(p1.y()-p2.y()))
        x = p2.x()
        y = p2.y()
        if p1.x() < p2.x():
            x = p1.x()
        if p1.y() < p2.y():
            y = p1.y()
        rect.setX(x)
        rect.setY(y)

        self.setRect(rect)
        self.isSet = True

    def setP1(self, point):
        rect = self.rect()
        rect.setX(point.x()-rect.width()/2.)
        rect.setY(point.y()+rect.height()/2.)
        self.setRect(rect)


if __name__ == '__main__':
    # ça marche --> voici deux examples de shapes
    test = Ellipse2D(0, 0, 100, 100)
    # print(test.x(), test.y(), test.width(), test.height())
    print(test.contains(QPointF(50, 50)))
    print(test.contains(QPointF(15, 15)))
    print(test.contains(QPointF(-1, -1)))
    print(test.contains(QPointF(0, 0)))
    print(test.contains(QPointF(100, 100)))
    print(test.contains(QPointF(100, 100.1)))
    print(test.x())
    print(test.y())
    print(test.translate(QPoint(10, 10)))
    print(test.x())
    print(test.y())

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
