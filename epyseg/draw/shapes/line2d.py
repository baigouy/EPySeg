from PyQt5.QtCore import QPointF, QLineF, QRectF, QPoint
from PyQt5.QtGui import QBrush, QPen, QColor
from math import sqrt

from epyseg.tools.logger import TA_logger

logger = TA_logger()

class Line2D(QLineF):

    def __init__(self, *args, color=0xFFFF00, fill_color=None, opacity=1., stroke=0.65, arrow=False, **kwargs):
        super(Line2D, self).__init__(*args)
        if not args:
            self.isSet = False
        else:
            self.isSet = True
        self.arrow = arrow
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
            painter.drawLine(self)
            painter.restore()

    def fill(self, painter, draw=True):
        if draw:
            painter.save()
        if self.fill_color is None:
            return
        painter.setBrush(QBrush(QColor(self.fill_color)))
        painter.setOpacity(self.opacity)
        if draw:
            painter.drawLine(self)
            painter.restore()

    def drawAndFill(self, painter):
        painter.save()
        self.draw(painter, draw=False)
        self.fill(painter, draw=False)
        painter.drawLine(self)
        painter.restore()

    def contains(self, *args):
        x = 0
        y = 0
        if isinstance(args[0], QPoint) or isinstance(args[0], QPointF):
            x = args[0].x()
            y = args[0].y()
        else:
            x = args[0]
            y = args[1]
        return self.distToSegment(QPointF(x, y), self.p1(), self.p2()) < 10 and self.boundingContains(*args)

    def lineFromPoints(self, x1, y1, x2, y2):
        a = y2 - y1
        b = x1 - x2
        c = a * x1 + b * y1
        return (a, b, c)

    def len(self, v, w):
        return (v.x() - w.x()) ** 2 + (v.y() - w.y()) ** 2

    def distToSegment(self, p, v, w):
        l2 = self.len(v, w)
        if l2 == 0:
            return self.len(p, v)
        t = ((p.x() - v.x()) * (w.x() - v.x()) + (p.y() - v.y()) * (w.y() - v.y())) / l2
        t = max(0, min(1, t))
        return sqrt(self.len(p, QPointF(v.x() + t * (w.x() - v.x()), v.y() + t * (w.y() - v.y()))))

    def boundingContains(self, *args):
        return self.boundingRect().contains(*args)

    def boundingRect(self):
        return QRectF(min(self.p1().x(), self.p2().x()), min(self.p1().y(), self.p2().y()),
                      abs(self.p2().x() - self.p1().x()), abs(self.p2().y() - self.p1().y()))

    def add(self, *args):
        point = args[1]
        self.setP2(point)
        self.isSet = True

if __name__ == '__main__':
    # ça marche --> voici deux examples de shapes
    test = Line2D(0, 0, 100, 100, arrow=True)

    print(test.lineFromPoints(0, 0, 100, 100))
    print(test.contains(0, 0))  # true
    print(test.contains(10, 10))  # true
    print(test.contains(-10, -10))  # false # on line with that equation but outside range
    print(test.contains(0, 18))  # false

    p1 = test.p1()
    print(p1.x(), p1.y())
    p2 = test.p2()
    print(p2.x(), p2.y())
    print(test.arrow)
    print(test.length())  # sqrt 2 --> 141
    # if it's an arrow I can add easily all the stuff I need

    test = Line2D(0, 0, 1, 1)
    p1 = test.p1()
    print(p1.x(), p1.y())
    p2 = test.p2()
    print(p2.x(), p2.y())
    print(test.arrow)
    import math

    print(test.length() == sqrt(2))  # sqrt 2

    test2 = Line2D()
    p1 = test2.p1()
    print(p1.x(), p1.y())
    p2 = test2.p2()
    print(p2.x(), p2.y())
    print(test2.arrow)
