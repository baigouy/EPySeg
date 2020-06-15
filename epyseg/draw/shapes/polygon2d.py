from PyQt5 import QtCore
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPolygonF
from PyQt5.QtGui import QPainter, QBrush, QPen, QImage, QColor
# from PyQt5.Qt
# from PyQt5.Qt import (QPaintEngine, QPaintDevice,  QTransform, QBrush)

from epyseg.tools.logger import TA_logger
logger = TA_logger()

class Polygon2D(QPolygonF):


    def __init__(self, *args, color=0xFFFF00, fill_color=None, opacity=1., stroke=0.65, **kwargs):
        super(Polygon2D, self).__init__()

        # print(args, len(args))
        if len(args) > 0:
            for i in range(0, len(args), 2):
                self.append(QPointF(args[i], args[i+1]))
                # print(QPointF(args[i], args[i+1]))

        self.color = color
        self.fill_color = fill_color
        self.stroke = stroke
        self.opacity = opacity
        # self.isSet = False

    def get_color(self):
        return self.color

    def get_fill_color(self):
        return self.fill_color

    def get_stroke_size(self):
        return self.stroke

    def get_points(self):
        points = []
        for point in self:
            points.append((point.x(), point.y()))
        return points

    def contains(self, *args):
        return self.containsPoint(*args, 0)

    def draw(self, painter, draw=True):
        if draw:
            painter.save()
        pen = QPen(QColor(self.color))
        if self.stroke is not None:
            pen.setWidthF(self.stroke)
        painter.setPen(pen)
        painter.setOpacity(self.opacity)
        if draw:
            painter.drawPolygon(self)
            painter.restore()

    def fill(self, painter, draw=True):
        if self.fill_color is None:
            return
        if draw:
            painter.save()
        painter.setBrush(QBrush(QColor(self.fill_color)))
        painter.setOpacity(self.opacity)
        if draw:
            painter.drawPolygon(self)
            painter.restore()

    def drawAndFill(self, painter):
        painter.save()
        self.draw(painter, draw=False)
        self.fill(painter, draw=False)
        painter.drawPolygon(self)
        painter.restore()

    def setP1(self, point):
        self.append(point)

    def add(self, *args, force=True):
        if self.count() > 1:
            self.remove(self.count()-1)
        self.append(args[1])
        self.isSet = True

    def listVertices(self):
        return [point for point in self]

if __name__ == '__main__':

    test = Polygon2D(0, 0, 10, 0, 10, 20, 0, 20, 0, 0)
    print(test.count()) # marche pas --> pas ajouté
    print(test)

    hexagon = Polygon2D()
    print(hexagon)
    hexagon.append(QPointF(10, 20))
    hexagon.append(QPointF(10, 30))
    hexagon.append(QPointF(20, 30))



    # hexagon.append(QPointF(10, 20))
    print(hexagon)
    print(hexagon.isEmpty())
    print(hexagon.count())
    print(hexagon.stroke)
    print(hexagon.get_stroke_size())
    # ça marche maintenant
    # print(hexagon.)
    # trop cool l'acces aux points
    for point in hexagon:
        print(point)

    print(hexagon.get_points())
    # print(hexagon.contains(10, 20))
    print(hexagon.contains(QPointF(10, 20)))
    print(hexagon.contains(QPointF(10, 21)))

    print(hexagon.isClosed()) #
    hexagon.append(QPointF(10, 20)) # closing the hexagon --> the last and first point should be the same
    print(hexagon.isClosed())  #

    # print(hexagon.translate(10, 20)) # why none ???
    # translate and so on can all be saved...

    image = QImage('./../data/handCorrection.png')
    painter = QPainter()
    painter.begin(image)
    # painter.setOpacity(0.3);
    painter.drawImage(0, 0, image)
    painter.setPen(QtCore.Qt.blue)
    painter.drawPolygon(hexagon)
    hexagon.opacity = 0.7
    painter.translate(10, 20)
    hexagon.draw(painter) # ça marche pourrait overloader ça avec du svg
    painter.end()

    # painter.save()
    # painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
    # painter.eraseRect(r)
    # painter.restore()

    image.save('./../trash/test_pyQT_draw.png', "PNG");

    #pas mal TODO faire une classe drawsmthg qui dessine n'importe quelle forme que l'on lui passe avec des parametres de couleur, transparence, ...

    # tt marche aps mal ça va très vite

