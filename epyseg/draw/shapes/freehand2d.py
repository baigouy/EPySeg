from PyQt5 import QtCore
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPainter, QBrush, QPen, QImage, QColor
from epyseg.draw.shapes.polygon2d import Polygon2D
# from PyQt5.Qt
# from PyQt5.Qt import (QPaintEngine, QPaintDevice,  QTransform, QBrush)
from epyseg.draw.shapes.polyline2d import PolyLine2D

from epyseg.tools.logger import TA_logger
logger = TA_logger()

class Freehand2D(PolyLine2D):

    def __init__(self, *args, color=0xFFFF00, opacity=1., stroke=0.65, line_style=None, theta=0, **kwargs):
        super(Freehand2D, self).__init__()
        self.isSet = False
        if len(args) > 0:
            for i in range(0, len(args), 2):
                self.append(QPointF(args[i], args[i+1]))
            self.isSet = True
        self.color = color
        self.fill_color = None
        self.stroke = stroke
        self.opacity = opacity
        self.line_style = line_style
        # rotation
        self.theta = theta

# it's simply a polygon so draw it as such

    # def draw(self, painter, draw=True):
    #     if draw:
    #         painter.save()
    #     pen = QPen(QColor(self.color))
    #     if self.stroke is not None:
    #         pen.setWidthF(self.stroke)
    #     painter.setPen(pen)
    #     painter.setOpacity(self.opacity)
    #     if draw:
    #         painter.drawPolyline(self)
    #         painter.restore()

    # def fill(self, painter, draw=True):
    #     if self.fill_color is None:
    #         return
    #     if draw:
    #         painter.save()
    #     painter.setBrush(QBrush(QColor(self.fill_color)))
    #     painter.setOpacity(self.opacity)
    #     if draw:
    #         painter.drawPolyline(self)
    #         painter.restore()
    #
    # # TODO pb will draw the shape twice.... ---> because painter drawpolygon is called twice
    # def drawAndFill(self, painter):
    #     painter.save()
    #     self.draw(painter, draw=False)
    #     self.fill(painter, draw=False)
    #     painter.drawPolyline(self)
    #     painter.restore()

    def setP1(self, point):
        self.append(point)

    def add(self, *args):
        self.append(args[1])
        self.isSet=True

if __name__ == '__main__':
    test = Freehand2D(0, 0, 10, 0, 10, 20, 0, 20, 0, 0)
    print(test.count()) # marche pas --> pas ajouté
    print(test)

    hexagon = Freehand2D()
    print(hexagon)
    hexagon.append(QPointF(10, 20))
    hexagon.append(QPointF(10, 30))
    hexagon.append(QPointF(20, 30))

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
    print(hexagon.contains(QPointF(10, 20)))
    print(hexagon.contains(QPointF(10, 21)))

    print(hexagon.isClosed()) #
    hexagon.append(QPointF(10, 20)) # closing the hexagon --> the last and first point should be the same
    print(hexagon.isClosed())  #

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

