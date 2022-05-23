from PyQt5 import QtCore
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QPainter, QBrush, QPen, QImage, QColor, QPolygonF
# from PyQt5.Qt
# from PyQt5.Qt import (QPaintEngine, QPaintDevice,  QTransform, QBrush)
from epyseg.draw.shapes.polygon2d import Polygon2D

from epyseg.tools.logger import TA_logger

logger = TA_logger()

class PolyLine2D(Polygon2D):

    def __init__(self, *args, color=0xFFFF00, opacity=1., stroke=0.65, line_style=None, theta=0, fill_color=None,  **kwargs):
        super(PolyLine2D, self).__init__()
        points = []
        if len(args) > 0:
            if isinstance(args[0],tuple):
                for i in range(0, len(args)):
                    self.append(QPointF(args[i][0], args[i][1]))
            else:
                for i in range(0, len(args), 2):
                    self.append(QPointF(args[i], args[i+1]))
        self.color = color
        self.fill_color = fill_color
        self.stroke = stroke
        self.opacity = opacity
        self.isSet = False
        self.line_style = line_style
        # rotation
        self.theta = theta

    def set_rotation(self, theta):
            self.theta = theta

    def set_opacity(self, opacity):
        self.opacity = opacity

    def set_line_style(self, style):
        '''allows lines to be dashed or dotted or have custom pattern

        :param style: a list of numbers or any of the following Qt.SolidLine, Qt.DashLine, Qt.DashDotLine, Qt.DotLine, Qt.DashDotDotLine but not Qt.CustomDashLine, Qt.CustomDashLine is assumed by default if a list is passed in. None is also a valid value that resets the line --> assume plain line
        :return:
        '''
        self.line_style = style
        # if style is a list then assume custom pattern otherwise apply solidline

    def draw(self, painter, draw=True):
        if self.color is None: # and self.fill_color is None
            return

        if draw:
            painter.save()
            pen = QPen(QColor(self.color))
            if self.stroke is not None:
                pen.setWidthF(self.stroke)
            if self.line_style is not None:
                if self.line_style in [Qt.SolidLine, Qt.DashLine, Qt.DashDotLine, Qt.DotLine, Qt.DashDotDotLine]:
                    pen.setStyle(self.line_style)
                elif isinstance(self.line_style, list):
                    pen.setStyle(Qt.CustomDashLine)
                    pen.setDashPattern(self.line_style)
            if self.color is not None:
                painter.setPen(pen)
            else:
                painter.setPen(Qt.NoPen)
            if self.fill_color is not None:
                # print('coloring')
                brush = QBrush(QColor(self.fill_color))
                painter.setBrush(brush)
            else:
                painter.setBrush(Qt.NoBrush)

            painter.setOpacity(self.opacity)
            polyline_to_draw = self.translated(0, 0)
            if self.scale is not None and self.scale != 1:
                polyline_to_draw = self.__scaled()
            # print('mid rect_to_plot', rect_to_plot)
            if self.translation is not None:
                # rect_to_plot.setX(rect_to_plot.x()+self.translation.x())
                # rect_to_plot.setY(rect_to_plot.y()+self.translation.y())
                polyline_to_draw.translate(self.translation.x(), self.translation.y())
            # painter.drawPolygon(polygon_to_draw)
            if self.theta is not None and self.theta != 0:
                painter.translate(polyline_to_draw.boundingRect().center())
                painter.rotate(self.theta)
                painter.translate(-polyline_to_draw.boundingRect().center())

            # painter.drawPolygon(polyline_to_draw)
            painter.drawPolyline(polyline_to_draw)
            painter.restore()
    #
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
    #
    # def setP1(self, point):
    #     self.append(point)

    def add(self, *args, force=True):
        if self.count() > 1:
            self.remove(self.count()-1)
        self.append(args[1])
        self.isSet = True

    def listVertices(self):
        return [point for point in self]

    def __scaled(self):
        vertices = self.listVertices()
        scaled_poly = QPolygonF()
        for vx in vertices:
            vx.setX(vx.x()*self.scale)
            vx.setY(vx.y()*self.scale)
            scaled_poly.append(vx)
        return scaled_poly

if __name__ == '__main__':
    test = PolyLine2D(0, 0, 10, 0, 10, 20, 0, 20, 0, 0)
    print(test.count()) # marche pas --> pas ajouté
    print(test)

    hexagon = PolyLine2D()
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
    #
    # image = QImage('./../data/handCorrection.png')
    # painter = QPainter()
    # painter.begin(image)
    # # painter.setOpacity(0.3);
    # painter.drawImage(0, 0, image)
    # painter.setPen(QtCore.Qt.blue)
    # painter.drawPolygon(hexagon)
    # hexagon.opacity = 0.7
    # painter.translate(10, 20)
    # hexagon.draw(painter) # ça marche pourrait overloader ça avec du svg
    # painter.end()
    #
    # # painter.save()
    # # painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
    # # painter.eraseRect(r)
    # # painter.restore()
    #
    # image.save('./../trash/test_pyQT_draw.png', "PNG");
    #
    # #pas mal TODO faire une classe drawsmthg qui dessine n'importe quelle forme que l'on lui passe avec des parametres de couleur, transparence, ...

    # tt marche aps mal ça va très vite

