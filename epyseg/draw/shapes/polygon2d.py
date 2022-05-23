from PyQt5 import QtCore
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QPolygonF, QTransform
from PyQt5.QtGui import QPainter, QBrush, QPen, QImage, QColor
# from PyQt5.Qt
# from PyQt5.Qt import (QPaintEngine, QPaintDevice,  QTransform, QBrush)

from epyseg.tools.logger import TA_logger
logger = TA_logger()

class Polygon2D(QPolygonF):


    def __init__(self, *args, color=0xFFFF00, fill_color=None, opacity=1., stroke=0.65, line_style=None, theta=0, **kwargs):
        super(Polygon2D, self).__init__()
        self.isSet = False
        # print(args, len(args))
        if len(args) > 0:
            for i in range(0, len(args), 2):
                self.append(QPointF(args[i], args[i+1]))
            self.isSet = True
                # print(QPointF(args[i], args[i+1]))

        self.color = color
        self.fill_color = fill_color
        self.stroke = stroke
        self.opacity = opacity
        self.scale = 1
        self.translation = QPointF()
        self.line_style = line_style
        # self.isSet = False
        # rotation
        self.theta = theta

    def set_rotation(self, theta):
        self.theta = theta

    def set_opacity(self, opacity):
        self.opacity = opacity

    def set_line_style(self,style):
        '''allows lines to be dashed or dotted or have custom pattern

        :param style: a list of numbers or any of the following Qt.SolidLine, Qt.DashLine, Qt.DashDotLine, Qt.DotLine, Qt.DashDotDotLine but not Qt.CustomDashLine, Qt.CustomDashLine is assumed by default if a list is passed in. None is also a valid value that resets the line --> assume plain line
        :return:
        '''
        self.line_style = style
        # if style is a list then assume custom pattern otherwise apply solidline

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
        if self.color is None and self.fill_color is None:
            return

        if draw:
            painter.save()
            painter.setOpacity(self.opacity)
            if self.color is not None:
                pen = QPen(QColor(self.color))
                if self.stroke is not None:
                    pen.setWidthF(self.stroke)
                if self.line_style is not None:
                    if self.line_style in [Qt.SolidLine, Qt.DashLine, Qt.DashDotLine, Qt.DotLine, Qt.DashDotDotLine]:
                        pen.setStyle(self.line_style)
                    elif isinstance(self.line_style, list):
                        pen.setStyle(Qt.CustomDashLine)
                        pen.setDashPattern(self.line_style)
                painter.setPen(pen)
            else:
                painter.setPen(Qt.NoPen)  # required to draw something filled without a border
            if self.fill_color is not None:
                painter.setBrush(QBrush(QColor(self.fill_color)))
            else:
                painter.setBrush(Qt.NoBrush)
            polygon_to_draw = self.translated(0, 0)
            if self.scale is not None and self.scale != 1:
                polygon_to_draw = self.__scaled()

            # print('mid rect_to_plot', rect_to_plot)
            if self.translation is not None:
                # rect_to_plot.setX(rect_to_plot.x()+self.translation.x())
                # rect_to_plot.setY(rect_to_plot.y()+self.translation.y())
                polygon_to_draw.translate(self.translation.x(), self.translation.y())

            if self.theta is not None and self.theta != 0:
                painter.translate(polygon_to_draw.center())
                painter.rotate(self.theta)
                painter.translate(-polygon_to_draw.center())

            painter.drawPolygon(polygon_to_draw)
            painter.restore()

    # def fill(self, painter, draw=True):
    #     if self.fill_color is None:
    #         return
    #     if draw:
    #         painter.save()
    #     painter.setBrush(QBrush(QColor(self.fill_color)))
    #     painter.setOpacity(self.opacity)
    #     if draw:
    #         painter.drawPolygon(self)
    #         painter.restore()
    #
    # def drawAndFill(self, painter):
    #     painter.save()
    #     self.draw(painter, draw=False)
    #     self.fill(painter, draw=False)
    #     painter.drawPolygon(self)
    #     painter.restore()

    # def setP1(self, point):
    #     self.append(point)
    def get_P1(self):
        return self.boundingRect().topLeft()

    def set_P1(self, point):# see how to do that
        #self.append(point)
        # get bounding box and set top left to P1
        current_pos = self.boundingRect().topLeft()
        self.translate(point.x()-current_pos.x(), point.y()-current_pos.y())

    def add(self, *args, force=False):
        if self.count() > 1 and not force:
            self.remove(self.count()-1)
        self.append(args[1])
        self.isSet = True

    def listVertices(self):
        return [point for point in self]

    def set_to_scale(self, factor):
        self.scale = factor

    def set_to_translation(self, translation):
        self.translation = translation

    def __scaled(self):
        vertices = self.listVertices()
        scaled_poly = QPolygonF()
        for vx in vertices:
            vx.setX(vx.x()*self.scale)
            vx.setY(vx.y()*self.scale)
            scaled_poly.append(vx)
        return scaled_poly

    def boundingRect(self):
        polygon_to_draw = self.translated(0, 0)
        try:
            # print('tada')
            if self.theta is not None and self.theta != 0:
                # print('entering')
                center = polygon_to_draw.boundingRect().center()
                # print('entering2')
                t = QTransform().translate(center.x(), center.y()).rotate(self.theta).translate(-center.x(),
                                                                                                -center.y())
                #  print('entering3')
                #  transformed = self.setTransform(t)
                #  print('entering4')
                #  print(transformed)
                #  print(QRectF(min(transformed.p1().x(), transformed.p2().x()), min(transformed.p1().y(), transformed.p2().y()),
                # abs(transformed.p2().x() - transformed.p1().x()), abs(transformed.p2().y() - transformed.p1().y())))
                #  return QRectF(min(transformed.p1().x(), transformed.p2().x()), min(transformed.p1().y(), transformed.p2().y()),
                # abs(transformed.p2().x() - transformed.p1().x()), abs(transformed.p2().y() - transformed.p1().y()))

                # copy.setT
                # print('entering')

                # t = QTransform().translate( center.x(), center.y()).rotate(self.theta).translate(-center.x(), -center.y())
                # # print('entersd')
                transformed = t.map(polygon_to_draw)  # // mapRect() returns the bounding rect of the rotated rect

                # print('rotated',rotatedRect )
                # return rotatedRect
                return transformed.boundingRect()
        except:
            pass
        return polygon_to_draw.boundingRect()

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

