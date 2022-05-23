from PyQt5.QtCore import QPointF, QLineF, QRectF, QPoint, Qt
from PyQt5.QtGui import QBrush, QPen, QColor, QTransform
from math import sqrt

from epyseg.tools.logger import TA_logger

logger = TA_logger()

class Line2D(QLineF):

    def __init__(self, *args, color=0xFFFF00, opacity=1., stroke=0.65, arrow=False, line_style=None, theta=0, **kwargs):
        super(Line2D, self).__init__(*args)
        if not args:
            self.isSet = False
        else:
            self.isSet = True
        self.arrow = arrow
        self.color = color
        self.stroke = stroke
        self.opacity = opacity
        self.scale = 1
        self.translation = QPointF()
        self.line_style = line_style
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

    def draw(self, painter, draw=True):
        if self.color is None:
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
        painter.setPen(pen)
        painter.setOpacity(self.opacity)
        if draw:
            # clone the line
            line_to_plot = self.translated(0, 0)
            if self.scale is not None and self.scale != 1:
                p1 = line_to_plot.p1()
                p2 = line_to_plot.p2()
                line_to_plot.setP1(QPointF(p1.x()*self.scale, p1.y()*self.scale))
                line_to_plot.setP2(QPointF(p2.x()*self.scale, p2.y()*self.scale))
            if self.translation is not None:
                line_to_plot.translate(self.translation)
            # print(line_to_plot)
            if self.theta is not None and self.theta != 0:
                painter.translate(line_to_plot.center())
                painter.rotate(self.theta)
                painter.translate(-line_to_plot.center())

            painter.drawLine(line_to_plot)
            painter.restore()
    #
    # def fill(self, painter, draw=True):
    #     if draw:
    #         painter.save()
    #     if self.fill_color is None:
    #         return
    #     painter.setBrush(QBrush(QColor(self.fill_color)))
    #     painter.setOpacity(self.opacity)
    #     if draw:
    #         painter.drawLine(self)
    #         painter.restore()
    #
    # def drawAndFill(self, painter):
    #     painter.save()
    #     self.draw(painter, draw=False)
    #     self.fill(painter, draw=False)
    #     painter.drawLine(self)
    #     painter.restore()

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

    # def boundingRect(self, scaled=True):
    #     scale = 1
    #     if not scaled and self.scale is not None:
    #         scale = self.scale
    #     return QRectF(min(self.p1().x(), self.p2().x()) * scale, min(self.p1().y(), self.p2().y()) * scale,
    #                   abs(self.p2().x() - self.p1().x()) * scale, abs(self.p2().y() - self.p1().y()) * scale)
    # TODO handle scale etc
    def boundingRect(self):
        rect = QRectF(min(self.p1().x(), self.p2().x()), min(self.p1().y(), self.p2().y()),
               abs(self.p2().x() - self.p1().x()), abs(self.p2().y() - self.p1().y()))

        try:
            # print('tada')
            if self.theta is not None and self.theta != 0:
                # print('entering')
                center = rect.center()
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
                transformed = t.map( self)  #// mapRect() returns the bounding rect of the rotated rect

                # print('rotated',rotatedRect )
                # return rotatedRect
                return QRectF(min(transformed.p1().x(), transformed.p2().x()),
                              min(transformed.p1().y(), transformed.p2().y()),
                abs(transformed.p2().x() - transformed.p1().x()), abs(transformed.p2().y() - transformed.p1().y()))
        except:
            pass
        return rect

    def add(self, *args):
        point = args[1]
        self.setP2(point)
        self.isSet = True

    def set_to_scale(self, factor):
        self.scale = factor

    def set_to_translation(self, translation):
        self.translation = translation

    def get_P1(self):
        return self.boundingRect().topLeft()

    # faut pas utiliser ça sinon pbs --> car en fait ce que je veux c'est postionned le point et pas le setter

    def set_P1(self, point):
        current_pos = self.boundingRect().topLeft()
        self.translate(point.x() - current_pos.x(), point.y() - current_pos.y())
        # self.translate(self.translation)
        # if not args:
        #     logger.error("no coordinate set...")
        #     return
        # if len(args) == 1:
        #     self.setP1(args[0])
        # else:
        #     self.setP1(QPointF(args[0], args[1]))

    # def set_P2(self,*args):
    #     if not args:
    #         logger.error("no coordinate set...")
    #         return
    #     if len(args) == 1:
    #         self.setP2(args[0])
    #     else:
    #         self.setP2(QPointF(args[0], args[1]))

    def erode(self, nb_erosion=1):
        self.__computeNewMorphology(sizeChange=-nb_erosion)

    def dilate(self, nb_dilation=1):
        self.__computeNewMorphology(sizeChange=nb_dilation)

    def __computeNewMorphology(self, sizeChange=1):
        currentBoundingRect = self.boundingRect()
        curWidth = currentBoundingRect.width()
        finalWitdth = curWidth + 2. * sizeChange

        if (finalWitdth < 1):
            finalWitdth = 1

        center2D = QPointF(currentBoundingRect.center().x(), currentBoundingRect.center().y())

        scale = finalWitdth / self.boundingRect(scaled=False).width()# divide by original width

        print('new scale', scale)
        self.set_to_scale(scale)

        # need translate according to center otherwise ok

        # self.setCenter(center2D)

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

    # TODO add preview as an image to gain time --> TODO
