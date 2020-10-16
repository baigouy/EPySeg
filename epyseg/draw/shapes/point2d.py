from math import sqrt

from PyQt5.QtCore import QRectF

from epyseg.draw.shapes.circle2d import *
from epyseg.tools.logger import TA_logger
logger = TA_logger()

class Point2D(QPointF):

    def __init__(self, *args, color=0xFFFF00, fill_color=None, opacity=1., stroke=0.65, line_style=None, **kwargs):
        self.isSet = True
        if len(args)==2:
            self.size = 5
            if stroke is not None and stroke > 2:
                self.size = stroke
            #TODO need fix size
            super(Point2D, self).__init__(*args)
        else:
            self.size = 5
            super(Point2D, self).__init__(*args) # create an empty point for drawing
        self.color = color
        self.fill_color = fill_color
        self.stroke = stroke
        self.opacity = opacity
        self.scale = 1
        self.translation = QPointF()
        self.line_style = line_style

    def set_opacity(self, opacity):
        self.opacity = opacity

    def set_line_style(self,style):
        '''allows lines to be dashed or dotted or have custom pattern

        :param style: a list of numbers or any of the following Qt.SolidLine, Qt.DashLine, Qt.DashDotLine, Qt.DotLine, Qt.DashDotDotLine but not Qt.CustomDashLine, Qt.CustomDashLine is assumed by default if a list is passed in. None is also a valid value that resets the line --> assume plain line
        :return:
        '''
        self.line_style = style
        # if style is a list then assume custom pattern otherwise apply solidline

    def contains(self, *args):
      x=0
      y=0
      if isinstance(args[0], QPoint) or isinstance(args[0], QPointF):
          x = args[0].x()
          y = args[0].y()
      if sqrt((x-self.x())**2+(y-self.y())**2)<10:
          return True
      return False

    def translate(self, translation):
        self.setX(self.x() + translation.x())
        self.setY(self.y() + translation.y())

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
        if draw:
            point_to_draw = QPointF(self.x(), self.y())
            if self.scale is not None and self.scale != 1:
                point_to_draw.setX(point_to_draw.x()*self.scale)
                point_to_draw.setY(point_to_draw.y()*self.scale)
            if self.translation is not None:
                point_to_draw.setX(point_to_draw.x()+self.translation.x())
                point_to_draw.setY(point_to_draw.y()+self.translation.y())

            painter.drawEllipse(point_to_draw.x()-self.stroke/2., point_to_draw.y()-self.stroke/2, self.stroke, self.stroke)
            painter.restore()

    # def fill(self, painter, draw=True):
    #     if self.fill_color is None:
    #         return
    #     if draw:
    #         painter.save()
    #     painter.setBrush(QBrush(QColor(self.fill_color)))
    #     painter.setOpacity(self.opacity)
    #     if draw:
    #         painter.drawEllipse(self.x()-self.stroke/2., self.y()-self.stroke/2, self.stroke, self.stroke)
    #         painter.restore()
    #
    # def drawAndFill(self, painter):
    #     painter.save()
    #     self.draw(painter, draw=False)
    #     self.fill(painter, draw=False)
    #     size = max(self.size, self.stroke)
    #     painter.drawEllipse(self.x()-size/2., self.y()-size/2, size, size) # drawEllipse (x, y, w, h)
    #     painter.restore()

    def boundingRect(self):
        return QRectF(self.x()-self.stroke/2., self.y()-self.stroke/2, self.stroke, self.stroke)

    def add(self, *args):
        point = args[1]
        self.setX(point.x())
        self.setY(point.y())

    def set_P1(self, *args):
        if not args:
            logger.error("no coordinate set...")
            return
        if len(args) == 1:
            # self.moveTo(args[0].x(), args[0].y())
            self.setX(args[0].x())
            self.setY(args[0].y())
        else:
            # self.moveTo(QPointF(args[0], args[1]))
            self.setX(args[0])
            self.setY(args[1])
        # self.setX(point.x())
        # self.setY(point.y())

    def get_P1(self):
        return self

    def width(self):
        return 0

    def height(self):
        return 0

    def set_to_scale(self, factor):
        self.scale = factor

    def set_to_translation(self, translation):
        self.translation = translation

if __name__ == '__main__':
    # ça marche --> voici deux examples de shapes
    test = Point2D(128, 128)
    # print(test.x(), test.y(), test.width(), test.height())
    print(test.contains(QPointF(128, 128)))
    print(test.contains(QPointF(129, 129)))
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
