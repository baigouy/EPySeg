from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint, QPointF, Qt, QRectF
from PyQt5.QtGui import QBrush, QPen, QColor, QTransform

from epyseg.tools.logger import TA_logger

logger = TA_logger()

class Ellipse2D(QtWidgets.QGraphicsEllipseItem):

    isSet = False

    def __init__(self, *args, color=0xFFFF00, fill_color=None, opacity=1., stroke=0.65, line_style=None,theta=0, **kwargs):
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
            painter.setPen(Qt.NoPen)
        if self.fill_color is not None:
            painter.setBrush(QBrush(QColor(self.fill_color)))
        if draw:
            rect_to_plot = self.rect().adjusted(0, 0, 0, 0)
            if self.scale is not None and self.scale != 1:
                # TODO KEEP THE ORDER THIS MUST BE DONE THIS WAY OR IT WILL GENERATE PLENTY OF BUGS...
                new_width = rect_to_plot.width() * self.scale
                new_height = rect_to_plot.height() * self.scale
                # TODO BE EXTREMELY CAREFUL AS SETX AND SETY CAN CHANGE WIDTH AND HEIGHT --> ALWAYS TAKE SIZE BEFORE OTHERWISE THERE WILL BE A PB AND ALWAYS RESET THE SIZE WHEN SETX IS CALLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # Sets the left edge of the rectangle to the given x coordinate. May change the width, but will never change the right edge of the rectangle. --> NO CLUE WHY SHOULD CHANGE WIDTH THOUGH BUT BE CAREFUL!!!
                rect_to_plot.setX(rect_to_plot.x() * self.scale)
                rect_to_plot.setY(rect_to_plot.y() * self.scale)
                rect_to_plot.setWidth(new_width)
                rect_to_plot.setHeight(new_height)
            if self.translation is not None:
                rect_to_plot.translate(self.translation)

            # if self.color is not None:
            #     painter.drawRect(rect_to_plot)
            # else:
            #     painter.fillRect(rect_to_plot, QColor(self.fill_color))
            if self.theta is not None and self.theta != 0:
                painter.translate(rect_to_plot.center())
                painter.rotate(self.theta)
                painter.translate(-rect_to_plot.center())

            painter.drawEllipse(rect_to_plot)
            painter.restore()




    # def fill(self, painter, draw=True):
    #     if self.fill_color is None:
    #         return
    #     if draw:
    #         painter.save()
    #     painter.setBrush(QBrush(QColor(self.fill_color)))
    #     painter.setOpacity(self.opacity)
    #     if draw:
    #         painter.drawEllipse(self.rect())
    #         painter.restore()
    #
    #     # TODO pb will draw the shape twice.... ---> because painter drawpolygon is called twice
    #
    # def drawAndFill(self, painter):
    #     painter.save()
    #     self.draw(painter, draw=False)
    #     self.fill(painter, draw=False)
    #     painter.drawEllipse(self.rect())
    #     painter.restore()

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

    def boundingRect(self):
        # should I return the scaled version or the orig --> think about it...
        rect_to_plot = self.rect().adjusted(0, 0, 0, 0)
        try:
            # print('tada')
            if self.theta is not None and self.theta!=0:
                # print('entering')
                center = rect_to_plot.center()
                # print('entering2')
                t = QTransform().translate(center.x(), center.y()).rotate(self.theta).translate(-center.x(),
                                                                                                -center.y())
                # print('entering3')
                # self.setTransform(t)
                # self.transform()


                # transformed = QRectF(self.rect())
                # print('entering5', transformed)
                # self.resetTransform()
                # print('entering5', rect_to_plot)
                # print('entering4')
                transformed = t.mapRect(rect_to_plot)

                # self.setTransform(t)
                # self.transform()
                #
                # print(self.shape().boundingRect())
                #
                # # print(self.rect(), transformed)
                #
                # transformed = QRectF(self.shape().boundingRect())
                # # self.resetTransform()

                # not perfect but ok for now though --> bounds are not sharp at the edges upon rotation
                # print('entering45', transformed)
                return transformed
        except:
            pass
        return rect_to_plot

    def get_P1(self):
        return self.boundingRect().topLeft()

    def set_P1(self, point):
        rect = self.rect()
        width = rect.width()
        height = rect.height()
        rect.setX(point.x())
        rect.setY(point.y())
        # required due to setX changing width and sety changing height
        rect.setWidth(width)
        rect.setHeight(height)
        self.setRect(rect)

    def set_to_scale(self, factor):
        self.scale = factor

    def set_to_translation(self, translation):
        self.translation = translation


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
