from PyQt5.QtCore import QPointF

from epyseg.draw.shapes.rect2d import Rect2D

from epyseg.tools.logger import TA_logger

logger = TA_logger()

class Square2D(Rect2D):

    def __init__(self, *args, color=0xFFFF00, fill_color=None, opacity=1., stroke=0.65, line_style=None, theta=0, **kwargs):
        if len(args) == 3:
            super(Square2D, self).__init__(*args, args[-1])
        elif len(args) == 4:
            logger.error("too many values, square pnly has, x,y and width")
        else:
            super(Square2D, self).__init__(*args)
        self.color = color
        self.fill_color = fill_color
        self.stroke = stroke
        self.opacity = opacity
        self.line_style = line_style
         # rotation
        self.theta = theta

    def add(self, *args):
            p1 = args[0]
            p2 = args[1]

            x = p2.x()
            y = p2.y()
            x2 = p1.x()
            y2 = p1.y()
            if p1.x() < p2.x():
                x = p1.x()
                x2 = p2.x()
            if p1.y() < p2.y():
                y = p1.y()
                y2 = p2.y()
            w = abs(x - x2)
            h = abs(y - y2)
            self.setX(x)
            self.setY(y)
            if w < h:
                self.setWidth(h)
                self.setHeight(h)
            else:
                self.setWidth(w)
                self.setHeight(w)
            self.isSet = True

if __name__ == '__main__':
    # ça marche --> voici deux examples de shapes
    test = Square2D(0, 0, 200)
    print(test.x(), test.y(), test.width(), test.height())
    print(test.contains(QPointF(50, 50)))
    print(test.contains(QPointF(-1, -1)))
    print(test.contains(QPointF(0, 0)))
    print(test.contains(QPointF(100, 100)))
    print(test.contains(QPointF(100, 100.1)))

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
