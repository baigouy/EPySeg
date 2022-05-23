# TODO pen.setJoinStyle(Qt.MiterJoin)# RoundJoin #BevelJoin # some day offer the handling of that but later
# TODO offer shear and scale at the level of the stuff
# Offer set to with also
# si double click sur une image --> edit it # offer drawing of shapes etc

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QPen, QColor, QTransform

from epyseg.tools.logger import TA_logger

logger = TA_logger()

class Rect2D(QRectF):

    def __init__(self, *args, color=0xFFFF00, fill_color=None, opacity=1., stroke=0.65, line_style=None, theta=0, **kwargs):
        super(Rect2D, self).__init__(*args)
        if not args:
            self.isSet = False
        else:
            self.isSet = True
        self.color = color
        self.fill_color = fill_color
        self.stroke = stroke
        self.opacity = opacity
        self.isSet = False
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


    def draw(self, painter, **kwargs):
        # print(kwargs)
        # if kwargs is not None and kwargs:
        #     pass
        # else:
        #     return

        if self.color is None and self.fill_color is None:
            return

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
            pen.setJoinStyle(Qt.MiterJoin)# RoundJoin #BevelJoin # some day offer the handling of that but later
            # if kwargs['draw']==True:

            # pen.setStyle(Qt.DashLine)
            # pen.setStyle(Qt.CustomDashLine)
            # pen.setDashPattern([1, 4, 5, 4])

            painter.setPen(pen)
        # if kwargs['fill']==True:
        else:
            painter.setPen(Qt.NoPen) # required to draw something filled without a border

        # needed to have a filling color in a shape without contour
        if self.fill_color is not None:
            painter.setBrush(QBrush(QColor(self.fill_color)))

        rect_to_plot = self.adjusted(0,0,0,0)
        # print('begin rect_to_plot', rect_to_plot, self.scale)
        # if kwargs['draw']==True or kwargs['fill']==True:
        # if self.scale is None or self.scale==1:
        #     painter.drawRect(self)
        # else:
            # on clone le rect
        if self.scale is not None and self.scale != 1:
            # TODO KEEP THE ORDER THIS MUST BE DONE THIS WAY OR IT WILL GENERATE PLENTY OF BUGS...
            new_width = rect_to_plot.width() * self.scale
            new_height = rect_to_plot.height() * self.scale
            # print(rect_to_plot.width(), rect_to_plot.height())  # here ok
            # setX changes width --> why is that

            # TODO BE EXTREMELY CAREFUL AS SETX AND SETY CAN CHANGE WIDTH AND HEIGHT --> ALWAYS TAKE SIZE BEFORE OTHERWISE THERE WILL BE A PB AND ALWAYS RESET THE SIZE WHEN SETX IS CALLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Sets the left edge of the rectangle to the given x coordinate. May change the width, but will never change the right edge of the rectangle. --> NO CLUE WHY SHOULD CHANGE WIDTH THOUGH BUT BE CAREFUL!!!
            rect_to_plot.setX(rect_to_plot.x()*self.scale)
            rect_to_plot.setY(rect_to_plot.y()*self.scale)
            # maybe to avoid bugs I should use translate instead rather that set x but ok anyways
            # print(rect_to_plot.width(), rect_to_plot.height())# bug here --> too big

            # print(new_height, new_height, self.width(), self.scale, self.scale* self.width())
            rect_to_plot.setWidth(new_width)
            rect_to_plot.setHeight(new_height)

        # print('mid rect_to_plot', rect_to_plot)
        if self.translation is not None:

            # rect_to_plot.setX(rect_to_plot.x()+self.translation.x())
            # rect_to_plot.setY(rect_to_plot.y()+self.translation.y())
            rect_to_plot.translate(self.translation)

        # print('rect to plot', rect_to_plot)

        # if self.color is not None:

        # rotation marche mais position est pas du tout bonne --> voir comment gérer ça ??
        # center = rect_to_plot.center()
        # painter.translate(center.x(), center.y())
        # painter.rotate(45)
        # rct = QRectF(-center.x(), -center.y(), rect_to_plot.width(), rect_to_plot.height())
        # painter.drawRect(rct)


        if self.theta is not None and self.theta != 0:
            # this is only if a rotation should be applied
            painter.translate(rect_to_plot.center())
            # print('center', rect_to_plot.center())
            # painter.setWorldTransform()
            # print(painter.combinedTransform().)



            # TODO use this maybe to get bounds of rotated shape properly...
            # center = rect_to_plot.center()
            # t = QTransform().translate( center.x(), center-y() ).rotate( angle ).translate( -center.x(), -center.y() )
            # rotatedRect =  t.mapToPolygon( rect_to_plot )  #// mapRect() returns the bounding rect of the rotated rect

            # on dirait que c'est ça faut faire deux translations en x

            painter.rotate(self.theta)

            # ça marche faut deux translations

            # painter.translate(rect_to_plot.width() / 2, -rect_to_plot.height() )


            #see https://www.qtcentre.org/threads/15540-Rotating-a-rectangle-about-its-origin --> maybe ok

            # difference = old center -
            # old_center = self.boundingRect().center()
            # trans = painter.combinedTransform()
            # transformed = trans.mapRect(rect_to_plot)

            # peut etre une piste mais a pas l'air de marcher
            # print(self.boundingRect(), transformed)
            # trans = transformed.center()-self.boundingRect().center()


            # width_dif = transformed.width()-self.boundingRect().width()
            # width_dif/=2
            # print('trans', trans, transformed.center(), self.boundingRect().center(), width_dif)

            # painter.translate(-trans)
            # only if rotation
            painter.translate(-rect_to_plot.center())
        painter.drawRect(rect_to_plot)
        # else:
        #     painter.fillRect(rect_to_plot, QColor(self.fill_color))

        painter.restore()
    #
    # def fill(self, painter, **kwargs):
    #     return self.draw(painter, **kwargs)
    #     if self.fill_color is None:
    #         return
    #     if draw:
    #         painter.save()
    #
    #     painter.setOpacity(self.opacity)
    #     if draw:
    #         painter.drawRect(self)
    #         painter.restore()
    #
    # def drawAndFill(self, painter,**kwargs):
    #     return self.draw(painter,**kwargs)
    #     painter.save()
    #     self.draw(painter, draw=False)
    #     self.fill(painter, draw=False)
    #     painter.drawRect(self)
    #     painter.restore()

    # TODO modify that to handle rotation etc...
    def boundingRect(self, scaled=True):
        # should I return the scaled version or the orig --> think about it...
        if scaled:
            # in fact scale only handles rotation and not scale --> the size is therefore wrong -->
            try:
                if self.theta is not None and self.theta!=0:
                    # print('entering')
                    center = self.center()
                    t = QTransform().translate( center.x(), center.y()).rotate(self.theta).translate(-center.x(), -center.y())
                    # print('entersd')
                    rotatedRect = t.mapRect( self )  #// mapRect() returns the bounding rect of the rotated rect
                    # print('rotated',rotatedRect )
                    return rotatedRect
            except:
                pass
        return self

    def add(self, *args):
        point = args[1]
        self.setWidth(point.x()-self.x())
        self.setHeight(point.y()-self.y())
        self.isSet = True

    def set_P1(self, *args):
        if not args:
            logger.error("no coordinate set...")
            return
        if len(args) == 1:
            self.moveTo(args[0].x(), args[0].y())
        else:
            self.moveTo(QPointF(args[0], args[1]))

    def get_P1(self):
        return QPointF(self.x(), self.y())

    def set_to_scale(self, factor):
        self.scale = factor

    def set_to_translation(self, translation):
        self.translation = translation

    #TODO rather use __computeNewMorphology --> as it is more likely to work and not damage the original shape in any way but need a functional scaling
    # def erode(self, nb_erosion=1):
    #     x = self.x()
    #     y = self.y()
    #     width = self.width()
    #     height = self.height()
    #     x += nb_erosion
    #     y += nb_erosion
    #
    #     width -= nb_erosion * 2
    #     height -= nb_erosion * 2
    #     if (width < 1):
    #         width = 1
    #         # x = rec2d.getCenterX() - 0.5
    #
    #     if (height < 1):
    #         height = 1;
    #         # y = rec2d.getCenterY() - 0.5
    #
    #     self.setX(x)
    #     self.setY(y)
    #     self.setWidth(width)
    #     self.setHeight(height)
    #
    # def dilate(self, nb_dilation=1):
    #     self.erode(nb_erosion=-nb_dilation)


    # hack to make column hashable to be able to add it to a dict, see https://stackoverflow.com/questions/10994229/how-to-make-an-object-properly-hashable
    def __eq__(self, other):
        try:
            return hash(str(self)) == other.__hash__()
        except:
            return False

    # hack to make column hashable to be able to add it to a dict, see https://stackoverflow.com/questions/10994229/how-to-make-an-object-properly-hashable
    def __hash__(self):
        return hash(str(self))

if __name__ == '__main__':
    # ça marche --> voici deux examples de shapes
    test = Rect2D(0, 0, 100, 100)
    print(test)

    rect = QRectF(0,0, 125,256)
    print(rect.x())
    print(rect.y())
    print(rect.width())
    print(rect.height())

    rect.translate(10,20) # ça marche
    print(rect)

    print(test)
    test.set_to_scale(0.5) #scale is not applied to the object so its size isn't the real size
    # pb is that if I modify this class there will be tremendous repercusion on other classes --> be careful...
    print('scaled test',test)
    print('scaled test',test.boundingRect())
    print('scaled test',test.boundingRect(scaled=True))

    (test.x(), test.y(), test.width(), test.height())
    print(test.contains(QPointF(50, 50)))
    print(test.contains(QPointF(-1, -1)))
    print(test.contains(QPointF(0, 0)))
    print(test.contains(QPointF(100, 100)))
    print(test.contains(QPointF(100, 100.1)))

    point = QPointF(50, 50)
    point.x()

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
