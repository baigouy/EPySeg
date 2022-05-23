from PyQt5 import QtCore
from epyseg.draw.shapes.polygon2d import Polygon2D
from epyseg.draw.shapes.line2d import Line2D
from epyseg.draw.shapes.rect2d import Rect2D
from epyseg.draw.shapes.square2d import Square2D
from epyseg.draw.shapes.ellipse2d import Ellipse2D
from epyseg.draw.shapes.circle2d import Circle2D
from epyseg.draw.shapes.freehand2d import Freehand2D
from epyseg.draw.shapes.point2d import Point2D
from epyseg.draw.shapes.polyline2d import PolyLine2D
from epyseg.draw.shapes.image2d import Image2D
from PyQt5.QtCore import QPointF, QRectF
from epyseg.tools.logger import TA_logger # logging
logger = TA_logger()

class VectorialDrawPane:

    def __init__(self, active=False, demo=False, scale=1.0, drawing_mode=False):
        self.shapes = []
        self.currently_drawn_shape = None
        self.shape_to_draw = None
        self.selected_shape = []
        self.active = active
        self.scale = scale
        self.drawing_mode = drawing_mode
        if demo:
            self.shapes.append(Polygon2D(0, 0, 10, 0, 10, 20, 0, 20, 0, 0, color=0x00FF00))
            self.shapes.append(
                Polygon2D(100, 100, 110, 100, 110, 120, 10, 120, 100, 100, color=0x0000FF, fill_color=0x00FFFF,
                          stroke=2))
            self.shapes.append(Line2D(0, 0, 110, 100, color=0xFF0000, stroke=3))
            self.shapes.append(Rect2D(200, 150, 250, 100, stroke=10))
            self.shapes.append(Square2D(300, 260, 250, stroke=3))
            self.shapes.append(Ellipse2D(0, 50, 600, 200, stroke=3))
            self.shapes.append(Circle2D(150, 300, 30, color=0xFF0000))
            self.shapes.append(PolyLine2D(10, 10, 20, 10, 20, 30, 40, 30, color=0xFF0000, stroke=2))
            self.shapes.append(PolyLine2D(10, 10, 20, 10, 20, 30, 40, 30, color=0xFF0000, stroke=2))
            self.shapes.append(Point2D(128, 128, color=0xFF0000, stroke=6))
            self.shapes.append(Point2D(128, 128, color=0x00FF00, stroke=1))
            self.shapes.append(Point2D(10, 10, color=0x000000, stroke=6))
            img0 = Image2D('./../data/counter/00.png')
            img1 = Image2D('./../data/counter/01.png')
            img2 = Image2D('./../data/counter/02.png')
            img3 = Image2D('./../data/counter/03.png')
            img4 = Image2D('./../data/counter/04.png')
            img5 = Image2D('./../data/counter/05.png')
            img6 = Image2D('./../data/counter/06.png')
            img7 = Image2D('./../data/counter/07.png')
            img8 = Image2D('./../data/counter/08.png')
            img9 = Image2D('./../data/counter/09.png')
            img10 = Image2D('./../data/counter/10.png')

            row = img1 + img2 + img10

            self.shapes.append(row)

            row2 = img4 + img5
            fig = row / row2
            # fig = Column(row, row2)
            #self.shapes.append(fig)
            self.drawing_mode = True
            # self.shape_to_draw = Line2D
            # self.shape_to_draw = Rect2D
            # self.shape_to_draw = Square2D
            # self.shape_to_draw = Ellipse2D
            # self.shape_to_draw = Circle2D
            # self.shape_to_draw = Point2D  # ok maybe small centering issue
            # self.shape_to_draw = Freehand2D
            # self.shape_to_draw = PolyLine2D
            # self.shape_to_draw = Polygon2D
            import random
            drawing_methods = [Line2D, Rect2D, Square2D, Ellipse2D, Circle2D, Point2D, Freehand2D, PolyLine2D, Polygon2D]
            self.shape_to_draw = random.choice(drawing_methods)

            # TODO freehand drawing
            # TODO broken line --> need double click for end

    def paintEvent(self, *args):
        painter = args[0]
        visibleRect = None
        if len(args) >= 2:
              visibleRect = args[1]

        painter.save()
        if self.scale != 1.0:
            painter.scale(self.scale, self.scale)

        for shape in self.shapes:
            # only draw shapes if they are visible --> requires a visiblerect to be passed
            if visibleRect is not None:
                # only draws if in visible rect
                if shape.boundingRect().intersects(QRectF(visibleRect)):
                    shape.draw(painter)
            else:
                shape.draw(painter)

        if self.currently_drawn_shape is not None:
            if self.currently_drawn_shape.isSet:
                self.currently_drawn_shape.draw(painter)

        sel = self.create_master_rect()
        if sel is not None:
            painter.drawRect(sel)
        painter.restore()
        # painter.end() # probably a good idea ????

    def group_contains(self, x, y):
        # checks if master rect for group contains click
        # get bounds and create union and compare
        master_rect = self.create_master_rect()
        if master_rect is None:
            return False
        return master_rect.contains(QPointF(x, y))

    def create_master_rect(self):
        master_rect = None
        if self.selected_shape:
            for shape in self.selected_shape:
                if master_rect is None:
                    master_rect = shape.boundingRect()
                else:
                    master_rect = master_rect.united(shape.boundingRect())
        return master_rect

    def removeCurShape(self):
        if self.selected_shape:
            self.shapes = [e for e in self.shapes if e not in self.selected_shape]
            self.selected_shape = []

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos() / self.scale
            self.firstPoint = event.pos() / self.scale

            shapeFound = False
            if self.currently_drawn_shape is None:
                for shape in reversed(self.shapes):
                    if shape.contains(self.lastPoint) and not shape in self.selected_shape:
                        logger.debug('you clicked shape:' + str(shape))
                        if event.modifiers() == QtCore.Qt.ControlModifier:
                            if shape not in self.selected_shape:  # avoid doublons
                                self.selected_shape.append(shape)  # add shape to group
                                logger.debug('adding shape to group')
                                shapeFound = True
                        else:
                            if not self.group_contains(self.lastPoint.x(), self.lastPoint.y()):
                                self.selected_shape = [shape]
                                logger.debug('only one element is selected')
                                shapeFound = True
                        return

                if not shapeFound and event.modifiers() == QtCore.Qt.ControlModifier:
                    for shape in reversed(self.shapes):
                        if shape.contains(self.lastPoint):
                            if shape in self.selected_shape:  # avoid doublons
                                logger.debug('you clicked again shape:' + str(shape))
                                self.selected_shape.remove(shape)  # add shape to group
                                logger.debug('removing a shape from group')
                                shapeFound = True
                # no shape found --> reset sel
                if not shapeFound and not self.group_contains(self.lastPoint.x(), self.lastPoint.y()):
                    logger.debug('resetting sel')
                    self.selected_shape = []

            # check if a shape is selected and only move that
            if self.drawing_mode and not self.selected_shape and self.currently_drawn_shape is None:
                # do not reset shape if not done drawing...
                if self.shape_to_draw is not None:
                    self.currently_drawn_shape = self.shape_to_draw()
                else:
                    self.currently_drawn_shape = None
            if self.drawing_mode and not self.selected_shape:
                if self.currently_drawn_shape is not None:
                    self.currently_drawn_shape.set_P1(QPointF(self.lastPoint.x(), self.lastPoint.y()))

    def mouseMoveEvent(self, event):
        if event.buttons() and QtCore.Qt.LeftButton:
            if self.selected_shape and self.currently_drawn_shape is None:
                logger.debug('moving' + str(self.selected_shape))
                for shape in self.selected_shape:
                    shape.translate(event.pos() / self.scale - self.lastPoint)

        if self.currently_drawn_shape is not None:
            self.currently_drawn_shape.add(self.firstPoint, self.lastPoint)

        self.lastPoint = event.pos() / self.scale

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = False
            if self.drawing_mode and self.currently_drawn_shape is not None:
                self.currently_drawn_shape.add(self.firstPoint, self.lastPoint)
                if isinstance(self.currently_drawn_shape, Freehand2D):
                    # this closes the freehand shape
                    self.currently_drawn_shape.add(self.lastPoint, self.firstPoint)
                # should not erase the shape if it's a polyline or a polygon by the way
                if isinstance(self.currently_drawn_shape, Freehand2D) or (not isinstance(self.currently_drawn_shape, PolyLine2D) and not isinstance(self.currently_drawn_shape, Polygon2D)):
                    self.shapes.append(self.currently_drawn_shape)
                    self.currently_drawn_shape = None

    def mouseDoubleClickEvent(self, event):
        if isinstance(self.currently_drawn_shape, PolyLine2D) or isinstance(self.currently_drawn_shape, Polygon2D):
            self.shapes.append(self.currently_drawn_shape)
            self.currently_drawn_shape = None

if __name__ == '__main__':
    VectorialDrawPane()
