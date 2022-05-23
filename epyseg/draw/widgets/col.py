#https://docs.python.org/2/library/operator.html

from epyseg.draw.shapes.rect2d import Rect2D
from PyQt5.QtCore import QPointF, QRectF
# logger
from epyseg.tools.logger import TA_logger
from epyseg.figure.row import Row  # KEEP Really required to avoid circular imports
logger = TA_logger()


# TODO hack it that it can handle rows or images if needed

class col(Rect2D):

    def __init__(self, *args, space=3, width=None):
        self.images = []
        self.widthInPixel = width
        self.space = space
        for arg in args:
        #     if isinstance(arg, Column):
        #         arg.packX(space=space)
        #         arg.setToWidth(self.widthInPixel)
            self.images.append(arg)
            # else:
            #     self.images.append(Column(arg, space=space, width=width))
        super(Rect2D, self).__init__()
        self.packY(space=space)

    def __or__(self, other):
        if isinstance(other, col):
            pos = self.get_P1()
            other_pos = other.get_P1()
            self.set_P1(other_pos)
            other.set_P1(pos)
        else:
            logger.error('swapping not implemented yet for ' + str(type(other)))

    def __sub__(self, other):
        if other in self.images:
            self.images.remove(other)
            return
        for row in self.images:
            row -= other
        for row in reversed(self.images):
            if row.isEmpty():
                self.images.remove(row)

        return self

    def __add__(self, other):
        from epyseg.figure.row import Row  # KEEP Really required to avoid circular imports # TODO is that still true ?
        if isinstance(other, Row):
            other.packX(space=self.space)
            other.setToWidth(self.widthInPixel)
            self.images.append(other)
            return self
        self.images.append(Row(other, space=self.space, width=self.widthInPixel))
        self.packY(space=self.space)
        return self

    def __len__(self):
        if self.images is None:
            return 0
        return len(self.images)

    def __iter__(self):
        ''' Returns the Iterator object '''
        self._index = -1
        return self

    def __next__(self):
        ''' Returns the next row of the col '''
        self._index += 1
        if self._index < len(self.images):
            return self.images[self._index]
        # End of Iteration
        raise StopIteration

    def setWidthInPixel(self, widthInPixel):
        self.widthInPixel = widthInPixel
        self.packY()

     # Forces the block to be of width (width_in_px)
    def setToWidth(self, width_in_px):
        for image in self:
            image.setToHeight(width_in_px)
        self.packY(space=self.space)

    def setToHeight(self, height_in_px):
        for image in self:
            image.setToHeight(height_in_px)
        self.packY(space=self.space)

    def getIncompressibleWidth(self):
        extra_space = 0;
        return extra_space;

    def packX(self, space=3):
        # pack shapes in x

        last_x = 0
        last_y = 0

        for i in range(len(self.images)):
            row = self.images[i]
            if i != 0:
                last_x += space
            row.set_P1(last_x, row.get_P1().y())
            # get all the bounding boxes and pack them with desired space in between
            # get first point and last point in x
            x = row.boundingRect().x()
            y = row.boundingRect().y()
            last_x = row.boundingRect().x() + row.boundingRect().width()
            last_y = row.boundingRect().y() + row.boundingRect().height()
        self.updateBoudingRect()


    def packY(self, space=3):

        if space is None:
            space = 0

        last_x = 0
        last_y = 0


        for i in range(len(self.images)):
            row = self.images[i]
            if i != 0:
                last_y += space
            row.set_P1(row.get_P1().x(), last_y)

            x = row.boundingRect().x()
            y = row.boundingRect().y()
            last_x = row.boundingRect().x() + row.boundingRect().width()
            last_y = row.boundingRect().y() + row.boundingRect().height()
        self.updateBoudingRect()

    def updateBoudingRect(self):
        '''updates the image bounding rect depending on content

        '''
        x = None
        y = None
        x2 = None
        y2 = None
        for row in self:
            topLeft = row.get_P1()
            if x is None:
                x = topLeft.x()
            if y is None:
                y = topLeft.y()
            x = min(topLeft.x(), x)
            y = min(topLeft.y(), y)
            if x2 is None:
                x2 = topLeft.x() + row.boundingRect().width()
            if y2 is None:
                y2 = topLeft.y() + row.boundingRect().height()
            x2 = max(topLeft.x() + row.boundingRect().width(), x2)
            y2 = max(topLeft.y() + row.boundingRect().height(), y2)

        if x is not None:
            self.setX(x)
        if y is not None:
            self.setY(y)
        if x2 is not None:
            self.setWidth(x2 - x)
        if y2 is not None:
            self.setHeight(y2 - y)

    def set_P1(self, *args):
        curP1 = self.get_P1()
        Rect2D.set_P1(self, *args)
        newP1 = self.get_P1()
        # need do a translation of all its content images
        for row in self:
            row.translate(newP1.x() - curP1.x(), newP1.y() - curP1.y())

    def translate(self, *args):
        if len(args)==1:
            point = args[0]
            QRectF.translate(self, point.x(), point.y())
            for img in self:
                img.translate(point.x(), point.y())
        else:
            QRectF.translate(self, args[0], args[1])
            for img in self:
                img.translate(QPointF(args[0], args[1]))


    def draw(self, painter, draw=True):
        if draw:
            painter.save()
        if draw:
            counter = 0
            for row in self:
                print('drawing', counter, row.boundingRect())
                counter+=1
                row.draw(painter, draw=draw)
            painter.restore()

    def fill(self, painter, draw=True):
        if self.fill_color is None:
            return
        if draw:
            painter.save()
        if draw:
            for row in self:
                row.fill(painter, draw=draw)
            painter.restore()

        # TODO pb will draw the shape twice.... ---> because painter drawpolygon is called twice

    def drawAndFill(self, painter):
        painter.save()
        self.draw(painter, draw=True)
        painter.restore()

if __name__ == '__main__':
    from epyseg.draw.shapes.image2d import Image2D  # KEEP Really required to avoid circular imports

    img1 = Image2D('./../data/counter/00.png')
    img2 = Image2D('./../data/counter/01.png')
    img3 = Image2D('./../data/counter/02.png')
    img4 = Image2D('./../data/counter/03.png')
    img5 = Image2D('./../data/counter/04.png')
    img6 = Image2D('./../data/counter/05.png')
    img8 = Image2D('./../data/counter/06.png')
    img9 = Image2D('./../data/counter/07.png')
    img10 = Image2D('./../data/counter/08.png')
    img11 = Image2D('./../data/counter/09.png')

    result = img1 + img2

    print(result)
    print(len(result.images))

    #    result += img3
    #    result += img4

    #    print(result)
    #    print(len(result.images)) # ça ça marche

    #    if True:
    #        sys.exit(0)

    # img3.pop()

    row2 = img3 + img4 + img8
    # row2 = row(img3, img4)

    print(row2)
    print(row2.images)  # why 4 images here ....
    print('row2', len(row2.images))  # pas bon trop d'images --> pkoi ????

    print(Image2D)

    fig = result + row2
    print(fig)
    print(len(fig.images))

    count = 0
    for img in fig.images:
        print(count, img.filename)
        count += 1

    final_result = fig - img4
    print(final_result)
    print(final_result.images)
    print(len(final_result.images))

    # ça marche

    final_result = img4 + fig  # ça ne marche pas
    print(final_result)
    print(final_result.images)
    print(len(final_result.images))

    # si je divide une row alors ça ajoute une colonne à la figure --> return 2 rows in a figure



    fig = img5 / img6

    print(fig)
    print(fig.rows)
    print(len(fig.rows))

    final_fig = fig - img5

    print(final_fig)
    print(final_fig.rows)
    print(len(final_fig.rows))

    final_fig += row2

    print(final_fig)
    print(final_fig.rows)
    print(len(final_fig.rows))

    fig = (img9) / (img10 + img11)
    print(fig)
    print(fig.rows)
    print(len(fig.rows))
    print('r1', len(fig.rows[0].images))
    print('r2', len(fig.rows[1].images))

    # try the pack and check it
    # almost there now in fact...

    for img in row2:
        print(img.boundingRect())

    print(row2)
    row2.packX()
    for img in row2:
        print(img.boundingRect())

    row2.packY()
    for img in row2:
        print(img.boundingRect())

    # ça marche

    # tester le draw sur une image --> balancer un qimage ???



    # print('#'*20)
    # from PyQt5.QtCore import QRectF, QPointF
    # test = QRectF(0,0,411,512)
    # print(test)
    # test.setX(414) # --> fucks width --> need reput it in correct position
    # # test.setWidth(411)
    # print(test)
    # test = QRectF(0,0,411,512)
    # print(test)
    # test.setTopLeft(QPointF(414,0))
    # print(test)
