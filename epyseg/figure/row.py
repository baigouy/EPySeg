# https://docs.python.org/2/library/operator.html

# TODO try remove row if possible to only handle panels
# in a way a figure can be a panel of panels --> if so the I would just have one element --> in fact that would really simplify things like crazy
# pas mal mais du coup faudrait pas packer par defaut ou bien si sauf si l'utilisateur veut pas
# peut être plus flexible --> supress fig and row just keep panel and make it more flexible
# faire un stacker(horizontal or vertical)
# that may work in a way
# seule difference est que le panel ne doit contenir que des images de même taille alors que la row non mais en gros la row est un panel de panel
# puis je appliquer la regle des rows aux panels
# --> j'ai besoin d'une orientation des le depart
# est ce que ça peut marcher --> peut etre
# should I put a warning if they don't have the same size initially --> ? think about it
# should I have linear panel and 2D panel and give them different rules panel1D panel2D #dans panel2D ttes les images doivent avoir la meme taille. Panel1D --> c'est un peu une row en fait

# panel1D --> orientation --> hor/ver
# panel2D --> more stuff and same size required

from epyseg.draw.shapes.image2d import Image2D
from epyseg.draw.shapes.point2d import Point2D
from epyseg.draw.shapes.rect2d import Rect2D
from PyQt5.QtCore import QPointF, QRectF
# logger
from epyseg.figure.alignment import alignTop, packX, updateBoudingRect, setToWidth, setToHeight
from epyseg.tools.logger import TA_logger

logger = TA_logger()


# keep this in mind: https://stackoverflow.com/questions/34827132/changing-the-order-of-operation-for-add-mul-etc-methods-in-a-custom-c

# do rows and columns and no need to take images --> they just pack horiz or vertically and that's it each image or stuff handles itslef
# a row like that cannot be appended another row or can it --> yes in fact just add to it
class Row(Rect2D):

    def __init__(self, *args, space=3, width=None):
        self.images = []
        for img in args:
            # if isinstance(img, Row):
            #     self.images += img.images
            # else:
            # prevent adding images that could not be loaded
            if img.isSet:
                self.images.append(img)
        # init super empty
        super(Rect2D, self).__init__()
        self.isSet = True  # required to allow dragging
        self.space = space
        self.sameHeight(space=space)
        self.widthInPixel = width
        self.setToWidth(width)

    # maybe also make the add a

    def __mod__(self, other):
        # formerly was a swap but in fact need better for a swap maybe == or <>
        if isinstance(other, Row):
            pos = self.get_P1()
            other_pos = other.get_P1()
            self.set_P1(other_pos)
            other.set_P1(pos)
        else:
            logger.error('swapping not implemented yet for ' + str(type(other)))

    def __ior__(self, other):
        return self.__or__(other=other)

    def __or__(self, other):
        # print('called')
        return self.__add__(other=other)
        # formerly was a swap but in fact need better for a swap maybe == or <> moved to modulo for now
        # if isinstance(other, Row):
        #     pos = self.get_P1()
        #     other_pos = other.get_P1()
        #     self.set_P1(other_pos)
        #     other.set_P1(pos)
        # else:
        #     logger.error('swapping not implemented yet for ' + str(type(other)))

    def __add__(self, other):
        print('here', other)
        # from epyseg.figure.panel import Panel
        if isinstance(other, Image2D):
            self.images.append(other)
            self.sameHeight(space=self.space)
            self.setToWidth(self.widthInPixel)
        # elif isinstance(other, Panel):
        #     self.images.append(other)
        #     self.sameHeight(space=self.space)
        #     self.setToWidth(self.widthInPixel)
        elif isinstance(other, Row):
            # print(self.images)
            # print(len(self.images))
            # print(other.images)
            # print(len(other.images))
            self.images = self.images + other.images
            self.sameHeight(space=self.space)
            self.setToWidth(self.widthInPixel)
        return self

    # TODO maybe make it smarter so that it also removes from lower panels
    def __sub__(self, other):
        if other in self.images:
            self.images.remove(other)
        self.setToWidth(self.widthInPixel)
        return self

    def is_empty(self):
        if self.images is None or not self.images:
            return True
        return False

    def __truediv__(self, other):
        return self.__floordiv__(other=other)

    # def __truediv__(self, other):
    def __floordiv__(self, other):
        from epyseg.figure.column import Column
        col = Column(self, other, space=self.space)
        # row.sameHeight(self.space)
        # row.setToWidth(self.widthInPixel)
        return col

    def __len__(self):
        if self.images is None:
            return 0
        return len(self.images)

    # def setWidthInPixel(self, widthInPixel):
    #     self.widthInPixel = widthInPixel
    #     self.packX()

    def packX(self, space=3):
        last_x = 0
        last_y = 0

        for i in range(len(self.images)):
            img = self.images[i]
            if i != 0:
                last_x += space
            img.set_P1(last_x, img.get_P1().y())
            last_x = img.boundingRect().x() + img.boundingRect().width()

        self.updateBoudingRect()

    def packY(self, space=3):
        last_x = 0
        last_y = 0

        for i in range(len(self.images)):
            img = self.images[i]
            if i != 0:
                last_y += space
            img.set_P1(img.get_P1().x(), last_y)
            # get all the bounding boxes and pack them with desired space in between
            # get first point and last point in x
            x = img.boundingRect().x()
            y = img.boundingRect().y()
            last_x = img.boundingRect().x() + img.boundingRect().width()
            last_y = img.boundingRect().y() + img.boundingRect().height()
        self.updateBoudingRect()

    def updateBoudingRect(self):
        '''updates the image bounding rect depending on content'''
        x = None
        y = None
        x2 = None
        y2 = None
        for img in self:
            topLeft = img.get_P1()
            if x is None:
                x = topLeft.x()
            if y is None:
                y = topLeft.y()
            x = min(topLeft.x(), x)
            y = min(topLeft.y(), y)

            # print(img, img.boundingRect(), type(img))
            # print(img, img.boundingRect(), type(img), img.boundingRect().height())
            if x2 is None:
                x2 = topLeft.x() + img.boundingRect().width()
            if y2 is None:
                y2 = topLeft.y() + img.boundingRect().height()
            x2 = max(topLeft.x() + img.boundingRect().width(), x2)
            y2 = max(topLeft.y() + img.boundingRect().height(), y2)
        self.setX(x)
        self.setY(y)
        self.setWidth(x2 - x)
        self.setHeight(y2 - y)

    def setOrigin(self, *args):
        self.set_P1(*args)

    def set_P1(self, *args):
        curP1 = self.get_P1()
        Rect2D.set_P1(self, *args)  # is this how you call super ???
        newP1 = self.get_P1()
        for img in self:
            img.translate(newP1.x() - curP1.x(), newP1.y() - curP1.y())

        self.updateBoudingRect()

    def translate(self, *args):
        if len(args) == 1:
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
            for img in self:
                img.draw(painter, draw=draw)
            painter.restore()

    def fill(self, painter, draw=True):
        if self.fill_color is None:
            return
        if draw:
            painter.save()
        if draw:
            for img in self:
                img.fill(painter, draw=draw)
            painter.restore()

    def drawAndFill(self, painter):
        painter.save()
        self.draw(painter, draw=True)
        painter.restore()

    def __iter__(self):
        ''' Returns the Iterator object '''
        self._index = -1
        return self

    def __next__(self):
        '''' Returns the next image or panel in the row '''
        self._index += 1
        if self._index < len(self.images):
            return self.images[self._index]
        # End of Iteration
        raise StopIteration

    def __lshift__(self, other):
        # move image left with respect to self
        if isinstance(other, Image2D):
            # swap the position of the two images and repack
            if other in self.images:
                pos = self.images.index(other)
                if pos - 1 >= 0:
                    self.images[pos - 1], self.images[pos] = self.images[pos], self.images[pos - 1]
                else:
                    return self
                self.packX(self.space)
                return self
        else:
            logger.error('not implemented yet swapping two objects ' + str(type(other)))

    def __rshift__(self, other):
        # move image left with respect to self
        if isinstance(other, Image2D):
            # swap the position of the two images and repack
            if other in self.images:
                pos = self.images.index(other)
                if pos + 1 < len(self.images):
                    self.images[pos + 1], self.images[pos] = self.images[pos], self.images[pos + 1]
                else:
                    return self
                self.packX(self.space)
                return self
        else:
            logger.error('not implemented yet swapping two objects ' + str(type(other)))

    # en fait ne peut pas marcher faut faire row per row
    def sameHeight(self, space):
        if space is None:
            space = 0
        max_height = None
        for img in self:
            if max_height is None:
                # print('TUTU',img)
                # print('TUTA',img, img.boundingRect(), img.boundingRect().height)
                # print('TOTO',img, img.boundingRect(), img.boundingRect().height())
                max_height = img.boundingRect().height()
            max_height = max(img.boundingRect().height(), max_height)
        for img in self:
            img.setToHeight(max_height)

        self.packX(space)
        self.alignTop(updateBounds=False)
        self.updateBoudingRect()

    # Aligns objects within the block
    def arrangeRow(self):
        self.alignTop()

    # Align vectorial objects to the top
    # in fact should align in y
    def alignTop(self, updateBounds=False):
        first_left = None
        for img in self:
            cur_pos = img.get_P1()
            if first_left is None:
                first_left = cur_pos
            img.set_P1(cur_pos.x(), first_left.y())
        if updateBounds:
            self.updateMasterRect()

    def get_shape_at_coord(self, x, y):
        '''returns the shape under the mouse or None if none is found

        :param x:
        :param y:
        :return:
        '''
        for img in self.images:
            if img.boundingRect().contains(x,y):
                return img
        return None

    # is that always correct irrespective of content ???
    # probably need to overwrite the set to scale of this
    # Forces the block to be of width (width_in_px)
    def setToWidth(self, width_in_px):
        # from timeit import default_timer as timer
        # start = timer()
        if width_in_px is None:
            return
        # print('setToWidth row 1', timer() - start)
        topleft = Point2D(self.boundingRect().topLeft())
        # print('setToWidth row 2', timer() - start)
        alignTop(topleft, *self.images)
        # print('setToWidth row 3', timer() - start)
        packX(self.space, topleft, *self.images)
        # print('setToWidth row 4', timer() - start)
        # print('bounds before', updateBoudingRect(*self.images))
        setToWidth(self.space, width_in_px, *self.images)
        # print('setToWidth row 5', timer() - start)
        bounds = updateBoudingRect(*self.images)
        # print('bounds after', updateBoudingRect(*self.images))
        # print('setToWidth row 6', timer() - start)

        max_height = -1

        cur_height = bounds.height()
        cur_width = bounds.width()
        all_heights_are_the_same = True
        # print('setToWidth row 7', timer() - start)
        for img in self.images:
            max_height = max(max_height, img.boundingRect().height())
            # print('cur height', img.boundingRect().height(), img.boundingRect().width())
            if cur_height != max_height:
                all_heights_are_the_same = False

        if not all_heights_are_the_same:
            setToHeight(self.space, max_height, *self.images)
            bounds = updateBoudingRect(*self.images)
            all_heights_are_the_same = True
        else:
            bounds = updateBoudingRect(*self.images)

        sign = 1
        if bounds.width() >= width_in_px:
            sign = -1

        # print(bounds.width(), 'vs desired', width_in_px, 'sign', sign)
        #
        # print(all_heights_are_the_same)
        # print(bounds.width() != width_in_px)
        # print(abs(bounds.width()-width_in_px)>0.3)


        # print('setToWidth row 8', timer() - start)
        # dirty height/width fix for complex panels figure out the math for it and replace this code
        # TODO also speed this up so that the action is only faked and not really done (just done once at the end to gain time)
        if not all_heights_are_the_same or (bounds.width() != width_in_px and abs(bounds.width()-width_in_px)>0.3):
            # print('setToWidth row 9', timer() - start)
            while True:
                setToHeight(self.space, max_height, *self.images)
                bounds = updateBoudingRect(*self.images)
                if sign<0:
                    if bounds.width() <= width_in_px:
                        break
                else:
                    if bounds.width() >= width_in_px:
                        break
                max_height += 1 * sign
            # print('setToWidth row 10', timer() - start)

            if bounds.width() != width_in_px:
                if bounds.width() >= width_in_px:
                    sign = -1
                else:
                    sign = 1
                # what if i remove refine step
                #
                # print('setToWidth row 10', timer() - start)
                # max_height -= 1
                # print('max_height', max_height, updateBoudingRect(*self.images))
                while True:
                    setToHeight(self.space, max_height, *self.images)
                    bounds = updateBoudingRect(*self.images)
                    if sign < 0:
                        if bounds.width() <= width_in_px:
                            break
                    else:
                        if bounds.width() >= width_in_px:
                            break
                    max_height += 0.25 *sign # critical step with this value --> timing is ok --> maybe ok for now
                # # print('setToWidth row 11', timer() - start)

        # print('setToWidth row 11', timer() - start)
        # print('setToWidth row 12', timer() - start)
        self.updateBoudingRect()
        # print('setToWidth row last', timer() - start)

    # def setToWidth(self, width_in_px):
    #     if width_in_px is None:
    #         return
    #     topleft = Point2D(self.boundingRect().topLeft())
    #     alignTop(topleft, *self.images)
    #     # because can use relative width and height
    #     # packX(self.space, topleft, *self.images)# do I need that --> I guess not
    #     # just fake it in fact --> faster
    #     fakeSetToWidth(width_in_px, *self.images)
    #     # compute fake nounds
    #     bounds = fakeUpdateBoudingRect(*self.images)
    #
    #
    #     # need compute fake height too
    #     min_height = 100_000_000
    #
    #     cur_height = bounds.height()
    #     cur_width = bounds.width()
    #     all_heights_are_the_same = True
    #     for img in self.images:
    #         min_height = min(min_height, img.boundingRect().height())
    #         if min_height<cur_height:
    #             all_heights_are_the_same=False
    #
    #     # dirty height/width fix for complex panels figure out the math for it and replace this code
    #     # TODO also speed this up so that the action is only faked and not really done (just done once at the end to gain time)
    #     if not all_heights_are_the_same or cur_width!=width_in_px:
    #         while True:
    #             setToHeight(min_height, *self.images)
    #             bounds = updateBoudingRect(*self.images)
    #             if bounds.width() >= width_in_px:
    #                 break
    #             min_height += 1
    #
    #         min_height -= 1
    #         while True:
    #             min_height += 0.05
    #             setToHeight(min_height, *self.images)
    #             bounds = updateBoudingRect(*self.images)
    #             if bounds.width() >= width_in_px:
    #                 break
    #
    #     self.updateBoudingRect()

    def setToHeight(self, height_in_px):
        if height_in_px is None:
            return
        # # nb_cols = len(self)
        # pure_image_height = (self.boundingRect().height()) - self.getIncompressibleHeight()
        # # print(width_in_px, self.getIncompressibleWidth(), (nb_cols - 1.) * self.space )
        # height_in_px -= self.getIncompressibleHeight()
        # ratio = height_in_px / pure_image_height
        # for img in self:
        #     img.setToHeight(img.boundingRect().height() * ratio)
        # self.packX(self.space)
        # self.updateBoudingRect()
        topleft = Point2D(self.boundingRect().topLeft())
        alignTop(topleft, *self.images)
        setToHeight(self.space, height_in_px, *self.images)
        self.updateBoudingRect()

    # @return the block incompressible width
    def getIncompressibleHeight(self):
        extra_space = 0
        return extra_space

    # @return the block incompressible width
    def getIncompressibleWidth(self):
        nb_cols = len(self)
        extra_space = (nb_cols - 1.) * self.space
        return extra_space

    # can I pass an ignoring char if some images need be ignored
    # probably need return the increment
    def setLettering(self, letter):
        # TODO implement increment
        # set letter for increasing stuff # check if letter is string or letter or array if so do or do not increase letters
        for img in self:
            img.setLettering(letter)
        # self.letter = letter


if __name__ == '__main__':
    img1 = Image2D('/D/Sample_images/sample_images_PA/trash_test_mem/counter/00.png')
    img2 = Image2D('/D/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
    img3 = Image2D('/D/Sample_images/sample_images_PA/trash_test_mem/counter/02.png')
    img4 = Image2D('/D/Sample_images/sample_images_PA/trash_test_mem/counter/03.png')
    result = img1 + img2

    print(result)
