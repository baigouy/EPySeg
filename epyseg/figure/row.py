# not so bad --> see how to add legends --> not so easy because I need to create various groups
# the ideal situtation would be to preview all the possible fusions and ask the user what should be done
# or offer a preview if user wants it and then ask him what to do
# otherwise fuse things
# NB add row names and column names to each image and offer the possibility to show or fuse them whenever necessary... --> will replace in an easier way the add labels around ...  --> think if smart or not ???
# do the math also below
# see ho to handle row labels
# preview might be slow --> ask how to do things ????


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
from PyQt5 import QtWidgets

from epyseg.draw.shapes.image2d import Image2D
from epyseg.draw.shapes.point2d import Point2D
from epyseg.draw.shapes.rect2d import Rect2D
from PyQt5.QtCore import QPointF, QRectF
# logger
# from epyseg.ezfig import MyWidget
# from epyseg.figure import fig_tools
from epyseg.figure.alignment import alignTop, packX, updateBoudingRect, setToHeight, _brute_force_find_height
from epyseg.figure.fig_tools import preview, get_master_bounds2, get_common_scaling_factor
from epyseg.tools.logger import TA_logger

logger = TA_logger()

# keep this in mind: https://stackoverflow.com/questions/34827132/changing-the-order-of-operation-for-add-mul-etc-methods-in-a-custom-c
# do rows and columns and no need to take images --> they just pack horiz or vertically and that's it each image or stuff handles itslef
# a row like that cannot be appended another row or can it --> yes in fact just add to it
# not a good idea to have rect2D and qrectF at the same time....
# Ideally choose one or allow a better sync between the two

class Row(Rect2D):

    def __init__(self, *args, space=3, width=None):
        super(QRectF, self).__init__()
        self.images = []
        for img in args:
            # if isinstance(img, Row):
            #     self.images += img.images
            # else:
            # prevent adding images that could not be loaded
            if img.isSet:
                self.images.append(img)
        # init super empty
        self.isSet = True  # required to allow dragging
        self.space = space
        # why did not work ???
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
        # print('here', other)
        # from epyseg.figure.panel import Panel
        if isinstance(other, Row):
            # print(self.images)
            # print(len(self.images))
            # print(other.images)
            # print(len(other.images))
            P1 = self.get_P1()
            final_width = self.width()
            self.images = self.images + other.images

            # self.sameHeight(space=self.space)
            # self.setToHeight(self.height())
            self.setToWidth(final_width)
            self.set_P1(P1)
        # elif isinstance(other, Image2D):
        else:
            P1 = self.get_P1()
            final_width = self.width()
            self.images.append(other)
            # self.sameHeight(space=self.space)
            # self.setToHeight(self.height())
            self.setToWidth(final_width)
            self.set_P1(P1)
        # elif isinstance(other, Panel):
        #     self.images.append(other)
        #     self.sameHeight(space=self.space)
        #     self.setToWidth(self.widthInPixel)


        return self

        # def __truediv__(self, other):

    def __floordiv__(self, other):
        from epyseg.figure.column import Column
        P1 = self.get_P1()
        # final_height = self.height()
        col = Column(self, other,
                     space=self.space)  # the size of this stuff is way too big and way bigger than its components --> do not allow that
        # col.setToHeight(final_height)
        # need update the bounding rect
        col.sameWidth(self.space)
        # row.setToWidth(self.widthInPixel)
        col.set_P1(P1)
        return col

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

            # topLeft = img.get_P1()
            topLeft = img.boundingRect().topLeft()
            # print('comparison', img.boundingRect(), img.get_P1(), img) # the bug is here the various objects are not in sync !!!! --> do not allow that

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

        # print('bounds from inside', self)
        # bounds = get_master_bounds2(self.images)
        # self.setX(bounds.x())
        # self.setY(bounds.y())
        # self.setWidth(bounds.width())
        # self.setHeight(bounds.height())
        # return bounds

    # def boundingRect(self):
    #     return self.updateBoudingRect()
    # return get_master_bounds2(self.images)

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
            # print('height after update', img.height())

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
            if img.boundingRect().contains(x, y):
                return img
        return None

    # ça marche super mais cree un bug pour les objects vectoriels --> voir où est le bug dans le truc
    # much smarter and mathematically correct set to width
    # Forces the block to be of width (width_in_px)
    def setToWidth(self, width_in_px):
        # from timeit import default_timer as timer
        # start = timer()
        if width_in_px is None:
            return

        topleft = Point2D(self.boundingRect().topLeft())
        self.sameHeight(self.space)

        # print('setToWidth row 1', timer() - start)

        # print('setToWidth row 2', timer() - start)
        # alignTop(topleft, *self.images)
        # print('setToWidth row 3', timer() - start)
        # packX(self.space, topleft, *self.images)

        # self.updateBoudingRect()  # I should also updat the bounding rect of output -> that is my error

        sizes = []
        for elm in self:
            sizes.append(elm.width())
            # print("inner size", elm)

        # new way to compute size --> much smarter than previous way --> start replace everywhere
        # sizes
        scaling = get_common_scaling_factor(sizes, width_in_px, self.space)
        # print('scaling, ', scaling)
        # need set to scale
        # need

        for elm in self:
            # elm.setWidth(scaling*elm.width())
            # elm.setHeight(scaling*elm.height())
            try:
                elm.scale *= scaling
            except:
                # scaling will not be working if elm has incompressible part --> need change scale again
                # c'est pas ce que je veux en fait --> c'est vraiment le mettre à la width desiree en fait
                elm.setToWidth(elm.width() * scaling)

        # pb there is a bug and the image and stuff don't match definitely my new code is better and smarter
        # this does noyt work but why ????
        # need pack them in x
        packX(self.space, None, *self.images)

        # need check that it worked

        # self.updateBoudingRect()
        # in fact that does not work and solving the equations can be difficult when the initial guess is wrong --> so brute force find it


        min_h = 10000000
        max_h = 0
        for elm in self:
            height = elm.height()
            # print('size of inner elm', )
            min_h = min(min_h, height)
            max_h = max(max_h, height)

        if max_h-min_h > 0.05:
            # closest = _brute_force_find_height(self, min_h - 1, max_h + 1, 1, width_in_px)

            # TODO --> need do something for gigantic images --> TODO
            # apparently this is painful with big images --> need a ratio
            # closest = _brute_force_find_height(self, min_h - 1, max_h + 1, (max_h - min_h) / 10, width_in_px)
            #
            # if closest is not None:
            #     if closest[1] - width_in_px < 0.05:
            #         self.setToHeight(closest[2])
            #         return closest
            # else:
            #     closest = _brute_force_find_height(self, min_h - 1, max_h + 1, 1, width_in_px)
            #
            # value = closest[1]
            # closest = _brute_force_find_height(self, value - 1, value + 1, 1, width_in_px)


            closest = _brute_force_find_height(self, min_h - 1, max_h + 1, 1, width_in_px)

            # print('closest1', closest)
            value = closest[1]
            # for height in np.arange(value - 1, value + 1, 0.01):
            #     col1.setToHeight(height)
            #     print('bob', height, col1.width(), col1.height())
            #     if (col1.width() - desired_width) <= mn and col1.width() - desired_width >= 0:
            #         closest = (col1.width(), col1.height(), height)
            #         mn = col1.width() - desired_width

            if closest[1]-width_in_px<0.05:
                self.setToHeight(closest[2])
                return closest

            closest = _brute_force_find_height(self, value - 0.5, value + 0.5, 0.1, width_in_px)


            # tres lent --> need skip steps if good enough
            # --> pas mal et assez rapide --> pourrait faire n iterations
            if closest[1]-width_in_px<0.05:
                self.setToHeight(closest[2])
                return closest

            value = closest[1]
            closest = _brute_force_find_height(self, value - 0.1, value + 0.1, 0.01, width_in_px)  # almost perfect

            # (128.00066196944778, 42.80999999999944, 42.80999999999944) --> solution est à peu pres ça --> comment la trouver
            # (128.5352647965376, 43.0, 43) # ineteger search then refinement --> could search between max height and min height of the shape of the column
            # (128.00066196944888, 42.80999999999984, 42.80999999999984)

            # this is a very good estimate

            self.setToHeight(closest[2])

        self.set_P1(topleft)
        self.updateBoudingRect()

    # def get_equa(self):
    #     sqdsqdsqdsqd

    # this is a really crappy method --> needs be changed
    # is that always correct irrespective of content ???
    # probably need to overwrite the set to scale of this
    # Forces the block to be of width (width_in_px)
    # def setToWidth_old(self, width_in_px):
    #     # from timeit import default_timer as timer
    #     # start = timer()
    #     if width_in_px is None:
    #         return
    #     # print('setToWidth row 1', timer() - start)
    #     topleft = Point2D(self.boundingRect().topLeft())
    #     # print('setToWidth row 2', timer() - start)
    #     alignTop(topleft, *self.images)
    #     # print('setToWidth row 3', timer() - start)
    #     packX(self.space, topleft, *self.images)
    #     # print('setToWidth row 4', timer() - start)
    #     # print('bounds before', updateBoudingRect(*self.images))
    #     setToWidth(self.space, width_in_px, *self.images)
    #     # print('setToWidth row 5', timer() - start)
    #     bounds = updateBoudingRect(*self.images)
    #     # print('bounds after', updateBoudingRect(*self.images))
    #     # print('setToWidth row 6', timer() - start)
    #
    #     max_height = -1
    #
    #     cur_height = bounds.height()
    #     cur_width = bounds.width()
    #     all_heights_are_the_same = True
    #     # print('setToWidth row 7', timer() - start)
    #     for img in self.images:
    #         max_height = max(max_height, img.boundingRect().height())
    #         # print('cur height', img.boundingRect().height(), img.boundingRect().width())
    #         if cur_height != max_height:
    #             all_heights_are_the_same = False
    #
    #     if not all_heights_are_the_same:
    #         setToHeight(self.space, max_height, *self.images)
    #         bounds = updateBoudingRect(*self.images)
    #         all_heights_are_the_same = True
    #     else:
    #         bounds = updateBoudingRect(*self.images)
    #
    #     sign = 1
    #     if bounds.width() >= width_in_px:
    #         sign = -1
    #
    #     # print(bounds.width(), 'vs desired', width_in_px, 'sign', sign)
    #     #
    #     # print(all_heights_are_the_same)
    #     # print(bounds.width() != width_in_px)
    #     # print(abs(bounds.width()-width_in_px)>0.3)
    #
    #
    #     # print('setToWidth row 8', timer() - start)
    #     # dirty height/width fix for complex panels figure out the math for it and replace this code
    #     # TODO also speed this up so that the action is only faked and not really done (just done once at the end to gain time)
    #
    #     # VERY DIRTY HERE --> need be hacked and replaced by new code
    #
    #     if not all_heights_are_the_same or (bounds.width() != width_in_px and abs(bounds.width()-width_in_px)>0.3):
    #
    #         # indeed this is super dirty code that need be replaced by the new code I have that doesn't use that
    #         # print('setToWidth row 9', timer() - start)
    #         while True:
    #             setToHeight(self.space, max_height, *self.images)
    #             bounds = updateBoudingRect(*self.images)
    #             if sign<0:
    #                 if bounds.width() <= width_in_px:
    #                     break
    #             else:
    #                 if bounds.width() >= width_in_px:
    #                     break
    #             max_height += 1 * sign
    #         # print('setToWidth row 10', timer() - start)
    #
    #         if bounds.width() != width_in_px:
    #             if bounds.width() >= width_in_px:
    #                 sign = -1
    #             else:
    #                 sign = 1
    #             # what if i remove refine step
    #             #
    #             # print('setToWidth row 10', timer() - start)
    #             # max_height -= 1
    #             # print('max_height', max_height, updateBoudingRect(*self.images))
    #             while True:
    #                 setToHeight(self.space, max_height, *self.images)
    #                 bounds = updateBoudingRect(*self.images)
    #                 if sign < 0:
    #                     if bounds.width() <= width_in_px:
    #                         break
    #                 else:
    #                     if bounds.width() >= width_in_px:
    #                         break
    #                 max_height += 0.25 *sign # critical step with this value --> timing is ok --> maybe ok for now
    #             # # print('setToWidth row 11', timer() - start)
    #
    #     # print('setToWidth row 11', timer() - start)
    #     # print('setToWidth row 12', timer() - start)
    #
    #
    #     self.updateBoudingRect() # I should also updat the bounding rect of output -> that is my error
    #
    #     # print('setToWidth row last', timer() - start)

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
        return 0

    # @return the block incompressible width
    def getIncompressibleWidth(self):
        if not self:
            return 0
        return (len(self) - 1.) * self.space

    # can I pass an ignoring char if some images need be ignored
    # probably need return the increment
    def setLettering(self, letter):
        # TODO implement increment
        # set letter for increasing stuff # check if letter is string or letter or array if so do or do not increase letters
        for img in self:
            img.setLettering(letter)
        # self.letter = letter

    # return all the solvable equations
    # all pairs of heights should be equal in a row
    # think how to do that
    # shall I actually build solvable equation sets ???
    def get_equation(self):
        from sympy import nsolve, exp, Symbol
        # TODO --> maybe get a width equation and a height one

        # need get the equa of inner also
        # maybe need even get the final equations to that I just append them all and get exactly what I need to solve
        # how should I increment that
        # UnevaluatedExpr(x + x)
        if not self:
            return None


        # in fact all heights should be equal and the sum of width would also be equal --> I need both sets of equations in fact --> maybe just do the get equation
        # maybe I should just append the equations
        # append both x and y equations and do the maths with them --> put them into a solvable form
        # what is the best way to do that ???
        equas_w = []
        equas_h = []
        equas_to_solve = []
        for img in self:
            equa_w, equa_h, equa_to_solve = img.get_equation()

            if equa_w:
                equas_w.extend(equa_w)
            if equa_h:
                equas_h.extend(equa_h)
            if equa_to_solve:
                equas_to_solve.extend(equa_to_solve)

            # do I need increment ??? --> maybe not or maybe yes
            # fig_tools.common_value+=1
        # do I need a formula there # that would be the sum of others
        # final equation is a concatenate of all equa for width
        # id(self) # to set the variable
        # --> is there an additional factor that would handle the resize if they are all the same size ???

        # in a row all heights need be the same --> must put the equas so that this is true
        # and in fact just return one equation for a row the top most row that needs be resized


        for iii in range(1, len(equas_h)):
            # print(iii)
            # first - sec = 0
            equa = equas_h[0] - equas_h[iii]
            equas_to_solve.append(equa)
        # if len(equa_h)==1:
        #     equa = equa_h[0] - equa_h[1]
        #     equas_to_solve.append(equa)

        final_equa = None
        for equa in equas_w:
            if final_equa is None:
                final_equa = equa
            else:
                final_equa += equa

        # see how I can do that
        # variable = Symbol('a'+str(id(self)))
        # final_equa

        # this is the equation I want
        final_equa += self.getIncompressibleWidth()
        equas_to_solve.append(final_equa) # pb is that for this equa I need to add the width --> how to I do that
        # this lists the variables in the formula --> do I don't need to store them separately
        # print(final_equa.free_symbols)  # returns all the variables contained in it

        # since they all have same height I need just return one equation
        return equas_w, [equas_h[0]], equas_to_solve

    # make no sense ??? la height n'est pas la somme mais juste une des valeurs de la colonne --> c'est tout je pense en fait
    # def get_height_equation(self):
    #     # TODO --> maybe get a width equation and a height one
    #
    #     # need get the equa of inner also
    #     # maybe need even get the final equations to that I just append them all and get exactly what I need to solve
    #     # how should I increment that
    #     # UnevaluatedExpr(x + x)
    #     equas = []
    #     for img in self:
    #         equas.append(img.get_height_equation())
    #         # do I need increment ??? --> maybe not or maybe yes
    #         # fig_tools.common_value+=1
    #     # do I need a formula there # that would be the sum of others
    #     # final equation is a concatenate of all equa for width
    #     # id(self) # to set the variable
    #     # --> is there an additional factor that would handle the resize if they are all the same size ???
    #     final_equa = None
    #     for equa in equas:
    #         if final_equa is None:
    #             final_equa = equa
    #         else:
    #             final_equa += equa
    #
    #     # see how I can do that
    #     # variable = Symbol('a'+str(id(self)))
    #     final_equa += self.getIncompressibleHeight() # useless because 0 but ok still
    #
    #     # this lists the variables in the formula --> do I don't need to store them separately
    #     print(final_equa.free_symbols)  # returns all the variables contained in it
    #
    #     return final_equa


    # # hack to make column hashable to be able to add it to a dict, see https://stackoverflow.com/questions/10994229/how-to-make-an-object-properly-hashable
    # def __eq__(self, other):
    #     try:
    #         return hash(str(self)) == other.__hash__()
    #     except:
    #         return False
    #
    # # hack to make column hashable to be able to add it to a dict, see https://stackoverflow.com/questions/10994229/how-to-make-an-object-properly-hashable
    # def __hash__(self):
    #     return hash(str(self))

    # allow to duplicate object
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

if __name__ == '__main__':
    img1 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/00.png')
    img2 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
    img3 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/02.png')
    img4 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/03.png')
    # result = img1 + img2

    row = Row(img1, img2)
    row2 = Row(img3, img4)

    # peut etre la aussi le scripting language sera la clef
    # faire aussi un code de packing et de truc de ce style


    # print(result)
    result = row + row2
    print(result)
    result.sameHeight(3)
    result.setToWidth(128)

    print('equa', result.get_equation())
    # print('equa2', result.get_height_equation())

    print(result)

    # preview([result])
    preview(result)

    # not bad --> also do a
    # by default add all to the image
    # maybe have a 'list' command that can return the content of the IQL dict --> so that one can interact with them and one can also list the sub elements in EZFIG --> can be useful

    # TODO --> do a show of the stuff so that one can see the stuff

    # TODO --> do a plot of this
    # --> maybe need open an ezfig window --> TODO

    # find easy ways to letter and autoincrement sutff --> TODO
    # maybe store all as a script --> easy to reexecute --> but need store all images in the desired format
