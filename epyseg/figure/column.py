# TODO --> handle inner selection on double click or synchronize to tab name --> if image --> go there, if row --> only allow rows, pb is that rows can also contain other rows --> makes it complex... --> maybe that will not work

# TODO --> do all
# in fact + adds an image / adds a new row or a new col in fact --> good idea
# should I have two figs --> one horiz and one vertical does that make sense and is that useful ??? think about it
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

# single row size is ok --> the bug only occurs when several rows are combined --> do a rapid demo sample

# panel1D --> orientation --> hor/ver
# panel2D --> more stuff and same size required
import random

from epyseg.draw.shapes.image2d import Image2D
from epyseg.draw.shapes.point2d import Point2D
from epyseg.draw.shapes.rect2d import Rect2D
from PyQt5.QtCore import QPointF, QRectF
# from sympy import nsolve, exp, Symbol
# logger
# from epyseg.figure import fig_tools
from epyseg.figure.alignment import alignLeft, setToWidth2, updateBoudingRect, packY, packX, _brute_force_find_height, \
    _brute_force_find_width
from epyseg.figure.fig_tools import preview, get_master_bounds2, get_common_scaling_factor
from epyseg.tools.logger import TA_logger

logger = TA_logger()


# keep this in mind: https://stackoverflow.com/questions/34827132/changing-the-order-of-operation-for-add-mul-etc-methods-in-a-custom-c

# do rows and columns and no need to take images --> they just pack horiz or vertically and that's it each image or stuff handles itslef
# a row like that cannot be appended another row or can it --> yes in fact just add to it

# shall i make it a qrectf ???? or not or shall I really keep it as that
# shall I really use p1 or the x coord of the rect --> maybe simpler to use qrectf in fact
# TODO check everything and clean things up
# think about the expected behaviour and handle that
# for scale --> be careful and always be sure to keep original

# scale --> see how to handle self scale --> shall it handle it or not
class Column(Rect2D):

    def __init__(self, *args, space=3, height=None):
        self.images = []
        # if args:
        for img in args:
            # if isinstance(img, Column):
            #     self.images += img.images
            # else:
            if img.isSet:
                self.images.append(img)
        # init super empty
        super(Rect2D, self).__init__()
        self.isSet = True  # required to allow dragging
        self.space = space
        self.sameWidth(space=space)
        self.heightInPixel = height
        self.setToHeight(height)
        self.updateBoudingRect()

    def __ior__(self, other):
        return self.__or__(other=other)

    def __or__(self, other):
        return self.__floordiv__(other=other)

    def __mod__(self, other):
        if isinstance(other, Column):
            pos = self.get_P1()
            other_pos = other.get_P1()
            self.set_P1(other_pos)
            other.set_P1(pos)
        else:
            logger.error('swapping not implemented yet for ' + str(type(other)))

    # the add should be append to last row or create last element as a row and append to it

    # def __add__(self, other):
    def __truediv__(self, other):
        # from epyseg.figure.panel import Panel

        if isinstance(other, Column):
            P1 = self.get_P1()
            final_height = self.height()
            # print(self.images)
            # print(len(self.images))
            # print(other.images)
            # print(len(other.images))
            self.images = self.images + other.images
            # self.sameWidth(space=self.space)
            self.setToHeight(final_height)
            self.set_P1(P1)
        # elif isinstance(other, Image2D):
        else:
            P1 = self.get_P1()
            final_height = self.height()
            self.images.append(other)
            # self.sameWidth(space=self.space)
            self.setToHeight(final_height)
            self.set_P1(P1)

        # elif isinstance(other, Panel):
        #     self.images.append(other)
        #     self.sameHeight(space=self.space)
        #     self.setToWidth(self.widthInPixel)
        return self

    # def __truediv__(self, other):
    def __floordiv__(self, other):
        from epyseg.figure.row import Row
        P1 = self.get_P1()
        # pas mal mais faudrait remettre à la taille souihaitée
        # final_width = self.width()
        row = Row(self, other, space=self.space)
        row.sameHeight(self.space)

        # print('in there')

        # row.setToWidth(final_width)
        row.set_P1(P1)
        return row

    # same here make it smarter
    def __sub__(self, other):
        if other in self.images:
            self.images.remove(other)
        self.setToHeight(self.heightInPixel)
        return self

    def is_empty(self):
        if self.images is None or not self.images:
            return True
        return False

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

    def __len__(self):
        if self.images is None:
            return 0
        return len(self.images)

    # def setWidthInPixel(self, heightInPixel):
    #     self.heightInPixel = heightInPixel
    #     self.packY()

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

    # I think there is a bug here --> see how to handle that in fact
    # TODO should I use qrectf or rect2D and how are the two in sync
    def updateBoudingRect(self):
        '''updates the image bounding rect depending on content'''
        x = None
        y = None
        x2 = None
        y2 = None

        # why are they not up o date ??? is that because components are not up to date
        # or is that because the rect is not ok ??? --> shall I use rect2D or rectf --> + do the code once for good
        # shall I update it ???? or not
        # why P1

        # if not self:
        #     return
        for img in self:

            # topLeft = img.get_P1()
            topLeft = img.boundingRect().topLeft()
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

        # print(x,y,x2-x, y2-y)
        self.setX(x)
        self.setY(y)
        self.setWidth(x2 - x)
        self.setHeight(y2 - y)

        # print('bounds from inside', self)
        #
        # bounds = get_master_bounds2(self.images)
        # self.setX(bounds.x())
        # self.setY(bounds.y())
        # self.setWidth(bounds.width())
        # self.setHeight(bounds.height())
        # return bounds

    # def boundingRect(self):
    #     return self.updateBoudingRect()
    #     # return get_master_bounds2(self.images)

    def setOrigin(self, *args):
        self.set_P1(*args)

    def set_P1(self, *args):
        curP1 = self.get_P1()
        Rect2D.set_P1(self, *args)
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

    def sameWidth(self, space):
        if space is None:
            space = 0
        max_width = None
        # if not self:
        #     return
        for img in self:
            if max_width is None:
                # print('TUTU',img)
                # print('TUTA',img, img.boundingRect(), img.boundingRect().height)
                # print('TOTO',img, img.boundingRect(), img.boundingRect().height())
                max_width = img.boundingRect().width()
            max_width = max(img.boundingRect().width(), max_width)
        for img in self:
            img.setToWidth(max_width)

        self.packY(space)
        self.alignLeft()  # updateBounds=False
        self.updateBoudingRect()

    # Aligns objects within the block
    def arrangeRow(self):
        self.alignLeft()

    # Align vectorial objects to the top
    # in fact should align in y
    def alignLeft(self):  # , updateBounds=False
        first_left = None
        for img in self:
            cur_pos = img.get_P1()
            if first_left is None:
                first_left = cur_pos
            img.set_P1(first_left.x(), cur_pos.y())

        # if updateBounds:
        self.updateBoudingRect()


    # ça marche mais peut surement etre grandement amelioré
    def setToHeight(self, height_in_px):
        # from timeit import default_timer as timer
        # start = timer()
        if height_in_px is None:
            return
        # pure_image_height = (self.boundingRect().height()) - self.getIncompressibleHeight()
        # # print(width_in_px, self.getIncompressibleWidth(), (nb_cols - 1.) * self.space )
        # height_in_px -= self.getIncompressibleHeight()
        # ratio = height_in_px / pure_image_height
        # for img in self:
        #     img.setToHeight(img.boundingRect().height() * ratio)
        # self.packY(self.space)
        # self.updateBoudingRect()
        # print('desired ',height_in_px)
        # topleft = Point2D(self.boundingRect().topLeft())

        topleft = Point2D(self.boundingRect().topLeft())
        # causes a bug I think
        self.sameWidth(self.space)
        # print('setToWidth row 1', timer() - start)

        # print('setToWidth row 2', timer() - start)
        # alignTop(topleft, *self.images)
        # print('setToWidth row 3', timer() - start)
        # packX(self.space, topleft, *self.images)

        # self.updateBoudingRect()  # I should also updat the bounding rect of output -> that is my error

        sizes = []
        for elm in self:
            sizes.append(elm.height())
            # print("inner size", elm)

        # new way to compute size --> much smarter than previous way --> start replace everywhere
        # sizes
        scaling = get_common_scaling_factor(sizes, height_in_px, self.space)
        # print('scaling, ', scaling)
        # need set to scale
        # need

        for elm in self:
            # elm.setWidth(scaling*elm.width())
            # elm.setHeight(scaling*elm.height())
            try:
                elm.scale *= scaling
            except:
                # elm.setToHeight(height_in_px)
                elm.setToHeight(elm.height() * scaling)
            # elm.scale *= scaling

        # pb there is a bug and the image and stuff don't match definitely my new code is better and smarter
        # this does noyt work but why ????
        # need pack them in x
        packY(self.space, None, *self.images)

        min_h = 10000000
        max_h = 0
        for elm in self:
            width = elm.width()
            # print('size of inner elm', )
            min_h = min(min_h, width)
            max_h = max(max_h, width)

        if max_h-min_h > 0.05:
            # TODO do something better for gigantic images
            # closest = _brute_force_find_width(self, min_h - 1, max_h + 1, (max_h - min_h) / 10, height_in_px)
            #
            # if closest is not None:
            #     if closest[0] - height_in_px < 0.05:
            #         self.setToWidth(closest[2])
            #         return closest
            # else:
            #     closest = _brute_force_find_width(self, min_h - 1, max_h + 1, 1, height_in_px)
            #
            # value = closest[0]
            # closest = _brute_force_find_width(self, value - 1, value + 1, 1, height_in_px)

            closest = _brute_force_find_width(self, min_h - 1, max_h + 1, 1, height_in_px)

            # print('closest1', closest)
            value = closest[0]
            if closest[0] - height_in_px < 0.05:
                self.setToWidth(closest[2])
                return closest
            # for height in np.arange(value - 1, value + 1, 0.01):
            #     col1.setToHeight(height)
            #     print('bob', height, col1.width(), col1.height())
            #     if (col1.width() - desired_width) <= mn and col1.width() - desired_width >= 0:
            #         closest = (col1.width(), col1.height(), height)
            #         mn = col1.width() - desired_width

            # it is never finding the solution
            closest = _brute_force_find_width(self, value - 0.5, value + 0.5, 0.1, height_in_px)

            if closest[0] - height_in_px < 0.05:
                self.setToWidth(closest[2])
                return closest
            # maybe skip steps if good enough --> close enough to desired value

            # print('closest, value-1, value+1',closest, value-1, value+1)

            # --> pas mal et assez rapide --> pourrait faire n iterations
            # bug can be 0 --> never crosses
            value = closest[0]
            closest = _brute_force_find_width(self, value - 0.1, value + 0.1, 0.01, height_in_px)  # almost perfect

            # (128.00066196944778, 42.80999999999944, 42.80999999999944) --> solution est à peu pres ça --> comment la trouver
            # (128.5352647965376, 43.0, 43) # ineteger search then refinement --> could search between max height and min height of the shape of the column
            # (128.00066196944888, 42.80999999999984, 42.80999999999984)

            # this is a very good estimate

            self.setToWidth(closest[2])

        packY(self.space, None, *self.images)
        self.set_P1(topleft)
        self.updateBoudingRect()

        #
        #
        # alignLeft(topleft, *self.images)
        # packY(self.space, topleft, *self.images)
        # setToHeight2(self.space, height_in_px, *self.images)
        # bounds = updateBoudingRect(*self.images)
        #
        # max_width = -1
        # cur_height = bounds.height()
        # cur_width = bounds.width()
        # all_widths_are_the_same = True
        # for img in self.images:
        #     max_width = max(max_width, img.boundingRect().width())
        #     if max_width != cur_width:
        #         all_widths_are_the_same = False
        #
        # if not all_widths_are_the_same:
        #     setToWidth2(self.space, max_width, *self.images)
        #     bounds = updateBoudingRect(*self.images)
        #     all_widths_are_the_same = True
        # else:
        #     bounds = updateBoudingRect(*self.images)
        #
        # sign = 1
        # if bounds.height() >= height_in_px:
        #     sign = -1
        #
        # # bug cause also cyclic inside --> need pass whether increase or decrease once for good initially
        # # print(bounds.height(), 'vs heigth desired', height_in_px, 'sign', sign)
        # # print('heigth', all_widths_are_the_same)
        # # dirty height/width fix for complex panels figure out the math for it and replace this code
        # # TODO also speed this up so that the action is only faked and not really done (just done once at the end to gain time)
        # if not all_widths_are_the_same or (
        #         bounds.height() != height_in_px and abs(bounds.height() - height_in_px) > 0.3):
        #     # print('setToHeight col 9', timer() - start)
        #     while True:
        #         # print(max_width)
        #         setToWidth2(self.space, max_width, *self.images)
        #         bounds = updateBoudingRect(*self.images)
        #         if sign < 0:
        #             if bounds.height() <= height_in_px:
        #                 break
        #         else:
        #             if bounds.height() >= height_in_px:
        #                 break
        #         max_width += 1 * sign
        #
        #     cur_height = bounds.height()
        #     if bounds.height() != height_in_px:
        #         if bounds.height() >= height_in_px:
        #             sign = -1
        #         else:
        #             sign = 1
        #         # print('setToHeight col 10', timer() - start)
        #         # max_width -= 1
        #         while True:
        #             setToWidth2(self.space, max_width, *self.images)
        #             bounds = updateBoudingRect(*self.images)
        #             if sign < 0:
        #                 if bounds.height() <= height_in_px:
        #                     break
        #             else:
        #                 if bounds.height() >= height_in_px:
        #                     break
        #             max_width += 0.25 * sign  # critical time consuming step --> seems ok with this value
        #
        # self.updateBoudingRect()

    # will not work with scale should I override scale here of the rect2D...
    # Forces the block to be of width (width_in_px)
    # def setToHeight_old(self, height_in_px):
    #     # from timeit import default_timer as timer
    #     # start = timer()
    #     if height_in_px is None:
    #         return
    #     # pure_image_height = (self.boundingRect().height()) - self.getIncompressibleHeight()
    #     # # print(width_in_px, self.getIncompressibleWidth(), (nb_cols - 1.) * self.space )
    #     # height_in_px -= self.getIncompressibleHeight()
    #     # ratio = height_in_px / pure_image_height
    #     # for img in self:
    #     #     img.setToHeight(img.boundingRect().height() * ratio)
    #     # self.packY(self.space)
    #     # self.updateBoudingRect()
    #     # print('desired ',height_in_px)
    #     topleft = Point2D(self.boundingRect().topLeft())
    #     alignLeft(topleft, *self.images)
    #     packY(self.space, topleft, *self.images)
    #     setToHeight2(self.space, height_in_px, *self.images)
    #     bounds = updateBoudingRect(*self.images)
    #
    #     max_width = -1
    #     cur_height = bounds.height()
    #     cur_width = bounds.width()
    #     all_widths_are_the_same = True
    #     for img in self.images:
    #         max_width = max(max_width, img.boundingRect().width())
    #         if max_width != cur_width:
    #             all_widths_are_the_same = False
    #
    #     if not all_widths_are_the_same:
    #         setToWidth2(self.space, max_width, *self.images)
    #         bounds = updateBoudingRect(*self.images)
    #         all_widths_are_the_same = True
    #     else:
    #         bounds = updateBoudingRect(*self.images)
    #
    #     sign = 1
    #     if bounds.height() >= height_in_px:
    #         sign = -1
    #
    #     # bug cause also cyclic inside --> need pass whether increase or decrease once for good initially
    #     # print(bounds.height(), 'vs heigth desired', height_in_px, 'sign', sign)
    #     # print('heigth', all_widths_are_the_same)
    #     # dirty height/width fix for complex panels figure out the math for it and replace this code
    #     # TODO also speed this up so that the action is only faked and not really done (just done once at the end to gain time)
    #     if not all_widths_are_the_same or (
    #             bounds.height() != height_in_px and abs(bounds.height() - height_in_px) > 0.3):
    #         # print('setToHeight col 9', timer() - start)
    #         while True:
    #             # print(max_width)
    #             setToWidth2(self.space, max_width, *self.images)
    #             bounds = updateBoudingRect(*self.images)
    #             if sign < 0:
    #                 if bounds.height() <= height_in_px:
    #                     break
    #             else:
    #                 if bounds.height() >= height_in_px:
    #                     break
    #             max_width += 1 * sign
    #
    #         cur_height = bounds.height()
    #         if bounds.height() != height_in_px:
    #             if bounds.height() >= height_in_px:
    #                 sign = -1
    #             else:
    #                 sign = 1
    #             # print('setToHeight col 10', timer() - start)
    #             # max_width -= 1
    #             while True:
    #                 setToWidth2(self.space, max_width, *self.images)
    #                 bounds = updateBoudingRect(*self.images)
    #                 if sign < 0:
    #                     if bounds.height() <= height_in_px:
    #                         break
    #                 else:
    #                     if bounds.height() >= height_in_px:
    #                         break
    #                 max_width += 0.25 * sign  # critical time consuming step --> seems ok with this value
    #
    #     self.updateBoudingRect()

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

    def setToWidth(self, width_in_px):
        if width_in_px is None:
            return
        # nb_cols = len(self)

        # probably need do more complex stuff here and take or not scaled size --> TODO
        # in fact c'est l'original size que je veux ???
        # pure_image_width = (self.boundingRect().width()) - self.getIncompressibleWidth()
        # # print(width_in_px, self.getIncompressibleWidth(), (nb_cols - 1.) * self.space )
        # width_in_px -= self.getIncompressibleWidth()
        # ratio = width_in_px / pure_image_width
        # for img in self:
        #     img.setToWidth(img.boundingRect().width() * ratio)
        # self.packY(self.space)
        # self.updateBoudingRect()
        topleft = Point2D(self.boundingRect().topLeft())
        alignLeft(topleft, *self.images)
        setToWidth2(self.space, width_in_px, *self.images)


        self.updateBoudingRect()  # this was my bug --> really needs be done # probably not needed now that I use set to height

    # @return the block incompressible width
    def getIncompressibleHeight(self):
        nb_cols = len(self)
        extra_space = (nb_cols - 1.) * self.space
        return extra_space

    def getIncompressibleWidth(self):
        extra_space = 0
        return extra_space

    # can I pass an ignoring char if some images need be ignored
    # probably need return the increment
    def setLettering(self, letter):
        # TODO implement increment
        # set letter for increasing stuff # check if letter is string or letter or array if so do or do not increase letters
        for img in self:
            img.setLettering(letter)
        # self.letter = letter

    def get_equation(self):
        # TODO --> maybe get a width equation and a height one
        from sympy import nsolve, exp, Symbol
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

        # print(len(self))
        for img in self:

            # print(img)

            equa_w, equa_h, equa_to_solve = img.get_equation()

            # print(equa_w)

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

        # in a cal all weidth need be the same --> must put the equas so that this is true
        # and in fact just return one equation for a row the top most row that needs be resized
        # print(equas_w)
        # print("tutu", len(equas_w), len(equas_h))

        for iii in range(1, len(equas_w)):
            # print('entering')
            # print(iii)
            # first - sec = 0
            equa = equas_w[0] - equas_w[iii]
            equas_to_solve.append(equa)
        # if len(equa_w)==1:
        #     equa = equa_w[0] - equa_w[1]
        #     equas_to_solve.append(equa)

        final_equa = None
        for equa in equas_h:
            if final_equa is None:
                final_equa = equa
            else:
                final_equa += equa

        # see how I can do that
        # variable = Symbol('a'+str(id(self)))
        # final_equa

        # this is the equation I want
        final_equa += self.getIncompressibleHeight()
        equas_to_solve.append(final_equa)  # pb is that for this equa I need to add the width --> how to I do that
        # this lists the variables in the formula --> do I don't need to store them separately
        print(final_equa.free_symbols)  # returns all the variables contained in it

        # since they all have the same width I need return just one width equation
        return [equas_w[0]], equas_h, equas_to_solve

    # def get_width_equation(self):
    #     # TODO --> maybe get a width equation and a height one
    #
    #     # need get the equa of inner also
    #     # maybe need even get the final equations to that I just append them all and get exactly what I need to solve
    #     # how should I increment that
    #     # UnevaluatedExpr(x + x)
    #     if not self:
    #         return None
    #
    #     equas = []
    #     for img in self:
    #         equas.append(img.get_width_equation())
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
    #     # final_equa
    #
    #     # this is the equation I want
    #     final_equa += self.getIncompressibleWidth()
    #     # this lists the variables in the formula --> do I don't need to store them separately
    #     print(final_equa.free_symbols)  # returns all the variables contained in it
    #
    #     return final_equa
    #
    #     # make no sense ??? la height n'est pas la somme mais juste une des valeurs de la colonne --> c'est tout je pense en fait
    #
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
    #     final_equa += self.getIncompressibleHeight()  # useless because 0 but ok still
    #
    #     # this lists the variables in the formula --> do I don't need to store them separately
    #     print(final_equa.free_symbols)  # returns all the variables contained in it
    #
    #     return final_equa


    # allows to duplicate an object
    # marche pas --> ilo faut le hard coder
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

if __name__ == '__main__':
    img1 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/00.png')
    img2 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
    img3 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/02.png')
    img4 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/03.png')
    img5 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/04.png')
    img6 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/05.png')
    img7 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/06.png')
    img8 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/07.png')
    img9 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/08.png')
    if False:
        result = img1 | img2
        result |= img3
        result |= img4

        print(result)
        preview(result)
    if False:
        result = Column(img4, img2, img3)  # ça a l'air de marcher
        result |= img1

        result.setToHeight(300)

        # all seems ok now --> just try to finalize all
        preview(result)

    if False:
        from epyseg.figure.row import Row

        row1 = Row(img1, img2)
        row2 = Row(img3, img4)
        row1 |= row2

        preview(row1)

    if False:
        from epyseg.figure.row import Row

        row1 = Row(img1, img2)
        row2 = Row(img3, img4)
        row1 /= row2

        preview(row1)

    if False:
        col1 = Column(img1, img2)
        col2 = Column(img3, img4)
        col1 |= col2

        preview(col1)


    if False:
        # attempt to solve a complex equation --> does not work because solution is too complex --> would need better estimates --> maybe do that later
        from epyseg.figure.row import Row

        # all  works if space is 0 but does not work otherwise!!!
        # need take incompressibility into account
        # or do it without space then fit the image within the same space taking into account incompressibility
        # --> will that work ---> not so easy in fact because incompressibility will change the AR and I need that
        # --> brute force is a good solution then

        # in both cases the width is ok, but in one case the height is bad --> need fix so that the height of the two is the same --> need decrease height of big and increase height of the other

        # space = random.choice([0,3])
        space = 3
        col1 = Column(img1, img2, space=space)
        print(type(col1))  # --> this is now a row

        # print('equa', col1.get_equation(), len(col1))
        # import sys
        # sys.exit(0)

        # col2 = Column(img3, img4, space=space)
        # col1 /= col2
        # a col with two images --> two width must be equal
        # print('equa', col1.get_equation(), len(col1))
        # import sys
        # sys.exit(0)

        row1 = Row(img5, img6, img7, space=space)

        col1 |= row1

        print(type(col1))  # --> this is now a row
        equas_w, equas_h, equas_to_solve = col1.get_equation()
        print('equa', (equas_w, equas_h, equas_to_solve), len(col1))

        # TODO --> try to solve all these equas
        # get all equas to solve in the end and edit the last one so that it matches the desired size
        # see if there is a bug ???
        variables = []
        for equa in equas_to_solve:
            variables.extend(equa.free_symbols)

        variables = list(
            set(variables))  # remove dupes # --> it cannot find the solution --> probably because estimate is too far away from the stuff --> need a better starting estimate --> need pack and get to the closest value
        # in a way maybe brute force is smarter .... in the end --> see

        estimates = [0.5 for _ in variables]

        print(variables)

        print(len(variables))
        print(len(equas_to_solve))

        # or I do have a bug in my equas --> maybe that is the problem too
        test = nsolve(equas_to_solve, variables, estimates)
        # Traceback (most recent call last):
        #   File "/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/figure/column.py", line 864, in <module>
        #     test = nsolve(equas_to_solve, variables, estimates)
        #   File "/usr/local/lib/python3.7/site-packages/sympy/utilities/decorator.py", line 88, in func_wrapper
        #     return func(*args, **kwargs)
        #   File "/usr/local/lib/python3.7/site-packages/sympy/solvers/solvers.py", line 2954, in nsolve
        #     x = findroot(f, x0, J=J, **kwargs)
        #   File "/usr/local/lib/python3.7/site-packages/mpmath/calculus/optimization.py", line 988, in findroot
        #     % (norm(f(*xl))**2, tol))
        # ValueError: Could not find root within given tolerance. (4.11873256516330155209 > 2.16840434497100886801e-19)
        # Try another starting point or tweak arguments.
        #
        # Process finished with exit code 1
        # -> this is probably a bug!!!
        # The second command tried to solve the same equations with a different initial guess, one farther from the actual solution. This time sympy failed hard. This is regrettably common when solving nonlinear equations with bad initial guesses for the solution. When we see that message:
        # Try another starting point or tweak arguments.
        # it means that we need to guess again on our starting point. It is often a good idea when solving nonlinear systems to plot the functions to get an idea of where they are zero.

        # gives me an error because estimates are too far from point -> maybe brute force is best and faster or start from brute force to get a rough estimate and solve it numerically!!!
        print(test)

        import sys
        sys.exit(0)

    if True:
        from epyseg.figure.row import Row

        # all  works if space is 0 but does not work otherwise!!!
        # need take incompressibility into account
        # or do it without space then fit the image within the same space taking into account incompressibility
        # --> will that work ---> not so easy in fact because incompressibility will change the AR and I need that
        # --> brute force is a good solution then

        # in both cases the width is ok, but in one case the height is bad --> need fix so that the height of the two is the same --> need decrease height of big and increase height of the other

        # space = random.choice([0,3])
        space = 3
        col1 = Column(img1, img2, space=space)
        print(type(col1))  # --> this is now a row

        # print('equa', col1.get_equation(), len(col1))
        # import sys
        # sys.exit(0)

        # col2 = Column(img3, img4, space=space)
        # col1 /= col2
        # a col with two images --> two width must be equal
        # print('equa', col1.get_equation(), len(col1))
        # import sys
        # sys.exit(0)


        row1 = Row(img5, img6, img7, space=space)

        col1 |= row1

        desired_height = 128

        col1.setToWidth(desired_height)  # --> need take into account the incompressible part of each and every object to compute the scaling --> TODO
        # so they don't have the same width and height which is a pb especially because I asked for it!!!!
        # they are not same height --> why is that
        for elm in col1:
            print('size of inner elm', elm)

        # brute force to really find same height of object
        import numpy as np



        # print('min_h, max_h',min_h, max_h)

        # en brute force ça marche mais pas top qd meme et en plus c'est lent --> comment puis-je calculer le truc numeriquement
        # col1.sameHeight(3)



        # ok then do the brute force stuff if needed

        # now this is really fast
        # so if I don't do crap in fact brute force is fast
        # I could also detect the sign inversion and stop after it by just taking the closest
        # or I could further refine by scanning in between the two
        # maybe get the max and min size of all the images in there and screen in between them


        #pas mal --> presque ça --> peut etre le plus simple c'est vraiment la brute force
        # could I estimate the 0 intersection from a set of values and use that ???

        # could do the same for a col
        # see how to do that






        # min_h = 10000000
        # max_h = 0
        # for elm in col1:
        #     height = elm.height()
        #     # print('size of inner elm', )
        #     min_h = min(min_h, height)
        #     max_h = max(max_h, height)
        # closest = _brute_force_find_height(col1, min_h - 1, max_h + 1,0.5, desired_height)
        #
        #
        # # print('closest1', closest)
        # value = closest[1]
        # # for height in np.arange(value - 1, value + 1, 0.01):
        # #     col1.setToHeight(height)
        # #     print('bob', height, col1.width(), col1.height())
        # #     if (col1.width() - desired_width) <= mn and col1.width() - desired_width >= 0:
        # #         closest = (col1.width(), col1.height(), height)
        # #         mn = col1.width() - desired_width
        #
        # closest = _brute_force_find_height(col1, value - 0.5, value + 0.5,0.01,desired_height)
        #
        # # --> pas mal et assez rapide --> pourrait faire n iterations
        # value = closest[1]
        # closest = _brute_force_find_height(col1, value - 0.1, value + 0.1, 0.001,desired_height) # almost perfect
        #
        #
        # # (128.00066196944778, 42.80999999999944, 42.80999999999944) --> solution est à peu pres ça --> comment la trouver
        # # (128.5352647965376, 43.0, 43) # ineteger search then refinement --> could search between max height and min height of the shape of the column
        # # (128.00066196944888, 42.80999999999984, 42.80999999999984)
        #
        # # this is a very good estimate
        #
        # col1.setToHeight(closest[2])
        print(type(col1))
        print(col1)

        # print(closest)



        preview(col1)

        # rewrite the equa with incompressibility... --> not so easy in fact
        # will then work if I set to min height in fact --> yes or no ??? --> not sure
        # in fact that is tough due to incompressible size
        # alternatively
        # the pb is that incompressibility changes the AR of the object --> so that makes my equation harder to handle and some of my assumptions wrong
        # need a smarter equation and need assume ratio width height not the same
        # TODO --> ...
        # need speed up the tracking
        # in tracking ask for fast vs precise

        # TODO --> add combinations of rows
