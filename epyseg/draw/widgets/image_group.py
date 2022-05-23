from epyseg.draw.shapes.rect2d import Rect2D
from PyQt5.QtCore import QPointF, QRectF
# logger
from epyseg.tools.logger import TA_logger

logger = TA_logger()

# en fait le pack vaudrait mieux le faire là ça serait bcp plus utile
# keep this in mind: https://stackoverflow.com/questions/34827132/changing-the-order-of-operation-for-add-mul-etc-methods-in-a-custom-c
class group(Rect2D):

    def __init__(self, *args, space=3):
        self.images = []
        for img in args:
            #     if isinstance(img, Column):
            #         self.images += img.images
            #     else:
            self.images.append(img)
        # init super empty
        super(Rect2D, self).__init__()
        # self.isSet = True # required to allow dragging
        self.space = space

        # self.sameHeight(space=space)
        # self.widthInPixel = width
        # self.setToWidth(width)
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

    def __iter__(self):
        ''' Returns the Iterator object '''
        self._index = -1
        return self

    def __next__(self):
        ''' Returns the next image or panel in the row '''
        self._index += 1
        if self._index < len(self.images):
            return self.images[self._index]
        # End of Iteration
        raise StopIteration

    # Forces the block to be of width (width_in_px)
    def setToWidth(self, width_in_px):

        # make all content same height then compute width and incompressible width of each object in fact

        if width_in_px is None:
            return
        nb_cols = len(self)
        pure_image_width = (self.boundingRect().width()) - (nb_cols - 1.) * self.space - self.getIncompressibleWidth()
        # print(width_in_px, self.getIncompressibleWidth(), (nb_cols - 1.) * self.space)
        width_in_px -= self.getIncompressibleWidth() + (nb_cols - 1.) * self.space
        ratio = width_in_px / pure_image_width
        for img in self:
            img.setToHeight(img.boundingRect().width() * ratio)


        self.packX(self.space)
        self.updateBoudingRect()


    # compute corresponding row height ???? so that it fits not so easy in fact

    # pb here l'incompressible space est le courant plus la somme des incompressible spaces du contenu
    # def setToWidth(self, width_in_px):
    #
    #     # make all content same height then compute width and incompressible width of each object in fact
    #
    #     if width_in_px is None:
    #         return
    #     nb_cols = len(self)
    #     # pure_image_height = self.boundingRect().height() - self.getIncompressibleHeight()
    #     # incompressible_width =  self.getIncompressibleWidth()
    #
    #     # for img in self:
    #     #     incompressible_width+=img.getIncompressibleWidth()
    #
    #     pure_image_width = (self.boundingRect().width()) -self.getIncompressibleWidth(), (nb_cols - 1.) * self.space
    #
    #     # print(width_in_px, self.getIncompressibleWidth(), (nb_cols - 1.) * self.space)
    #     width_in_px -= self.getIncompressibleWidth(), (nb_cols - 1.) * self.space
    #     # height_in_px -= self.getIncompressibleHeight()
    #     ratio = width_in_px / pure_image_width
    #     for img in self:
    #         print('target width before', width_in_px, img.boundingRect(), ratio)
    #         img.setToWidth(img.boundingRect().width() * ratio)
    #         print('target width after', width_in_px, img.boundingRect())
    #
    #
    #     self.packX(self.space)
    #     self.updateBoudingRect()

    # Forces the block to be of width (width_in_px)
    # tjrs faire ça row by row
    def setToHeight(self, height_in_px):
        # celui la marche mais celui du dessus ne marche pas --> pkoi ???

        if height_in_px is None:
            return
        nb_cols = len(self) #ce ne sera pas tjrs correct --> faudrait faire mieux que ça en fait
        # pure_image_width = (self.boundingRect().width()) -self.getIncompressibleWidth(), (nb_cols - 1.) * self.space
        # incompressible_width = self.getIncompressibleWidth()

        # for img in self:
        #     incompressible_width += img.getIncompressibleWidth()
        pure_image_width = (self.boundingRect().width()) - self.getIncompressibleWidth(), (nb_cols - 1.) * self.space

        pure_image_height = self.boundingRect().height()- self.getIncompressibleHeight()
        # print(height_in_px, self.getIncompressibleHeight(), (nb_cols - 1.) * self.space)
        height_in_px -= self.getIncompressibleHeight()
        ratio = height_in_px / pure_image_height
        for img in self:
            img.setToHeight(img.boundingRect().height() * ratio)

        print('values resize', pure_image_width, pure_image_height, height_in_px, ratio)

        self.packX(self.space)
        self.updateBoudingRect()

    def __len__(self):
        if self.images is None:
            return 0
        return len(self.images)

     # @return the block incompressible width
    def getIncompressibleWidth(self):
        return 0
        #     # in fact need get nb of cols a
        #     nb_cols = len(self)
        #     extra_space = (nb_cols - 1.) * self.space
        #     extra_space += self.getExtraIncompressibleWidth()
        #     return extra_space
        #
    # def getExtraIncompressibleWidth(self):
    #     extra_space = 0
    #     return extra_space
    #
    # # @return the block incompressible height
    def getIncompressibleHeight(self):
        return 0
        #     # in fact need get nb of cols a
        #     nb_rows= len(self)
        #     extra_space = (nb_rows - 1.) * self.space
        #     extra_space += self.getExtraIncompressibleHeight()
        #     return extra_space
    #
    # def getExtraIncompressibleHeight(self):
    #     extra_space = 0
    #     return extra_space

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

    def alignTop(self, updateBounds=False):
        first_left = None
        for img in self:
            cur_pos = img.get_P1()
            if first_left is None:
                first_left = cur_pos
            img.set_P1(cur_pos.x(), first_left.y())
        if updateBounds:
            self.updateMasterRect()