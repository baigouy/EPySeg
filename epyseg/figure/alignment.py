# align objects left right top or bottom

# maybe just need methods and not even a class
# logger
from PyQt5.QtCore import QRectF

from epyseg.draw.shapes.point2d import Point2D
from epyseg.tools.logger import TA_logger

logger = TA_logger()

# TODO replace all functions by that to avoid duplicated code

def setToWidth(space, width_in_px, *objects):
    if width_in_px is None:
        return
    bounds = updateBoudingRect(*objects)
    left = Point2D(bounds.topLeft())
    incompressible_width = 0
    for img in objects:
        incompressible_width+=img.getIncompressibleWidth()
    # si row doit maintenir same height until reached desired width is reached --> change its width --> doit aussi connaitre la incompressible height des objects
    pure_image_width = (bounds.width()) - incompressible_width
    # print(width_in_px, self.getIncompressibleWidth(), (nb_cols - 1.) * self.space )
    width_in_px -= incompressible_width
    ratio = width_in_px / pure_image_width
    for img in objects:
        img.setToWidth(img.boundingRect().width() * ratio)
    packX(space, left, *objects)
    # self.updateBoudingRect()

def setToHeight2(space, height_in_px, *objects):
    if height_in_px is None:
        return
    bounds = updateBoudingRect(*objects)
    left = Point2D(bounds.topLeft())
    incompressible_height = 0
    for img in objects:
        incompressible_height+=img.getIncompressibleHeight()
    # si row doit maintenir same height until reached desired width is reached --> change its width --> doit aussi connaitre la incompressible height des objects
    pure_image_height = (bounds.height()) - incompressible_height
    # print(width_in_px, self.getIncompressibleWidth(), (nb_cols - 1.) * self.space )
    height_in_px -= incompressible_height
    ratio = height_in_px / pure_image_height
    for img in objects:
        img.setToHeight(img.boundingRect().height() * ratio)
    packY(space, left, *objects)
    # self.updateBoudingRect()



#from timeit import default_timer as timer see why so slow --> see what part of computation is slow and remove it
def setToHeight(space, height_in_px, *objects):
    if height_in_px is None:
        return
    bounds = updateBoudingRect(*objects)
    left = Point2D(bounds.topLeft())
    for img in objects:
        img.setToHeight(height_in_px)
    packX(space, left, *objects)
    # self.updateBoudingRect()

def setToWidth2(space, width_in_px, *objects):
    if width_in_px is None:
        return
    bounds = updateBoudingRect(*objects)
    left = Point2D(bounds.topLeft())
    for img in objects:
        img.setToWidth(width_in_px)
    packY(space, left, *objects)
    # self.updateBoudingRect()

def updateBoudingRect(*objects):
        bounding_rect = QRectF()
        '''updates the image bounding rect depending on content'''
        x = None
        y = None
        x2 = None
        y2 = None
        for img in objects:
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

        bounding_rect.setX(x)
        bounding_rect.setY(y)
        bounding_rect.setWidth(x2 - x)
        bounding_rect.setHeight(y2 - y)
        return bounding_rect


def packX(space=3, reference=None, *objects_to_align_with_respect_to_ref):
    skip_first = False
    if objects_to_align_with_respect_to_ref is None or len(objects_to_align_with_respect_to_ref) == 0:
        logger.warning("Nothing to align...")
        return
    if reference is None:
        if len(objects_to_align_with_respect_to_ref) < 2:
            logger.warning("Reference is None, can't align anything...")
            return
        else:
            reference = objects_to_align_with_respect_to_ref[0]
            skip_first = True
            # del objects_to_align_with_respect_to_ref[0]

    last_x = reference.get_P1().x()+reference.width()
    last_y = reference.get_P1().y()

    for i in range(len(objects_to_align_with_respect_to_ref)):
            if i == 0 and skip_first:
                continue
            img = objects_to_align_with_respect_to_ref[i]
            # print('begin', last_x)
            if i != 0:
                last_x += space


            # print('before', last_x, img.boundingRect(), space)
            img.set_P1(last_x, img.get_P1().y())
            # print(img.boundingRect(), last_x, img.get_P1().x())
            last_x = img.boundingRect().x() + img.boundingRect().width()
            # print('end', last_x)
    # self.updateBoudingRect()

    # align everything with respect to ref
    # get first point and align so that the all have the same x

ORIENTATION_TOP_TO_BOTTOM = 0
ORIENTATION_BOTTOM_TO_TOP = 1

# need do a code for two orientations --> from top to bottom and reverse --> can be useful to do...


def packYreverse(space=3, reference=None, *objects_to_align_with_respect_to_ref):
    skip_first = False
    if objects_to_align_with_respect_to_ref is None or len(objects_to_align_with_respect_to_ref) == 0:
        logger.warning("Nothing to align...")
        return
    if reference is None:
        if len(objects_to_align_with_respect_to_ref) < 2:
            logger.warning("Reference is None, can't align anything...")
            return
        else:
            reference = objects_to_align_with_respect_to_ref[0]
            skip_first = True
            # del objects_to_align_with_respect_to_ref[0]

    # last_x = reference.get_P1().x()
    last_y = reference.get_P1().y()

    for i in range(len(objects_to_align_with_respect_to_ref)):
        if i==0 and skip_first:
            continue
        img = objects_to_align_with_respect_to_ref[i]
        if i != 0:
            last_y -= space
            # last_y -= img.boundingRect().height()
        # print(last_y, img.get_P1())
        img.set_P1(img.get_P1().x(), last_y-img.boundingRect().height())
        # get all the bounding boxes and pack them with desired space in between
        # get first point and last point in x
        x = img.boundingRect().x()
        y = img.boundingRect().y()
        # last_x = img.boundingRect().x() + img.boundingRect().width()
        last_y = img.boundingRect().y()
    # self.updateBoudingRect()

def packY(space=3, reference=None, *objects_to_align_with_respect_to_ref):
    skip_first = False
    if objects_to_align_with_respect_to_ref is None or len(objects_to_align_with_respect_to_ref) == 0:
        logger.warning("Nothing to align...")
        return
    if reference is None:
        if len(objects_to_align_with_respect_to_ref) < 2:
            logger.warning("Reference is None, can't align anything...")
            return
        else:
            reference = objects_to_align_with_respect_to_ref[0]
            skip_first = True
            # del objects_to_align_with_respect_to_ref[0]

    # last_x = reference.get_P1().x()
    last_y = reference.get_P1().y() + reference.height()

    for i in range(len(objects_to_align_with_respect_to_ref)):
        if i==0 and skip_first:
            continue
        img = objects_to_align_with_respect_to_ref[i]
        if i != 0:
            last_y += space
        # print(last_y, img.get_P1())
        img.set_P1(img.get_P1().x(), last_y)
        # get all the bounding boxes and pack them with desired space in between
        # get first point and last point in x
        x = img.boundingRect().x()
        y = img.boundingRect().y()
        # last_x = img.boundingRect().x() + img.boundingRect().width()
        last_y = img.boundingRect().y() + img.boundingRect().height()
    # self.updateBoudingRect()


    # align everything with respect to ref
    # get first point and align so that the all have the same x


def alignLeft(reference=None, *objects_to_align_with_respect_to_ref):
    skip_first = False
    if objects_to_align_with_respect_to_ref is None or len(objects_to_align_with_respect_to_ref) == 0:
        logger.warning("Nothing to align...")
        return
    if reference is None:
        if len(objects_to_align_with_respect_to_ref) < 2:
            logger.warning("Reference is None, can't align anything...")
            return
        else:
            reference = objects_to_align_with_respect_to_ref[0]
            skip_first = True
            # del objects_to_align_with_respect_to_ref[0]

    last_x = reference.get_P1().x()
    # last_y = reference.get_P1().y()

    for i in range(len(objects_to_align_with_respect_to_ref)):
        if i==0 and skip_first:
            continue
        img = objects_to_align_with_respect_to_ref[i]

        img.set_P1(last_x, img.get_P1().y())
        # get all the bounding boxes and pack them with desired space in between
        # get first point and last point in x
        # x = img.boundingRect().x()
        # y = img.boundingRect().y()
        # last_x = img.boundingRect().x() + img.boundingRect().width()
        # last_y = img.boundingRect().y() + img.boundingRect().height()
    # self.updateBoudingRect()


    # align everything with respect to ref
    # get first point and align so that the all have the same x


def alignTop(reference=None, *objects_to_align_with_respect_to_ref):
    skip_first = False
    if objects_to_align_with_respect_to_ref is None or len(objects_to_align_with_respect_to_ref) == 0:
        logger.warning("Nothing to align...")
        return
    if reference is None:
        if len(objects_to_align_with_respect_to_ref) < 2:
            logger.warning("Reference is None, can't align anything...")
            return
        else:
            reference = objects_to_align_with_respect_to_ref[0]
            skip_first = True
            # del objects_to_align_with_respect_to_ref[0]

    # last_x = reference.get_P1().x()
    last_y = reference.get_P1().y()

    for i in range(len(objects_to_align_with_respect_to_ref)):
        if i==0 and skip_first:
            continue
        img = objects_to_align_with_respect_to_ref[i]

        img.set_P1(img.get_P1().x(), last_y)
        # get all the bounding boxes and pack them with desired space in between
        # get first point and last point in x
        # x = img.boundingRect().x()
        # y = img.boundingRect().y()
        # last_x = img.boundingRect().x() + img.boundingRect().width()
        # last_y = img.boundingRect().y() + img.boundingRect().height()
    # self.updateBoudingRect()


    # align everything with respect to ref
    # get first point and align so that the all have the same x


def alignBottom(reference=None, *objects_to_align_with_respect_to_ref):
    skip_first = False
    if objects_to_align_with_respect_to_ref is None or len(objects_to_align_with_respect_to_ref) == 0:
        logger.warning("Nothing to align...")
        return
    if reference is None:
        if len(objects_to_align_with_respect_to_ref) < 2:
            logger.warning("Reference is None, can't align anything...")
            return
        else:
            reference = objects_to_align_with_respect_to_ref[0]
            skip_first = True
            # del objects_to_align_with_respect_to_ref[0]

    # last_x = reference.get_P1().x()
    last_y = reference.get_P1().y()+reference.height() # should be the end point of every stuff

    for i in range(len(objects_to_align_with_respect_to_ref)):
        if i==0 and skip_first:
            continue
        img = objects_to_align_with_respect_to_ref[i]


        pos_y = img.get_P1().y()+img.height()
        difference_y = last_y-pos_y
        img.set_P1(img.get_P1().x(), img.get_P1().y()+difference_y)
        # get all the bounding boxes and pack them with desired space in between
        # get first point and last point in x
        # x = img.boundingRect().x()
        # y = img.boundingRect().y()
        # last_x = img.boundingRect().x() + img.boundingRect().width()
        # last_y = img.boundingRect().y() + img.boundingRect().height()
    # self.updateBoudingRect()


    # align everything with respect to ref
    # get first point and align so that the all have the same x


# bug here now with the images
def alignRight(reference=None, *objects_to_align_with_respect_to_ref):
    skip_first = False
    if objects_to_align_with_respect_to_ref is None or len(objects_to_align_with_respect_to_ref) == 0:
        logger.warning("Nothing to align...")
        return
    if reference is None:
        if len(objects_to_align_with_respect_to_ref) < 2:
            logger.warning("Reference is None, can't align anything...")
            return
        else:
            reference = objects_to_align_with_respect_to_ref[0]
            skip_first = True
            # del objects_to_align_with_respect_to_ref[0]

    last_x = reference.get_P1().x()+reference.width() # should be the end point of every stuff
    # last_y = reference.get_P1().y()+reference.height() # should be the end point of every stuff

    for i in range(len(objects_to_align_with_respect_to_ref)):
        if i==0 and skip_first:
            continue
        img = objects_to_align_with_respect_to_ref[i]

        pos_x = img.get_P1().x()+img.width()
        difference_x = last_x-pos_x
        img.set_P1(img.get_P1().x()+difference_x, img.get_P1().y())
        # get all the bounding boxes and pack them with desired space in between
        # get first point and last point in x
        # x = img.boundingRect().x()
        # y = img.boundingRect().y()
        # last_x = img.boundingRect().x() + img.boundingRect().width()
        # last_y = img.boundingRect().y() + img.boundingRect().height()
    # self.updateBoudingRect()


    # align everything with respect to ref
    # get first point and align so that the all have the same x


def alignCenterH(reference=None, *objects_to_align_with_respect_to_ref):
    skip_first = False
    if objects_to_align_with_respect_to_ref is None or len(objects_to_align_with_respect_to_ref) == 0:
        logger.warning("Nothing to align...")
        return
    if reference is None:
        if len(objects_to_align_with_respect_to_ref) < 2:
            logger.warning("Reference is None, can't align anything...")
            return
        else:
            reference = objects_to_align_with_respect_to_ref[0]
            skip_first = True
            # del objects_to_align_with_respect_to_ref[0]

    last_x = reference.get_P1().x()+reference.width()/2. # should be the end point of every stuff
    # last_y = reference.get_P1().y()+reference.height() # should be the end point of every stuff

    for i in range(len(objects_to_align_with_respect_to_ref)):
        if i==0 and skip_first:
            continue
        img = objects_to_align_with_respect_to_ref[i]

        pos_x = img.get_P1().x()+img.width()/2.
        difference_x = last_x-pos_x
        img.set_P1(img.get_P1().x()+difference_x, img.get_P1().y())
        # get all the bounding boxes and pack them with desired space in between
        # get first point and last point in x
        # x = img.boundingRect().x()
        # y = img.boundingRect().y()
        # last_x = img.boundingRect().x() + img.boundingRect().width()
        # last_y = img.boundingRect().y() + img.boundingRect().height()
    # self.updateBoudingRect()


    # align everything with respect to ref
    # get first point and align so that the all have the same x

def alignCenterV(reference=None, *objects_to_align_with_respect_to_ref):
    skip_first = False
    if objects_to_align_with_respect_to_ref is None or len(objects_to_align_with_respect_to_ref) == 0:
        logger.warning("Nothing to align...")
        return
    if reference is None:
        if len(objects_to_align_with_respect_to_ref) < 2:
            logger.warning("Reference is None, can't align anything...")
            return
        else:
            reference = objects_to_align_with_respect_to_ref[0]
            skip_first = True
            # del objects_to_align_with_respect_to_ref[0]

    # last_x = reference.get_P1().x()
    last_y = reference.get_P1().y()+reference.height()/2. # should be the end point of every stuff

    for i in range(len(objects_to_align_with_respect_to_ref)):
        if i==0 and skip_first:
            continue
        img = objects_to_align_with_respect_to_ref[i]


        pos_y = img.get_P1().y()+img.height()/2.
        difference_y = last_y-pos_y
        img.set_P1(img.get_P1().x(), img.get_P1().y()+difference_y)
        # get all the bounding boxes and pack them with desired space in between
        # get first point and last point in x
        # x = img.boundingRect().x()
        # y = img.boundingRect().y()
        # last_x = img.boundingRect().x() + img.boundingRect().width()
        # last_y = img.boundingRect().y() + img.boundingRect().height()
    # self.updateBoudingRect()


    # align everything with respect to ref
    # get first point and align so that the all have the same x
#
# def alignRight(reference=None, *objects_to_align_with_respect_to_ref):
#     skip_first = False
#     if objects_to_align_with_respect_to_ref is None or len(objects_to_align_with_respect_to_ref) == 0:
#         logger.warning("Nothing to align...")
#         return
#     if reference is None:
#         if len(objects_to_align_with_respect_to_ref) < 2:
#             logger.warning("Reference is None, can't align anything...")
#             return
#         else:
#             reference = objects_to_align_with_respect_to_ref[0]
#             skip_first = True
#             # del objects_to_align_with_respect_to_ref[0]
#
#     last_x = reference.get_P1().x()
#     # last_y = reference.get_P1().y()
#
#     for i in range(len(objects_to_align_with_respect_to_ref)):
#         if i==0 and skip_first:
#             continue
#         img = objects_to_align_with_respect_to_ref[i]
#
#         img.set_P1(last_x, img.get_P1().y())
#         # get all the bounding boxes and pack them with desired space in between
#         # get first point and last point in x
#         # x = img.boundingRect().x()
#         # y = img.boundingRect().y()
#         # last_x = img.boundingRect().x() + img.boundingRect().width()
#         # last_y = img.boundingRect().y() + img.boundingRect().height()
#     # self.updateBoudingRect()
#
#
#     # align everything with respect to ref
#     # get first point and align so that the all have the same x

