# TODO allow add or remove ROIs --> create IJ compatible ROIs... --> in a way that is simpler than the crop I was proposing --> think about how to implement that
# NB I have a bug with czi files that have channels --> need fix so that they appear as being channel last !!! --> TODO rapidly
from epyseg.tools.logger import TA_logger  # logging

logger = TA_logger()  # logging_level=TA_logger.DEBUG

import random
import os
import read_lif  # read Leica .lif files (requires numexpr)
from builtins import super, int
import warnings
import skimage
from skimage import io
from PIL import Image
import tifffile  # open Zeiss .tif and .lsm files
import czifile  # open .czi spim files
import glob
from skimage.transform import rescale
from skimage.util import img_as_ubyte
import scipy.signal  # convolution of images
import numpy as np
import json
from PyQt5.QtGui import QImage, QColor  # allows for qimage creation
from natsort import natsorted  # sort strings as humans would do
import xml.etree.ElementTree as ET  # to handle xml metadata of images
import base64
import io
import matplotlib.pyplot as plt
import traceback
from skimage.morphology import white_tophat, black_tophat, disk
from skimage.morphology import square, ball, diamond, octahedron, rectangle
import pathlib
import platform
import datetime as dt
import elasticdeform


# for future development
# np = None
# try:
#     np = __import__('cupy') # 3d accelerated numpy
# except:
#     np = __import__('numpy')

# mode : ({nearest, wrap, reflect, mirror, constant})
# mode reflect is cool so that I don't have to zoom to increase the size  by default
# assumes image is hwc except if two channels then expect hw
# nb could be useful to deform only in the Z axis to do 3D data aug!!!!
def elastic_deform(image, displacement=None, axis=None, order=0, zoom=None, rotation=None, mode='reflect',
                   return_deformation_matrix=False):
    if axis is None:
        if len(image.shape) == 4:
            # assume 'dhwc'
            axis = (1, 2)
        else:
            # assume 'hw' or 'hwc'
            axis = (0, 1)
    if displacement is None:
        displacement = np.random.randn(2, 3, 3) * 25
    X_deformed = elasticdeform.deform_grid(image, displacement, axis=axis, order=order, mode=mode, rotate=rotation, zoom=zoom)
    if return_deformation_matrix:
        # maybe would be smart to return the comple parameter dict some day but ok for now
        return displacement, X_deformed  # diplacement can be used to pass all the parameters to another image --> quite easy to do
    else:
        return X_deformed

def avg_proj(image, axis=0):
    return np.mean(image, axis=axis)

def max_proj(image, axis=0):  # z proj for a 3D image
    return np.max(image, axis=axis)

# we check if the image is a binary image --> retrun True if yes, False otherwise
def is_binary(image):
    mx = image.max()
    mn = image.min()
    if mx == mn:
        return True
    return image.size == np.count_nonzero((image == mn) | (image == mx))


def to_stack(images):
    # Nothing to do
    if images is None or not images:
        return
    # we need to determine the max width and height of the images so that we can create an image that contains all images if they haven't the same size already

    # if this are just paths --> convert them to images
    if isinstance(images[0], str):
        images = [Img(image) for image in images]

    for iii, image in enumerate(images):
        image = image[np.newaxis, ...]
        images[iii] = image
    # try create a stack and if fails try find a solution
    try:
        # stack = np.dstack(images)#, axis=np.newaxis)
        # stack = np.moveaxis(stack,-1, 0)
        # NB this will fail if not all dims are the same --> need more control ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 51 and the array at index 1 has size 48
        stack = np.concatenate(images, axis=0)
        return stack
    except:
        # if fails then see how to fix that
        traceback.print_exc()
        print('conversion to stack failed')
        pass

def fake_n_channels(image, n_channels=3):
    if len(image.shape)==2:
        tmp = np.zeros((*image.shape,n_channels), dtype=image.dtype)
        for ch in range(3):
            tmp[...,ch]=image
        return tmp
    else:
        return image

# returns True if an image is of Img class and has metadata, if it was converted to a classical nd array it will have lost it
def has_metadata(im):
    return hasattr(im, 'metadata')


def numpy_to_PIL(im, force_RGB=True):
    img = Image.fromarray(im)
    return img


def PIL_to_numpy(PIL_image):
    return np.array(PIL_image)


# converts a matplotlib fig to a numpy array
def fig_to_numpy(fig, tight=True):
    buf = io.BytesIO()
    if tight:
        # save with tight layout, very useful for saving LUTs for example
        # the good thing about png is that the size is saved in it --> avoids pbs I was having with saving as raw images
        fig.savefig(buf, format='png', bbox_inches='tight')
    else:
        fig.savefig(buf, format='png')
    buf.seek(0)
    # open in mem image
    im = Image.open(buf)
    img = np.array(im)
    # close buffer
    buf.close()
    return img


# TODO check on the different oses maybe replace mtime by mtime_nano
# https://docs.python.org/3/library/os.html#os.stat_result
def get_file_creation_time(filename, return_datetime_object=False):
    fname = pathlib.Path(filename)
    assert fname.exists(), f'No such file: {fname}'  # check that the file exists # shall I replace with try catch to be more flexible
    mtime = dt.datetime.fromtimestamp(fname.stat().st_mtime)
    if 'indow' in platform.system():
        ctime = dt.datetime.fromtimestamp(fname.stat().st_ctime)
        if return_datetime_object:
            return ctime
        else:
            return str(ctime)
    else:
        if return_datetime_object:
            return mtime
        else:
            return str(mtime)


def RGB_to_int24(RGBimg):
    RGB24 = (RGBimg[..., 0].astype(np.uint32) << 16) | (RGBimg[..., 1].astype(np.uint32) << 8) | RGBimg[..., 2].astype(
        np.uint32)
    return RGB24


def int24_to_RGB(RGB24):
    RGBimg = np.zeros(shape=(*RGB24.shape, 3), dtype=np.uint8)
    for c in range(RGBimg.shape[-1]):
        RGBimg[..., c] = (RGB24 >> ((RGBimg.shape[-1] - c - 1) * 8)) & 0xFF
    return RGBimg


def _normalize_8bits(img, mode='min_max'):
    try:
        if mode == 'min_max':
            if img.max() == img.min():
                return np.zeros_like(img, dtype=np.uint8)
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
            img = img.astype(np.uint8)
        else:
            img /= img.max() / 255.0
            img = img.astype(np.uint8)
    except:
        pass
    return img


# NB I ASSUME IMAGE IS hw or hwc and nothing else which may be wrong!!! but then conversion should take place before the image is passed there
def toQimage(img, autofix_always_display2D=True, normalize=True):
    '''get a qimage from ndarray

    Returns
    -------
    qimage
        a pyqt compatible image
    '''
    qimage = None

    logger.debug('Creating a qimage from a numpy image')
    dimensions = None
    if isinstance(img, Img):
        try:
            dimensions = img.get_dimensions_as_string()
        except:
            # fall back to hw dimensions if unknown as it is the default TA format anyawqy --> there are probably better solutions and I should not let images be built without dimensions but ok for now
            # dimensions = 'hw'
            if len(img.shape) == 2:
                dimensions = 'hw'
            elif len(img.shape) == 3:
                dimensions = 'hwc'
            else:
                logger.error('unknown image dimensions')
                return

    img = np.copy(img)  # need copy the array

    if autofix_always_display2D and dimensions is not None:
        # probably need more fixes
        if 't' in dimensions:
            img = img[0]  # get first time point


        # bug here cause my stuff is not very smart because the image is not having d as the first dimension...
        if 'd' in dimensions:
            if dimensions.index('d') != 0:
                logger.error('xyzt images not supported, only xyz images are supported.')
                return None
            # reduce dimensionality --> take central image
            img = img[int(img.shape[dimensions.index('d')] / 2)]

    if img.dtype != np.uint8:
        # just to remove the warning raised by img_as_ubyte
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                # need manual conversion of the image so that it can be read as 8 bit or alike
                # force image between 0 and 1 then do convert
                img = img_as_ubyte((img - img.min()) / (
                        img.max() - img.min()))  # do I really need that --> probably better to do it myself ???
            except:
                try:
                    img = img.astype(
                        np.uint8)  # error is probably due to a full black image --> can be fixed by converting it to uint8 dircetly!!!
                except:
                    logger.error('error converting image to 8 bits')
                    return None

    # KEEP UNCHANGED
    # DIRTY HACK FOR BUG https://doc.qt.io/qtforpython-5/PySide2/QtGui/QImage.html Warning  Painting on a QImage with the format Format_Indexed8 is not supported.
    if len(img.shape) == 2:
        # convert to RGB because of bug https://doc.qt.io/qtforpython-5/PySide2/QtGui/QImage.html Warning  Painting on a QImage with the format Format_Indexed8 is not supported.
        tmp = np.zeros((*img.shape, 3), dtype=np.uint8)
        tmp[..., 0] = img
        tmp[..., 1] = img
        tmp[..., 2] = img
        img = tmp
        del tmp

    bytesPerLine = img.strides[1]

    if len(img.shape) == 3:
        # print('in here channels toqimage')

        nb_channels = img.shape[-1]
        logger.debug('Image has ' + str(nb_channels) + ' channels')
        # print('Image has ' + str(nb_channels) + ' channels')

        if nb_channels == 3:
            # print( img.shape, bytesPerLine)

            # NB check that I haven't inverted stuff --> PB CANNOT LOAD 'ON 290119.lif - Series003.tif_t000.tif'
            # qimage = QImage(img.data,  img.shape[-2], img.shape[-3], bytesPerLine, QImage.Format_RGB888) # --> bug fix but no display --> why ???

            # bug fix is here
            # https://stackoverflow.com/questions/55468135/what-is-the-difference-between-an-opencv-bgr-image-and-its-reverse-version-rgb-i

            if normalize:
                img = _normalize_8bits(img)

            bytesPerLine = 3 * img.shape[-2]

            # qimage = QImage(img.data.tobytes(),  img.shape[-3], img.shape[-2], bytesPerLine, QImage.Format_RGB888) # --> bug fix but no display --> why ???
            qimage = QImage(img.data.tobytes(), img.shape[-2], img.shape[-3], bytesPerLine,
                            QImage.Format_RGB888)  # --> bug fix but no display --> why ???

            # bug here --> why
        elif nb_channels < 3:
            # add n dimensions
            bgra = np.zeros((img.shape[-3], img.shape[-2], 3), np.uint8, 'C')
            if img.shape[2] >= 1:
                bgra[..., 0] = img[..., 0]
            if img.shape[2] >= 2:
                bgra[..., 1] = img[..., 1]
            if img.shape[2] >= 3:
                bgra[..., 2] = img[..., 2]
            bytesPerLine = 3 * bgra.shape[-2]
            if normalize:
                bgra = _normalize_8bits(bgra)
            qimage = QImage(bgra.data.tobytes(), img.shape[-2], img.shape[-3], bytesPerLine, QImage.Format_RGB888)
        else:
            # this does not make sense to handle ARGB in science but when I deactivate this all my images are black --> WHY
            if nb_channels == 4:
                # if False:
                bgra = np.zeros((img.shape[-3], img.shape[-2], 4), np.uint8, 'C')
                bgra[..., 0] = img[..., 0]
                bgra[..., 1] = img[..., 1]
                bgra[..., 2] = img[..., 2]
                if img.shape[2] >= 4:
                    logger.debug('using 4th numpy color channel as alpha for qimage')
                    bgra[..., 3] = img[..., 3]
                else:
                    bgra[..., 3].fill(255)
                if normalize:
                    bgra = _normalize_8bits(bgra)
                bytesPerLine = 4 * bgra.shape[-2]
                qimage = QImage(bgra.data.tobytes(), img.shape[-2], img.shape[-3], bytesPerLine,
                                QImage.Format_ARGB32)
            else:
                # here we have images with many channels
                # logger.error("not implemented yet!!!!, too many channels")
                # best here is to average all the channels --> good idea
                # print(img.shape)
                bgra = np.average(img, axis=-1)

                # print(img.shape)
                tmp = np.zeros((*bgra.shape, 3), dtype=np.uint8)
                tmp[..., 0] = bgra
                tmp[..., 1] = bgra
                tmp[..., 2] = bgra
                bgra = tmp
                del tmp
                # pb can't be drawn over --> need force it to be RGB
                # qimage = QImage(img.data.tobytes(), img.shape[1], img.shape[0], bytesPerLine,
                #        QImage.Format_Indexed8)

                if normalize:
                    bgra = _normalize_8bits(bgra)
                bytesPerLine = 3 * bgra.shape[-2]

                # qimage = QImage(img.data.tobytes(),  img.shape[-3], img.shape[-2], bytesPerLine, QImage.Format_RGB888) # --> bug fix but no display --> why ???
                qimage = QImage(bgra.data.tobytes(), bgra.shape[-2], bgra.shape[-3], bytesPerLine, QImage.Format_RGB888)

                # print('in')
    else:
        # should never be reached because can't be painted over --> useless stuff and creates seg faults

        # print('in here 2') #reached"
        # there is a seg fault bug with the indexed version --> never use that --> rely on RGB maybe instead

        '''KEEP MEGA IMPORTANT
        # https://doc.qt.io/qtforpython-5/PySide2/QtGui/QImage.html
        # Warning
        # Painting on a QImagewith the format Format_Indexed8 is not supported. --> NEVER USE THIS THAT'S IT AND FIX ALWAYS
        '''

        logger.warning('this should never have been reached --> errors may occur')
        # seg fault was due to inversion in width and height of the image --> need be careful and fix always
        qimage = QImage(img.data.tobytes(), img.shape[1], img.shape[0], bytesPerLine,
                        QImage.Format_Indexed8)

        # qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)

        # print('in here 3')  # reached"

        # required to allow creation of a qicon --> need keep --> in fact cause some crashes --> do not keep that
        # the crash is here !!!! --> WHY AND DO I NEED THAT
        # for i in range(256):
        #     qimage.setColor(i, QColor(i, i, i).rgb())
        # print('in here 4')  # reached"
    # print('out qimage') # never reaches there!!!
    return qimage


def _get_white_bounds(img):
    coords = np.where(img != 0)
    if coords[0].size == 0:
        return None
    return np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1])


def get_white_bounds(imgs):
    if isinstance(imgs, list):
        bounds = [10000000, 0, 10000000, 0]
        for img in imgs:
            curbounds = _get_white_bounds(img)
            if curbounds is None:
                continue
            # print(curbounds)
            bounds[0] = min(curbounds[0], bounds[0])
            bounds[1] = max(curbounds[1], bounds[1])
            bounds[2] = min(curbounds[2], bounds[2])
            bounds[3] = max(curbounds[3], bounds[3])
        if bounds[0] == 10000000:
            return None
    else:
        bounds = _get_white_bounds(imgs)
    return bounds


def crop_smartly(imgs, bounds=None, bounds_around=0):
    if bounds is None:
        bounds = get_white_bounds(imgs)
        # print(bounds)
    if isinstance(imgs, list):
        for iii, img in enumerate(imgs):
            imgs[iii] = img[bounds[0] - bounds_around:bounds[1] + bounds_around,
                        bounds[2] - bounds_around:bounds[3] + bounds_around]
        return imgs
    else:
        return imgs[bounds[0] - bounds_around:bounds[1] + bounds_around,
               bounds[2] - bounds_around:bounds[3] + bounds_around]


# nb it assumes the image has a c channel or is two at most # or use pos x and pos y
# can also use this to make a black border I guess
# this is much smarter than the setBorder code --> replace it everywhere
# shall I make it from within ??? using self so that I keep the dims ???
# if mode is a value --> not a string then assume constant
def pad_border_xy(img, dim_x=-2, dim_y=-3, size=1, mode='symmetric'):
    if size <= 0:
        # null or negative padding --> nothing to do --> return original
        return img

    if img is None:
        logger.error('Image is None -> nothing to do')
        return None

    smart_slices = []
    pad_seq = []

    if dim_y == -3 and abs(dim_y) > len(img.shape):
        if dim_x == -2:
            dim_x = -1
            dim_y = -2
    # print(dim_x, dim_y)

    for dim in range(len(img.shape)):
        smart_slices.append(slice(None))
        pad_seq.append((0, 0))

    smart_slices[dim_x] = slice(size, -size)
    smart_slices[dim_y] = slice(size, -size)

    pad_seq[dim_x] = ((size, size))
    pad_seq[dim_y] = ((size, size))

    if isinstance(mode, str):
        img = np.pad(img[tuple(smart_slices)], tuple(pad_seq), mode=mode)
    else:
        # print(smart_slices)
        # print(pad_seq)
        img = np.pad(img[tuple(smart_slices)], tuple(pad_seq), mode='constant', constant_values=mode)
    return img


# could also add forbidden colors that are ultimately converted to masks that can be used in blending
# NB IF I IMPLEMENT FORBIDDEN COLORS I REALLY NEED TO HAVE THE MASKS COMPUTED INITIALLY COULD RENAME MASK AS MASK_OR_FORBIDDEN_COLOR --> would be clearer

# NB what I do is not very smart masking should ideally be applied before compositing --> that would be more efficient --> think about it because not so easy to do and that works well already
# this is work in progress --> probably still full of bugs
def blend(bg, fg, alpha=0.3, mask_or_forbidden_colors=None):
    # TODO do conversion so that all images are all in the same range --> not very hard in fact
    # create compatible images in shape TODO probably need also make sure images match at the level of dtype otherwise things will not work
    bg, fg = create_compatible_image(bg, fg, auto_sort=True)

    # if bg

    # trick to normalize the bg image so that it is in the same range as fg image that is an RGB image --> maybe need do that more cleanly but ok for now
    # added that if image is not of type uint 8 --> most likely type float it must be converted to 0 255 so that the blending can be applied to it
    if bg.max() > 255 or bg.dtype != np.uint8:
        bg = np.interp(bg, (bg.min(), bg.max()), (0, 255))
    # --> indeed bug if not same range --> make sure range is same of make them fit --> TODO --> convert to RGB the image or single channel whichever format it is
    # --> test

    # print('fg.shape, bg.shape, fg.dtype, bg.dtype',fg.shape, bg.shape, fg.dtype, bg.dtype)

    # plt.imshow(fg)
    # plt.show()
    #
    # plt.imshow(bg)
    # plt.show()

    blended = fg * (alpha) + bg * (1. - alpha)
    blended = np.clip(blended, 0, 255)
    blended = blended.astype(np.uint8)

    if mask_or_forbidden_colors is None:
        return blended

    # do the masking based on colors or on mask
    if not isinstance(mask_or_forbidden_colors, np.ndarray):
        # assume colors must have been passed --> get a mask from them
        mask_or_forbidden_colors = mask_colors(fg, colors_to_mask=mask_or_forbidden_colors)
        # the mask is full of bugs
        # if mask_or_forbidden_colors is not None:
        #     plt.imshow(mask_or_forbidden_colors.astype(np.uint8)*255)
        #     plt.show()
        # else:
        #     print('EROOR EMPTY')

    if mask_or_forbidden_colors is None:
        # no color found or error --> return default blended image
        return blended

    # not easy --> can create a lot of errors --> need a smart fix some day
    # maybe force also the mask to be compatible with bg

    # marche pas --> il y a un bug qq part mais j'en suis pas loin en fait
    bg, mask_or_forbidden_colors = create_compatible_image(bg, mask_or_forbidden_colors, auto_sort=False)
    if mask_or_forbidden_colors.dtype != bool:
        # need normalization
        if mask_or_forbidden_colors.max() != 0:
            mask_or_forbidden_colors = mask_or_forbidden_colors / mask_or_forbidden_colors.max()

    bg = bg * mask_or_forbidden_colors
    bg = np.clip(bg, 0, 255)
    bg = bg.astype(np.uint8)
    # restore bg for masked colors
    blended[bg != 0] = bg[bg != 0]

    return blended


# TODO maybe add range and dtype check at some point
# NB if auto --> match directly to the image with the max nb of channels/dimensions
# in fact need return just the image to be changed
# but need identify which image it is in fact --> need return an idx --> such as 0 or 1

# maybe fix image if it has more dims than 3
def create_compatible_image(desired_image, image_to_change, auto_sort=False):
    # print('desired_image.shape',desired_image.shape) # --> pb if image has 4 channels --> need a fix --> see how to do --> probably convert to 3 channels...  ????
    # print('image_to_change.shape',image_to_change.shape)

    # in case the bg image has too many channels convert it to a 3 channel image --> simpler or remove the last channel --> ???
    # dirty fix when source image has more than 3 channels (e.g. 4 channels) is there any more elegant way of doing that ???? not sure
    if len(desired_image.shape) >= 3 and desired_image.shape[-1] > 3:
        # convert it first to a RGB image
        # original_dtype = desired_image.dtype
        desired_image = np.average(desired_image, axis=-1)
        # desired_image = desired_image.astype(original_dtype)

        # tmp = np.zeros((*desired_image.shape, 3), dtype=np.uint8)
        # tmp[..., 0] = desired_image
        # tmp[..., 1] = desired_image
        # tmp[..., 2] = desired_image
        # desired_image = tmp
        # nb there seems to still be a bug in the blending algo because the image is too bright -->

    # NB if auto --> match directly to the image with the max nb of channels/dimensions
    if len(desired_image.shape) == len(image_to_change.shape) and desired_image.shape == image_to_change.shape:
        # images are already compatible --> nothing todo
        return desired_image, image_to_change
    # else modify image so that it matches the desired stuff

    # otherwise too complex and return an error
    if abs(len(desired_image.shape) - len(image_to_change.shape)) > 1:
        logger.error('Images cannot be rendered compatible easily --> ignoring')
        return desired_image, image_to_change
    # check that all dimensions but channels match otherwise return an error
    # assume just the nb of channels differ --> do fix them
    try:
        if not desired_image.shape[0] == image_to_change.shape[0] and desired_image.shape[1] == image_to_change.shape[
            1]:
            logger.error(
                'Images cannot be rendered compatible easily, because the two first dimensions are not equal between the two images')
            return desired_image, image_to_change
    except:
        logger.error(
            'Images cannot be rendered compatible easily, because the two first dimensions are not equal between the two images')
        return desired_image, image_to_change

    # now enter the serious stuff --> image really needs be changed
    need_swap = False
    desired_image_clone, image_to_change_clone = desired_image, image_to_change
    if auto_sort:
        # find image withthe more dims or nb of channels and match the other to it

        # print(desired_image_clone.shape, image_to_change_clone.shape)  #(512, 512, 3) (512, 512) -->should not swap in fact --> bug is here
        # print(len(image_to_change_clone.shape)>len(desired_image_clone.shape), image_to_change_clone.shape[-1]>desired_image_clone.shape[-1])

        if len(image_to_change.shape) > len(desired_image.shape):
            desired_image_clone, image_to_change_clone = image_to_change_clone, desired_image_clone
            need_swap = True
            # print('swappped', desired_image_clone.shape,image_to_change_clone.shape)
        elif len(image_to_change.shape) == len(desired_image.shape) and image_to_change.shape[-1] > desired_image.shape[
            -1]:
            desired_image_clone, image_to_change_clone = image_to_change_clone, desired_image_clone
            need_swap = True
            # print('swappped', desired_image_clone.shape, image_to_change_clone.shape)
        # idx = 1

    # print(desired_image_clone.shape, image_to_change_clone.shape)# bug here --> wrong image in fact

    # print('desired_image_clone.shape',desired_image_clone.shape, 'image_to_change_clone.shape',image_to_change_clone.shape)

    tmp = np.zeros_like(desired_image_clone, dtype=image_to_change_clone.dtype)
    # print()

    # print(tmp.shape) # --> ok
    # just do copy the image several times --> should work
    # si ça ne marche pas je fais quoi ????
    # si image a top de channels que dois-je faire dois-je forcer l'image à etre exactement 3 channels anyway cannot show so many channels ...

    for c in range(tmp.shape[-1]):
        tmp[..., c] = image_to_change_clone

    if not need_swap:
        return desired_image_clone, tmp
    else:
        return tmp, desired_image_clone


# not bad and can be useful !!!
# colors to mask should be an array colors/values
def mask_colors(colored_image, colors_to_mask, invert_mask=False, warn_on_color_not_found=False):
    if colors_to_mask is None:
        logger.warning('No color to be masked was specified --> ignoring mask')
        return None
    if not (isinstance(colors_to_mask, list) or isinstance(colors_to_mask, tuple)):
        colors_to_mask = [colors_to_mask]
    if not invert_mask:
        # if len(mask.shape) > 3
        mask = np.zeros_like(colored_image, dtype=bool)
    else:
        mask = np.ones((colored_image.shape), dtype=bool)
    mask_value = True
    if invert_mask:
        mask_value = False

    # print(colored_image.shape, mask.shape)
    for color in colors_to_mask:
        # print(color)
        # if not isinstance(color, tuple):
        #     print('before 1')
        #     plt.imshow(mask.astype(np.uint8)*255)
        #     plt.show()

        # mask[np.where(colored_image==color)]=mask_value

        #  is that even working --> need test

        if len(mask.shape) == 3:
            mask[np.where(np.all(colored_image == color, axis=-1))] = mask_value  # cretin
        else:
            mask[np.where(colored_image == color)] = mask_value

        # print(np.where(np.all(colored_image == color, axis=-1)))

        # coords = np.where(np.all(colored_image == color, axis=-1))
        # mask[coords]=mask_value

        # print('after 1')
        # plt.imshow(mask.astype(np.uint8)*255)
        # plt.show()
    # else:
    #     print('in')
    #
    #     print(color)
    #
    #     for ccc, col in enumerate(color):
    #         mask[...,ccc][colored_image[...,ccc]==col]=mask_value
    # print('out')

    if mask.max() == mask.min():
        # colors not found --> ignore
        if warn_on_color_not_found:
            logger.warning('Colors not found --> no mask created')
        return None
    return mask


# TODO maybe make this a more generic stuff

def get_nb_of_series_in_lif(lif_file_name):
    if not lif_file_name or not lif_file_name.lower().endswith('.lif'):
        logger.error('Error only lif file supported')
        return None
    reader = read_lif.Reader(lif_file_name)
    series = reader.getSeries()
    return len(series)


# TODO --> do a code to reinject the metadata from one file into another especially the dimensions or save them as a .meta file that is independent --> would be a good idea
# see if and how I can reinject the metadata in the stuff


# TODO split my code into metadata gathering data and image opener --> ideally make this separately ????
# see how to do that
# can I get just the metadata of the lif in order to gain time and not process everything, can I also get the name of the series
def get_series_names_in_lif(lif_file_name):
    if not lif_file_name or not lif_file_name.lower().endswith('.lif'):
        logger.error('Error only lif file supported')
        return None
    reader = read_lif.Reader(lif_file_name)
    series = reader.getSeries()
    for serie in series:
        print('name', serie.getName())  # --> ok
        print('ts',
              len(serie.getTimeStamps()))  # so it would be easy to associate a time or a time stamp to every image --> maybe that is what I want
        print('tl', serie.getTimeLapse())  # is filled when a time lapse is made
        # print('rts',len(serie.getRelativeTimeStamps())) # always empty --> exclude all of this
        print('times',
              serie.getNbFrames())  # ça c'est super utile pr recupérer les temps en fait aussi --> peut etre aussi l'extraire et le stocker qq part ds les metadata (en fait c'est deja stocke dans une des dimensions)
        # Voir comment finaliser les temps
        # tester un fichier de height map --> faire un and entre le mask et le fichier de height map et voir comment je peux faire mon analyse
        # dupliquer la classe pr éviter de casser des trucs
        # could maybe format this
        # need get the ratio xy over z in order to compute the area --> simpler TODO
        print('Zx ratio', serie.getZXratio())  # --> exactly what I want and need
        # dimensions = serie.getDimensions()
        # for dim in range(len(dimensions)):
        #     try:
        #         print('Voxel size dim', dim, serie.getVoxelSize(dimensions[dim])) # --> exactly what I want and need
        #     except:
        #         pass
        metadata = serie.getMetadata()
        print('voxel_size_x', metadata['voxel_size_x'])
        print('voxel_size_y', metadata['voxel_size_y'])
        print('voxel_size_z', metadata['voxel_size_z'])

        print('nb series', len(series))
    # print(series)
    return len(series)


# work in progress to transfer metadata from one file to another --> DO NOT USE A LOT MORE WORK IS NEEDED HERE AND THINGS HAVE NOT BEEN MADE PROPERLY!!!
def _transfer_voxel_size_metadata(input_file_with_correct_metadata, output_file_with_missing_metadata):
    img1 = Img(input_file_with_correct_metadata)
    img2 = Img(output_file_with_missing_metadata)

    relevant_metadatas = ['vx', 'vy', 'vz', 'AR',
                          'creation_time']  # , 'times']--> in lsm time is an array --> not serializable --> ignore for now --> fix that in future

    meta_changed = False
    for relevant_metadata in relevant_metadatas:
        if relevant_metadata in img1.metadata:
            if relevant_metadata in img2.metadata:
                if img2.metadata[relevant_metadata] == img1.metadata[relevant_metadata]:
                    continue
                else:
                    img2.metadata[relevant_metadata] = img1.metadata[relevant_metadata]
                    meta_changed = True
            else:
                img2.metadata[relevant_metadata] = img1.metadata[relevant_metadata]
                meta_changed = True
        else:
            logger.warning('relevant metadata "' + relevant_metadata + '" not found --> ignoring')
            meta_changed = True

    # print(img2.metadata)
    # return

    if meta_changed:
        if 'vx' in img2.metadata and 'vy' in img2.metadata:
            # just to make resolution rationale

            # TODO replace resolution by fractions.Fraction https://docs.python.org/3/library/fractions.html
            tifffile.imwrite(output_file_with_missing_metadata, img2, imagej=False, resolution=(
                (int(1_000_000 / img2.metadata['vx']), 1000000), (int(1_000_000 / img2.metadata['vy']), 1000000)),
                             # resolution=(1_000_000/img2.metadata['vx'], 1_000_000/img2.metadata['vy']), # resolution is almost ok but I do have a small bug in fact
                             metadata=img2.metadata)
        else:
            tifffile.imwrite(output_file_with_missing_metadata, img2,
                             imagej=False, metadata=img2.metadata)


# TODO maybe make one that does the same with ROIs ???
# if return mask then I just return the mask with boolean and not the
def mask_rows_or_columns(img, spacing_X=2, spacing_Y=None, masking_value=0, return_boolean_mask=False,
                         initial_shiftX=0, initial_shiftY=0, random_start=False):  # , dimension_h=-2, dimension_w=-1

    if isinstance(img, tuple):
        mask = np.zeros(img, dtype=bool)
    else:
        mask = np.zeros(img.shape, dtype=bool)

    if mask.ndim < 3:  # assume no channel so add one
        mask = mask[..., np.newaxis]

    if spacing_X is not None:
        if spacing_X <= 1:
            spacing_X = None
    if spacing_Y is not None:
        if spacing_Y <= 1:
            spacing_Y = None

    if initial_shiftX == 0 and initial_shiftY == 0 and random_start:
        if spacing_X is not None:
            initial_shiftX = random.randint(0, spacing_X)
        if spacing_Y is not None:
            initial_shiftY = random.randint(0, spacing_Y)

    # assume all images are with a channel --> probably the best way to do things
    for c in range(mask.shape[-1]):
        if spacing_Y is not None:
            if mask.ndim > 3:
                mask[..., initial_shiftY::spacing_Y, :, c] = True
            else:
                mask[initial_shiftY::spacing_Y, :, c] = True
        if spacing_X is not None:
            mask[..., initial_shiftX::spacing_X, c] = True

    if return_boolean_mask or isinstance(img, tuple):
        return mask

    if img.ndim < 3:  # assume no channel so add one
        img = img[..., np.newaxis]

    # apply mask to image
    img[mask] = masking_value

    return img


# TODO in development --> code that better and check whether it keeps the intensity range or not
def resize(img, new_size, order=1, preserve_range=False):
    from skimage.transform import resize
    img = resize(img, new_size, order=order, preserve_range=preserve_range)
    return img


# nb there seems to be a bug in white top hat --> infinite loop or bug ???
def __top_hat(image, type='black', structuring_element=square(50), preserve_range=True):
    logger.debug('bg subtraction ' + str(type) + '_top_hat')
    try:
        # TODO crappy bug fix for 3D images in tensorflow --> need some more love
        # TODO NB will only work for tensorflow like images or maybe always load and treat images as in tensorflow by adding 1 for channel dimension even if has only one channel?? --> MAY MAKE SENSE
        # for some reason top hat does not work with 3D images --> why --> in fact that does work but if image is very noisy and filter is big then it does nothing
        if len(image.shape) == 4:
            out = np.zeros_like(image)  # , dtype=image.dtype
            for zpos, zimg in enumerate(image):
                for ch in range(zimg.shape[-1]):
                    out[zpos, ..., ch] = __top_hat_single_channel__(zimg[..., ch], type=type,
                                                                    structuring_element=structuring_element,
                                                                    preserve_range=preserve_range)
            return out
        elif len(image.shape) == 3:
            out = np.zeros_like(image)  # , dtype=image.dtype
            for ch in range(image.shape[-1]):
                out[..., ch] = __top_hat_single_channel__(image[..., ch], type=type,
                                                          structuring_element=structuring_element,
                                                          preserve_range=preserve_range)
            return out
        elif len(image.shape) == 2:
            out = __top_hat_single_channel__(image, type=type, structuring_element=structuring_element,
                                             preserve_range=preserve_range)
            return out
        else:
            print('invalid shape --> ' + type + ' top hat failed, sorry...')
    except:
        print(str(type) + ' top hat failed, sorry...')
        traceback.print_exc()
        return image


def black_top_hat(image, structuring_element=square(50), preserve_range=True):
    return __top_hat(image, type='black', structuring_element=structuring_element, preserve_range=preserve_range)


def white_top_hat(image, structuring_element=square(50), preserve_range=True):
    return __top_hat(image, type='white', structuring_element=structuring_element, preserve_range=preserve_range)


# somehow tophat does not work for 3D but why ???
def __top_hat_single_channel__(single_channel_image, type, structuring_element=square(50), preserve_range=True):
    dtype = single_channel_image.dtype
    min = single_channel_image.min()
    max = single_channel_image.max()
    if type == 'white':
        out = white_tophat(single_channel_image, structuring_element)
    else:
        out = black_tophat(single_channel_image, structuring_element)
    # TODO NB check if correct also
    if preserve_range and (out.min() != min or out.max() != max):
        out = out / out.max()
        out = (out * (max - min)) + min
        out = out.astype(dtype)
    return out


class Img(np.ndarray):  # subclass ndarray
    background_removal = ['No', 'White bg', 'Dark bg']
    # see https://en.wikipedia.org/wiki/Feature_scaling
    normalization_methods = ['Rescaling (min-max normalization)', 'Standardization (Z-score Normalization)',
                             'Mean normalization', 'Max normalization (auto)', 'Max normalization (x/255)',
                             'Max normalization (x/4095)', 'Max normalization (x/65535)',
                             'Rescaling based on defined lower and upper percentiles',
                             'None']  # should I add vgg, etc for pretrained encoders ??? maybe put synonyms
    normalization_ranges = [[0, 1], [-1, 1]]

    clipping_methods = ['ignore outliers', '+', '+/-', '-']

    # TODO allow load list of images all specified as strings one by one
    # TODO allow virtual stack --> open only one image at a time from a series, can probably do that with text files
    def __new__(cls, *args, t=0, d=0, z=0, h=0, y=0, w=0, x=0, c=0, bits=8, serie_to_open=None, dimensions=None,
                metadata=None, **kwargs) -> object:
        '''Creates a new instance of the Img class
        
        The image class is a numpy ndarray. It is nothing but a matrix of pixel values.

        Parameters
        ----------
        t : int
            number of time points of the image
        d, z : int
            number of z stacks of an image
        h, y : int
            image height
        w, x : int
            image width
        c : int
            number of color channels 
        bits : int
            bits per pixel
        dimensions : string
            order and name of the dimensions of the image
        metadata : dict
            dict containing metadata entries and their corresponding values

        '''

        img = None

        meta_data = {'dimensions': None,  # image dimensions
                     'bits': None,  # bits per pixel
                     'vx': None,  # voxel x size
                     'vy': None,  # voxel y size
                     'vz': None,  # voxel z size
                     'AR': None,  # vz/vx ratio
                     'LUTs': None,  # lut
                     'cur_d': 0,  # current z/depth pos
                     'cur_t': 0,  # current time
                     'Overlays': None,  # IJ overlays
                     'ROI': None,  # IJ ROIs
                     'timelapse': None,  # time between frames when in a time lapse movie-->
                     'creation_time': None,
                     }

        if metadata is not None:
            # if user specified some metadata update them
            meta_data.update(metadata)
        else:
            # recover old metadata from original image # is that the correct way
            if isinstance(args[0], Img):
                try:
                    meta_data.update(args[0].metadata)
                except:
                    pass

        if len(args) == 1:
            # case 1: Input array is an already an ndarray
            if isinstance(args[0], np.ndarray):
                img = np.asarray(args[0]).view(cls)
                img.metadata = meta_data
                if dimensions is not None:
                    img.metadata['dimensions'] = dimensions

            elif isinstance(args[0], str):
                logger.debug('loading ' + args[0])

                # input is a string, i.e. a link to one or several files
                if '*' not in args[0]:
                    # single image
                    creation_time = get_file_creation_time(args[0])
                    meta, img = ImageReader.read(args[0], serie_to_open=serie_to_open)
                    meta_data.update(meta)
                    meta_data['path'] = args[0]  # add path to metadata
                    # meta_data['creation_time'] = creation_time
                    if not 'creation_time' in meta_data or meta_data['creation_time'] == None:
                        meta_data['creation_time'] = str(dt.datetime.now())
                    img = np.asarray(img).view(cls)
                    img.metadata = meta_data
                    # img.metadata.update({'creation_time': creation_time})
                else:
                    # series of images
                    image_list = [img for img in glob.glob(args[0])]
                    image_list = natsorted(image_list)
                    # maybe create an array of creation time in this case
                    creation_time = []
                    for file in image_list:
                        creation_time.append(get_file_creation_time(file))
                    img = ImageReader.imageread(image_list)  # TODO add metadata here too for w,h d and channels
                    meta_data['path'] = args[
                        0]  ## add path to metadata # TODO make this an array of files instead --> smarter in a way
                    # meta_data['creation_time'] = creation_time
                    if not 'creation_time' in meta_data or meta_data['creation_time'] == None:
                        meta_data['creation_time'] = str(dt.datetime.now())
                    img = np.asarray(img).view(cls)
                    img.metadata = meta_data
                    # img.metadata.update({'creation_time':creation_time})
        else:
            # custom image creation : setting the dimensions
            dims = []
            dimensions = []
            if t != 0:
                dimensions.append('t')
                dims.append(t)
            if z != 0 or d != 0:
                dimensions.append('d')
                dims.append(max(z, d))
            if h != 0 or y != 0:
                dimensions.append('h')
                dims.append(max(h, y))
            if w != 0 or x != 0:
                dimensions.append('w')
                dims.append(max(w, x))
            if c != 0:
                dimensions.append('c')
                dims.append(c)

            dimensions = ''.join(dimensions)

            meta_data['dimensions'] = dimensions  # add dimensions to metadata
            dtype = np.uint8  # default is 8 bits
            if bits == 16:
                dtype = np.uint16  # 16 bits
            if bits == 32:
                dtype = np.float32  # 32 bits
            meta_data['bits'] = bits
            if not 'creation_time' in meta_data or meta_data['creation_time'] == None:
                meta_data['creation_time'] = str(dt.datetime.now())
            img = np.asarray(np.zeros(tuple(dims), dtype=dtype)).view(cls)
            # array = np.squeeze(array) # TODO may be needed especially if people specify 1 instead of 0 ??? but then need remove some stuff
            # img = array
            img.metadata = meta_data

        if img is None:
            # TODO do that better
            logger.critical(
                "Error, can't open image invalid arguments, file not supported or file does not exist...")  # TODO be more precise
            return None

        return img

    # TODO do implement it more wisely or drop it because it's simpler to access the numpy array directly...
    def get_pixel(self, *args):
        '''get pixel value

        TODO
        '''
        if len(args) == self.ndim:
            return self[tuple(args)]
        logger.critical('wrong nb of dimensions')
        return None

    # TODO do implement it more wisely or drop it because it's simpler to access the numpy array directly...
    def set_pixel(self, x, y, value):
        '''sets pixel value

        TODO
        '''
        # if len(args) == self.ndim:
        self[x, y] = value

    def get_dimension(self, dim):
        '''gets the specified image dimension length

        Parameters
        ----------
        dim : single char string
            dimension of interest

        Returns
        -------
        int
            dimension length
	    '''

        # force dimensions compatibility (e.g. use synonyms)
        if dim == 'z':
            dim = 'd'
        elif dim == 'x':
            dim = 'w'
        elif dim == 'y':
            dim = 'h'
        elif dim == 'f':
            dim = 't'

        if self.metadata['dimensions'] is None:
            logger.error('dimension ' + str(dim) + ' not found!!!')
            return None
        if dim in self.metadata['dimensions']:
            idx = self.metadata['dimensions'].index(dim)
            idx = idx - len(self.metadata['dimensions'])
            if self.ndim >= abs(idx) >= 1:
                return self.shape[idx]
            else:
                logger.error('dimension ' + str(dim) + ' not found!!!')
                return None
        else:
            logger.error('dimension ' + str(dim) + ' not found!!!')
            return None

    def get_dimensions(self):
        '''gets the length of all dimensions

        Returns
        -------
        dict
            a dict containing dimension name along with its length

        '''
        dimension_parameters = {}
        for d in self.metadata['dimensions']:
            dimension_parameters[d] = self.get_dimension(d)
        return dimension_parameters

    def get_dimensions_as_string(self):
        return self.metadata['dimensions']

    def get_dim_idx(self, dim):
        # force dimensions compatibility (e.g. use synonyms)
        if dim == 'z':
            dim = 'd'
        elif dim == 'x':
            dim = 'w'
        elif dim == 'y':
            dim = 'h'
        elif dim == 'f':
            dim = 't'
        if not dim in self.metadata['dimensions']:
            return None
        return self.metadata['dimensions'].index(dim)

    # TODO code this better
    def pop(self, pause=1, lut='gray', interpolation=None, show_axis=False, preserve_AR=True):
        '''pops up an image using matplot lib

        Parameters
        ----------
        pause : int
            time the image should be displayed

        interpolation : string or None
            interpolation for image display (e.g. 'bicubic', 'nearest', ...)

        show_axis : boolean
            TODO

        preserve_AR : boolean
            keep image AR upon display

        '''

        if self.ndim > 3:
            logger.warning("too many dimensions can't pop image")
            return

        plt.ion()
        plt.axis('off')
        plt.margins(0)

        plt.clf()
        plt.axes([0, 0, 1, 1])

        ax = plt.gca()
        ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
        ax.get_yaxis().set_visible(False)
        ax.margins(0)

        if self.ndim == 3 and self.shape[2] <= 2:
            # create a 3 channel array from the 2 channel array image provided
            rgb = np.concatenate(
                (self[..., 0, np.newaxis], self[..., 1, np.newaxis], np.zeros_like(self[..., 0, np.newaxis])), axis=-1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                plt.imshow(img_as_ubyte(rgb), interpolation=interpolation)
                # logger.debug("popping image method 1")
        else:
            if self.ndim == 2:
                # if image is single channel display it as gray instead of with a color lut by default
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    plt.imshow(img_as_ubyte(self), cmap=lut, interpolation=interpolation)  # self.astype(np.uint8)
                    # logger.debug("popping image method 2")
            else:
                # split channels if more than 3 channels maybe or remove the alpha channel ??? or not ??? see how to do that
                if self.shape[2] == 3:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        plt.imshow(img_as_ubyte(self),
                                   interpolation=interpolation)
                        # logger.debug("popping image method 3")
                else:
                    for c in range(self.shape[2]):
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            plt.imshow(img_as_ubyte(self[:, :, c]), cmap=lut, interpolation=interpolation)
                            if c != self.shape[2] - 1:
                                plt.show()
                                plt.draw()
                                plt.pause(pause)
                            # logger.debug("popping image method 4")

        if not preserve_AR:
            ax.axis('tight')  # required to preserve AR but this necessarily adds a bit of white around the image
        ax.axis('off')

        plt.show()
        plt.draw()
        plt.pause(pause)

    def _pad_border_xy(self, *args, **kwargs):
        # will that work ??? --> yes that works and that is relatively simple to implement --> maybe this is the way to proceed for all functions --> MEGA TODO
        self[...] = pad_border_xy(self, *args, **kwargs)[...]

    # @deprecated --> remove all of this soon --> use pad_border_xy instead
    def setBorder(self, distance_from_border_in_px=1, color=0):
        ''' Set n pixels at the border of the image to the defined color

        Parameters
        ----------
        distance_from_border_in_px : int
            Distance in pixels from the borders of the image.
        color : int or tuple
            new color (default is black = 0)
        '''
        logger.warning(
            'NB: setBorder is deprecated and will be removed soon --> use Img._pad_border_xy or pad_border_xy instead.')

        if distance_from_border_in_px <= 0:
            # ignore when distance < 0
            return

        val = color

        if self.has_c() and self.get_dimension('c') > 1 and not isinstance(color, tuple):
            # convert int color to tuple when tuple is required, i.e. when an img has several channels
            val = tuple([color] * self.get_dimension('c'))

        all_dims_before_hwc = []
        for d in self.metadata['dimensions']:  # keep all dimensions before hwc unchanged
            if d not in ['w', 'h', 'c', 'x', 'y']:
                all_dims_before_hwc.append(slice(None))

        # recolor the border
        for v in range(distance_from_border_in_px):
            all_dims_before_hwc.append(slice(None))
            all_dims_before_hwc.append(v)
            self[tuple(all_dims_before_hwc)] = val
            all_dims_before_hwc = all_dims_before_hwc[:-2]
            all_dims_before_hwc.append(v)
            all_dims_before_hwc.append(slice(None))
            self[tuple(all_dims_before_hwc)] = val
            all_dims_before_hwc = all_dims_before_hwc[:-2]
            all_dims_before_hwc.append(-(v + 1))
            all_dims_before_hwc.append(slice(None))
            self[tuple(all_dims_before_hwc)] = val
            all_dims_before_hwc = all_dims_before_hwc[:-2]
            all_dims_before_hwc.append(slice(None))
            all_dims_before_hwc.append(-(v + 1))
            self[tuple(all_dims_before_hwc)] = val
            all_dims_before_hwc = all_dims_before_hwc[:-2]

    # TODO in fact that is more complex I should not donwsample the channel or color dimension nor the time dimension --> so I need many more parameters and controls  --> quite good already but finalize that later
    def downsample(self, dimensions_to_downsample, downsampling_factor=2):
        '''Downsamples an image along the specified dimension by the specified factor

        Parameters
        ----------
        dimensions_to_downsample : string
            chars representing the dimension to downsample

        downsampling_factor : int
            downsampling factor

        Returns
        -------
        ndarray
            a downsampled image
        '''

        if downsampling_factor == 1:
            logger.error("downsampling with a factor = 1 means no downsampling, thereby ignoring...")
            return self

        if self.metadata['dimensions'] is None:
            logger.error("Image dimensions not specified!!!")
            return self
        idx = None

        for dim in self.metadata['dimensions']:
            if dim in dimensions_to_downsample:
                if idx is None:
                    idx = np.index_exp[::downsampling_factor]
                else:
                    idx += np.index_exp[::downsampling_factor]
            else:
                if idx is None:
                    idx = np.index_exp[:]
                else:
                    idx += np.index_exp[:]

        if idx is None:
            return self
        return self[idx]

    def rescale(self, factor=2):
        '''rescales an image (using scipy)

        Parameters
        ----------
        factor : int
            rescaling factor

        Returns
        -------
        ndarray
            a rescaled image

        '''
        return skimage.transform.rescale(self, 1. / factor, preserve_range=True, anti_aliasing=False, multichannel=True)

    # ideally should make it return an image but maybe too complicated --> ok for now let's wait for my python skills to improve
    def convolve(self, kernel=np.array([[-1, -1, -1],
                                        [-1, 8, -1],
                                        [-1, -1, -1]])):
        '''convolves an image (using scipy)

        Parameters
        ----------
        kernel : np.array
          a convolution kernel

        Returns
        -------
        ndarray
          a convolved image

        '''

        convolved = scipy.signal.convolve2d(self, kernel, 'valid')
        return convolved

    def has_dimension(self, dim):

        '''Returns True if image has the specified dimension, False otherwise

        Parameters
        ----------
        dim : single char string
            dimension of interest

        Returns
        -------
        boolean
            True if dimension of interest exist in image
        '''

        # use dimension synonyms
        if dim == 'x':
            dim = 'w'
        if dim == 'y':
            dim = 'h'
        if dim == 'z':
            dim = 'd'
        if dim in self.meta_data['dimensions']:
            return True
        return False

    def is_stack(self):
        '''returns True if image has a z/d dimension, False otherwise

        '''
        return self.has_d()

    def has_channels(self):
        '''returns True if image has a c dimension, False otherwise

        '''
        return self.has_c()

    def get_t(self, t):
        '''returns an image at time t, None otherwise

        Parameters
        ----------
        t : int
            time point of interest

        Returns
        -------
        ndarray
            image at time t or None
        '''
        if not self.is_time_series():
            return None
        if t < self.get_dimension('t'):  # TODO check code
            return self.imCopy(t=t)
        return None

    # set the current time frame
    def set_t(self, t):
        self.metadata['cur_t'] = t

    def get_d_scaling(self):
        '''gets the z/d scaling factor for the current image

        Returns
        -------
        float
            the depth scaling factor
        '''
        return self.z_scale

    def set_d_scaling(self, scaling_factor):
        '''sets the z/d scaling factor for the current image

        Parameters
        ----------
        scaling_factor : float
            the new image scaling factor

        '''
        self.z_scale = scaling_factor

    def has_t(self):
        '''returns True if the image is a time series, False otherwise

        '''
        return self.has_dimension('t')

    def is_time_series(self):
        '''returns True if the image is a time series, False otherwise

        '''
        return self.has_t()

    def has_d(self):
        '''returns True if the image is a Z-stack, False otherwise

        '''
        return self.has_dimension('d') or self.has_dimension('z')

    def has_dimension(self, d):
        '''returns True if the image has the specified dimension, False otherwise

        Parameters
        ----------
        dim : single char string
            dimension of interest

        Returns
        -------
        boolean
            True if dim exists
        '''
        return d in self.metadata['dimensions']

    # check for the presence of LUTs
    def has_LUTs(self):
        return 'LUTs' in self.metadata and self.metadata['LUTs'] is not None

    # get LUTs
    def get_LUTs(self):
        if 'LUTs' in self.metadata:
            return self.metadata['LUTs']
        return None

    # set LUTs
    def set_LUTs(self, LUTs):
        self.metadata['LUTs'] = LUTs

    def has_c(self):
        '''returns True if the image has color channels, False otherwise

        '''
        return 'c' in self.metadata['dimensions']

    def _create_dir(self, output_name):
        # create dir if does not exist
        if output_name is None:
            return
        output_folder, filename = os.path.split(output_name)
        # bug fix in case just a filename and no parent folder
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

    @staticmethod
    def img2Base64(img):
        # save it as png and encode it
        if img is not None:
            # assume image
            buf = io.BytesIO()
            im = Image.fromarray(img)
            im.save(buf, format='png')
            buf.seek(0)  # rewind file
            figdata_png = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()
            return figdata_png
        else:
            # assume pyplot image then
            print('Please call this before plt.show() to avoid getting a blank output')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')  # TO REMOVE UNNECESSARY WHITE SPACE AROUND GRAPH...
            buf.seek(0)  # rewind file
            figdata_png = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()
            return figdata_png

    # mode can be IJ or raw --> if raw --> set IJ to false and save directly TODO clean the mode and mode is only for tif so far --> find a way to make it better and more optimal --> check also how mode would behave with z stacks, etc...
    def save(self, output_name, print_file_name=False, ijmetadata='copy', mode='IJ'):
        '''saves the current image

        Parameters
        ----------
        output_name : string
            name of the file to save

        '''

        if print_file_name:
            print('saving', output_name)

        if output_name is None:
            logger.error("No output name specified... ignoring...")
            return

        # TODO maybe handle tif with stars in their name here to avoid loss of data but ok for now...
        if not '*' in output_name and (output_name.lower().endswith('.tif') or output_name.lower().endswith('.tiff')):
            self._create_dir(output_name)
            if mode != 'IJ':  # TODO maybe do a TA mode or alike instead...
                out = self
                tifffile.imwrite(output_name, out)
            else:
                # create dir if does not exist
                out = self
                # apparently int type is not supported by IJ
                if out.dtype == np.int32:
                    out = out.astype(np.float32)  # TODO check if correct with real image but should be
                if out.dtype == np.int64:
                    out = out.astype(np.float64)  # TODO check if correct with real image but should be
                # IJ does not support bool type too
                if out.dtype == bool:
                    out = out.astype(np.uint8) * 255
                if out.dtype == np.double:
                    out = out.astype(np.float32)
                # if self.has_c():
                #     if not self.has_d() and self.has_t():
                #         out = np.expand_dims(out, axis=-1)
                #         out = np.moveaxis(out, -1, 1)
                #     out = np.moveaxis(out, -1, -3)
                #     tifffile.imwrite(output_name, out, imagej=True)  # make the data compatible with IJ
                # else:
                #     # most likely a big bug here --> fix it --> if has d and no t does it create a bug ???? --> maybe
                #     if not self.has_d() and self.has_t():
                #         out = np.expand_dims(out, axis=-1)
                #         out = np.moveaxis(out, -1, 1)
                #     out = np.expand_dims(out, axis=-1)
                #     # reorder dimensions in the IJ order
                #     out = np.moveaxis(out, -1, -3)
                #     tifffile.imwrite(output_name, out, imagej=True)  # this is the way to get the data compatible with IJ
                # should work better now and fix several issues... but need test it with real images
                # if image has no c --> assume all ok
                if self.metadata['dimensions'] is not None:
                    # print('in dims')
                    # print(self.has_c())  # why has no c channel ???
                    if not self.has_c():
                        out = out[..., np.newaxis]
                    if not self.has_d():
                        out = out[np.newaxis, ...]
                    if not self.has_t():
                        out = out[np.newaxis, ...]
                else:
                    # print('othyer')
                    # no dimension specified --> assume always the same order that is tzyxc --> TODO maybe ...tzyxc
                    if out.ndim < 3:
                        out = out[..., np.newaxis]
                    if out.ndim < 4:
                        out = out[np.newaxis, ...]
                    if out.ndim < 5:
                        out = out[np.newaxis, ...]

                # print('final', out.shape)

                out = np.moveaxis(out, -1, -3)  # need move c channel before hw (because it is default IJ style)

                # TODO maybe offer compression at some point to gain space ???
                # imageJ order is TZCYXS order with dtype is uint8, uint16, or float32. Is S a LUT ???? probably yes because (S=3 or S=4) must be uint8. can I use compression with ImageJ's Bio-Formats import function.
                # TODO add the possibility to save ROIs if needed...
                #        Parameters 'append', 'byteorder', 'bigtiff', and 'imagej', are passed             #         to TiffWriter(). Other parameters are passed to TiffWriter.save().
                # print(ijmetadata)

                # working version 2021.11.2

                ijmeta = {}
                if ijmetadata == 'copy':
                    if self.metadata['Overlays']:
                        ijmeta['Overlays'] = self.metadata['Overlays']
                    if self.metadata['ROI']:
                        ijmeta['ROI'] = self.metadata['ROI']
                    # TODO add support for Luts some day --> make sure IJ luts and epyseg lust are not incompatible or define an IJ_LUTs in metadata and get it
                    # make sure this does not create trouble
                    if self.metadata['LUTs']:
                        ijmeta['LUTs'] = self.metadata['LUTs']
                if not ijmeta:
                    ijmeta = None

                # old save code with deprecated ijmetadata
                if tifffile.__version__ < '2022.4.22':
                    tifffile.imwrite(output_name, out, imagej=True, ijmetadata=ijmeta,metadata={'mode': 'composite'} if self.metadata['dimensions'] is not None and self.has_c() else {})  # small hack to keep only non RGB images as composite and self.get_dimension('c')!=3
                else:
                    try:
                        # somehow this code doesn't seem to work with old tiffile but works with new one
                        from imageio.plugins._tifffile import imagej_metadata_tags
                        # fix for ijmetadata deprecation in recent tifffile
                        ijtags = imagej_metadata_tags(ijmeta, '>') if ijmeta is not None else {}
                        # nb can add and save lut to the metadata --> see https://stackoverflow.com/questions/50258287/how-to-specify-colormap-when-saving-tiff-stack

                        # quick hack to force images to display as composite in IJ if they have channels -> probably needs be improved at some point
                        tifffile.imwrite(output_name, out, imagej=True, metadata={'mode': 'composite'} if self.metadata['dimensions'] is not None and self.has_c() else {}, extratags=ijtags)  # small hack to keep only non RGB images as composite and self.get_dimension('c')!=3
                        # TODO at some point handle support for RGB 24-32 bits images saving as IJ compatible but skip for now
                        # nb tifffile.imwrite(os.path.join(filename0_without_ext,'tra_test_saving_24bits_0.tif'), tracked_cells_t0, imagej=True,                      metadata={}) --> saves as RGB if image RGB 3 channels

                        # TODO --> some day do the saving smartly with the dimensions included see https://pypi.org/project/tifffile/
                        # imwrite('temp.tif', data, bigtiff=True, photometric='minisblack',  compression = 'deflate', planarconfig = 'separate', tile = (32, 32),    metadata = {'axes': 'TZCYX'})
                        # imwrite('temp.tif', volume, imagej=True, resolution=(1. / 2.6755, 1. / 2.6755),        metadata = {'spacing': 3.947368, 'unit': 'um', 'axes': 'ZYX'})
                    except:
                        traceback.print_exc()
                        tifffile.imwrite(output_name, out, imagej=True, metadata={'mode': 'composite'} if self.metadata[
                                                                                                              'dimensions'] is not None and self.has_c() else {})  # small hack to keep only non RGB images as composite and self.get_dimension('c')!=3
        else:
            if output_name.lower().endswith('.npy') or output_name.lower().endswith('.epyseg'):
                # directly save as .npy --> the numpy default array format
                self._create_dir(output_name)
                np.save(output_name, self,
                        allow_pickle=False)  # set allow pickle false to avoid pbs as pickle is by def not stable

                if self.metadata is not None and 'times' in self.metadata.keys():
                    times = self.metadata['times']
                    # force serialisation of times
                    self.metadata['times'] = str(times)
                    with open(output_name + '.meta', 'w') as outfile:
                        json.dump(self.metadata, outfile)

                    # restore time metadata
                    self.metadata['times'] = times
                    # print('exporting metadata', self.metadata)  # metadata is not set --> too bad --> why
                # np.savez_compressed(output_name, self ) allow_pickle=False {'allow_pickle':False} --> maybe pass that
                return

            # the huge pb with this is that it is not portable --> because it necessarily uses pickle --> very dangerous save and too bad cause would allow saving metadata easily if passed as an array...
            if output_name.lower().endswith('.npz'):
                # directly save as .npy --> the numpy default array format
                self._create_dir(output_name)
                # VERY GOOD IDEA TODO data is saved as data.npy inside the npz --> could therefore also save metadata ... --> VERY GOOD IDEA
                np.savez_compressed(output_name,
                                    data=self)  # set allow pickle false to avoid pbs as pickle is by def not stable
                return

            if not '*' in output_name and (self.has_t() or self.has_d()):
                logger.warning(
                    "image is a stack and cannot be saved as a single image use a geneic name like /path/to/img*.png instead")
                return
            else:
                self._create_dir(output_name)
                if not self.has_t() and not self.has_d():
                    new_im = Image.fromarray(self)
                    new_im.save(output_name)
                    self.save_IJ_ROIs_or_overlays(output_name)
                    # try save IJ ROIs and overlays if they exist
                else:
                    # TODO recode below to allow any number of dimensions
                    if self.has_t():
                        t_counter = 0
                        # loop over all times of the image
                        for t in self[:]:
                            z_counter = 0
                            # loop over all z of the image
                            for z in t[:]:
                                if z.ndim == 3 and z.shape[2] <= 2:
                                    # create a 3 channel array from the 2 channel array image provided
                                    z = np.concatenate((z[..., 0, np.newaxis], z[..., 1, np.newaxis],
                                                        np.zeros_like(z[..., 0, np.newaxis])), axis=-1)
                                with warnings.catch_warnings():  # force it to be 8 bits for these formats
                                    warnings.simplefilter('ignore')
                                    z = img_as_ubyte(z)
                                new_im = Image.fromarray(z)
                                new_im.save(output_name.replace('*', 't{:03d}_z{:04d}'.format(t_counter,
                                                                                              z_counter)))  # replace * by tover 3 digit and z over 4 digits
                                z_counter += 1
                            t_counter += 1
                        self.save_IJ_ROIs_or_overlays(output_name)
                    elif self.has_d():
                        # loop over all z of the image
                        z_counter = 0
                        for z in self[:]:
                            if z.ndim == 3 and z.shape[2] <= 2:
                                # create a 3 channel array from the 2 channel array image provided
                                z = np.concatenate((z[..., 0, np.newaxis], z[..., 1, np.newaxis],
                                                    np.zeros_like(z[..., 0, np.newaxis])), axis=-1)
                            with warnings.catch_warnings():  # force it 8 bits for these rough formats
                                warnings.simplefilter('ignore')
                                z = img_as_ubyte(z)
                            new_im = Image.fromarray(z)
                            new_im.save(
                                output_name.replace('*', 'z{:04d}'.format(z_counter)))  # replace * by z over 4 digits
                            z_counter += 1
                        self.save_IJ_ROIs_or_overlays(output_name)

    # returns IJ ROIs from metadata
    def get_IJ_ROIs(self):
        try:
            # trying to save ROIs from ij images
            from roifile import ImagejRoi
            rois = []
            if self.metadata['Overlays'] is not None:
                overlays = self.metadata['Overlays']
                if isinstance(overlays, list):
                    if overlays:
                        overlays = [ImagejRoi.frombytes(roi) for roi in overlays]
                        rois.extend(overlays)
                else:
                    overlays = ImagejRoi.frombytes(overlays)
                    rois.append(overlays)
            if self.metadata['ROI'] is not None:
                rois_ = self.metadata['ROI']
                print(len(rois_), rois_)
                if isinstance(rois_, list):
                    if rois_:
                        rois_ = [ImagejRoi.frombytes(roi) for roi in rois_]
                        rois.extend(rois_)
                else:
                    rois_ = ImagejRoi.frombytes(rois_)
                    rois.append(rois_)
            if not rois:
                return None

            return rois
        except:
            # no big deal if it fails --> just print error for now
            traceback.print_exc()

    # maybe do an IJ ROI editor some day ????
    # saves IJ ROIs as a .roi file or .zip file
    def save_IJ_ROIs_or_overlays(self, filename):
        try:
            # trying to save ROIs from ij images
            rois = self.get_IJ_ROIs()
            if not rois:
                return
            output_filename = filename
            if len(rois) > 1:
                # delete file if exists
                output_filename += '.zip'
                if os.path.exists(output_filename):
                    os.remove(output_filename)
            else:
                output_filename += '.roi'
            if rois is not None and rois:
                for roi in rois:
                    roi.tofile(output_filename)
        except:
            # no big deal if it fails --> just print error for now
            traceback.print_exc()

    def get_width(self):
        return self.get_dimension('w')

    def get_height(self):
        return self.get_dimension('h')

    def projection(self, type='max'):
        '''creates projection

        TODO add more proj

        Parameters
        ----------
        type : string
            projection type

        '''

        # TODO implement that more wisely asking just which dimension should be projected and projection type
        proj_dimensions = []
        if self.has_t():
            proj_dimensions.append(self.get_dimension('t'))
        proj_dimensions.append(self.get_height())
        proj_dimensions.append(self.get_width())
        if self.has_c():
            proj_dimensions.append(self.get_dimension('c'))

        projection = np.zeros(tuple(proj_dimensions), dtype=self.dtype)

        if type == 'max':
            if self.has_t():
                # do proj for each channel
                if self.has_c():
                    for t in range(self.shape[0]):
                        if self.has_d():
                            for z in self[t][:]:
                                for i in range(z.shape[-1]):
                                    projection[t, ..., i] = np.maximum(projection[t, ..., i], z[..., i])
                    # print(projection.shape)
                    return Img(projection, dimensions='thwc')
                else:
                    for t in range(self.shape[0]):
                        if self.has_d():
                            for z in self[t]:
                                projection[t] = np.maximum(projection[t], z)
                    return Img(projection, dimensions='thw')
            elif self.has_c():
                if self.has_d():
                    for z in self[:]:
                        for i in range(z.shape[-1]):
                            projection[..., i] = np.maximum(projection[..., i], z[..., i])
                return Img(projection, dimensions='hwc')
            else:
                if self.has_d():
                    for z in self[:]:
                        projection = np.maximum(projection, z)
                return Img(projection, dimensions='hw')
        else:
            logger.critical("projection type " + type + " not supported yet")
            return None
        return self

    # TODO DANGER!!!! OVERRIDING __str__ CAUSES HUGE TROUBLE BUT NO CLUE WHY
    #  --> this messes the whole class and the slicing of the array --> DO NOT PUT IT BACK --> NO CLUE WHY THOUGH
    # def __str__(self):
    def to_string(self):
        '''A string representation of this image

        '''
        description = '#' * 20
        description += '\n'
        description += 'Image:'
        description += '\n'
        description += 'vx=' + str(self.metadata['vx']) + ' vy=' + str(self.metadata['vy']) + ' vz=' + str(
            self.metadata['vz'])
        description += '\n'
        description += 'dimensions=' + self.metadata['dimensions']
        description += '\n'
        description += 'shape=' + str(self.shape)
        description += '\n'
        description += self.metadata.__str__()
        description += '\n'
        dimensions_sizes = self.get_dimensions()
        for k, v in dimensions_sizes.items():
            description += k + '=' + str(v) + ' '
        description += '\n'
        description += str(super.__str__(self))
        description += '\n'
        description += '#' * 20
        return description

    # below assumes channels last
    @staticmethod
    def BGR_to_RGB(bgr):
        return bgr[..., ::-1]

    @staticmethod
    def RGB_to_BGR(rgb):
        return rgb[..., ::-1]

    @staticmethod
    def RGB_to_GBR(rgb):
        return rgb[..., [2, 0, 1]]

    @staticmethod
    def RGB_to_GRB(rgb):
        return rgb[..., [1, 0, 2]]

    @staticmethod
    def RGB_to_RBG(rgb):
        return rgb[..., [0, 2, 1]]

    @staticmethod
    def RGB_to_BRG(rgb):
        return rgb[..., [2, 0, 1]]

    # TODO code that better and make a version that directly takes images
    # assume images must be 2D or 2D + color --> see how to do that!!!
    # buggy and deprecated --> use toQimage
    # def getQimage(self):
    #     '''get a qimage from ndarray
    #
    #     Returns
    #     -------
    #     qimage
    #         a pyqt compatible image
    #     '''
    #     logger.debug('Creating a qimage from a numpy image')
    #     img = self
    #     dims = []
    #     for d in self.metadata['dimensions']:
    #         if d in ['w', 'h', 'c', 'x', 'y']:
    #             dims.append(slice(None))
    #         else:
    #             dims.append(0)
    #     img = img[tuple(dims)]
    #     img = np.ndarray.copy(img)  # need copy the array
    #
    #     if img.dtype != np.uint8:
    #         # just to remove the warning raised by img_as_ubyte
    #         with warnings.catch_warnings():
    #             warnings.simplefilter('ignore')
    #             try:
    #                 # need manual conversion of the image so that it can be read as 8 bit or alike
    #                 # force image between 0 and 1 then do convert
    #                 img = img_as_ubyte((img - img.min()) / (img.max() - img.min()))
    #             except:
    #                 print('error converting image to 8 bits')
    #                 return None
    #
    #     bytesPerLine = img.strides[0]
    #
    #     if self.has_c() and self.get_dimension('c') is not None and self.get_dimension('c') != 0:
    #         nb_channels = self.get_dimension('c')
    #         logger.debug('Image has ' + str(nb_channels) + ' channels')
    #
    #         if nb_channels == 3:
    #             qimage = QImage(img.data, self.get_width(), self.get_height(), bytesPerLine,
    #                             QImage.Format_RGB888)
    #         elif nb_channels < 3:
    #             # add n dimensions
    #             bgra = np.zeros((self.get_height(), self.get_width(), 3), np.uint8, 'C')
    #             if img.shape[2] >= 1:
    #                 bgra[..., 0] = img[..., 0]
    #             if img.shape[2] >= 2:
    #                 bgra[..., 1] = img[..., 1]
    #             if img.shape[2] >= 3:
    #                 bgra[..., 2] = img[..., 2]
    #             qimage = QImage(bgra.data, self.get_width(), self.get_height(), bgra.strides[0], QImage.Format_RGB888)
    #         else:
    #             if nb_channels == 4:
    #                 bgra = np.zeros((self.get_height(), self.get_width(), 4), np.uint8, 'C')
    #                 bgra[..., 0] = img[..., 0]
    #                 bgra[..., 1] = img[..., 1]
    #                 bgra[..., 2] = img[..., 2]
    #                 if img.shape[2] >= 4:
    #                     logger.debug('using 4th numpy color channel as alpha for qimage')
    #                     bgra[..., 3] = img[..., 3]
    #                 else:
    #                     bgra[..., 3].fill(255)
    #                 qimage = QImage(bgra.data, self.get_width(), self.get_height(), bgra.strides[0],
    #                                 QImage.Format_ARGB32)
    #             else:
    #                 # TODO
    #                 logger.error("not implemented yet!!!!, too many channels")
    #     else:
    #         qimage = QImage(img.data, self.get_width(), self.get_height(), bytesPerLine,
    #                         QImage.Format_Indexed8)
    #         # required to allow creation of a qicon --> need keep
    #         for i in range(256):
    #             qimage.setColor(i, QColor(i, i, i).rgb())
    #     return qimage

    @staticmethod
    def interpolation_free_rotation(img, angle=90):
        '''performs a rotation that does not require interpolation

        :param img: image to be rotated
        :param angle: int in [90, 180, 270] or 'random' string
        :return: a rotated image without interpolation
        '''
        if angle is 'random':
            angle = random.choice([90, 180, 270])
            return Img.interpolation_free_rotation(img, angle=angle)
        else:
            if angle < 0:
                angle = 360 + angle

            if angle == 270:
                return np.rot90(img, 3)
            elif angle == 180:
                return np.rot90(img, 2)
            else:
                return np.rot90(img)

    @staticmethod
    def get_2D_tiles_with_overlap(inp, width=512, height=512, overlap=0, overlap_x=None, overlap_y=None, dimension_h=0,
                                  dimension_w=1, force_to_size=False):
        '''split 2 and 3D images with h/w overlap

                Parameters
                ----------
                inp : ndarray
                    input image to be cut into tiles
                width : int
                    desired tile width
                height : int
                    desired tile width
                overlap : int
                    tile w and h overlap
                overlap_x : int
                    tile overlap w axis (if set overrides overlap)
                overlap_y : int
                    tile overlap y axis (if set overrides overlap)
                dimension_h : int
                    position of the h dimension in the ndarray
                dimension_w : int
                    position of the w dimension in the ndarray
                force_to_size : boolean
                    if True add empty pixels around the image to force image to have width and height

                Returns
                -------
                dict, 2D list
                    a dict containing instructions to reassemble the tiles, and a 2D list containing all the tiles
                '''

        if overlap_x is None:
            overlap_x = overlap
        if overlap_y is None:
            overlap_y = overlap

        # for debug
        # overlap_x = 32
        # overlap_y = 32

        if dimension_h < 0:
            dimension_h = len(inp.shape) + dimension_h
        if dimension_w < 0:
            dimension_w = len(inp.shape) + dimension_w

        # print('inpshape', inp.shape, width, height, dimension_h, dimension_w)

        final_height = inp.shape[dimension_h]
        final_width = inp.shape[dimension_w]

        if overlap_x % 2 != 0 or overlap_y % 2 != 0:
            logger.error(
                'Warning overlap in x or y dimension is not even, this will cause numerous errors please do change this!')

        last_idx = 0
        cuts_y = []
        end = 0

        # print(overlap_x, overlap_y, 'overlap')
        if height >= inp.shape[dimension_h]:
            overlap_y = 0
        if width >= inp.shape[dimension_w]:
            overlap_x = 0

        # print(overlap_x, overlap_y, 'overlap', height, width, inp.shape[dimension_w], inp.shape[dimension_h])

        if height + overlap_y < inp.shape[dimension_h]:
            for idx in range(height, inp.shape[dimension_h], height):
                begin = last_idx
                end = idx + overlap_y
                if begin < 0:
                    begin = 0
                if end >= inp.shape[dimension_h]:
                    end = inp.shape[dimension_h]
                cuts_y.append((begin, end))
                last_idx = idx
            if end < inp.shape[dimension_h] - 1:
                begin = last_idx
                end = inp.shape[dimension_h]
                if begin < 0:
                    begin = 0
                cuts_y.append((begin, end))
        elif height + overlap_y > inp.shape[dimension_h]:
            height += overlap_y
            overlap_y = 0
            padding = []
            for dim in range(len(inp.shape)):
                padding.append((0, 0))
            # padding_required = False
            padding[dimension_h] = (0, height - inp.shape[dimension_h])
            # padding_required = True
            # bigger = np.zeros(
            #     (*inp.shape[:dimension_h], height + overlap_y, inp.shape[dimension_w], *inp.shape[dimension_w + 1:]),
            #     dtype=inp.dtype)
            # if dimension_h == 2:
            #     bigger[:, :, :inp.shape[dimension_h], :inp.shape[dimension_w]] = inp
            # elif dimension_h == 1:
            #     bigger[:, :inp.shape[dimension_h], :inp.shape[dimension_w]] = inp
            # elif dimension_h == 0:
            #     bigger[:inp.shape[dimension_h], :inp.shape[dimension_w]] = inp
            bigger = np.pad(inp, pad_width=tuple(padding), mode='symmetric')
            inp = bigger
            del bigger
            cuts_y.append((0, inp.shape[dimension_h]))
        else:
            cuts_y.append((0, inp.shape[dimension_h]))

        # now split image along x direction
        last_idx = 0
        cuts_x = []
        if width + overlap_x < inp.shape[dimension_w]:
            for idx in range(width, inp.shape[dimension_w], width):
                begin = last_idx
                end = idx + overlap_x
                if begin < 0:
                    begin = 0
                if end >= inp.shape[dimension_w]:
                    end = inp.shape[dimension_w]
                cuts_x.append((begin, end))
                last_idx = idx
            if end < inp.shape[dimension_w] - 1:
                begin = last_idx
                end = inp.shape[dimension_w]
                if begin < 0:
                    begin = 0
                cuts_x.append((begin, end))
        elif width + overlap_x > inp.shape[dimension_w]:
            width += overlap_x
            overlap_x = 0
            # bigger = np.zeros((*inp.shape[:dimension_w], width + overlap_x, *inp.shape[dimension_w + 1:]),
            #                   dtype=inp.dtype)
            # if dimension_w == 3:
            #     bigger[:, :, :inp.shape[dimension_h], :inp.shape[dimension_w]] = inp
            # elif dimension_w == 2:
            #     bigger[:, :inp.shape[dimension_h], :inp.shape[dimension_w]] = inp
            # elif dimension_w == 1:
            #     bigger[:inp.shape[dimension_h], :inp.shape[dimension_w]] = inp
            padding = []
            for dim in range(len(inp.shape)):
                padding.append((0, 0))
            # padding_required = False
            padding[dimension_w] = (0, width - inp.shape[dimension_w])
            bigger = np.pad(inp, pad_width=tuple(padding), mode='symmetric')
            inp = bigger
            del bigger
            cuts_x.append((0, inp.shape[dimension_w]))
        else:
            cuts_x.append((0, inp.shape[dimension_w]))

        nb_tiles = 0
        final_splits = []
        for x_begin, x_end in cuts_x:
            cols = []
            for y_begin, y_end in cuts_y:
                # try crop with real data if possible otherwise add black area around
                if (y_end == inp.shape[0] or x_end == inp.shape[1]) and (
                        width + overlap_x <= inp.shape[1] and height + overlap_y <= inp.shape[0]):
                    if dimension_h == 2:
                        cur_slice = inp[:, :, y_end - (height + overlap_y):y_end, x_end - (width + overlap_x):x_end]
                    elif dimension_h == 1:
                        cur_slice = inp[:, y_end - (height + overlap_y):y_end, x_end - (width + overlap_x):x_end]
                    elif dimension_h == 0:
                        cur_slice = inp[y_end - (height + overlap_y):y_end, x_end - (width + overlap_x):x_end]
                else:
                    if dimension_h == 2:
                        cur_slice = inp[:, :, y_begin:y_end, x_begin:x_end]
                    elif dimension_h == 1:
                        cur_slice = inp[:, y_begin:y_end, x_begin:x_end]
                    elif dimension_h == 0:
                        cur_slice = inp[y_begin:y_end, x_begin:x_end]
                nb_tiles += 1

                if not force_to_size:
                    cols.append(cur_slice)
                else:
                    # if size is still smaller than desired resize
                    padding = []
                    for dim in range(len(cur_slice.shape)):
                        padding.append((0, 0))
                    padding_required = False
                    if cur_slice.shape[dimension_h] < height + overlap_y:
                        padding[dimension_h] = (0, (height + overlap_y) - cur_slice.shape[dimension_h])
                        padding_required = True
                        # bigger = np.zeros(
                        #     (*cur_slice.shape[:dimension_h], height + overlap_y, cur_slice.shape[dimension_w],
                        #      *cur_slice.shape[dimension_w + 1:]), dtype=cur_slice.dtype)
                        # if dimension_h == 2:
                        #     bigger[:, :, :cur_slice.shape[dimension_h], :cur_slice.shape[dimension_w]] = cur_slice
                        # elif dimension_h == 1:
                        #     bigger[:, :cur_slice.shape[dimension_h], :cur_slice.shape[dimension_w]] = cur_slice
                        # elif dimension_h == 0:
                        #     bigger[:cur_slice.shape[dimension_h], :cur_slice.shape[dimension_w]] = cur_slice
                    if cur_slice.shape[dimension_w] < width + overlap_x:
                        padding[dimension_w] = (0, (width + overlap_x) - cur_slice.shape[dimension_w])
                        padding_required = True
                    # print('padding_required', padding_required, cur_slice.shape[dimension_h],cur_slice.shape[dimension_w], width + overlap_x, height+overlap_x)
                    if padding_required:
                        # print('dding here', padding)
                        bigger = np.pad(cur_slice, pad_width=tuple(padding), mode='symmetric')
                        cur_slice = bigger
                        del bigger

                    # if cur_slice.shape[dimension_w] < width + overlap_x:
                    #     bigger = np.zeros(
                    #         (*cur_slice.shape[:dimension_w], width + overlap_x, *cur_slice.shape[dimension_w + 1:]),
                    #         dtype=cur_slice.dtype)
                    #     if dimension_w == 3:
                    #         bigger[:, :, :cur_slice.shape[dimension_h], :cur_slice.shape[dimension_w]] = cur_slice
                    #     elif dimension_w == 2:
                    #         bigger[:, :cur_slice.shape[dimension_h], :cur_slice.shape[dimension_w]] = cur_slice
                    #     elif dimension_w == 1:
                    #         bigger[:cur_slice.shape[dimension_h], :cur_slice.shape[dimension_w]] = cur_slice
                    #     cur_slice = bigger
                    cols.append(cur_slice)
            final_splits.append(cols)

        crop_params = {'overlap_y': overlap_y, 'overlap_x': overlap_x, 'final_height': final_height,
                       'final_width': final_width, 'n_cols': len(final_splits[0]), 'n_rows': len(final_splits),
                       'nb_tiles': nb_tiles}

        return crop_params, final_splits

    @staticmethod
    def tiles_to_linear(tiles):
        '''converts tiles to a 1D list

        Parameters
        ----------
        tiles : 2D list
            image tiles

        Returns
        -------
        list
            1D list containing tiles
        '''
        linear = []
        for idx in range(len(tiles)):
            for j in range(len(tiles[0])):
                linear.append(tiles[idx][j])
        return linear

    @staticmethod
    def tiles_to_batch(tiles):
        '''converts 2D list of tiles to an ndarray with a batch dimension (for tensorflow input)

        Parameters
        ----------
        tiles : 2D list
            tiled image

        Returns
        -------
        ndarray
            ndarray with a batch dimension as the first dimension
        '''
        linear = Img.tiles_to_linear(tiles)
        out = np.concatenate(tuple(linear), axis=0)
        return out

    @staticmethod
    def normalization(img, method=None, range=None, individual_channels=False, clip=False,
                      normalization_minima_and_maxima=None):
        '''normalize an image

        Parameters
        ----------
        img : ndarray
            input image

        method : string
            normalization method

        range : list
            range of the image after normalization (e.g. [0, 1], [-1,1]

        individual_channels : boolean
            if True normalization is per channel (i.e. max and min are computed for each channel individually, rather than globally)

        Returns
        -------
        ndarray
            a normalized image
        '''
        if img is None:
            logger.error("'None' image cannot be normalized")
            return

        logger.debug('max before normalization=' + str(img.max()) + ' min before normalization=' + str(img.min()))
        if method is None or method == 'None':
            logger.debug('Image is not normalized')
            return img

        if 'ercentile' in method:
            logger.debug('Image will be normalized using percentiles')
            img = img.astype(np.float32)
            img = Img._nomalize(img, individual_channels=individual_channels, method=method,
                                norm_range=range, clip=clip,
                                normalization_minima_and_maxima=normalization_minima_and_maxima)  # TODO if range is list of list --> assume per channel data and do norm that way --> TODO --> think about the best way to do that
            logger.debug('max after normalization=' + str(img.max()) + ' min after normalization=' + str(img.min()))
            return img
        elif 'ormalization' in method and not 'tandardization' in method:
            logger.debug('Image will be normalized')
            img = img.astype(np.float32)
            img = Img._nomalize(img, individual_channels=individual_channels, method=method,
                                norm_range=range)
            logger.debug('max after normalization=' + str(img.max()) + ' min after normalization=' + str(img.min()))
            return img
        elif 'tandardization' in method:
            logger.debug('Image will be standardized')
            img = img.astype(np.float32)
            img = Img._standardize(img, individual_channels=individual_channels, method=method,
                                   norm_range=range)
            logger.debug('max after standardization=' + str(img.max()) + ' min after standardization=' + str(img.min()))
            return img
        else:
            logger.error('unknown normalization method ' + str(method))
        return img

    # https://en.wikipedia.org/wiki/Feature_scaling
    @staticmethod
    def _nomalize(img, individual_channels=False, method='Rescaling (min-max normalization)', norm_range=None,
                  clip=False, normalization_minima_and_maxima=None):
        eps = 1e-20  # for numerical stability avoid division by 0
        if individual_channels:
            for c in range(img.shape[-1]):
                norm_min_max = None
                if normalization_minima_and_maxima is not None:
                    # if list of list then use that --> in fact could also check if individual channel or not...
                    if isinstance(normalization_minima_and_maxima[0], list):
                        norm_min_max = normalization_minima_and_maxima[c]
                    else:
                        norm_min_max = normalization_minima_and_maxima
                img[..., c] = Img._nomalize(img[..., c], individual_channels=False, method=method,
                                            norm_range=norm_range, clip=clip,
                                            normalization_minima_and_maxima=norm_min_max)
        else:
            # that should work
            if 'percentile' in method:
                # direct_range ??? --> think how to do that ???
                # TODO here in some cases need assume passed directly the percentiles and in that case need not do that again... --> think how to do that --> shall I pass a second parameter directly --> maybe direct_range that bypasses the percentiles if set --> TODO --> check that
                if normalization_minima_and_maxima is None:
                    lowest_percentile = np.percentile(img, norm_range[0])
                    highest_percentile = np.percentile(img, norm_range[1])
                else:
                    lowest_percentile = normalization_minima_and_maxima[0]
                    highest_percentile = normalization_minima_and_maxima[1]
                try:
                    import numexpr
                    img = numexpr.evaluate(
                        "(img - lowest_percentile) / ( highest_percentile - lowest_percentile + eps )")
                except:
                    img = (img - lowest_percentile) / (highest_percentile - lowest_percentile + eps)
                if clip:
                    img = np.clip(img, 0, 1)
            elif method == 'Rescaling (min-max normalization)':
                max = img.max()
                min = img.min()
                # if max != 0 and max != min:
                if norm_range is None or norm_range == [0, 1] or norm_range == '[0, 1]' or norm_range == 'default' \
                        or isinstance(norm_range, int):
                    try:
                        import numexpr
                        img = numexpr.evaluate("(img - min) / (max - min + eps)")
                    except:
                        img = (img - min) / (
                                max - min + eps)  # TODO will it take less memory if I split it into two lines
                elif norm_range == [-1, 1] or norm_range == '[-1, 1]':
                    try:
                        import numexpr
                        img = numexpr.evaluate("-1 + ((img - min) * (1 - -1)) / (max - min + eps)")
                    except:
                        img = -1 + ((img - min) * (1 - -1)) / (max - min + eps)
            elif method == 'Mean normalization':
                # TODO should I implement range too here ??? or deactivate it
                max = img.max()
                min = img.min()
                if max != 0 and max != min:
                    img = (img - np.average(img)) / (max - min)
            elif method.startswith('Max normalization'):  # here too assume 0-1 no need for range
                if 'auto' in method:
                    max = img.max()
                elif '255' in method:
                    max = 255
                elif '4095' in method:
                    max = 4095
                elif '65535' in method:
                    max = 65535

                if max != 0:
                    try:
                        import numexpr
                        img = numexpr.evaluate("img / max")
                    except:
                        img = img / max
            else:
                logger.error('Unknown normalization method "' + str(method) + '" --> ignoring ')
        return img

    @staticmethod
    def _standardize(img, individual_channels=False, method=None, norm_range=range):
        if individual_channels:
            for c in range(img.shape[-1]):
                img[..., c] = Img._standardize(img[..., c], individual_channels=False, method=method,
                                               norm_range=norm_range)
        else:
            mean = np.mean(img)
            std = np.std(img)
            # print('mean', mean, 'std', std)
            if std != 0.0:
                img = (img - mean) / std
            else:
                print('error empty image')
                if mean != 0.0:
                    img = (img - mean)

        if norm_range == [0, 1] or norm_range == [-1, 1] or norm_range == '[0, 1]' or norm_range == '[-1, 1]':
            img = Img._nomalize(img, method='Rescaling (min-max normalization)',
                                individual_channels=individual_channels, norm_range=[0, 1])

        if norm_range == [-1, 1] or norm_range == '[-1, 1]':
            img = (img - 0.5) * 2.

        logger.debug('max after standardization=' + str(img.max()) + ' min after standardization=' + str(img.min()))
        return img

    @staticmethod
    def reassemble_tiles(tiles, crop_parameters, three_d=False):
        '''Changes image contrast using scipy

        Parameters
        ----------
        tiles : list
            input tiles

        crop_parameters : dict
            parameters required to reassemble the tiles

        three_d : boolean
            if True assume image is 3D (dhw), 2D (hw) otherwise

        Returns
        -------
        ndarray
            a reassembled image from individual tiles

        '''

        overlap_y = crop_parameters['overlap_y']
        overlap_x = crop_parameters['overlap_x']
        final_height = crop_parameters['final_height']
        final_width = crop_parameters['final_width']

        cols = []
        for i in range(len(tiles)):
            cur_size = 0
            for j in range(len(tiles[0])):
                if j == 0:
                    if overlap_y != 0:
                        y_slice = slice(None, -int(overlap_y / 2))
                    else:
                        y_slice = slice(None, None)
                elif j == len(tiles[0]) - 1:
                    if overlap_y != 0:
                        y_slice = slice(int(overlap_y / 2), None)
                    else:
                        y_slice = slice(None, None)
                else:
                    if overlap_y != 0:
                        y_slice = slice(int(overlap_y / 2), -int(overlap_y / 2))
                    else:
                        y_slice = slice(None, None)
                if not three_d:
                    tiles[i][j] = tiles[i][j][y_slice, ...]
                    cur_size += tiles[i][j].shape[0]
                else:
                    tiles[i][j] = tiles[i][j][:, y_slice, ...]
                    cur_size += tiles[i][j].shape[1]
            if not three_d:
                cols.append(np.vstack(tuple(tiles[i])))
            else:
                cols.append(np.hstack(tuple(tiles[i])))
        cur_size = 0
        for i in range(len(cols)):
            if i == 0:
                if overlap_x != 0:
                    x_slice = slice(None, -int(overlap_x / 2))
                else:
                    x_slice = slice(None, None)
            elif i == len(cols) - 1:
                if overlap_x != 0:
                    x_slice = slice(int(overlap_x / 2), None)  # orig
                else:
                    x_slice = slice(None, None)
            else:
                if overlap_x != 0:
                    x_slice = slice(int(overlap_x / 2), -int(overlap_x / 2))
                else:
                    x_slice = slice(None, None)

            if not three_d:
                if len(cols[i].shape) == 3:
                    cols[i] = cols[i][:, x_slice]
                else:
                    cols[i] = cols[i][:, x_slice, ...]
                cur_size += cols[i].shape[1]
            else:
                if len(cols[i].shape) == 3:
                    cols[i] = cols[i][:, :, x_slice]
                else:
                    cols[i] = cols[i][:, :, x_slice, ...]
                cur_size += cols[i].shape[2]

        if not three_d:
            return np.hstack(tuple(cols))[:final_height, :final_width]
        else:
            return np.dstack(tuple(cols))[:, :final_height, :final_width]

    @staticmethod
    def linear_to_2D_tiles(tiles, crop_parameters):
        '''converts a 1D list to a 2D list

        Parameters
        ----------
        tiles : list
            1D list containing tiles

        crop_parameters : dict
            parameters to recreate a 2D list from a 1D (i.e. nb or rows and cols)

        Returns
        -------
        list
            a 2D list containing tiles

        '''
        n_rows = crop_parameters['n_rows']
        n_cols = crop_parameters['n_cols']
        nb_tiles = crop_parameters['nb_tiles']  # nb tiles is unused --> so why do I save it --> why not removing it
        # if I exceeded the stuff then skip things

        # do I have a bug in there ???? no clue --> and how can I fix it

        output = []
        counter = 0
        for i in range(n_rows):
            try:
                cols = []
                for j in range(n_cols):
                    # BUG PREVENTING HANDLING MULTIPLE OUTPUTS IS THERE --> TODO
                    cols.append(tiles[counter])
                    counter += 1
                output.append(cols)
            except:
                # bug preventing multiple output seems to be somewhere there or in function that calls it
                traceback.print_exc()
                pass

        return output

    # should dynamically crop images
    def crop(self, **kwargs):
        '''crops an image

        Parameters
        ----------
        kwargs : dict
            a dict containing the top left corner and the bottom right coordinates of the crop x1, y1, x2, y2

        Returns
        -------
        ndarray
            a crop of the image
        '''
        img = self
        corrected_metadata = dict(self.metadata)
        dims = []
        for i in range(len(img.shape)):
            dims.append(slice(None))

        # get the dim and its begin and end and create the appropriate slice
        for key, value in kwargs.items():
            if key in self.metadata['dimensions']:
                idx = self.metadata['dimensions'].index(key)
                if isinstance(value, list):
                    if len(value) == 2:
                        dims[idx] = slice(value[0], value[1])
                    elif len(value) == 3:
                        dims[idx] = slice(value[0], value[1], value[2])
                        # update the width and height parameters then or suppress w and h parameters from the data to avoid pbs
                    elif len(value) == 1:
                        dims[idx] = value
                        corrected_metadata.update(
                            {'dimensions': corrected_metadata['dimensions'].replace(key, '')})  # do remove dimension
                else:
                    if value is not None:
                        dims[idx] = value
                        corrected_metadata.update(
                            {'dimensions': corrected_metadata['dimensions'].replace(key, '')})  # do remove dimension
                    else:
                        dims[idx] = slice(None)
                # TODO need reduce size dim for the stuff in the metadata to avoid bugs

        img = np.ndarray.copy(img[tuple(dims)])
        output = Img(img, metadata=corrected_metadata)
        return output

    # should be able to parse any dimension in fact by its name
    # IMPORTANT NEVER CALL IT COPY OTHERWISE OVERRIDES THE DEFAULT COPY METHOD OF NUMPY ARRAY THAT CREATES ERRORS
    def imCopy(self, t=None, d=None, c=None):
        '''Changes image contrast using scipy

        Parameters
        ----------
        t : int
            the index of the time series to copy

        d : int
            the index of the z/d to copy

        c : int
            the channel to copy

        Returns
        -------
        Img
            a (sub)copy of the image
        '''
        img = self

        corrected_metadata = dict(self.metadata)

        dims = []
        for i in range(len(img.shape)):
            dims.append(slice(None))

        if t is not None and self.has_t():
            idx = self.metadata['dimensions'].index('t')
            dims[idx] = t
            corrected_metadata.update({'dimensions': corrected_metadata['dimensions'].replace('t', '')})
        if d is not None and self.has_d():
            idx = self.metadata['dimensions'].index('d')
            dims[idx] = d
            corrected_metadata.update({'dimensions': corrected_metadata['dimensions'].replace('d', '')})
        if c is not None and self.has_c():
            idx = self.metadata['dimensions'].index('c')
            dims[idx] = c
            corrected_metadata.update({'dimensions': corrected_metadata['dimensions'].replace('c', '')})

        # TODO finalize this to handle any slicing possible --> in fact it's relatively easy

        img = np.ndarray.copy(img[tuple(dims)])
        output = Img(img, metadata=corrected_metadata)
        return output

    def within(self, x, y):
        ''' True if a pixel within the image, False otherwise

        '''
        if x >= 0 and x < self.get_width() and y >= 0 and y < self.get_height():
            return True
        return False

    @staticmethod
    def clip(img, tuple=None, min=None, max=None):
        # clip an image to a defined range
        if tuple is not None:
            min = tuple[0]
            max = tuple[1]
        img = np.clip(img, a_min=min, a_max=max)
        return img

    @staticmethod
    def invert(img):
        # should take the negative of an image should always work I think but try and see if not wise making a version that handles channels # does it even make sense ??? need to think a bit about it
        max = img.max()
        img = np.negative(img) + max
        return img

    @staticmethod
    def clip_by_frequency(img, lower_cutoff=None, upper_cutoff=0.05, channel_mode=True):
        logger.debug(' inside clip ' + str(lower_cutoff) + str(upper_cutoff) + str(channel_mode))

        if lower_cutoff == upper_cutoff == 0:
            logger.debug('clip: keep image unchanged')
            return img
        if lower_cutoff is None and upper_cutoff == 0:
            logger.debug('clip: keep image unchanged')
            return img
        if upper_cutoff is None and lower_cutoff == 0:
            logger.debug('clip: keep image unchanged')
            return img
        if lower_cutoff == upper_cutoff == None:
            logger.debug('clip: keep image unchanged')
            return img

        logger.debug('chan mode ' + str(channel_mode))

        if channel_mode:
            for ch in range(img.shape[-1]):
                img[..., ch] = Img.clip_by_frequency(img[..., ch], lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff,
                                                     channel_mode=False)
            return img

        # print('min', img.min(), 'max', img.max())

        if img.max() == img.min():
            return img

        logger.debug('Removing image outliers/hot pixels')

        # hist, bins = np.histogram(img, bins=np.arange(img.min(), img.max()+1),
        #                           density=True)

        # print(np.percentile(img, 100*(lower_cutoff)))
        # print(np.percentile(img, 100*(1-upper_cutoff)))

        # print('hist', hist)
        # print(hist.sum()) # sums to 1
        # print('bins', bins)

        if upper_cutoff is not None:  # added this to avoid black images
            # cum_freq = 0.
            # max = bins[-1]
            # for idcs, val in enumerate(hist[::-1]):
            #     cum_freq += val
            #     if cum_freq >= upper_cutoff:
            #         max = bins[len(bins) - 1 - idcs]
            #         break
            # print(np.percentile(img, lower_cutoff))
            max = np.percentile(img, 100. * (1. - upper_cutoff))
            img[img > max] = max

        if lower_cutoff is not None:
            # cum_freq = 0.
            # min = bins[0]
            # for idcs, val in enumerate(hist):
            #     cum_freq += val
            #     if cum_freq >= lower_cutoff:
            #         min = bins[idcs]
            #         break
            min = np.percentile(img, 100. * lower_cutoff)
            img[img < min] = min
        # print('--> min', img.min(), 'max', img.max())
        return img


class ImageReader:

    def read(f, serie_to_open=None):

        width = None
        height = None
        depth = None
        channels = None
        voxel_x = None
        voxel_y = None
        voxel_z = None
        times = None
        bits = None
        t_frames = None
        luts = None
        ar = None
        overlays = None
        roi = None
        timelapse = None
        creation_time = None

        dimensions_string = ''

        metadata = {'w': width, 'h': height, 'c': channels, 'd': depth, 't': t_frames, 'bits': bits, 'vx': voxel_x,
                    'vy': voxel_y, 'vz': voxel_z, 'AR': ar, 'dimensions': dimensions_string, 'LUTs': luts,
                    'times': times, 'Overlays': overlays, 'ROI': roi, 'timelapse': timelapse,
                    'creation_time': creation_time}  # TODO check always ok

        logger.debug('loading' + str(f))

        if f.lower().endswith('.tif') or f.lower().endswith('.tiff') or f.lower().endswith(
                '.lsm'):
            with tifffile.TiffFile(f) as tif:

                # TODO need handle ROIs there!!!
                # just copy stuff
                # --> can then use it and pass it directly then if needed --> maybe need a smart handling in case there is a reduction of the number of dimensions to only keep the correct ROIs

                # if image is IJ image preserve ROIs and overlays
                if tif.is_imagej:
                    if 'Overlays' in tif.imagej_metadata:
                        overlays = tif.imagej_metadata['Overlays']
                        metadata['Overlays'] = overlays
                    if 'ROI' in tif.imagej_metadata:
                        roi = tif.imagej_metadata['ROI']
                        metadata['ROI'] = roi
                    if 'LUTs' in tif.imagej_metadata:
                        luts = tif.imagej_metadata['LUTs']
                        metadata['LUTs'] = luts

                tif_tags = {}
                for tag in tif.pages[0].tags.values():

                    # if name == 'vx':
                    #     width = value
                    # elif name == 'vy':
                    #     height = value
                    # elif name == 'vz':
                    #     depth = value
                    name, value = tag.name, tag.value
                    tif_tags[name] = value
                    logger.debug(''' + name + ''' + '\'' + str(value) + '\'')
                    if name == 'ImageWidth':
                        width = value
                    elif name == 'ImageLength':
                        height = value
                    elif name == 'BitsPerSample':
                        if not isinstance(value, tuple):
                            bits = value
                        else:
                            bits = value[0]
                    elif name == 'XResolution':
                        val = value[1] / value[0]
                        if val != 1.0 or voxel_x is None:
                            voxel_x = value[1] / value[0]
                    elif name == 'YResolution':
                        val = value[1] / value[0]
                        if val != 1.0 or voxel_x is None:
                            voxel_y = val
                    elif name == 'ImageDescription':
                        if not value.startswith('{'):
                            # print(value)
                            lines = value.split()
                            # print(len(lines), lines)
                            # print(type(value), value)
                            #
                            # if value.startswith('{'):
                            #     res = json.loads(value)
                            #     print(res)
                            #     print(res['vx']+1)
                            # then it is a dict and I need parse it back
                            # if description starts with a {
                            for l in lines:
                                logger.debug('1'' + l + ''1')
                                if l.startswith('channels'):
                                    _, val = l.split('=')
                                    channels = int(val)
                                elif l.startswith('slices'):  # Z slices
                                    _, val = l.split('=')
                                    depth = int(val)
                                elif l.startswith('frames'):  # time frames
                                    _, val = l.split('=')
                                    t_frames = int(val)
                                elif l.startswith('spacing'):
                                    _, val = l.split('=')
                                    voxel_z = float(val)

                                    # print(name, value)
                                    # TODO if there is an ImageDescription I could parse it and get the data out of it
                                    # ImageDescription
                        else:
                            # TODO improve that
                            # metadata added is in fact epyseg metadata --> recover it
                            epyseg_meta = json.loads(value)
                            if 'vx' in epyseg_meta:
                                voxel_x = epyseg_meta['vx']
                            if 'vy' in epyseg_meta:
                                voxel_y = epyseg_meta['vy']
                            if 'vz' in epyseg_meta:
                                voxel_z = epyseg_meta['vz']
                            if 'creation_time' in epyseg_meta:
                                creation_time = epyseg_meta['creation_time']
                            if 'timelapse' in epyseg_meta:
                                timelapse = epyseg_meta['timelapse']
                            if 'times' in epyseg_meta:
                                times = epyseg_meta['times']
                            # TODO maybe get more metadata from epyseg --> check which meta is smart to keep and which isn't...

                    # read lsm
                    if isinstance(value, dict):
                        for name, value in value.items():
                            logger.debug(name + ' ' + str(value))
                            # THE 3 BELOW ARE MY OWN TAGS --> ADD MORE AND MAYBE STICK TO OMERO FOR CONSISTENCY MAYBE BUT OK FOR NOW
                            if name == 'DimensionZ':
                                depth = value
                            elif name == 'DimensionX':
                                width = value
                            elif name == 'DimensionY':
                                height = value
                            elif name == 'DimensionTime':
                                t_frames = value
                                if t_frames == 1:
                                    t_frames = None
                            elif name == 'DimensionChannels':
                                channels = value
                            elif name == 'VoxelSizeX':
                                voxel_x = value * 1_000_000
                            elif name == 'VoxelSizeY':
                                voxel_y = value * 1_000_000
                            elif name == 'VoxelSizeZ':
                                voxel_z = value * 1_000_000
                            elif name == 'TimeStamps':
                                times = value
                            elif name == 'ChannelColors':
                                luts = value['Colors']

        # TODO also recover my own tags maybe # TODO recode all of this properly some day because it becomes a huge mess now

        # very dumb thing here is that I open the tiff file twice --> really not Smart --> change this --> I opne it here and above
        if f.lower().endswith('.tif') or f.lower().endswith('.tiff') or f.lower().endswith('.lsm'):
            image_stack = tifffile.imread(f)
            # has more properties than that
            image = image_stack
            image = np.squeeze(image)
        elif f.lower().endswith('.czi'):
            with czifile.CziFile(f) as czi:
                meta_data = czi.metadata(
                    raw=False)  # raw=False --> there is a bug it can't read properly the dimension xyz there --> parse myself the xml --> easy # retrun metadata as dict --> recover parameters # set it to false to get xml

                logger.debug(meta_data)
                xml_metadata = czi.metadata()
                root = ET.fromstring(xml_metadata)

                # manually parse xml as dict is erroneous to get the x, y and z voxel sizes
                for l in root.findall('./*/Scaling/Items/Distance'):
                    rank = l.find('Value').text
                    name = l.get('Id')
                    if name == 'X':
                        voxel_x = float(rank) * 1_000_000
                    if name == 'Y':
                        voxel_y = float(rank) * 1_000_000
                    if name == 'Z':
                        voxel_z = float(rank) * 1_000_000

                image = czi.asarray()
                bits = meta_data['ImageDocument']['Metadata']['Information']['Image']['ComponentBitCount']
                width = meta_data['ImageDocument']['Metadata']['Information']['Image']['SizeX']
                height = meta_data['ImageDocument']['Metadata']['Information']['Image']['SizeY']
                depth = meta_data['ImageDocument']['Metadata']['Information']['Image']['SizeZ']
                image = np.squeeze(image)  # removes all the empty dimensions
        elif f.lower().endswith('.lif'):
            # reader = read_lif.Reader(f)
            # series = reader.getSeries()
            # # print('series', len(series))
            # chosen = series[0]
            #
            # meta_data = chosen.getMetadata()
            # voxel_x = meta_data['voxel_size_x']
            # voxel_y = meta_data['voxel_size_y']
            # voxel_z = meta_data['voxel_size_z']
            # width = meta_data['voxel_number_x']
            # height = meta_data['voxel_number_y']
            # depth = meta_data['voxel_number_z']
            # channels = meta_data['channel_number']
            # times = chosen.getTimeStamps()
            # t_frames = chosen.getNbFrames()
            #
            # image = None
            # for i in range(channels):
            #     cur_image = chosen.getFrame(channel=i)
            #     dimName = {1: 'X',
            #                2: 'Y',
            #                3: 'Z',
            #                4: 'T',
            #                5: 'Lambda',
            #                6: 'Rotation',
            #                7: 'XT Slices',
            #                8: 'TSlices',
            #                10: 'unknown'}
            #     cur_image = np.moveaxis(cur_image, -1, 0)
            #     if image is None:
            #         image = cur_image
            #     else:
            #         image = np.stack((image, cur_image), axis=-1)
            image = None
            reader = read_lif.Reader(f)
            series = reader.getSeries()
            # print('series', len(series))
            if serie_to_open is None:
                chosen = series[0]
            else:
                if serie_to_open >= len(series) or serie_to_open < 0:
                    logger.error('Out of range serie nb for current lif file, returning None')
                    return None
                chosen = series[serie_to_open]

            meta_data = chosen.getMetadata()
            voxel_x = meta_data['voxel_size_x']
            voxel_y = meta_data['voxel_size_y']
            voxel_z = meta_data['voxel_size_z']
            width = meta_data['voxel_number_x']
            height = meta_data['voxel_number_y']
            depth = meta_data['voxel_number_z']
            channels = meta_data['channel_number']
            times = chosen.getTimeStamps()  # shall I try getRelativeTimeStamps????
            # print('relative times', chosen.getRelativeTimeStamps()) # marche pas peut etre sur non chosen --> still useless for lif
            timelapse = chosen.getTimeLapse()
            t_frames = chosen.getNbFrames()

            # print('t_frames', t_frames)
            # TODO check time points cause I think they are not ok for the t frames

            # stack = None
            for T in range(t_frames):
                zstack = None
                for i in range(channels):
                    cur_image = chosen.getFrame(T=T, channel=i)
                    # dimName = {1: 'X',
                    #            2: 'Y',
                    #            3: 'Z',
                    #            4: 'T',
                    #            5: 'Lambda',
                    #            6: 'Rotation',
                    #            7: 'XT Slices',
                    #            8: 'TSlices',
                    #            10: 'unknown'}
                    cur_image = np.moveaxis(cur_image, -1, 0)
                    if zstack is None:
                        zstack = cur_image
                    else:
                        zstack = np.stack((zstack, cur_image), axis=-1)
                if image is None:
                    image = zstack[np.newaxis, ...]
                    # stack = image
                else:
                    # print(image.shape, zstack.shape)
                    image = np.vstack((image, zstack[np.newaxis, ...]))
                    # stack = np.vstack((stack, image), axis = np.newaxis)

            # if only one T --> reduce dimensionality
            if t_frames == 1:
                t_frames = None

            # print('before squeeze', image.shape)
            image = np.squeeze(image)
            # image = stack
        else:
            if not f.lower().endswith('.npy') and not f.lower().endswith('.npz'):
                # for some reason this stuff reads 8 bits images as RGB and that causes some trouble
                image = skimage.io.imread(f)
            else:
                # load numpy image directly
                if f.lower().endswith('.npy'):
                    image = np.load(f)
                    try:
                        with open(f + '.meta') as json_file:
                            metadata = json.load(json_file)
                    except:
                        logger.debug('could not load metadata ' + str(f + '.meta'))
                    # replace metadata from this file
                    return metadata, image
                else:
                    all_data = np.load(f)
                    image = all_data['data']
                    # Dirty way to recover first data in an image if data does not exist...
                    if image is None:
                        for dat in all_data:
                            image = dat
                            break
                    # TODO allow support for metadata some day
                    return None, image

        if voxel_x is not None and voxel_z is not None:
            ar = voxel_z / voxel_x

        logger.debug('original dimensions:' + str(image.shape))

        if image.shape[1] != height and image.ndim == 4 and t_frames is None:
            image = np.moveaxis(image, [1], -1)

        if image.ndim >= 3 and image.shape[2] != height and image.ndim == 5:
            image = np.moveaxis(image, [2], -1)

        if channels is not None and image.ndim == 3 and image.shape[0] == channels:
            image = np.moveaxis(image, [0], -1)

        if channels is not None and image.ndim == 4 and image.shape[1] == channels:
            image = np.moveaxis(image, [1], -1)

        dimensions_string += 'hw'

        # bug fix for images having d=1 incompatible with squeeze; need be done for every dimension by the way...
        if depth is not None and depth > 1:
            dimensions_string = 'd' + dimensions_string

        if channels is None and width != image.shape[-1] and len(image.shape) > 2:
            channels = image.shape[-1]

        if channels is not None and channels > 1:
            dimensions_string += 'c'

        # bug fix for images having t=1 incompatible with squeeze; need be done for every dimension by the way...
        if t_frames is not None and t_frames > 1:
            dimensions_string = 't' + dimensions_string
        else:
            if image.ndim > len(dimensions_string):
                dimensions_string = 't' + dimensions_string
                t_frames = image.shape[0]

        if width is None and image.ndim >= 3:
            width = image.shape[-2]
        if height is None and image.ndim >= 3:
            height = image.shape[-3]

        if width is None and image.ndim == 2:
            width = image.shape[-1]
        if height is None and image.ndim == 2:
            height = image.shape[-2]

        # update metadata
        metadata.update({'w': width, 'h': height, 'c': channels, 'd': depth, 't': t_frames, 'bits': bits, 'vx': voxel_x,
                         'vy': voxel_y, 'vz': voxel_z, 'AR': ar, 'dimensions': dimensions_string, 'LUTs': luts,
                         'times': times, 'Overlays': overlays, 'ROI': roi, 'timelapse': timelapse,
                         'creation_time': creation_time})
        # print(metadata)

        logger.debug('image params:' + str(metadata))
        logger.debug('final shape:' + str(image.shape))

        return metadata, image

    # def _fix_dimensions1(self):
    #     if

    def imageread(self, filePath):
        # TODO return other stuff here such as nb of frames ... do I need skimage to read or should I use smthg else
        temp = skimage.io.imread(filePath[0])
        h, w, c = temp.shape
        d = len(filePath)
        volume = np.zeros((d, w, h, c), dtype=np.uint16)  # TODO why np.uint16 especially if imag is not ? FIX
        k = 0
        for img in filePath:  # assuming tif
            im = skimage.io.imread(img)
            volume[k, :, :, :] = np.swapaxes(im[:, :, :], 0, 1)
            k += 1
        return volume


if __name__ == '__main__':
    if True:
        f = '/E/Sample_images/test_IJ_metadata_n_ROIs_tifffile/IJ_input.tif'
        f = '/E/Sample_images/test_IJ_metadata_n_ROIs_tifffile/IJ_input_2.tif'
        f = '/E/Sample_images/test_IJ_metadata_n_ROIs_tifffile/IJ_input_noROI.tif'
        f = '/E/Sample_images/test_IJ_metadata_n_ROIs_tifffile/IJ_input_noROI_channels_changed.tif'
        f = '/E/Sample_images/test_IJ_metadata_n_ROIs_tifffile/IJ_input_ROIs_n_LUTs.tif'

        # bug ROI is not there but why ???
        # marche pas mais pkoi ????
        img = Img(f)
        print(img.shape)
        Img(img, dimensions='dhwc').save('/E/Sample_images/test_IJ_metadata_n_ROIs_tifffile/test.tif')

        import sys
        sys.exit(0)

    if True:
        # now epyseg reads the image properly but IJ does not for the voxel size --> need hack it a bit in order to get the stuff done properly
        img = Img('/E/Sample_images/sample_images_pyta/surface_projection/210219.lif_t000.tif')

        print(img.max(), img.min())  # 1.7492598 -0.0020951158 --> ok
        # see the RGB version of the image and maybe get that all the time

        SQL_plot = Img(
            '/E/Sample_images/sample_images_pyta/surface_projection/210219.lif_t000/tracked_cells_resized.tif')

        # img = img*255 + 1 # that fixes the bug --> see how I can do it in a cleaner and generic way

        composite = blend(img, SQL_plot, alpha=0.3, mask_or_forbidden_colors=0x000000)

        print(img.shape)
        print(SQL_plot.shape)
        print(composite.shape)

        print(img.dtype)
        print(SQL_plot.dtype)
        print(composite.dtype)

        print(img.max(), img.min())  # 1.7492598 -0.0020951158
        print(SQL_plot.max(), SQL_plot.min())  # 255 0
        print(composite.max(), composite.min())  # 77 0

        # in fact the original image should be between 0 and 255 most likely

        plt.imshow(composite)
        plt.show()
        # plt.imshow(img)
        # plt.show()

        import sys

        sys.exit(0)

    data = np.zeros((1024, 1024), dtype=np.uint8)

    if True:
        # now epyseg reads the image properly but IJ does not for the voxel size --> need hack it a bit in order to get the stuff done properly
        import sys

        sys.exit(0)
