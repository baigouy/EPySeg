# seems ok and better working again
import random
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from epyseg.img import Img
from skimage import measure
from skimage.util import invert, img_as_ubyte
from skimage import filters
from skimage.segmentation import watershed
from skimage.util import invert
from timeit import default_timer as timer
from epyseg.tools.logger import TA_logger  # logging
from numba import jit, njit
from scipy.ndimage import generic_filter

logger = TA_logger()

__DEBUG__ = False

def wshed(img, channel=None, weak_blur=None, strong_blur=None, seeds='mask', min_seed_area=None, is_white_bg=False, fix_scipy_wshed=False, force_dual_pass=False):
    """
    Perform watershed segmentation on an image.

    Args:
        img (numpy.ndarray): The input image.
        channel (int): The channel index to be used from the image (optional).
        weak_blur (float): The sigma value for weak blurring (optional).
        strong_blur (float): The sigma value for strong blurring (optional).
        seeds (str, list, or numpy.ndarray): The seeds for watershed segmentation. Can be 'mask', a list of 2D coordinates, or a labeled image (optional).
        min_seed_area (int): The minimum seed area for dual-pass watershed (optional).
        is_white_bg (bool): Whether the image has a white background (optional).
        fix_scipy_wshed (bool): Whether to fix missing watershed lines in scipy (optional).
        force_dual_pass (bool): Force dual-pass watershed even if min_seed_area is not specified (optional).

    Returns:
        numpy.ndarray: The segmented image.

    """
    if __DEBUG__:
        start = timer()

    if min_seed_area is None and force_dual_pass:
        min_seed_area = 0

    if weak_blur is None and strong_blur is not None:
        weak_blur = strong_blur
        strong_blur = None

    if strong_blur is not None or weak_blur is not None:
        if isinstance(seeds, str):
            seeds = None

    if channel:
        if len(img.shape) > 2:
            img = img[..., channel]

    if is_white_bg:
        img = invert(img)

    strong = None
    if weak_blur:
        weak = filters.gaussian(img, sigma=weak_blur, preserve_range=True, mode='wrap')
    else:
        weak = img
    if strong_blur:
        strong = filters.gaussian(img, sigma=strong_blur, preserve_range=True, mode='wrap')

    if __DEBUG__:
        print('blur ' + str(timer() - start) + ' s')

    if isinstance(seeds, str) and seeds == 'mask':
        markers = measure.label(weak, connectivity=1, background=255)
    elif isinstance(seeds, list):
        marker = np.zeros_like(weak, dtype=np.int32)
        seeds = np.asarray(seeds)
        marker[seeds[:, 0], seeds[:, 1]] = 1
        markers = measure.label(marker, connectivity=1, background=0)
    elif isinstance(seeds, np.ndarray):
        if seeds.shape == weak.shape:
            markers = seeds
        else:
            marker = np.zeros_like(weak, dtype=np.int32)
            marker[seeds[:, 0], seeds[:, 1]] = 1
            markers = measure.label(marker, connectivity=1, background=0)
    else:
        if is_white_bg:
            if strong_blur:
                local_maxi = ndi.maximum_filter(strong, size=3) == strong
                del strong
            else:
                local_maxi = ndi.maximum_filter(weak, size=3) == weak
        else:
            if strong_blur:
                local_maxi = ndi.minimum_filter(strong, size=3) == strong
                del strong
            else:
                local_maxi = ndi.minimum_filter(weak, size=3) == weak
        markers = measure.label(local_maxi)
        if __DEBUG__:
            print('local max ' + str(timer() - start) + ' s')

    labels_ws = watershed(weak, markers=markers, watershed_line=True)
    del weak
    del markers

    if fix_scipy_wshed:
        labels_ws = full_numba_wshed_fixer(labels_ws)

    tmp = np.zeros_like(labels_ws)
    tmp[labels_ws == 0] = 255
    labels_ws = tmp
    del tmp

    if __DEBUG__:
        print('wshed 1 ' + str(timer() - start) + ' s')

    if (min_seed_area is not None and min_seed_area > 0) or force_dual_pass:
        wshed_mask = np.copy(labels_ws)
        del labels_ws
        label = measure.label(wshed_mask, connectivity=1, background=255)
        most_frequent_segmentation_mask, count = np.unique(label, return_counts=True)
        cells_to_remove = most_frequent_segmentation_mask[count <= min_seed_area]

        if cells_to_remove.size == 0 and not force_dual_pass:
            final_mask = wshed_mask
            del label
        else:
            for cell in cells_to_remove:
                wshed_mask[label == cell] = 255
            del label

            wshed_mask = watershed(wshed_mask, markers=None, watershed_line=True)
            final_mask = np.zeros_like(wshed_mask)
            final_mask[wshed_mask == 0] = 255
            del wshed_mask
    else:
        final_mask = labels_ws

    if final_mask.max() == final_mask.min():
        final_mask = np.zeros_like(final_mask, dtype=np.uint8)
    else:
        final_mask = final_mask.astype(np.uint8)

    if __DEBUG__:
        print('final wshed ' + str(timer() - start) + ' s')

    return final_mask

@njit
def full_numba_wshed_fixer(img):
    height = img.shape[0]
    width = img.shape[1]
    for jjj in range(height):
        last_pixel = -1
        for iii in range(width):
            pixel = img[jjj, iii]
            if pixel == 0:
                last_pixel = -1
                continue
            if last_pixel == -1:
                last_pixel = pixel
            else:
                if pixel == 0:
                    last_pixel = -1
                elif last_pixel != pixel and pixel != 0:
                    img[jjj, iii] = 0
                    last_pixel = pixel

    for iii in range(width):
        last_pixel = -1
        for jjj in range(height):
            pixel = img[jjj, iii]
            if pixel == 0:
                last_pixel = -1
                continue
            if last_pixel == -1:
                last_pixel = pixel
            else:
                if pixel == 0:
                    last_pixel = -1
                elif last_pixel != pixel and pixel != 0:
                    img[jjj, iii] = 0
                    last_pixel = pixel

    return img

@njit
def detect_wshed_line(P):
    if P[4] == 0:
        return 0
    corrected_ids = set(P)
    size = len(corrected_ids)
    if size == 3:
        if np.count_nonzero(P) == 7:
            if P[1] == 0 and P[7] == 0:
                return 255
            if P[3] == 0 and P[5] == 0:
                return 255
            if P[0] == 0 and P[8] == 0:
                return 255
            if P[2] == 0 and P[6] == 0:
                return 255
    return 0

if __name__ == '__main__':

    # all works now try to run the local wshed on the image
    # --> need a small test stuff
    # test apply ctrl + M to the image --> see how I can do that properly

    if True:
        # 7.7357873320002 seconds --> passe à 19 secs avec le fix du wshed --> probably not worth it but put it as an option
        # testing seeds are already a label in the proper shape of the image
        img = Img('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif')
        # test seeded wshed
        start = timer()
        seeds = np.zeros_like(img)

        # out = watershed(img, seeds=[[10,20],[60,20]])
        # out = watershed(img, seeds=[[0,0],[800,1900]])
        # out = watershed(img, seeds=[[780,1800],[800,1900]])
        for i in range(1, 3000):
            seeds[random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1)] = i

        # print(tuple(seeds))
        # print(seeds[:, 0])

        # out = wshed(img, weak_blur=1,seeds=seeds)
        # out = wshed(img, seeds=seeds) # --> 7.7357873320002 seconds
        # out = wshed(img, seeds=seeds, fix_scipy_wshed=True) # --> 19 secs
        out = wshed(img, seeds=seeds, fix_scipy_wshed=True) # --> new algo full numba --> 15secs --> not that bad... just double the time

        # not so bad --> all is ok now
        # check what seeds I can make # maybe do local stuff too --> would need a bounding box then !!!

        print('total time', timer() - start)
        plt.imshow(out)
        plt.show()

    if False:
        # testing new kinds of seeds --> just for a test
        img = Img('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif')
        # test seeded wshed
        start = timer()

        # out = watershed(img, seeds=[[10,20],[60,20]])
        # out = watershed(img, seeds=[[0,0],[800,1900]])
        # out = watershed(img, seeds=[[780,1800],[800,1900]])
        seeds = []
        for i in range(2000):
            seeds.append([random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1)])

        if random.random() > 0.5:
            # print(seeds)
            seeds = np.asarray(seeds)  # 2D seeds

        # print(tuple(seeds))

        # print(seeds[:, 0])

        out = wshed(img, seeds=seeds)

        # check what seeds I can make # maybe do local stuff too --> would need a bounding box then !!!

        print('total time', timer() - start)
        plt.imshow(out)
        plt.show()

    if False:
        # ça marche ça c'est vraiment la correction de TA et c'est pas trop slow --> ok
        # le removing of seeds est slow --> if I could make it faster would be cool but ok
        img = Img('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif')
        # test seeded wshed
        start = timer()
        out = wshed(img, min_seed_area=0, seeds='mask')  # self instead of seeds

        # check what seeds I can make # maybe do local stuff too --> would need a bounding box then !!!

        print('total time', timer() - start)
        plt.imshow(out)
        plt.show()

    # do i need seeds
    if False:
        # ça marche ça c'est vraiment la correction de TA et c'est pas trop slow --> ok
        # le removing of seeds est slow --> if I could make it faster would be cool but ok
        img = Img('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif')
        # test seeded wshed
        start = timer()
        out = wshed(img, min_seed_area=3, seeds='mask')  # self instead of seeds
        print('total time', timer() - start)
        plt.imshow(out)
        plt.show()
    # test default wshed
    if False:
        # img = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012.png')[...,1]
        # img = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012.png')
        img = Img('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/Optimized_projection_018.png')[
            ..., 1]  # 27.9 secs --> maybe not so bad secs per image --> slow
        start = timer()
        # nb first and second blur are inverted --> rather use weak and strong blur --> TODO
        # fix also the minimal size stuff , can also use np.unique to speed up things
        out = wshed(img, weak_blur=1, strong_blur=5, min_seed_area=3, is_white_bg=False)
        Img(out).save('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif')

        # if only one blur then must set weak blur
        # out = watershed(img, weak_blur=5,  min_seed_area=3, is_white_bg=False) # --> 3 secs # --> sans filtering --> gain de 7 secs

        # TODO try to play with seeds to support all sorts of interesting stuff such as manual seeds

        print('total time', timer() - start)
        plt.imshow(out)
        plt.show()
