from skimage.filters import threshold_sauvola
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
from epyseg.img import Img
import numpy as np
from epyseg.postprocess.superpixel_methods import get_optimized_mask2


def sauvola(img, window_size=25, min_threshold=0.02):
    """
    Applies the Sauvola thresholding method to an image.

    Args:
        img (numpy.ndarray): The input image.
        window_size (int): The size of the window for local thresholding. Defaults to 25.
        min_threshold (float): The minimum threshold value. Defaults to 0.02.

    Returns:
        numpy.ndarray: The thresholded image.

    """
    k = 0.25
    r = 0.5

    t = threshold_sauvola(img, window_size=window_size, k=k, r=r)
    if min_threshold is not None:
        t[t <= min_threshold] = min_threshold
    return t


def segment_cells(image, __DEBUG=False, __VISUAL_DEBUG=False, stop_at_threshold_step=False,
                  min_unconnected_object_size=None, min_threshold=None, window_size=25, real_avg_mode=False):
    """
    Segments cells in an image using the Sauvola thresholding method.

    Args:
        image (numpy.ndarray): The input image.
        __DEBUG (bool): Debug flag. Defaults to False.
        __VISUAL_DEBUG (bool): Visual debug flag. Defaults to False.
        stop_at_threshold_step (bool): Flag to stop at thresholding step and return the thresholded image. Defaults to False.
        min_unconnected_object_size (int): The minimum size of unconnected objects to be removed. Defaults to None.
        min_threshold (float): The minimum threshold value. Defaults to None.
        window_size (int): The size of the window for local thresholding. Defaults to 25.
        real_avg_mode (bool): Flag indicating whether to use real average mode. Defaults to False.

    Returns:
        numpy.ndarray: The segmented image.

    """
    original = image.copy()

    t = sauvola(image, min_threshold=min_threshold, window_size=window_size)

    if __VISUAL_DEBUG:
        plt.imshow(t)
        plt.show()

    image[image >= t] = 1
    image[image < t] = 0

    if __VISUAL_DEBUG and min_unconnected_object_size == 12:
        plt.imshow(image)
        plt.title('before')
        plt.show()

    if min_unconnected_object_size is not None and min_unconnected_object_size >= 1:
        image = remove_small_objects(image.astype(bool), min_size=min_unconnected_object_size, connectivity=2).astype(np.uint8)
        if __VISUAL_DEBUG and min_unconnected_object_size == 12:
            plt.imshow(image)
            plt.title('after')
            plt.show()

    if stop_at_threshold_step:
        image = image * 255
        return image

    return get_optimized_mask2(original, sauvola_mask=None, score_before_adding=True)


if __name__ == '__main__':
    from timeit import default_timer as timer

    # image = Img('/D/final_folder_scoring/predict_hybrid/mini_test.tif')
    # image = Img('/D/final_folder_scoring/predict_hybrid/AVG_StackFocused_Endocad-GFP(6-12-13)#19_000.tif')
    # image = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/122.tif')[...,0]
    # image = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/image_plant_best-zoomed.tif')[...,0]
    image = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/5.tif')[..., 0]
    # image = Img('/D/final_folder_scoring/predict_hybrid/tmp11.png')
    # image = Img('/D/final_folder_scoring/predict_hybrid/11-1_nuclei_1.tif')
    start = timer()

    # final_mask = segment_cells(image, __DEBUG=False, __VISUAL_DEBUG=False, stop_at_threshold_step=False)
    final_mask = segment_cells(image, __DEBUG=False, __VISUAL_DEBUG=False, stop_at_threshold_step=False, min_unconnected_object_size=12)

    duration = timer() - start
    print(duration)

    plt.imshow(final_mask)
    plt.show()
    Img(final_mask, dimensions='hw').save('/home/aigouy/Bureau/trash/test_new_seeds_seg_stuff/final_rewatershed.tif')
