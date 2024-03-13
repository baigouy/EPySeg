# this will contain a set of binary tools
from skimage.measure import label, regionprops
from skimage.draw import ellipse
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters
from skimage.filters.thresholding import threshold_otsu



def ellipsoidify_dots(binary_dots):
    """
    Replaces all 2D dots detected by an equivalent ellipse using bbox radius to draw the dot.

    Args:
        binary_dots (ndarray): An image that contains dots (typically obtained through binarization of an image).

    Returns:
        ndarray: The modified image with dots replaced by ellipses.

    Examples:
        >>> binary_dots = np.array([[0, 0, 0, 0],
        ...                         [0, 1, 0, 0],
        ...                         [0, 0, 0, 0]])
        >>> result = ellipsoidify_dots(binary_dots)
        >>> print(result)
        [[0 0 0 0]
         [0 1 0 0]
         [0 0 0 0]]
    """
    # Check if the image is 3D
    if len(binary_dots.shape) == 3:
        # If image is 3D, loop over all images and apply ellipsoidify_dots to each channel
        for iii, img_2D in enumerate(binary_dots):
            binary_dots[iii] = ellipsoidify_dots(img_2D)
        return binary_dots

    # Label the dots in the binary image
    lab_dots = label(binary_dots, connectivity=None, background=0)
    rps_dots = regionprops(lab_dots)

    # Iterate over each labeled region (dot)
    for region in rps_dots:
        centroid = region.centroid
        bbox = region.bbox
        # Draw an ellipse that fits the dot
        rr, cc = ellipse(centroid[0], centroid[1], r_radius=abs(bbox[0] - bbox[2]) / 2,
                         c_radius=abs(bbox[1] - bbox[3]) / 2, shape=binary_dots.shape)
        binary_dots[rr, cc] = binary_dots.max()

    return binary_dots
# should also work for 3D even though the 3D version is not super elegant

def tst_all_auto_thresholds(img):
    # Apply all thresholding methods and display results
    fig, ax = filters.try_all_threshold(img, figsize=(8, 5), verbose=False)

    # Show plot
    plt.show()


def yen_thresholding(img,return_threshold=False):
    # Apply Yen thresholding method
    thresh = filters.threshold_yen(img)
    if return_threshold:
        return thresh
    # Apply threshold to image
    return apply_threshold(img, thresh)

def triangle_threshold(img, return_threshold=False):
    # Apply triangle thresholding method
    thresh = filters.threshold_triangle(img)
    if return_threshold:
        return thresh
    # Apply threshold to image
    return apply_threshold(img, thresh)

def isodata_threshold(img, return_threshold=False):
    # Apply triangle thresholding method
    thresh = filters.threshold_isodata(img)
    # thresh = threshold_otsu(img)
    if return_threshold:
        return thresh
    # Apply threshold to image
    return apply_threshold(img, thresh)

def otsu_threshold(img, return_threshold=False):
    # Apply triangle thresholding method
    # thresh = filters.otsu_threshold(img)
    thresh = threshold_otsu(img)
    if return_threshold:
        return thresh
    # Apply threshold to image
    return apply_threshold(img, thresh)

def mean_threshold(img, return_threshold=False):
    # Apply triangle thresholding method
    thresh = filters.threshold_mean(img)
    if return_threshold:
        return thresh
    # Apply threshold to image
    return apply_threshold(img, thresh)


def apply_threshold(img, threshold):
    # Apply threshold to image
    binary = img > threshold
    return binary

def smart_threshold_above(img,max_fraction, increment=0.01):
    norm = np.copy(img)
    factor = norm.max()
    norm= norm/norm.max()
    # for vvv in range(0,1,increment):
    threshold = 1.
    while threshold-increment>0.:
        if np.count_nonzero(norm>=threshold)/img.size >= max_fraction:
            return threshold*factor
        threshold-=increment


if __name__ == '__main__':
    from epyseg.img import Img
    img =Img('/E/Sample_images/cells_escaping_silencing_test_count/Apotome/MAX_IP-ApoTome-01.tif')
    # img =Img('/E/Sample_images/cells_escaping_silencing_test_count/R1-14,5kb T2A DsRed X2  TgA/R1-14,5kb T2A DsRed X2  TgA 0004/wing_deep.tif')
    # img =Img('/E/Sample_images/cells_escaping_silencing_test_count/HT 14,5kbT2ADsRed/A5 No Wari 14,5T2A DsRed X1 Female 0001/wing_deep.tif')
    print(smart_threshold_above(img, 0.36))