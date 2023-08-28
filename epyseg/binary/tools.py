# this will contain a set of binary tools
from skimage.measure import label, regionprops
from skimage.draw import ellipse
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters

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

def test_all_auto_thresholds(img):
    # Apply all thresholding methods and display results
    fig, ax = filters.try_all_threshold(img, figsize=(8, 5), verbose=False)

    # Show plot
    plt.show()


def yen_thresholding(img):
    # Apply Yen thresholding method
    thresh = filters.threshold_yen(img)
    # Apply threshold to image
    binary = img > thresh
    return binary

if __name__ == '__main__':
    pass