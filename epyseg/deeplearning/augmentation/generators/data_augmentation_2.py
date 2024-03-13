# TODO --> do random crops that through reflect can create images of any given size --> smarter for training
# TODO --> try the pipes now

# MEGA TODO CHECK THAT THIS IS REALLY SMART to set EXTRAPOLATION_MASKS to 1 instead of 0 !!!

# this contains all the data augmentations
# and should make sure the same nb of random are called so that everything keeps being in sync even when piped --> TODO
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt
from skimage import filters
from skimage.util import random_noise

import epyseg.img
from epyseg.img import Img, mask_rows_or_columns, elastic_deform, \
    create_random_intensity_graded_perturbation, apply_2D_gradient, array_equal, array_different, pad_border_xy, \
    interpolation_free_rotation
import random
from scipy import ndimage
from skimage import transform
import numpy as np
from skimage import exposure
from epyseg.tools.logger import TA_logger # logging
# TODO add augmentation randomly swap channels --> can learn by itself which channel is containing epithelia
from epyseg.utils.commontools import execute_chained_functions


logger = TA_logger()

# TODO --> find trick to get it to work as I want !!! --> below need be tunable parameters
EXTRAPOLATION_MASKS = 1  # shall I set it to 1 or TO 0 --> think about it important in fact --> THIS MAY NEED BE CHANGED --> can maybe also get it from the kwargs --> TODO

# see how to set the range since it's global !!! is it reset upon a new call ????
# if so need add kwargs to all --> to get the default range
def blur(orig, is_mask, **kwargs):
    """
    Apply Gaussian blur to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The blurred image or the original mask.

    # Examples:
    #     >>> image = np.array([[1, 2, 3],
    #     ...                   [4, 5, 6],
    #     ...                   [7, 8, 9]])
    #     >>> result = blur(image, False, blur=1.5)
    #     >>> print(result)
    #     [[1.00132017 2.00099013 3.00066008]
    #      [4.00033004 5.         5.99966996]
    #      [6.99933992 7.99900987 8.99867983]]
    """
    # Generate a random Gaussian blur value within the specified range
    gaussian_blur = random.uniform(0, kwargs['blur'])

    if not is_mask:
        # Apply Gaussian blur to the input image
        if len(orig.shape) == 4:
            # Apply 2D Gaussian blur to each slice of the 3D image
            for n, slc in enumerate(orig):
                orig[n] = filters.gaussian(slc, gaussian_blur, preserve_range=True, mode='wrap')
            out = orig
        else:
            # Apply 2D Gaussian blur to the input image
            out = filters.gaussian(orig, gaussian_blur, preserve_range=True, mode='wrap')
    else:
        # Return the original mask without blurring
        out = orig

    return out


def graded_intensity_modification(orig, is_mask, **kwargs):
    """
    Modify the intensity of the input image or mask using a graded perturbation.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The modified image or the original mask.

    # Examples:
    #     >>> image = np.array([[1, 2, 3],
    #     ...                   [4, 5, 6],
    #     ...                   [7, 8, 9]])
    #     >>> result = graded_intensity_modification(image, False)
    #     >>> print(result)
    #     [[0 1 2]
    #      [3 4 5]
    #      [6 7 8]]
    """
    # Apply a graded perturbation to the input image or mask
    out = apply_2D_gradient(orig, create_random_intensity_graded_perturbation(orig, off_centered=True, mock=is_mask))

    return out



# TODO --> make use of that
# nb could do random crops too
# nb does crop work for images with channels ??? not so sure --> need check
def crop_augmentation(orig, is_mask, coords_dict, **kwargs):
    """
    Perform crop augmentation on the input image or mask based on the provided coordinates.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        coords_dict (dict): A dictionary containing the crop coordinates.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The cropped image or mask.

    # Examples:
    #     >>> image = np.array([[1, 2, 3],
    #     ...                   [4, 5, 6],
    #     ...                   [7, 8, 9]])
    #     >>> coords = {'x1': 0, 'x2': 2, 'y1': 0, 'y2': 2}
    #     >>> result = crop_augmentation(image, False, coords)
    #     >>> print(result)
    #     [[1 2]
    #      [4 5]]
    """
    # Extract the crop coordinates
    startx = coords_dict['x1']
    starty = coords_dict['y1']

    # Random crop if startx and starty are None
    if startx is None:
        try:
            width = coords_dict['w']
        except:
            width = coords_dict['width']
        min = orig.shape[-2] - width
        if min < 0:
            min = 0
        startx = random.randrange(start=0, stop=min, step=1)
    if starty is None:
        try:
            height = coords_dict['h']
        except:
            height = coords_dict['height']
        min = orig.shape[-3] - height
        if min < 0:
            min = 0
        starty = random.randrange(start=0, stop=min, step=1)

    # Determine the end coordinates
    if 'w' in coords_dict or 'width' in coords_dict:
        if 'w' in coords_dict:
            endx = startx + coords_dict['w']
        else:
            endx = startx + coords_dict['width']
    else:
        endx = coords_dict['x2']
    if 'h' in coords_dict or 'height' in coords_dict:
        if 'h' in coords_dict:
            endy = starty + coords_dict['h']
        else:
            endy = starty + coords_dict['height']
    else:
        endy = coords_dict['y2']

    # Adjust for negative coordinates
    if endx < 0:
        endx = orig.shape[-2] + endx
    if endy < 0:
        endy = orig.shape[-3] + endy

    # Swap coordinates if necessary
    if starty > endy:
        tmp = starty
        starty = endy
        endy = tmp
    if startx > endx:
        tmp = startx
        startx = endx
        endx = tmp

    # Crop the image or mask based on the coordinates
    if len(orig.shape) == 3:
        return orig[starty:endy, startx:endx]
    else:
        return orig[::, starty:endy, startx:endx]


def invert(orig, is_mask, **kwargs):
    """
    Invert the colors of the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The inverted image or mask.

    Examples:
        >>> image = np.array([[0, 1, 0],
        ...                   [1, 0, 1],
        ...                   [0, 1, 0]])
        >>> result = invert(image, False)
        >>> print(result)
        [[1 0 1]
         [0 1 0]
         [1 0 1]]
    """
    if not is_mask:
        inverted_image =  epyseg.img.invert(orig)
    else:
        inverted_image = orig
    return inverted_image


def rotate(orig, is_mask, **kwargs):
    """
    Rotate the input image or mask by a random angle.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The rotated image or mask.

    # Examples:
    #     >>> import math
    #     >>> image = np.array([[1, 2, 3],
    #     ...                   [4, 5, 6],
    #     ...                   [7, 8, 9]])
    #     >>> result = rotate(image, False,**{'rotate':0.2})
    #     >>> print(result)
    #     [[7 4 1]
    #      [8 5 2]
    #      [9 6 3]]
    """
    # Rotate by random angle between 0 and 360
    angle = random.randint(0, kwargs['rotate'] * 360)
    angle *= 1 if random.random() < 0.5 else -1
    order = random.choice([0, 1, 3])

    final_order = order
    if is_mask:
        final_order = EXTRAPOLATION_MASKS

    # Bug fix for 3D images
    # Rotating the image using ndimage.rotate with the specified angle, axes, reshape, and order
    rot_orig = ndimage.rotate(orig, angle, axes=(-3, -2), reshape=False, order=final_order)

    return rot_orig


def strong_elastic_with_zoom_and_translation(orig, is_mask, **kwargs):
    """
    Apply strong elastic deformation with zoom and translation to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The transformed image or mask.

    # Examples:
    #     >>> image = np.array([[1, 2, 3],
    #     ...                   [4, 5, 6],
    #     ...                   [7, 8, 9]])
    #     >>> result = strong_elastic_with_zoom_and_translation(image, False)
    #     >>> print(result)
    #     [[5 5 5]
    #      [5 5 5]
    #      [5 5 5]]
    """
    # Apply elastic deformation with zoom and translation
    return elastic_with_zoom_and_translation(orig, is_mask, strength=100, **kwargs)



def elastic_with_zoom_and_translation(orig, is_mask, strength=25, **kwargs):
    """
    Apply elastic deformation with zoom and translation to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        strength (int): The strength of the elastic deformation.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The transformed image or mask.


    """
    # Set numpy random seed based on the provided random seed
    np.random.seed(random.randint(0, 10000000))

    # Generate a random displacement matrix
    displacement = np.random.randn(2, 3, 3) * strength

    # Select the order for interpolation
    order = random.choice([0, 1])

    # Generate random rotation and zoom values
    rotation = random.randint(0, kwargs['rotate'] * 360)
    rotation *= 1 if random.random() < 0.5 else -1
    zoom = random.uniform(1. - kwargs['zoom'], 1. + kwargs['zoom'])

    # Adjust order for masks
    if is_mask:
        order = EXTRAPOLATION_MASKS

    # Apply elastic deformation with rotation and zoom
    if len(orig.shape) == 4:
        # Assume 'dhwc' format
        rot_orig = elastic_deform(orig, displacement, axis=(1, 2), order=order, rotation=rotation, zoom=zoom)
    else:
        # Assume 'hwc' format
        rot_orig = elastic_deform(orig, displacement, axis=(0, 1), order=order, rotation=rotation, zoom=zoom)

    return rot_orig


def strong_elastic(orig, is_mask, **kwargs):
    """
    Apply strong elastic deformation to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The transformed image or mask.

    """
    # Call elastic function with a strength of 100
    return elastic(orig, is_mask, strength=100, **kwargs)

def elastic(orig, is_mask, strength=25, **kwargs):
    """
    Apply elastic deformation to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        strength (float): The strength of the elastic deformation.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The transformed image or mask.

    """
    # Set the random seed for numpy based on the random seed to ensure reproducibility
    np.random.seed(random.randint(0, 10000000))

    # Generate a random deformation matrix based on the strength parameter
    displacement = np.random.randn(2, 3, 3) * strength

    # Select the order for interpolation
    order = random.choice([0, 1])

    # Adjust the order for masks
    if is_mask:
        order = EXTRAPOLATION_MASKS

    # Apply elastic deformation based on the dimensions of the input
    if len(orig.shape) == 4:
        # Assume 'dhwc'
        transformed_orig = elastic_deform(orig, displacement, axis=(1, 2), order=order)
    else:
        # Assume 'hwc'
        transformed_orig = elastic_deform(orig, displacement, axis=(0, 1), order=order)

    return transformed_orig


# nb will that create a bug when images don't have same width and height --> most likely yes but test it --> TODO
# in fact I should not allow that except if image has same width and height or I should use a crop of it --> can I do that ???
def rotate_interpolation_free(orig, is_mask, **kwargs):
    """
    Apply rotation to the input image or mask without using interpolation.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The rotated image or mask.

    # Examples:
    #     >>> image = np.array([[1, 2, 3],
    #     ...                   [4, 5, 6],
    #     ...                   [7, 8, 9]])
    #     >>> result = rotate_interpolation_free(image, False)
    #     >>> print(result)
    #     [[3 6 9]
    #      [2 5 8]
    #      [1 4 7]]
    """
    # Determine the rotation angle based on the dimensions of the input
    if orig.shape[-2] != orig.shape[-3]:
        angle = 180
    else:
        angle = random.choice([90, 180, 270])

    # Apply interpolation-free rotation using the provided angle
    rot_orig = interpolation_free_rotation(orig, angle=angle)

    return rot_orig


def translate(orig, is_mask, **kwargs):
    """
    Apply translation to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The translated image or mask.

    """
    # Generate random translation values
    trans_x = np.random.randint(-int(orig.shape[-2] * kwargs['translate']),
                                int(orig.shape[-2] * kwargs['translate']))
    trans_y = np.random.randint(-int(orig.shape[-3] * kwargs['translate']),
                                int(orig.shape[-3] * kwargs['translate']))

    # Choose the interpolation order randomly
    order = random.choice([0, 1, 3])

    # Create an affine transformation matrix for translation
    afine_tf = transform.AffineTransform(translation=(trans_y, trans_x))

    # Determine the final interpolation order based on the input type
    final_order = order
    if is_mask:
        final_order = EXTRAPOLATION_MASKS

    # Apply translation to the input image or mask
    if len(orig.shape) <= 3:
        zoomed_orig = transform.warp(orig, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                     mode='reflect')
    else:
        zoomed_orig = np.zeros_like(orig, dtype=orig.dtype)
        for i, slice in enumerate(orig):
            zoomed_orig[i] = transform.warp(slice, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                            mode='reflect')

    return zoomed_orig


def stretch(orig, is_mask,**kwargs):
    raise Exception('TODO --> needs a lot of reworking !!!')
    scale = random.uniform(self.stretch_range[0],
                           self.augmentation_types_and_values['stretch'])
    order = random.choice([0, 1, 3])
    orientation = random.choice([0, 1])
    if orientation == 0:
        afine_tf = transform.AffineTransform(scale=(scale, 1))
    else:
        afine_tf = transform.AffineTransform(scale=(1, scale))

    final_order = order
    if is_mask:
        final_order = EXTRAPOLATION_MASKS
        # TODO check that I'm not doing unnecessary stuff here but seems to work... maybe recheck dtype too
        if self.is_output_1px_wide:
            dilation = 0
            if scale >= 2 and scale <= 3:
                # do 1 dilat
                dilation = 1
            elif scale >= 3:
                # do 2 dilat
                dilation = 2
            if dilation != 0:
                s = ndimage.generate_binary_structure(2, 1)
                # Apply dilation to every channel then reinject
                for c in range(orig.shape[-1]):
                    dilated = orig[..., c]
                    if len(orig.shape) == 4:
                        for n, slc in enumerate(dilated):
                            for dil in range(dilation):
                                dilated[n] = ndimage.grey_dilation(slc, footprint=s)
                        orig[..., c] = dilated
                    else:
                        for dil in range(dilation):
                            dilated = ndimage.grey_dilation(dilated, footprint=s)
                        orig[..., c] = dilated

    if len(orig.shape) == 4:
        # handle 3D images
        for n, slc in enumerate(orig):
            orig[n] = transform.warp(slc, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                     mode='reflect')
        stretched_orig = orig
    else:
        stretched_orig = transform.warp(orig, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                        mode='reflect')  # nb should I do this per channel to avoid issues --> MAYBE

    # print('test3', stretched_orig.max(), stretched_orig.min())
    return stretched_orig

# CODER KEEP TIP: SCALE IS INVERTED (COMPARED TO MY OWN LOGIC SCALE 2 MEANS DIVIDE SIZE/DEZOOM BY 2 AND 0.5 MEANS INCREASE SIZE/ZOOM BY 2 --> # shall I use 1/scale
def zoom(orig, is_mask, **kwargs):
    """
    Apply zooming to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The zoomed image or mask.

    """
    # Generate a random scale factor for zooming
    scale = random.uniform(1. - kwargs['zoom'], 1. + kwargs['zoom'])

    # Choose the interpolation order randomly
    order = random.choice([0, 1, 3])

    # Create an affine transformation matrix for zooming
    afine_tf = transform.AffineTransform(scale=(scale, scale))

    # Determine the final interpolation order based on the input type
    final_order = order
    if is_mask:
        final_order = EXTRAPOLATION_MASKS

    # Apply zooming to the input image or mask
    if len(orig.shape) == 4:
        # Handle 3D images
        for n, slc in enumerate(orig):
            orig[n] = transform.warp(slc, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                     mode='reflect')
        zoomed_orig = orig
    else:
        zoomed_orig = transform.warp(orig, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                     mode='reflect')

    return zoomed_orig


def shear(orig, is_mask, **kwargs):
    """
    Apply shearing to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The sheared image or mask.

    """
    # Generate a random shear factor within the specified range
    shear = random.uniform(-kwargs['shear'], kwargs['shear'])

    # Choose the interpolation order randomly
    order = random.choice([0, 1, 3])

    # Create an affine transformation matrix for shearing
    afine_tf = transform.AffineTransform(shear=shear)

    # Determine the final interpolation order based on the input type
    final_order = order
    if is_mask:
        final_order = EXTRAPOLATION_MASKS

    # Apply shearing to the input image or mask
    if len(orig.shape) == 4:
        # Handle 3D images
        for n, slc in enumerate(orig):
            orig[n] = transform.warp(slc, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                     mode='reflect')
        sheared_orig = orig
    else:
        sheared_orig = transform.warp(orig, inverse_map=afine_tf, order=final_order, preserve_range=True,
                                      mode='reflect')

    return sheared_orig


# allow for shift mask but only for the mask
# this cannot be an augmentation --> it is more complex than that
def mask_pixels_and_compute_for_those(orig, is_mask, shift_mask=False, **kwargs):
    """
    Mask pixels in the input image or mask and compute for those pixels.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        shift_mask (bool): A flag indicating whether to shift the mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The masked and computed image or mask.

    """
    if not is_mask:
        # Mask rows or columns in the input image
        rolled_orig = mask_rows_or_columns(orig, spacing_X=2, spacing_Y=5)
    else:
        if shift_mask:
            # Determine initial shifts for shifting the mask
            initial_shift_X = 0
            initial_shift_Y = 0
            if random.uniform(0, 1) < 0.5:
                if random.uniform(0, 1) < 0.5:
                    if random.uniform(0, 1) < 0.5:
                        initial_shift_X = 1
                    else:
                        initial_shift_Y = 1
                else:
                    initial_shift_X = 1
                    initial_shift_Y = 1

            # Mask rows or columns in the input mask with initial shifts
            rolled_orig = mask_rows_or_columns(orig, spacing_X=2, spacing_Y=5, initial_shiftX=initial_shift_X,
                                               initial_shiftY=initial_shift_Y, return_boolean_mask=True)
            orig[rolled_orig == False] = np.nan
            rolled_orig = orig
        else:
            # Mask rows or columns in the input mask without shifting
            rolled_orig = mask_rows_or_columns(orig, spacing_X=2, spacing_Y=5, return_boolean_mask=True)
            orig[rolled_orig == False] = np.nan
            rolled_orig = orig

    return rolled_orig


# rolls along the Z axis of a 3D image ignores for 2D images and ignores for masks
def rollZ(orig, is_mask, **kwargs):
    """
    Roll the input image or mask along the z-axis.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The rolled image or mask.

    # Examples:
    #     >>> image = np.array([[[1, 2, 3],
    #     ...                    [4, 5, 6],
    #     ...                    [7, 8, 9]],
    #     ...                   [[10, 11, 12],
    #     ...                    [13, 14, 15],
    #     ...                    [16, 17, 18]]])
    #     >>> result = rollZ(image, False)
    #     >>> print(result)
    #     [[[10 11 12]
    #       [13 14 15]
    #       [16 17 18]]
    #      [[1 2 3]
    #       [4 5 6]
    #       [7 8 9]]]
    """
    # Generate a random roll value for the z-axis
    random_roll = np.random.randint(0, orig.shape[0])
    if random_roll != 0:
        random_roll = random_roll if random.random() < 0.5 else -random_roll

    if len(orig.shape) != 4:
        # Do not perform roll for non-4D images
        random_roll = 0

    if random_roll != 0 and (not is_mask or (is_mask and len(orig.shape) == 4)) and len(orig.shape) == 4 and \
            random_roll != orig.shape[0] and random_roll != -orig.shape[0]:
        # Roll the image along the z-axis if conditions are met
        rolled_orig = np.roll(orig, random_roll, axis=0)
    else:
        # Keep the original image or mask
        rolled_orig = orig

    return rolled_orig


# shuffle images along the Z axis, may be useful for best focus algorithms
def shuffleZ(orig, is_mask, **kwargs):
    """
    Shuffle the input image or mask along the z-axis.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The shuffled image or mask.

    # Examples:
    #     >>> image = np.array([[[1, 2, 3],
    #     ...                    [4, 5, 6],
    #     ...                    [7, 8, 9]],
    #     ...                   [[10, 11, 12],
    #     ...                    [13, 14, 15],
    #     ...                    [16, 17, 18]]])
    #     >>> result = shuffleZ(image, False)
    #     >>> print(result)
    #     [[[10 11 12]
    #       [13 14 15]
    #       [16 17 18]]
    #      [[1 2 3]
    #       [4 5 6]
    #       [7 8 9]]]
    """
    if len(orig.shape) == 4 and (not is_mask or (is_mask and len(orig.shape) == 4)):
        # Shuffle the image along the z-axis
        np.random.shuffle(orig)

    return orig




# mega TODO --> allow flip 3D and if so check that masks that can be 2D remain ok
# in fact that should already work works
def flip(orig, is_mask, **kwargs):
    """
    Flip the input image or mask along a random axis.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The flipped image or mask.


    """
    if len(orig.shape) == 4:
        axis = random.choice([-2, -3, -4])  # Randomly choose an axis to flip
    else:
        axis = random.choice([-2, -3])

    if axis == -4 and len(orig.shape) == 3:
        # If the axis is -4 and the input is a 3D image, no flipping is performed
        flipped_orig = orig
    else:
        # Flip the image or mask along the chosen axis using np.flip
        flipped_orig = np.flip(orig, axis)

    return flipped_orig


def high_noise(orig, is_mask, **kwargs):
    """
    Add high levels of noise to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The image or mask with high levels of noise added.

    """
    return _add_noise(orig, is_mask, 'high')


def low_noise(orig, is_mask, **kwargs):
    """
    Add low levels of noise to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The image or mask with low levels of noise added.

    """
    return _add_noise(orig, is_mask, 'low')


# really a crappy augmenter probably not even worth keeping ????
def add_z_frames(orig, is_mask, z_frames_to_add_below=0, z_frames_to_add_above=0, **kwargs):
    """
    Add additional Z frames to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        z_frames_to_add_below (int): The number of Z frames to add below the original image or mask.
        z_frames_to_add_above (int): The number of Z frames to add above the original image or mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The image or mask with additional Z frames added.

    # Examples:
    #     >>> image = np.array([[[1, 2, 3],
    #     ...                    [4, 5, 6],
    #     ...                    [7, 8, 9]]])
    #     >>> result = add_z_frames(image, False, z_frames_to_add_below=1, z_frames_to_add_above=2)
    #     >>> print(result.shape)
    #     (3, 3, 5)
    """
    # Ignore for masks
    if is_mask and not len(orig.shape) == 4:
        return orig

    if z_frames_to_add_below == 0 and z_frames_to_add_above == 0:
        # Nothing to do
        return orig

    zpos = -4
    try:
        zpos = orig.get_dim_idx('d')
    except:
        logger.error('dimension d/z not found, assuming Z pos = -4, i.e. ...dhwc image')

    if orig.shape[zpos] == 1:
        # Image is 2D, nothing to do
        return orig

    if z_frames_to_add_above != 0:
        # Add Z frames before
        smallest_shape = list(orig.shape)
        smallest_shape[zpos] = z_frames_to_add_above
        missing_frames_above = np.zeros(smallest_shape, dtype=orig.dtype)

        orig = np.append(missing_frames_above, orig, axis=zpos)

    if z_frames_to_add_below != 0:
        # Add Z frames after
        smallest_shape[zpos] = z_frames_to_add_below
        missing_frames_below = np.zeros(smallest_shape, dtype=orig.dtype)

        orig = np.append(orig, missing_frames_below, axis=zpos)

    return orig


# use skimage to add noise
def _add_noise(orig, is_mask, strength='low'):
    """
    Add noise to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        strength (str): The strength of the noise. Valid values are 'low' or 'high'.

    Returns:
        ndarray: The image or mask with noise added.

    """
    mode = random.choice(['gaussian', 'localvar', 'poisson', 's&p', 'speckle'])

    if not is_mask:
        extra_params = {}
        if mode in ['salt', 'pepper', 's&p']:
            if strength == 'low':
                extra_params['amount'] = 0.1  # Weak amount of salt and pepper noise
            else:
                extra_params['amount'] = 0.25  # Strong amount of salt and pepper noise
        if mode in ['gaussian', 'speckle']:
            if strength == 'low':
                extra_params['var'] = 0.025  # Weak variance for Gaussian noise
            else:
                extra_params['var'] = 0.1  # Strong variance for Gaussian noise
            if mode == 'speckle':
                if strength == 'low':
                    extra_params['var'] = 0.1  # Weak variance for speckle noise
                else:
                    extra_params['var'] = 0.40  # Strong variance for speckle noise

        min = orig.min()
        max = orig.max()

        noisy_image = random_noise(orig, mode=mode, clip=True, **extra_params)

        if min == 0 and max == 1 or max == min:
            pass
        else:
            # We preserve the original image range
            noisy_image = (noisy_image * (max - min)) + min
    else:
        noisy_image = orig

    return noisy_image


def random_intensity_gamma_contrast_changer(orig, is_mask, **kwargs):
    """
    Apply random intensity, gamma, and contrast changes to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.

    Returns:
        ndarray: The image or mask with random intensity, gamma, and contrast changes applied.

    """
    return _random_intensity_gamma_contrast_changer(orig, is_mask)


# use skimage to change gamma/contrast, ... pixel intensity and/or scale
def _random_intensity_gamma_contrast_changer(orig, is_mask):
    """
    Apply random intensity, gamma, and contrast changes to the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.

    Returns:
        ndarray: The image or mask with random intensity, gamma, and contrast changes applied.

    """
    # fraction of image that should contain salt black or pepper white pixels
    mode = random.choice([exposure.rescale_intensity, exposure.adjust_gamma, exposure.adjust_log,
                          exposure.adjust_sigmoid])
    strength = random.choice(['medium', 'strong'])

    if not is_mask:
        if orig.min() >= 0:
            extra_params = {}
            if mode == exposure.rescale_intensity:
                if strength == 'strong':
                    v_min, v_max = np.percentile(orig, (5, 95))
                else:
                    v_min, v_max = np.percentile(orig, (0.9, 98))
                extra_params['in_range'] = (v_min, v_max)
            elif mode == exposure.adjust_gamma:
                if strength == 'strong':
                    extra_params['gamma'] = 0.4
                    extra_params['gain'] = 0.9
                else:
                    extra_params['gamma'] = 0.8
                    extra_params['gain'] = 0.9
            elif mode == exposure.adjust_sigmoid:
                if strength == 'strong':
                    extra_params['gain'] = 5
                else:
                    extra_params['gain'] = 2

            contrast_changed_image = mode(orig, **extra_params)
        else:
            logger.warning(
                'Negative intensities detected --> ignoring random_intensity_gamma_contrast_changer augmentation.')
            contrast_changed_image = orig
    else:
        contrast_changed_image = orig

    return contrast_changed_image


def change_image_intensity_and_shift_range(orig, is_mask, **kwargs):
    """
    Change the intensity and shift the range of the input image or mask.

    Args:
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The image or mask with changed intensity and shifted range.

    """
    scaling_factor = random.uniform(kwargs['intensity'], 1)

    for c in range(orig.shape[-1]):
        cur = orig[..., c]
        if not is_mask:
            min_before = cur.min()
            initial_range = cur.max() - min_before

            try:
                import numexpr
                cur = numexpr.evaluate("cur * scaling_factor")
            except:
                cur = cur * scaling_factor

            new_min = cur.min()
            shift_min = min_before - new_min

            try:
                import numexpr
                cur = numexpr.evaluate("cur + shift_min")
            except:
                cur += shift_min

        shift_range = bool(random.getrandbits(1))
        if shift_range:
            if not is_mask:
                new_range = cur.max() - cur.min()
                possible_range_increase = initial_range - new_range
                random_range_shift = random.uniform(0., possible_range_increase)

                try:
                    import numexpr
                    cur = numexpr.evaluate("cur + random_range_shift")
                except:
                    cur += random_range_shift

        orig[..., c] = cur

    out = orig

    if is_mask:
        out = orig

    return out


def execute_chained_augmentations(chained_augmentations, orig, is_mask, **kwargs):
    """
    Execute a chain of augmentations on the input image or mask.

    Args:
        chained_augmentations (list): A list of augmentation functions to apply.
        orig (ndarray): The input image or mask.
        is_mask (bool): A flag indicating whether the input is a mask.
        **kwargs: Additional keyword arguments.

    Returns:
        ndarray: The augmented image or mask.

    """

    from epyseg.deeplearning.augmentation.generators.data import DataGenerator

    if chained_augmentations is None:
        logger.error('No augmentation in the chain. Ignoring.')
        return orig

    if not isinstance(chained_augmentations, list):
        chained_augmentations = [chained_augmentations]

    for aug in chained_augmentations:
        if aug is None:
            continue
        if isinstance(aug, str):
            if aug.lower() == 'none':
                continue
            if aug in DataGenerator.augmentation_types_and_methods:
                aug = DataGenerator.augmentation_types_and_methods[aug]

        orig = aug(orig, is_mask=is_mask, **kwargs)

    return orig



def tst_all_augs():
    """
    A basic test method for all augs.

    # Examples:
    #     >>> tst_all_augs()
    """
    from epyseg.deeplearning.augmentation.generators.data import DataGenerator
    # TODO --> could do all the tests here and check that they are always in sync despite not being used ---> can I call random after to check in sync --> yes -> very smart anc non costly idea
    orig = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012.png')[..., 1]
    mask = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012/handCorrection.tif')

    # try with wings
    orig = Img('/E/Sample_images/wings/adult_wings/pigmented_wings_benjamin_all/scaled_half/N01_YwrGal4_males_wing0_projected.tif')
    mask = Img('/E/Sample_images/wings/adult_wings/pigmented_wings_benjamin_all/scaled_half/predict/N01_YwrGal4_males_wing0_projected.tif')[...,0]
    # print(orig.shape, mask.shape)

    orig_3D = Img('/E/Sample_images/Amrutha_gut/trash_output_tests/test_2D_to_3D/orig.tif')
    mask_3D = Img('/E/Sample_images/Amrutha_gut/trash_output_tests/test_2D_to_3D/segmented_gut.tif')

    # we fake a set of parameters here

    augmentation_types_and_ranges = DataGenerator.augmentation_types_and_ranges
    augmentation_types_and_values =  DataGenerator.augmentation_types_and_values

    if True:
        from personal.ensemble.preview_many_images_side_by_side import preview_as_panel
        # raise Exception('THERE IS A BUG HERE IN THE RANGE OF THE MASK --> HOW CAN I CHANGE THIS --> TODO')
        print('testing crop')
        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)

        crop_with_parameters = partial(crop_augmentation, coords_dict={'x1': 10, 'x2': -10, 'y1': 10,
                                                                       'y2': -10})  # does that work with negative values by the way

        # NB image need hwc
        orig_2D_plus = orig[..., np.newaxis]
        augmented_orig = crop_with_parameters(orig=np.copy(orig_2D_plus), is_mask=False,
                                              **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        # NB image need hwc
        mask_2D_plus = mask[..., np.newaxis]
        augmented_mask = crop_with_parameters(orig=np.copy(mask_2D_plus), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        # that does not work --> but why is that
        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_2D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_different(mask_2D_plus, augmented_mask, success_message='mask is augmented')
        # import sys
        # sys.exit(0)

    if True:

        # raise Exception('THERE IS A BUG HERE IN THE RANGE OF THE MASK --> HOW CAN I CHANGE THIS --> TODO')
        print('testing elastic_with_zoom_and_translation')
        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = elastic_with_zoom_and_translation(orig=np.copy(orig), is_mask=False,
                                                           **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = elastic_with_zoom_and_translation(orig=np.copy(mask), is_mask=True,
                                                           **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        # that does not work --> but why is that
        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_different(mask, augmented_mask, success_message='mask is augmented')
        # import sys
        # sys.exit(0)

    if True:
        # raise Exception('THERE IS A BUG HERE IN THE RANGE OF THE MASK --> HOW CAN I CHANGE THIS --> TODO')
        print('testing elastic')
        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = elastic(orig=np.copy(orig), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = elastic(orig=np.copy(mask), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        # that does not work --> but why is that
        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_different(mask, augmented_mask, success_message='mask is augmented')
        # import sys
        # sys.exit(0)

    if True:
        print('testing change_image_intensity_and_shift_range')
        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)
        # NB image need hwc
        orig_2D_plus = orig[..., np.newaxis]
        augmented_orig = change_image_intensity_and_shift_range(orig=np.copy(orig_2D_plus), is_mask=False,
                                                                **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        # NB image need hwc
        mask_2D_plus = mask[..., np.newaxis]
        augmented_mask = change_image_intensity_and_shift_range(orig=np.copy(mask_2D_plus), is_mask=True,
                                                                **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        # that does not work --> but why is that
        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_2D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_equal(mask_2D_plus, augmented_mask, success_message='mask was kept unchanged')
        # import sys
        # sys.exit(0)

    if True:
        print('testing augmentation of original')
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = blur(orig=np.copy(orig), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = blur(orig=np.copy(mask), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_equal(mask, augmented_mask, success_message='mask is kept unchanged')

    if True:
        print('testing graded_intensity_modification')
        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = graded_intensity_modification(orig=np.copy(orig), is_mask=False,
                                                       **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = graded_intensity_modification(orig=np.copy(mask), is_mask=True,
                                                       **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_equal(mask, augmented_mask, success_message='mask is kept unchanged')

    if True:
        print('testing invert')
        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = invert(orig=np.copy(orig), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = invert(orig=np.copy(mask), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_equal(mask, augmented_mask, success_message='mask is kept unchanged')

    if True:
        print('testing rotate')
        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)
        # NB image need dhwc
        orig_2D_plus = orig[..., np.newaxis]
        augmented_orig = rotate(orig=np.copy(orig_2D_plus), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        # NB image need dhwc
        mask_2D_plus = mask[..., np.newaxis]
        augmented_mask = rotate(orig=np.copy(mask_2D_plus), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_2D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is changed
        array_different(mask_2D_plus, augmented_mask, success_message='mask was augmented')

    if True:
        print('testing rotate_interpolation_free')
        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)
        # NB image need hwc
        orig_2D_plus = orig[..., np.newaxis]
        augmented_orig = rotate_interpolation_free(orig=np.copy(orig_2D_plus), is_mask=False,
                                                   **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        #  NB image need hwc
        mask_2D_plus = mask[..., np.newaxis]
        augmented_mask = rotate_interpolation_free(orig=np.copy(mask_2D_plus), is_mask=True,
                                                   **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_2D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_different(mask_2D_plus, augmented_mask, success_message='mask has been augmented')

    if True:
        print('testing translate')
        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)
        #  NB image need hwc
        orig_2D_plus = orig[..., np.newaxis]
        augmented_orig = translate(orig=np.copy(orig_2D_plus), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        #  NB image need hwc
        mask_2D_plus = mask[..., np.newaxis]
        augmented_mask = translate(orig=np.copy(mask_2D_plus), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_2D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_different(mask_2D_plus, augmented_mask, success_message='mask was augmented')

    if True:
        print('testing zoom')
        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = zoom(orig=np.copy(orig), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = zoom(orig=np.copy(mask), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_different(mask, augmented_mask, success_message='mask was augmented')

    if True:
        print('testing shear')
        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = shear(orig=np.copy(orig), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = shear(orig=np.copy(mask), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_different(mask, augmented_mask, success_message='mask was augmented')

    if True:
        # NB THIS IS NOT A DATA AUG !!!!!!
        print('testing mask_pixels_and_compute_for_those')
        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)
        # assumes image si à minima hwc
        #  NB image need hwc
        orig_2D_plus = orig[..., np.newaxis]
        augmented_orig = mask_pixels_and_compute_for_those(orig=np.copy(orig_2D_plus), is_mask=False,
                                                           **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        # NB image needs be float for it to work and this is definitely not a data augmentation --> do not put it in data aug but rather in post processing and can apply it to compute losses
        mask_2D_plus = mask[..., np.newaxis]
        augmented_mask = mask_pixels_and_compute_for_those(orig=np.copy(mask_2D_plus).astype(float), is_mask=True,
                                                           **augmentation_types_and_values)
        augmented_mask = augmented_mask.astype(bool)  # nb it does return a True or False mask
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_2D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_different(mask_2D_plus, augmented_mask, success_message='mask was augmented')

    if True:
        print('testing rollZ')
        # testing augmentation of original
        # NB image need dhwc
        orig_3D_plus = orig_3D[..., np.newaxis]
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = rollZ(orig=np.copy(orig_3D_plus), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        # NB image need dhwc
        mask_3D_plus = mask_3D[..., np.newaxis]
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = rollZ(orig=np.copy(mask_3D_plus), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        mask_2D = mask[..., np.newaxis]
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask2 = rollZ(orig=np.copy(mask_2D), is_mask=True, **augmentation_types_and_values)
        val_after_third = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second == val_after_third, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            try:
                # TODO --> maybe plot both side by side
                plt.imshow(augmented_mask2)
                plt.show()
                # plt.imshow(augmented_mask)
                # plt.show()
                # plt.imshow(augmented_orig)
                # plt.show()
                preview_as_panel([augmented_orig, augmented_mask])
            except:
                # plt.imshow(augmented_mask[int((augmented_mask.shape[0])/2)])
                # plt.show()
                # plt.imshow(augmented_orig[int((augmented_orig.shape[0])/2)])
                # plt.show()
                preview_as_panel([augmented_orig[int((augmented_orig.shape[0]) / 2)],
                                  augmented_mask[int((augmented_mask.shape[0]) / 2)]])
                # preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)
        print('augmented_mask2', augmented_mask2.max(), augmented_mask2.min(), augmented_mask2.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_3D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_different(mask_3D_plus, augmented_mask, success_message='mask  was augmented')

        array_equal(mask_2D, augmented_mask2, success_message='2D mask was unchanged')

    if True:
        print('testing shuffleZ')
        # testing augmentation of original
        # NB image need dhwc
        orig_3D_plus = orig_3D[..., np.newaxis]
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = shuffleZ(orig=np.copy(orig_3D_plus), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        # NB image need dhwc
        mask_3D_plus = mask_3D[..., np.newaxis]
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = shuffleZ(orig=np.copy(mask_3D_plus), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        mask_2D = mask[..., np.newaxis]
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask2 = shuffleZ(orig=np.copy(mask_2D), is_mask=True, **augmentation_types_and_values)
        val_after_third = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second == val_after_third, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            try:
                # TODO --> maybe plot both side by side
                plt.imshow(augmented_mask2)
                plt.show()
                # plt.imshow(augmented_mask)
                # plt.show()
                # plt.imshow(augmented_orig)
                # plt.show()
                preview_as_panel([augmented_orig, augmented_mask])
            except:
                # plt.imshow(augmented_mask[int((augmented_mask.shape[0])/2)])
                # plt.show()
                # plt.imshow(augmented_orig[int((augmented_orig.shape[0])/2)])
                # plt.show()
                preview_as_panel([augmented_orig[int((augmented_orig.shape[0]) / 2)],
                                  augmented_mask[int((augmented_mask.shape[0]) / 2)]])
                # preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)
        print('augmented_mask2', augmented_mask2.max(), augmented_mask2.min(), augmented_mask2.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_3D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_different(mask_3D_plus, augmented_mask, success_message='mask  was augmented')

        array_equal(mask_2D, augmented_mask2, success_message='2D mask was unchanged')

    if True:
        print('testing flip')
        # testing augmentation of original
        # NB image need dhwc
        orig_2D_plus = orig[..., np.newaxis]
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = flip(orig=np.copy(orig_2D_plus), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        # NB image need dhwc
        mask_2D_plus = mask[..., np.newaxis]
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = flip(orig=np.copy(mask_2D_plus), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_2D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_different(mask_2D_plus, augmented_mask, success_message='mask was augmented')

    if True:
        # NB why is max noise 254 and not 255 --> MEGA TODO CHECK --> did i do a mistake ????
        print('testing high_noise')
        # testing augmentation of original
        # NB image need dhwc
        orig_2D_plus = orig[..., np.newaxis]
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = high_noise(orig=np.copy(orig_2D_plus), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        # NB image need dhwc
        mask_2D_plus = mask[..., np.newaxis]
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = high_noise(orig=np.copy(mask_2D_plus), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_2D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_equal(mask_2D_plus, augmented_mask, success_message='mask was not augmented')

    if True:
        print('testing low_noise')
        # testing augmentation of original
        # NB image need dhwc
        orig_2D_plus = orig[..., np.newaxis]
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = low_noise(orig=np.copy(orig_2D_plus), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        # NB image need dhwc
        mask_2D_plus = mask[..., np.newaxis]
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = low_noise(orig=np.copy(mask_2D_plus), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_2D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_equal(mask_2D_plus, augmented_mask, success_message='mask was not augmented')

    if True:
        print('testing random_intensity_gamma_contrast_changer')
        # testing augmentation of original
        # NB image need dhwc
        orig_2D_plus = orig[..., np.newaxis]
        seed = datetime.now()
        random.seed(seed)
        augmented_orig = random_intensity_gamma_contrast_changer(orig=np.copy(orig_2D_plus), is_mask=False,
                                                                 **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        # NB image need dhwc
        mask_2D_plus = mask[..., np.newaxis]
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = random_intensity_gamma_contrast_changer(orig=np.copy(mask_2D_plus), is_mask=True,
                                                                 **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_2D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_equal(mask_2D_plus, augmented_mask, success_message='mask was not augmented')

    # seems ok but could do the test more smartly by checking that the size of some of the dimensions matches the expected
    # this aug anyways sucks!!!
    # maybe add a parameter force mask in some cases because that could be useful --> but what I do is probably ok still in fact !!!
    if True:
        print('testing add_z_frames')
        # testing augmentation of original
        # NB image need dhwc
        orig_3D_plus = orig_3D[..., np.newaxis]
        seed = datetime.now()
        random.seed(seed)

        # the values should be passed through the stuff
        augmentation_types_and_values['z_frames_to_add_below'] = 2
        augmentation_types_and_values['z_frames_to_add_above'] = 4

        augmented_orig = add_z_frames(orig=np.copy(orig_3D_plus), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        # NB image need dhwc
        mask_3D_plus = mask_3D[..., np.newaxis]
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask = add_z_frames(orig=np.copy(mask_3D_plus), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        mask_2D = mask[..., np.newaxis]
        random.seed(seed)  # reseed to be sure the same random are applied
        augmented_mask2 = shuffleZ(orig=np.copy(mask_2D), is_mask=True, **augmentation_types_and_values)
        val_after_third = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second == val_after_third, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if False:
            try:
                # TODO --> maybe plot both side by side
                plt.imshow(augmented_mask2)
                plt.show()
                # plt.imshow(augmented_mask)
                # plt.show()
                # plt.imshow(augmented_orig)
                # plt.show()
                preview_as_panel([augmented_orig, augmented_mask])
            except:
                # plt.imshow(augmented_mask[int((augmented_mask.shape[0])/2)])
                # plt.show()
                # plt.imshow(augmented_orig[int((augmented_orig.shape[0])/2)])
                # plt.show()
                preview_as_panel([augmented_orig[int((augmented_orig.shape[0]) / 2)],
                                  augmented_mask[int((augmented_mask.shape[0]) / 2)]])
                # preview_as_panel([augmented_orig, augmented_mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)
        print('augmented_mask2', augmented_mask2.max(), augmented_mask2.min(), augmented_mask2.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_3D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        array_different(mask_3D_plus, augmented_mask, success_message='mask was augmented')

        array_equal(mask_2D, augmented_mask2, success_message='2D mask was unchanged')


if __name__ == '__main__':

    from epyseg.deeplearning.augmentation.generators.data import DataGenerator
    # TODO --> could do all the tests here and check that they are always in sync despite not being used ---> can I call random after to check in sync --> yes -> very smart anc non costly idea
    orig = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012.png')[..., 1]
    mask = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012/handCorrection.tif')

    # try with wings
    orig = Img('/E/Sample_images/wings/adult_wings/pigmented_wings_benjamin_all/scaled_half/N01_YwrGal4_males_wing0_projected.tif')
    mask = Img('/E/Sample_images/wings/adult_wings/pigmented_wings_benjamin_all/scaled_half/predict/N01_YwrGal4_males_wing0_projected.tif')[...,0]
    # print(orig.shape, mask.shape)

    orig_3D = Img('/E/Sample_images/Amrutha_gut/trash_output_tests/test_2D_to_3D/orig.tif')
    mask_3D = Img('/E/Sample_images/Amrutha_gut/trash_output_tests/test_2D_to_3D/segmented_gut.tif')

    # we fake a set of parameters here

    # TODO should I also store increment there ???
    augmentation_types_and_ranges = DataGenerator.augmentation_types_and_ranges
    augmentation_types_and_values = DataGenerator.augmentation_types_and_values

    # if True:
    #     preview_as_panel([orig, mask])
    #     import sys
    #     sys.exit(0)


    # if True:
    #     # they are not the same so I need to seed them
    #     import sys
    #     seed = datetime.now()
    #     random.seed(seed)
    #     np.random.seed(random.randint(0, 10000000))
    #     displacement1 = np.random.randn(2, 3, 3) * 25
    #     random.seed(seed)
    #     np.random.seed(random.randint(0, 10000000))
    #     displacement2 = np.random.randn(2, 3, 3) * 25
    #     array_equal(displacement1, displacement2,success_message='arrays are the same')
    #     import sys
    #     sys.exit(0)

    if False:
        # in fact elastic deform is super weak for big images such as wings --> how can I increase its force so that it becomes a useful augmentation

        # orig = np.random.rand(1289,2001)
        orig = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012.png')[..., 1] #strong deformation
        orig = Img('/E/Sample_images/wings/adult_wings/pigmented_wings_benjamin_all/scaled_half/N01_YwrGal4_males_wing0_projected.tif') #very weak deformation
        augmented_orig = strong_elastic_with_zoom_and_translation(orig, False, **augmentation_types_and_values)
        preview_as_panel([augmented_orig,  orig])

        import sys
        sys.exit(0)


    if True:
        from personal.ensemble.preview_many_images_side_by_side import preview_as_panel
        # that works
        # try pipes with string names --> can I do that ??? --> probably yes
        # it seems to work --> now just plot them side by side --> try that --> TODO
        print('testing piped data aug')

        # parfait tt marche --> cool --> implementer ça un peu partout maintenant et voir ce que ça fait si je le passe à un data aug --> à faire

        # chained_augmentation = [pad_border_xy, invert, rotate_interpolation_free, flip, elastic]  #, elastic] --> elastic does not work --> it fucks the pipe --> how can I do that then ????
        crop_with_parameters = partial(crop_augmentation, coords_dict={'x1':10, 'x2':-10, 'y1':10, 'y2':-10}) # does that work with negative values by the way
        chained_augmentation = [crop_with_parameters, strong_elastic]  # elastic_with_zoom_and_translation  #, elastic] --> elastic does not work --> it fucks the pipe --> how can I do that then ????
        # chained_augmentation = [pad_border_xy, 'invert', 'rotate (interpolation free)', 'flip', 'elastic']  #, elastic] --> elastic does not work --> it fucks the pipe --> how can I do that then ????

        # testing augmentation of original
        seed = datetime.now()
        random.seed(seed)
        # NB image need dhwc
        orig_2D_plus = orig[..., np.newaxis]
        # pb is how to pass the immutable params to all the functions --> probably need make my own function here once for good
        augmented_orig = execute_chained_augmentations(chained_augmentation, orig=np.copy(orig_2D_plus), is_mask=False, **augmentation_types_and_values) #blur(orig=np.copy(orig), is_mask=False, **augmentation_types_and_values)
        val_after_first = random.random()  # will be used to check that random is still in sync after the call

        # testing augmentation/non-augmentation of mask
        random.seed(seed)  # reseed to be sure the same random are applied
        # NB image need hwc
        mask_2D_plus = mask[..., np.newaxis]
        augmented_mask = execute_chained_augmentations(chained_augmentation,  orig=np.copy(mask_2D_plus), is_mask=True, **augmentation_types_and_values)
        val_after_second = random.random()  # will be used to check that random is still in sync after the call

        # try:
        assert val_after_first == val_after_second, "NB Random is NOT IN SYNC, the nb of random calls is not equal"
        print('EVERYTHING OK: Random is in SYNC')
        #        except AssertionError as e:
        #            print(e)

        if True:
            # TODO --> maybe plot both side by side
            # plt.imshow(augmented_mask)
            # plt.show()
            # plt.imshow(augmented_orig)
            # plt.show()
            preview_as_panel([augmented_orig, augmented_mask, orig, mask])

        print('orig', orig.max(), orig.min(), orig.shape)
        print('mask', mask.max(), mask.min(), mask.shape)

        print('augmented_orig', augmented_orig.max(), augmented_orig.min(), augmented_orig.shape)
        print('augmented_mask', augmented_mask.max(), augmented_mask.min(), augmented_mask.shape)

        # asserts changes were applied to the original (is that smart ???) --> since it's random and value can be 0 then may make no sense but most often it will be a valid check
        array_different(orig_2D_plus, augmented_orig, success_message='original image was augmented')

        # assert the mask is unchanged
        try:
            array_equal(mask_2D_plus, augmented_mask, success_message='mask is kept unchanged')
        except:
            array_different(mask_2D_plus, augmented_mask, success_message='mask is changed')

        import sys
        sys.exit(0)