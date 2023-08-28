from skimage.registration import phase_cross_correlation # change in scipy 0.19
# image = data.camera()
from skimage import transform
import numpy as np

__DEBUG__ = False

def apply_translation(img, y, x):
    """
    Applies translation to an image.

    Args:
        img (numpy.ndarray): The input image.
        y (float): Translation along the y-axis.
        x (float): Translation along the x-axis.

    Returns:
        numpy.ndarray: The translated image.
    """
    afine_tf = transform.AffineTransform(translation=(x, y))
    translated = transform.warp(img, inverse_map=afine_tf, order=0, preserve_range=True)
    return translated

def pre_register_images(orig_t0, orig_t1, apply_to_orig_t0=False):
    """
    Performs pre-registration between two images.

    Args:
        orig_t0 (numpy.ndarray): The first image.
        orig_t1 (numpy.ndarray): The second image.
        apply_to_orig_t0 (bool, optional): Whether to apply the translation to orig_t0. Defaults to False.

    Returns:
        tuple or numpy.ndarray: If apply_to_orig_t0 is False, returns the global translations (gloabl_t0, gloabl_t1).
        If apply_to_orig_t0 is True, returns the translated orig_t0 image.

    Notes:
        - The function uses phase cross-correlation to estimate the global translation between the two images.
        - The global translation is then applied to either orig_t0 or both orig_t0 and orig_t1, depending on the
          value of apply_to_orig_t0.
    """
    global_translation, error, diffphase = phase_cross_correlation(orig_t0, orig_t1)
    gloabl_t0 = -global_translation[0]
    gloabl_t1 = -global_translation[1]
    if not apply_to_orig_t0:
        return gloabl_t0, gloabl_t1
    else:
        return apply_translation(orig_t0, -gloabl_t0, -gloabl_t1)

def get_pyramidal_registration(orig_t0, orig_t1, depth=2, threshold_translation=20):
    """
    Performs pyramidal registration between two images.

    Args:
        orig_t0 (numpy.ndarray): The first image.
        orig_t1 (numpy.ndarray): The second image.
        depth (int, optional): The depth of the pyramid. Defaults to 2.
        threshold_translation (float, optional): The threshold for ignoring large translations. Defaults to 20.

    Returns:
        numpy.ndarray: The translation matrix representing the translations for each pixel.

    Notes:
        - The function applies pre-registration between orig_t0 and orig_t1.
        - It then iteratively performs registration on blocks of decreasing size.
        - Translations are accumulated in the translation matrix.
    """

    translation_matrix = np.zeros((*orig_t0.shape, 2))

    gloabl_t0, gloabl_t1 = pre_register_images(orig_t0, orig_t1)

    translation_matrix[..., 0] += gloabl_t0
    translation_matrix[..., 1] += gloabl_t1

    height = orig_t0.shape[0]
    width = orig_t0.shape[1]

    block_size = 256

    for i in range(depth):
        try:
            tile_height = block_size // (i * 2)
            tile_width = block_size // (i * 2)
        except:
            tile_height = block_size
            tile_width = block_size

        for y in range(0, height, tile_height):
            for x in range(0, width, tile_width):
                afine_tf = transform.AffineTransform(
                    translation=(-translation_matrix[y, x, 1], -translation_matrix[y, x, 0]))
                translated = transform.warp(orig_t0, inverse_map=afine_tf, order=0, preserve_range=True)

                tile_t0 = translated[y:y + tile_height, x:x + tile_width]
                tile_t1 = orig_t1[y:y + tile_height, x:x + tile_width]

                shift, error, diffphase = phase_cross_correlation(tile_t0, tile_t1)
                t0 = -shift[0]
                t1 = -shift[1]

                if abs(t0) > threshold_translation or abs(t1) > threshold_translation:
                    continue

                translation_matrix[y:y + tile_height, x:x + tile_width, 0] += t0
                translation_matrix[y:y + tile_height, x:x + tile_width, 1] += t1

    return translation_matrix
