import numpy as np
from numpy.ma.testutils import assert_array_equal

from epyseg.img import Img, to_stack, is_binary
from epyseg.utils.loadlist import loadlist


def invert(img):
    # should take the negative of an image should always work I think but try and see if not wise making a version that handles channels # does it even make sense ??? need to think a bit about it
    max = img.max()
    min = img.min()
    # print(np.negative(img))
    if not img.dtype == bool:
        img = np.negative(img) + max + min
    else:
        img = ~img
    return img

if __name__ == '__main__':
    print(is_binary(np.asarray([[True, True],[False, True]])))