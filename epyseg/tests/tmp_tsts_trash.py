import numpy as np
from numpy.ma.testutils import assert_array_equal


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
    tmp = np.asarray([[123, 123],
                      [65300, 123]])
    assert_array_equal(invert(tmp), np.asarray([[65300, 65300], [123, 65300]]))

    tmp = np.asarray([[-123, -123],
                      [65300, -123]])
    assert_array_equal(invert(tmp), np.asarray([[65300, 65300], [-123, 65300]]))

    tmp = np.asarray([[True, True],
                      [False, True]])
    assert_array_equal(invert(tmp), np.asarray([[False, False], [True, False]]))