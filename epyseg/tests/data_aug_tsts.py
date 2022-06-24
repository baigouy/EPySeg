# Here I will gather all the unit tests for data augmentation

import unittest

from numpy.ma.testutils import assert_array_equal

from epyseg.img import mask_rows_or_columns
import numpy as np


class TestSum(unittest.TestCase):

    # tmp = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png')
    # self.assertIsInstance(tmp, Img)
    # self.assertIsInstance(tmp, np.ndarray)
    # self.assertEqual(tmp.get_dimensions_as_string(),'hwc')
    # self.assertEqual(tmp.get_dimensions_as_string(),'dhwc')

    # nb it assume the image is hwc
    def test_mask_rows_or_columns(self):
        image = np.asarray([[255, 255, 64],
                            [0, 128, 255],
                            [255, 32, 255],
                            [255, 32, 255]])
        masked = np.squeeze(mask_rows_or_columns(image, spacing_X=2, spacing_Y=2)) # NB converts image to hwc if not
        assert_array_equal(masked, np.asarray([[0, 0, 0],
                                               [0, 128, 0],
                                               [0, 0, 0],
                                               [0, 32, 0]]))

        image = np.asarray([[255, 255, 64],
                            [0, 128, 255],
                            [255, 32, 255],
                            [255, 32, 255]])
        masked = np.squeeze(mask_rows_or_columns(image, spacing_X=2, spacing_Y=2, masking_value=-1)) # NB converts image to hwc if not
        assert_array_equal(masked, np.asarray([[-1,-1, -1],
                                               [-1, 128, -1],
                                               [-1, -1, -1],
                                               [-1, 32, -1]]))


if __name__ == '__main__':
    unittest.main()
