# Here I will gather all the unit tests for images

# NB: DEVELOPER TIP: keep in mind that in order that to get things tested it must start with 'test' !!!

import unittest

from numpy.ma.testutils import assert_array_equal

from epyseg.img import Img
import numpy as np

class TestSum(unittest.TestCase):

    def test_open_local_img(self):
        tmp = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png')
        self.assertIsInstance(tmp, Img)
        self.assertIsInstance(tmp, np.ndarray)
        self.assertEqual(tmp.get_dimensions_as_string(),'hwc')
        # self.assertEqual(tmp.get_dimensions_as_string(),'dhwc')

    def test_open_image_series(self):
        img = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini_asym/*.png')
        self.assertEqual(img.shape, (6, 312, 512, 3)) # h=312

    def test_invert(self):
        tmp = np.asarray([[0,0],[255,0]])
        assert_array_equal(Img.invert(tmp),np.asarray([[255,255],[0,255]]))

        tmp = np.asarray([[0, 0],[65300, 0]])
        assert_array_equal(Img.invert(tmp), np.asarray([[65300, 65300], [0, 65300]]))

        # that does not work as I want --> shall I fix it
        tmp = np.asarray([[123,123],[65300, 123]])
        assert_array_equal(Img.invert(tmp), np.asarray([[65300, 65300], [123, 65300]]))

        tmp = np.asarray([[-123, -123],[65300, -123]])
        assert_array_equal(Img.invert(tmp), np.asarray([[65300, 65300], [-123, 65300]]))

        tmp = np.asarray([[True, True],[False, True]])
        assert_array_equal(Img.invert(tmp), np.asarray([[False, False], [True, False]]))

    def test_read_image_from_the_web(self):
        # all the files below are working --> cool
        # img = Img('https://samples.fiji.sc/colocsample1b.lsm')
        # img = Img('https://samples.fiji.sc/12_9.tif')
        # img = Img('https://samples.fiji.sc/5602-01_4_568_633_x63_stack2_Channel_BLUE.tif')
        # img = Img('https://samples.fiji.sc/150707_WTstack.lsm')
        # img = Img('https://samples.fiji.sc/MessedUpColoc.png')
        # img = Img('https://samples.fiji.sc/blobs.png')
        # img = Img('https://samples.fiji.sc/colocsample1bRGB_BG.tif')
        # img = Img('http://www.cellimagelibrary.org/pic/interactive_cells/eukaryotic/default.png')
        # print(img.shape)
        img = Img('https://samples.fiji.sc/new-lenna.jpg')
        self.assertEqual(img.shape,(1279, 853, 3))

        img = Img('https://samples.fiji.sc/colocsample1b.lsm')
        self.assertEqual(img.shape, (33, 152, 172, 2))

if __name__ == '__main__':
    unittest.main()
