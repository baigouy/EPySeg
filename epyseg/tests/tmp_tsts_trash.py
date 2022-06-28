import math
from random import gauss

import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.testutils import assert_array_equal

from epyseg.img import Img,  pop
from epyseg.utils.loadlist import loadlist
import random

if __name__ == '__main__':
    if True:
        img = Img('/E/Sample_images/sample_images_FIJI/150707_WTstack.lsm')
        # auto_scale(img)
        pop(img)

        img = Img('/E/Sample_images/sample_images_FIJI/AuPbSn40.jpg')
        pop(img)
        # ok but then where is the bug???

        import sys
        sys.exit(0)
