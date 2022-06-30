import math
from random import gauss

import matplotlib.pyplot as plt
import numpy as np
import qtpy
from numpy.ma.testutils import assert_array_equal

from epyseg.img import Img,  pop
from epyseg.settings.global_settings import print_default_qtpy_UI_really_in_use
from epyseg.utils.loadlist import loadlist
import random


def tst_pyside6():
    import os
    import sys
    os.environ['QT_API'] = 'pyside6'
    print(os.environ['QT_API'])
    print('testing UI', os.environ['QT_API'])
    # somehow it is changed
    # set_UI()
    # print_default_qtpy_UI_really_in_use() # --> fucks it
    print('vsss', os.environ['QT_API'])
    # maybe due to some wird default settings of qtpy --> forget for now ---> just test things manually
    from epyseg.ta.GUI.pyta import \
        TissueAnalyzer  # --> this force changes to pyQT6 here but why ???? --> works well outside of tests but not from inside --> really no clue why ???
    print('vsss2', os.environ['QT_API'])
    from qtpy.QtWidgets import QApplication
    app = QApplication(sys.argv)
    # bug but why is that
    print_default_qtpy_UI_really_in_use()
    w = TissueAnalyzer()
    w.show()
    if w.isVisible():
        QApplication.exit()
        # self.assertTrue(True)
    else:
        QApplication.exit()
        # self.assertFalse(True)

def tst_pyside2():
    import os
    import sys
    os.environ['QT_API'] = 'pyside2'
    print(os.environ['QT_API'])
    print('testing UI', os.environ['QT_API'])
    # somehow it is changed
    # set_UI()
    # print_default_qtpy_UI_really_in_use() # --> fucks it
    print('vsss', os.environ['QT_API'])
    # maybe due to some wird default settings of qtpy --> forget for now ---> just test things manually
    from epyseg.ta.GUI.pyta import \
        TissueAnalyzer  # --> this force changes to pyQT6 here but why ???? --> works well outside of tests but not from inside --> really no clue why ???
    print('vsss2', os.environ['QT_API'])
    from qtpy.QtWidgets import QApplication
    app = QApplication(sys.argv)
    # bug but why is that
    print_default_qtpy_UI_really_in_use()
    w = TissueAnalyzer()
    w.show()
    if w.isVisible():
        QApplication.exit()
        # self.assertTrue(True)
    else:
        QApplication.exit()
        # self.assertFalse(True)

def tst_pyqt6():
    import os
    import sys
    os.environ['QT_API'] = 'pyqt6'
    print(os.environ['QT_API'])
    print('testing UI', os.environ['QT_API'])
    # somehow it is changed
    # set_UI()
    # print_default_qtpy_UI_really_in_use() # --> fucks it
    print('vsss', os.environ['QT_API'])

    # bug is
    # maybe due to some wird default settings of qtpy --> forget for now ---> just test things manually
    from epyseg.ta.GUI.pyta import \
        TissueAnalyzer  # --> this force changes to pyQT6 here but why ???? --> works well outside of tests but not from inside --> really no clue why ???
    print('vsss2', os.environ['QT_API'])
    from qtpy.QtWidgets import QApplication
    app = QApplication(sys.argv)
    # bug but why is that
    print_default_qtpy_UI_really_in_use()
    w = TissueAnalyzer()
    w.show()
    if w.isVisible():
        QApplication.exit()
        # self.assertTrue(True)
    else:
        QApplication.exit()
        # self.assertFalse(True)


def tst_pyqt5():
    import os
    import sys
    os.environ['QT_API'] = 'pyqt5'
    print(os.environ['QT_API'])
    print('testing UI', os.environ['QT_API'])
    # somehow it is changed
    # set_UI()
    # print_default_qtpy_UI_really_in_use() # --> fucks it
    print('vsss', os.environ['QT_API'])
    # maybe due to some wird default settings of qtpy --> forget for now ---> just test things manually
    from epyseg.ta.GUI.pyta import \
        TissueAnalyzer  # --> this force changes to pyQT6 here but why ???? --> works well outside of tests but not from inside --> really no clue why ???
    print('vsss2', os.environ['QT_API'])
    from qtpy.QtWidgets import QApplication
    # print(qtpy.API)
    app = QApplication(sys.argv)
    # bug but why is that
    print_default_qtpy_UI_really_in_use()
    w = TissueAnalyzer()
    w.show()
    if w.isVisible():
        QApplication.exit()
        # self.assertTrue(True)
    else:
        QApplication.exit()
        # self.assertFalse(True)

# do that for all GUIs --> TODO

if __name__ == '__main__':
    if True:
        # incroyable ça marche ici mais pas en unittest ... --> comprend rien

        # KEEP ALL ça marche
        # tst_pyqt5()
        tst_pyqt6() # now it's always pyqt5 --> no clue IN FACT BUG IS STILL HERE TOO --> I REALLY DON'T GET IT ...
        # tst_pyside2()
        # tst_pyside6()
        # there are weird bugs but globally ok


        import sys
        sys.exit(0)

    if True:
        img = Img('/E/Sample_images/sample_images_FIJI/150707_WTstack.lsm')
        # auto_scale(img)
        pop(img)

        img = Img('/E/Sample_images/sample_images_FIJI/AuPbSn40.jpg')
        pop(img)
        # ok but then where is the bug???

        import sys
        sys.exit(0)
