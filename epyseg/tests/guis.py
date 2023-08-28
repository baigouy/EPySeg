# Here I will gather all the unit tests for GUIs
# make sure all runs with pyside, pyside2, pyqt5 and pyqt6
import unittest
import os
import sys
import warnings
import random
from epyseg.settings.global_settings import print_default_qtpy_UI_really_in_use, list_UIs, set_UI, \
    force_qtpy_to_use_user_specified

class TestGUI(unittest.TestCase):

    # DEV COMMENT TO KEEP: uncomment the lines below to check one specific UI
    # UI = 'pyqt5'
    UI = 'pyqt6'
    # UI = 'pyside2'
    # UI = 'pyside6'
    GUIs = ['TissueAnalyzer',
            'EPySeg',
            # 'DeepTools', # deprecated
            'EZFIG_GUI',
            'scrollable_EZFIG',
            # 'PyQT_markdown',
            'GT_editor',
            'ListGUI',
            'Createpaintwidget',
            'scrollable_paint',
            'dualList',
            'tascrollablepaint',
            # gut_SEG, # not public yet
            # Branchpaint, # not public yet
            # TEM_SEG, # not public yet
            # ImageSelectionPaint, # not public yet
            ] # GUIs to test


    def test_qtpy_force(self):
        import os
        force_qtpy_to_use_user_specified()
        os.environ['QT_API']=self.UI
        self.assertEqual(os.environ['QT_API'],self.UI)
        from qtpy.QtWidgets import QWidget
        self.assertEqual( os.environ['QT_API'],self.UI)

    def test_GUIs(self):
        force_qtpy_to_use_user_specified()
        os.environ['QT_API']=self.UI

        from epyseg.epygui import EPySeg
        from epyseg.figure.ezfgui import EZFIG_GUI
        from epyseg.figure.scrollableEZFIG import scrollable_EZFIG
        # from epyseg.gui.pyqtmarkdown import PyQT_markdown
        from epyseg.ta.GUI.GT_editor_GUI import GT_editor
        from epyseg.ta.GUI.list_gui import ListGUI
        from epyseg.ta.GUI.paint2 import Createpaintwidget
        from epyseg.ta.GUI.pyta import \
            TissueAnalyzer  # --> this force changes to pyQT6 here but why ???? --> works well outside of tests but not from inside --> really no clue why ???
        from epyseg.ta.GUI.scrollablepaint import scrollable_paint
        from epyseg.ta.GUI.stackedduallist import dualList
        from epyseg.ta.GUI.tascrollablepaint import tascrollablepaint
        # from personal.amrutha_gut.GUT_seg_GUI import gut_SEG
        # from personal.amrutha_gut.branchpaint import Branchpaint
        # from personal.laurence_had_neurons.TEM_seg_GUI import TEM_SEG
        # from personal.pyTA.GUI.display.imageSelectionPaint import ImageSelectionPaint

        from qtpy.QtWidgets import QApplication
        app = QApplication(sys.argv)
        print_default_qtpy_UI_really_in_use()
        self.assertEqual(os.environ['QT_API'], self.UI)
        # for gui in self.GUIs:
        for iii, gui in enumerate(self.GUIs):
            print(iii, gui) #  even though it only prints ta it seems to run for all
            # w = gui()
            w = locals()[gui]()
            w.show()
            self.assertTrue(w.isVisible())
            # if random.uniform(0, 1)<0.5:
            #     self.assertTrue(False)
        #


if __name__ == '__main__':
    # DEV COMMENT KEEP nothing below works --> must uncomment at the beginning of the code

    # import os
    # os.environ['QT_API']='pyqt5'
    # TestGUI.UI ='pyqt5'
    # TestGUI.UI ='pyqt6'
    # TestGUI.UI ='pyside6'
    # TestGUI.UI ='pyside2'
    # TestGUI.UI = os.environ.get('TEST_UI', TestGUI.UI)
    # ne marche pas... --> need change it directly at the tip of the code
    unittest.main()
