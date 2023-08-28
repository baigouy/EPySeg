# Here I will gather all the unit tests for GUIs
# make sure all runs with pyside, pyside2, pyqt5 and pyqt6
import unittest
import os
import sys
import warnings
import random

from epyseg.settings.global_settings import print_default_qtpy_UI_really_in_use, list_UIs, set_UI, \
    force_qtpy_to_use_user_specified

# it works but I had to deactivate some values --> see why and what is the pb with the deactivated one!!!

class TestDialogs(unittest.TestCase):

    # DEV COMMENT TO KEEP: uncomment the lines below to check one specific UI
    # UI = 'pyqt5'
    UI = 'pyqt6'
    # UI = 'pyside2'
    # UI = 'pyside6'

    dialogs = ['DenoiseRecursionDialog',
               'crop_or_preview',
               'DefineROI',
               'minisel',               # GreekSelectorDialog
               'image_input_settings',  # will that work ???
               'Multiple_inputs',  # will that work ???
               'OpenFileOrFolderWidget',
               'PostProcessGUI',
               'DenoiseRecursionDialog',
               'FinishAllDialog',
               'QPlainTextInputDialog',
               'TrackingDialog',
               'WshedDialog',
               'Example', # buggy too
               'SQL_column_name_and_type',
               'DataAugmentationGUI', # this one is buggy --> remove
               ]  # dialogs to test


    def test_dialogs(self):
        force_qtpy_to_use_user_specified()
        os.environ['QT_API']=self.UI
        from epyseg.ta.GUI.denoise_recursion_dialog import DenoiseRecursionDialog
        from epyseg.gui.preview import crop_or_preview
        from epyseg.gui.defineROI import DefineROI
        from epyseg.gui.img import image_input_settings
        from epyseg.gui.mini.miniGUI_selection_dialog import minisel
        from epyseg.gui.multi_inputs_img import Multiple_inputs
        from epyseg.gui.open import OpenFileOrFolderWidget
        from epyseg.postprocess.gui import PostProcessGUI
        from epyseg.ta.GUI.denoise_recursion_dialog import DenoiseRecursionDialog
        from epyseg.ta.GUI.finish_all_dialog import FinishAllDialog
        from epyseg.ta.GUI.input_text_dialog import QPlainTextInputDialog
        from epyseg.ta.GUI.tracking_static_dialog import TrackingDialog
        from epyseg.ta.GUI.watershed_dialog import WshedDialog
        from epyseg.ta.database.preview_and_edit_sqlite_db import Example # buggy
        from epyseg.ta.database.sql_column_name_dialog import SQL_column_name_and_type
        from epyseg.gui.augmenter import DataAugmentationGUI #creates a bug probably misses some parameters!!!!
        from qtpy.QtWidgets import QApplication
        app = QApplication(sys.argv)
        print_default_qtpy_UI_really_in_use()
        self.assertEqual(os.environ['QT_API'], self.UI)


        # debug
        # gui = self.dialogs[4] # 0,1,2 -> ok
        # w = locals()[gui]()
        # # w = DenoiseRecursionDialog()
        # w.show()
        # self.assertTrue(w.isVisible())

        #
        #
        for iii,gui in enumerate(self.dialogs):
            print(iii, gui)
            w = locals()[gui]()
            w.show()
            # result = w.exec_()
            self.assertTrue(w.isVisible())
            # if iii == 1:
            #     break
            # w.close()
        # # self.w.btn.click()
        #
        #
        # # self.assertTrue(DenoiseRecursionDialog.isVisible())
        #
        #     # if random.uniform(0, 1)<0.5:
        #     #     self.assertTrue(False)
        # self.assertEqual(os.environ['QT_API'], self.UI)

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
