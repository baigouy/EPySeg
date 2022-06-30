# Here I will gather all the unit tests for GUIs
# make sure all runs with pyside, pyside2, pyqt5 and pyqt6
import gc
# import os
import unittest
import sys

from epyseg.settings.global_settings import print_default_qtpy_UI_really_in_use, list_UIs, set_UI


class TestGUI(unittest.TestCase):

    # ça marche pas --> voir comment faire !!!
    # NOW that seems to work...
    # @unittest.skip("skipping")
    # def test_pyTA(self):
    #     # tmp = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png')
    #     # self.assertIsInstance(tmp, Img)
    #     # self.assertIsInstance(tmp, np.ndarray)
    #     # self.assertEqual(tmp.get_dimensions_as_string(),'hwc')
    #     # self.assertEqual(tmp.get_dimensions_as_string(),'dhwc')
    #     # import multiprocessing
    #     # import platform
    #
    #     # NB I could use that to test several APIs --> TODO
    #     print_default_qtpy_UI_really_in_use()
    #     # could even loop over all environments --> à tester...
    #     lst_UIs = list_UIs()
    #     print(lst_UIs)
    #     #
    #     for UI in lst_UIs:
    #         # need a subtest to have a loop # https://stackoverflow.com/questions/26458634/for-loop-in-unittest
    #         with self.subTest(line=UI):
    #             os.environ['QT_API'] = UI
    #
    #             print('testing UI', UI)
    #
    #             from epyseg.ta.GUI.pyta import TissueAnalyzer
    #             from qtpy.QtWidgets import QApplication
    #             app = QApplication(sys.argv)
    #
    #             print_default_qtpy_UI_really_in_use()
    #
    #             w = TissueAnalyzer()
    #             w.show()
    #             # w.close()
    #             if w.isVisible():
    #                 # assume if visible everything ran ok
    #                 self.assertTrue(True)
    #                 # sys.exit(0)
    #                 # return
    #             else:
    #                 self.assertFalse(True)
    #
    #             del app
    #             gc.collect()
    #             self.assertTrue(True)
    #     # do the same for all other GUIs

    # def test_pyTA_pyqt6(self):
    #     os.environ['QT_API'] = 'pyqt6'
    #     print('testing UI',  os.environ['QT_API'])
    #     self.tst_pyTA()

    def tst_pyTA_pyqt5(self):
        self.test_pyTA_pyqt5()

    def test_pyTA_pyqt5(self):

        # HUGE BUG --> ALWAYS DEFAULTING TO pyQT6 --> forget about that test --> do it manually outside

        # why is qt6 used instead of QT 5 I really don't get it
        import os
        import sys
        os.environ['QT_API'] = 'pyqt5'
        print(os.environ['QT_API'])
        print('testing UI',  os.environ['QT_API'])
        # somehow it is changed
        # set_UI()
        # print_default_qtpy_UI_really_in_use() # --> fucks it
        print('vsss',os.environ['QT_API'])
        # maybe due to some wird default settings of qtpy --> forget for now ---> just test things manually
        from epyseg.ta.GUI.pyta import TissueAnalyzer # --> this force changes to pyQT6 here but why ???? --> works well outside of tests but not from inside --> really no clue why ???
        print('vsss2', os.environ['QT_API'])
        from qtpy.QtWidgets import QApplication
        app = QApplication(sys.argv)
        # bug but why is that
        print_default_qtpy_UI_really_in_use()
        w = TissueAnalyzer()
        w.show()
        if w.isVisible():
            QApplication.exit()
            self.assertTrue(True)
        else:
            QApplication.exit()
            self.assertFalse(True)

    # def test_pyTA_pyside2(self):
    #     os.environ['QT_API'] = 'pyside2'
    #     print('testing UI',  os.environ['QT_API'])
    #     self.tst_pyTA()

    # def test_pyTA_pyside6(self):
    #     os.environ['QT_API'] = 'pyside6'
    #     print('testing UI',  os.environ['QT_API'])
    #     self.tst_pyTA()

    # def tst_pyTA(self):
    #     from epyseg.ta.GUI.pyta import TissueAnalyzer
    #     from qtpy.QtWidgets import QApplication
    #     app = QApplication(sys.argv)
    #     print_default_qtpy_UI_really_in_use()
    #     w = TissueAnalyzer()
    #     w.show()
    #     if w.isVisible():
    #         QApplication.exit()
    #         self.assertTrue(True)
    #     else:
    #         QApplication.exit()
    #         self.assertFalse(True)


if __name__ == '__main__':
    unittest.main()
