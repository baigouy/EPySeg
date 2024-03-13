from epyseg.ta.GUI.scrollablepaint import scrollable_paint
from epyseg.ta.GUI.paint2 import Createpaintwidget
from epyseg.settings.global_settings import set_UI
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QIcon, QPixmap
from qtpy.QtCore import Qt
import sys

class tascrollablepaint(scrollable_paint):
    """
    A customized version of the scrollable_paint class for TA-like drawing area.

    Args:
        overriding_paint_widget (Createpaintwidget, optional): Custom paint widget to override the default behavior. Defaults to None.
    """

    def __init__(self, overriding_paint_widget=None):
        if overriding_paint_widget is None:
            class overriding_apply(Createpaintwidget):
                """
                Custom paint widget with TA-like functions.
                """

                def apply(self):
                    """
                    Apply the drawing with a minimal cell size of 0.
                    """
                    self.apply_drawing(minimal_cell_size=0)

                def shift_apply(self):
                    """
                    Apply the drawing with a minimal cell size of 10.
                    """
                    self.apply_drawing(minimal_cell_size=10)

                def ctrl_m_apply(self):
                    """
                    Manually reseed the watershed with the selected channel.
                    """
                    self.manually_reseeded_wshed()

                def m_apply(self):
                    """
                    Toggle the visibility of the mask.
                    """
                    self.maskVisible = not self.maskVisible
                    self.update()

                def save(self):
                    """
                    Save the mask.
                    """
                    self.save_mask()

            super().__init__(custom_paint_panel=overriding_apply())
        else:
            super().__init__(custom_paint_panel=overriding_paint_widget)


if __name__ == '__main__':
    import os
    from epyseg.settings.global_settings import set_UI  # set the UI to be used py qtpy
    set_UI()
    import sys
    from qtpy.QtWidgets import QApplication
    if False:
        app = QApplication(sys.argv)
        qimage = toQimage(Img('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif'))
        # seg fault here
        print('before seg fault')
        icon = QIcon(QPixmap.fromImage(qimage))  # b
        print(qimage, icon)
        sys.exit(0)


    # TODO add a main method so it can be called directly
    # maybe just show a canvas and give it interesting props --> TODO --> really need fix that too!!!

    app = QApplication(sys.argv)

    w = tascrollablepaint()  # ça marche --> permet de mettre des paint panels avec des proprietes particulieres --> assez facile en fait
    # w.set_image('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/Optimized_projection_018.png')
    w.set_image('/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/210219/predict/predict_model_nb_0/210219.lif_t000.tif')
    w.set_mask('/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/210219/predict/predict_model_nb_0/210219.lif_t000/handCorrection.tif')
    # w.set_image('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif') # nb if alone it is causing a crash --> why # --> incomprehensible why bugs sometimes
    # w.set_mask('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif')


    w.show()
    sys.exit(app.exec_())


