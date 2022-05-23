from epyseg.ta.GUI.paint2 import Createpaintwidget
from epyseg.ta.GUI.scrollablepaint import scrollable_paint

# this is the perfect TA like drawing area --> just need replace it
class tascrollablepaint(scrollable_paint):
    def __init__(self):
        class overriding_apply(Createpaintwidget):
            # all seems ok now and functions as in TA
            # just do the shift enter to get rid of small cells

            # avec ça ça marche 100% à la TA ... --> cool --> maybe make it a TA drawing pad
            def apply(self):
                self.apply_drawing(minimal_cell_size=0)

            def shift_apply(self):
                # MEGA TODO IMPLEMENT SIZE within this stuff!!!
                self.apply_drawing(minimal_cell_size=10)

            def ctrl_m_apply(self):
                self.manually_reseeded_wshed() # how can I pass the channel to the stuff ???
                # channel = self.get_selected_channel()

            def m_apply(self):
                self.maskVisible = not self.maskVisible
                self.update()

            def save(self):
                self.save_mask()

        super().__init__(custom_paint_panel=overriding_apply())


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
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
    w.set_image('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif') # nb if alone it is causing a crash --> why # --> incomprehensible why bugs sometimes
    # w.set_mask('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif')


    w.show()
    sys.exit(app.exec_())


