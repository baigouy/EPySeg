from epyseg.ta.GUI.paint2 import Createpaintwidget
from epyseg.ta.GUI.scrollablepaint import scrollable_paint

# this is the perfect TA like drawing area --> just need replace it
class scrollablepaint_multichannel_edit(scrollable_paint):
    def __init__(self, force_nb_of_channels=None):

        self.force_nb_of_channels = force_nb_of_channels

        class overriding_apply(Createpaintwidget):
            # all seems ok now and functions as in TA
            # just do the shift enter to get rid of small cells

            # avec ça ça marche 100% à la TA ... --> cool --> maybe make it a TA drawing pad
            def apply(self):
                # do nothing or add current mask to the stuff ??? ...
                # self.apply_drawing(minimal_cell_size=0)
                # add drawing on top of current channel
                self.edit_drawing()
                return

            def shift_apply(self):
                # do nothing or add current mask to the stuff ??? ...
                # self.apply_drawing(minimal_cell_size=10)
                self.edit_drawing()
                return

            def ctrl_m_apply(self):
                # self.manually_reseeded_wshed() # how can I pass the channel to the stuff ???
                # channel = self.get_selected_channel()
                pass

            def m_apply(self):
                self.maskVisible = not self.maskVisible
                self.update()

            def save(self):
                self.edit_drawing()
                self.save_mask(multichannel_save=True, forced_nb_of_channels=force_nb_of_channels)
                # print(self.channel)
                # ça marche en fait --> juste hacker un peu ça
                # load the save image and replace the channel of interest ???
                # sqqsdsqdsqd


            # def channelChange(self, i):
            #     print('updated one')
            #     # self.channelChange(i)
            #
            #     print('chan', self.channel)
            # #     # self.paint.channelChange(i)
            #     # try run a multichannel change

        super().__init__(custom_paint_panel=overriding_apply())


    def _update_channels(self, img):
        # print('boso')
        selection = self.channels.currentIndex()
        self.channels.disconnect()
        self.channels.clear()
        comboData = ['merge']

        # hack to force the nb of channels
        if self.force_nb_of_channels is not None and self.force_nb_of_channels>0:
            for i in range(self.force_nb_of_channels):
                comboData.append(str(i))
        elif img is not None:
            try:
                if img.has_c():
                    for i in range(img.get_dimension('c')):
                        comboData.append(str(i))
            except:
                # assume image has no channel and return None
                # or assume last stuff is channel if image has more than 2
                pass
        # logger.debug('channels found ' + str(comboData))
        self.channels.addItems(comboData)
        # index = self.channels.findData(selection)

        self.channels.currentIndexChanged.connect(self.channelChange)
        # print("data", index)
        if selection != -1 and selection < self.channels.count():
            self.channels.setCurrentIndex(selection)
            # self.channelChange(selection)
        else:
            self.channels.setCurrentIndex(0)
            # self.channelChange(0)

    # def set_mask(self, mask):
    #     # force 0 1 float masks to be binarized --> good idea --> in fact
    #     self.paint.set_mask(mask, auto_convert_float_to_binary=True)


if __name__ == '__main__':
    # what needs be changed --> save should only save the current channel in save
    # when channel is changed I need load just the appropriate channel
    # need design a multi-mask mode that is not enabled by default !!!

    # maybe the only thing I miss is an undo ???
    # if click and shortcut --> then allow erase also for example --> useful with a pen
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


    # pb is the image is single channel whereas the mask is 4 --> how can I hack to force n channels even if the image does not have as many
    # or hack the image to fake the nb of channels to match --> ok but a waste of resources
    # force n channels
    nb_of_channels_of_deep_learning_model = 5


    # list = ['/E/Sample_images/sample_images_PA/test_complete_wing_raphael/Optimized_projection_018 (copie).png', '/E/Sample_images/sample_images_PA/test_complete_wing_raphael/Optimized_projection_018.png']

    # for img in list:


    w = scrollablepaint_multichannel_edit(force_nb_of_channels=nb_of_channels_of_deep_learning_model)  # ça marche --> permet de mettre des paint panels avec des proprietes particulieres --> assez facile en fait
    w.paint.multichannel_mode = True  # activate multichannel edit...
    # w.set_image('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/Optimized_projection_018.png')
    # w.set_image('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/Optimized_projection_018 (copie).png')
    # w.set_image(img)
    # w.set_mask('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/Optimized_projection_018/handCorrection.tif')
    # w.set_image('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif') # nb if alone it is causing a crash --> why # --> incomprehensible why bugs sometimes
    # w.set_mask('/E/Sample_images/sample_images_PA/test_complete_wing_raphael/neo_mask.tif')


    # --> this made with a list would allow me to edit GT
    # ar can I loop that over a list...



    w.show()


    sys.exit(app.exec_())


