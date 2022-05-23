# will ultimately allow to load several inputs/outputs models
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QScrollArea, QVBoxLayout, QGroupBox, QPushButton
from epyseg.gui.open import OpenFileOrFolderWidget
import sys
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()

# I think this stuff is ready to be overridden by the other stuff
# could add a popping preview otherwise...

# for what I wanna do I really need this stuff to implement all the functions of the OpenFileOrFolderWidget Class

class Multiple_inputs(QWidget):

    def __init__(self, nb_of_inputs=1, generic_label='input#', parent_window=None, finalize_text_change_method_overrider=None):
        super().__init__(parent=parent_window)
        self.nb_of_inputs = nb_of_inputs
        self.generic_label =generic_label
        self.finalize_text_change_method_overrider = finalize_text_change_method_overrider
        self.initUI()

    def initUI(self):
        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.models_container = QWidget()

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)

        # self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.models = []
        # self.model_weights = []
        # add window to it --> TODO
        # self.input_model = OpenFileOrFolderWidget(parent_window=self, is_file=True,
        #                                           extensions="Supported Files (*.h5 *.H5 *.hdf5 *.HDF5 *.json *.JSON *.model);;All Files (*)",
        #                                           tip_text='Drag and drop a single model file here')
        # self.input_weights = OpenFileOrFolderWidget(parent_window=self, label_text='Load weights',
        #                                             is_file=True,
        #                                             extensions="Supported Files (*.h5 *.H5 *.hdf5 *.HDF5);;All Files (*)",
        #                                             tip_text='Drag and drop a single weight file here')  # TODO shall i add *.model ???
        #

        self.models_container.layout = QVBoxLayout()
        # self.models_container.layout.setAlignment(Qt.AlignTop)
        # self.models_container.layout.setColumnStretch(0, 25)
        # self.models_container.layout.setColumnStretch(1, 75)
        # self.models_container.layout.setHorizontalSpacing(3)
        # self.models_container.layout.setVerticalSpacing(3)

        for iii in range(self.nb_of_inputs):
            self.add_new_model(iii)
            # self.add_new_model()
        # self.add_new_model()
        # self.add_new_model()
        # self.add_new_model()
        # self.add_new_model()

        self.models_container.setLayout(self.models_container.layout)


        self.scrollArea.setWidget(self.models_container )

        # self.scrollArea.setGeometry(QRect(0, 0, self.prev_width, self.prev_height))
        # self.setGeometry(QRect(0, 0, self.prev_width, self.prev_height))
        # self.setFixedSize(self.size())

        layout.addWidget(self.scrollArea)
        # layout.addWidget(self.add_model_to_ensemble)
        # layout.addWidget(self.remove_model_to_ensemble)

        self.setLayout(layout)
        # self.setMinimumHeight(60)
        self.setFixedHeight(60)
        self.setMinimumWidth(400)




    #################################### in this block I reimplement all the methods of the Openfileorfolderwidget stuff

    # smartest would be to get the last modified model and update it --> see how this can be done...

    # weird command that returns the path to the first input that is gonna be used for the saving and also for the controls -> the idea is to mimmick a the behaviour of a single OpenFileOrFolderWidget object

    def get_items(self):
        if self.models is None:
            return None
        return [model.text() for model in self.models]


    # maybe all is ok this way
    def text(self):
        return self.models[0].text()

    # do override this to change the behaviour of inputs...
    # dangerous to override --> do it in a cleaner way
    # def finalize_text_change(self):
        # print('in', self.path.text())
        # pass

    def __get_last_modified_model(self):
        last_modified_time = 0
        last_modified_item = None
        for model in self.models:
            last_mod =model.get_time_of_last_change()
            if last_mod is not None:
                if last_mod>last_modified_time:
                    last_modified_time = last_mod
                    last_modified_item = model
        return last_modified_item


    def set_icon_ok(self, ok):
        model = self.__get_last_modified_model()
        if model is not None:
            model.set_icon_ok(ok)


        # self.models[0].set_icon_ok(ok)
        # if not self.show_ok_or_not_icon:
        #     return
        # if ok:
        #     self.ok_or_not_ico.setPixmap(self.ok_ico)
        # else:
        #     self.ok_or_not_ico.setPixmap(self.not_ok_ico)

    def set_size(self, size):
        model = self.__get_last_modified_model()
        # self.size_label.setText(size)
        if model is not None:
        # self.models[0].size_label.setText(size)
            model.size_label.setText(size)

    def get_list_using_glob(self):
        model = self.__get_last_modified_model()
        # self.size_label.setText(size)
        if model is not None:
            # self.models[0].size_label.setText(size)
            return model.get_list_using_glob()



    # def get_list_using_glob(self):
    #     pass

    #################################### end of the block I reimplement all the methods of the Openfileorfolderwidget stuff

    # this will be used to define the nb of inputs
    def set_nb_of_items(self, nb_items=1):
        # reset inputs
        self.models = []
        # empty layout then add new inputs
        try:
            for i in reversed(range(self.models_container.layout.count())):
                self.models_container.layout.itemAt(i).widget().setParent(None)
        except:
            pass
        for iii in range(nb_items):
            self.add_new_model(count=iii)

    def add_new_model(self, count=None):
        # self.groupBox_model = QGroupBox('Model '+str(len(self.models)), objectName='Model_'+str(len(self.models)))
        # self.groupBox_model.setEnabled(True)
        # groupBox layout
        self.groupBox_model_layout = QVBoxLayout()
        # self.groupBox_model_layout.setAlignment(Qt.AlignTop)
        # self.groupBox_model_layout.setColumnStretch(0, 25)
        # self.groupBox_model_layout.setColumnStretch(1, 25)
        # self.groupBox_model_layout.setColumnStretch(2, 50)
        # self.groupBox_model_layout.setColumnStretch(3, 2)
        # self.groupBox_model_layout.setHorizontalSpacing(3)
        # self.groupBox_model_layout.setVerticalSpacing(3)

        # add a groupbox per model

        # TODO maybe allow this to be stored and or reupdated --> not so easy would need to design specific stuff
        #TODO should I allow it to be saved in ini, most likely not...

        # add them the main checking function
        # difficulty is how to show preview of all models at once ???? --> Not so easy todo... --> do this later but ok for now


        # pb is that I miss all the features to update here...
        # shall I add the image on the side of the input maybe ??? --> maybe there is an idea there???
        # or find a trick

        # qqsdqdqsdqsd

        # do override the default method with the current one

        # EN FAIT C4EST VRAIMENT DANGEREUX --> FAIRE 9A PROPREMENT... --> DEFINE AN OVERRIDING METHOD THAT IS CALLED INSTEAD OF THE MAIN METHOD IF NOT NONE --> SUPER EASY AND CLEAN



        self.input_model = OpenFileOrFolderWidget(parent_window=self, add_timer_to_changetext=True,
                                                        show_ok_or_not_icon=True,  # label_text=label_input,
                                                        show_size=True,
                                                        tip_text='Drag and drop a single file or folder here',
                                                        objectName='open_input_button',
                                                        finalize_text_change_method_overrider=self.finalize_text_change_method_overrider
                                                        ) #objectName +'open_input_button'
        # self.input_weights = OpenFileOrFolderWidget(parent_window=self, label_text='Load weights (otpional)',
        #                                             is_file=True,
        #                                             extensions="Supported Files (*.h5 *.H5 *.hdf5 *.HDF5);;All Files (*)",
        #                                             tip_text='Drag and drop a single weight file here')  # TODO shall i add *.model ???


        self.models.append(self.input_model)
        # self.model_weights.append(self.input_weights)

        # add it to the layout too

        # self.groupBox_model_layout.addWidget(self.input_model)
        # self.groupBox_model_layout.addWidget(self.input_weights)

        # self.groupBox_model.setLayout(self.groupBox_model_layout)



        self.models_container.layout.addWidget(self.input_model)


    def remove_last_model(self):
        # for i in range(self.models_container.layout.count()):
        #     print(i)
        # for i in reversed(range(models_container.count())):
        try:
            self.models_container.layout.itemAt( self.models_container.layout.count()-1).widget().setParent(None)
        except:
            pass
        # self.models_container.layout.itemAt(self.models_container.layout.count()-1).widget().setParent(None)

        if self.models:
            self.models.pop()

    def get_models(self):
        # loop of other model lists
        # else return path to models and weights
        # models = {}
        # for iii,model in enumerate(self.models):
        #     if model.text() is not None:
        #         models_and_weights[iii]=[model.text(), self.model_weights[iii].text()]
        # if not models_and_weights:
        #     logger.error('No model loaded --> nothing can be done, sorry...')
        #     return None
        models = []
        for iii, model in enumerate(self.models):
            models.append(model.text())
        return models


if __name__ == '__main__':
    # just for a test

    def printous():
        print('heeloow')

    app = QApplication(sys.argv)
    ex = Multiple_inputs(nb_of_inputs=3, finalize_text_change_method_overrider=printous)
    ex.show()
    app.exec_()

    # that seems to work
    print(ex.get_models())
