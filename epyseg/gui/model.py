# a small GUI to load an ensemble/a series of models

from PyQt5.QtCore import QRect, Qt, QRectF
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QScrollArea, QVBoxLayout, QGroupBox, QPushButton
from epyseg.gui.open import OpenFileOrFolderWidget
import sys
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()

class Ensemble_Models_Loader(QWidget):

    def __init__(self, parent_window=None):
        super().__init__(parent=parent_window)
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
        self.model_weights = []
        # add window to it --> TODO
        # self.input_model = OpenFileOrFolderWidget(parent_window=self, is_file=True,
        #                                           extensions="Supported Files (*.h5 *.H5 *.hdf5 *.HDF5 *.json *.JSON *.model);;All Files (*)",
        #                                           tip_text='Drag and drop a single model file here')
        # self.input_weights = OpenFileOrFolderWidget(parent_window=self, label_text='Load weights',
        #                                             is_file=True,
        #                                             extensions="Supported Files (*.h5 *.H5 *.hdf5 *.HDF5);;All Files (*)",
        #                                             tip_text='Drag and drop a single weight file here')  # TODO shall i add *.model ???
        #

        self.add_model_to_ensemble = QPushButton('+')
        self.add_model_to_ensemble.clicked.connect(self.add_new_model)

        self.remove_model_to_ensemble = QPushButton('-')
        self.remove_model_to_ensemble.clicked.connect(self.remove_last_model)


        self.models_container.layout = QVBoxLayout()
        # self.models_container.layout.setAlignment(Qt.AlignTop)
        # self.models_container.layout.setColumnStretch(0, 25)
        # self.models_container.layout.setColumnStretch(1, 75)
        # self.models_container.layout.setHorizontalSpacing(3)
        # self.models_container.layout.setVerticalSpacing(3)

        self.add_new_model()
        self.add_new_model()
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
        layout.addWidget(self.add_model_to_ensemble)
        layout.addWidget(self.remove_model_to_ensemble)

        self.setLayout(layout)
        self.setFixedHeight(200)

    def add_new_model(self):
        self.groupBox_model = QGroupBox('Model '+str(len(self.models)), objectName='Model_'+str(len(self.models)))
        self.groupBox_model.setEnabled(True)
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
        self.input_model = OpenFileOrFolderWidget(parent_window=self,  label_text='Load model',is_file=True,
                                                  extensions="Supported Files (*.h5 *.H5 *.hdf5 *.HDF5 *.json *.JSON *.model);;All Files (*)",
                                                  tip_text='Drag and drop a single model file here')
        self.input_weights = OpenFileOrFolderWidget(parent_window=self, label_text='Load weights (otpional)',
                                                    is_file=True,
                                                    extensions="Supported Files (*.h5 *.H5 *.hdf5 *.HDF5);;All Files (*)",
                                                    tip_text='Drag and drop a single weight file here')  # TODO shall i add *.model ???


        self.models.append(self.input_model)
        self.model_weights.append(self.input_weights)

        # add it to the layout too

        self.groupBox_model_layout.addWidget(self.input_model)
        self.groupBox_model_layout.addWidget(self.input_weights)

        self.groupBox_model.setLayout(self.groupBox_model_layout)

        self.models_container.layout.addWidget(self.groupBox_model)


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
        if self.model_weights:
            self.model_weights.pop()

    def get_models(self):
        # loop of other model lists
        if not self.model_weights:
            logger.error('No model --> nothing can be done, sorry...')
            return None
        # else return path to models and weights
        models_and_weights = {}
        for iii,model in enumerate(self.models):
            if model.text() is not None:
                models_and_weights[iii]=[model.text(), self.model_weights[iii].text()]
        if not models_and_weights:
            logger.error('No model loaded --> nothing can be done, sorry...')
            return None
        return models_and_weights


if __name__ == '__main__':
    # just for a test
    app = QApplication(sys.argv)
    ex = Ensemble_Models_Loader()
    ex.show()
    app.exec_()

    # that seems to work
    print(ex.get_models())
