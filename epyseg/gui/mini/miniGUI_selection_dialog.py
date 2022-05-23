import traceback
import logging
from itertools import zip_longest
from epyseg.deeplearning.docs.doc2html import markdown_file_to_html, browse_tip
from epyseg.gui.defineROI import DefineROI
from epyseg.utils.loadlist import loadlist
from epyseg.postprocess.gui import PostProcessGUI
from epyseg.uitools.blinker import Blinker
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QPushButton, QToolTip, QHBoxLayout, QFrame
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QTabWidget
from PyQt5 import QtCore
from epyseg.deeplearning.augmentation.generators.data import DataGenerator
from epyseg.gui.open import OpenFileOrFolderWidget
from epyseg.gui.preview import crop_or_preview
from PyQt5.QtWidgets import QSpinBox, QComboBox, QVBoxLayout, QLabel, QCheckBox, QRadioButton, QButtonGroup, QGroupBox, \
    QDoubleSpinBox
from PyQt5.QtCore import Qt, QPoint
from epyseg.img import Img
import sys
import json
import os
from epyseg.gui.multi_inputs_img import Multiple_inputs
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()

# TODO set a min size for this stuff so that size fits in smaller windows
class minisel(QDialog):

    def __init__(self, parent_window=None):
        super().__init__(parent=parent_window)
        # d = QDialog()

        # ask whether images are to be used in combination with TA or not
        self.auto_output = QRadioButton('Minimal interface (just use pretrained EPySeg, CARE, ... models)',
                                   objectName='auto_output')
        self.custom_output = QRadioButton('Advanced interface (to train a model or use cutsom parameters for prediction)',
                                     objectName='custom_output')
        predict_output_radio_group = QButtonGroup()
        # predict_output_radio_group.buttonClicked.connect(self.predict_output_mode_changed)

        # default is build a new model
        self.auto_output.setChecked(True)

        # connect radio to output_predictions_to text
        self.auto_output.toggled.connect(self.predict_output_mode_changed)
        self.custom_output.toggled.connect(self.predict_output_mode_changed)

        predict_output_radio_group.addButton(self.auto_output)
        predict_output_radio_group.addButton(self.custom_output)

        # b1 = QPushButton("ok", d)
        # b1.move(50, 50)
        self.setWindowTitle("Dialog")
        # self.setWindowModality(QtCore.Qt.ApplicationModal)

        layout = QGridLayout()

        layout.addWidget(self.auto_output)
        layout.addWidget(self.custom_output)

        line_sep_predict = QFrame()
        line_sep_predict.setFrameShape(QFrame.HLine)
        line_sep_predict.setFrameShadow(QFrame.Sunken)

        layout.addWidget(line_sep_predict)

        self.setLayout(layout)

        # OK and Cancel buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                     QtCore.Qt.Horizontal, self)
        self.buttons.accepted.connect(self. accept)
        self.buttons.rejected.connect(self.reject)
        self.layout().addWidget(self.buttons)

    def get_parameters(self):
        return self.auto_output.isChecked()

    @staticmethod
    def getDataAndParameters(parent_window=None):  #
        # get all the params for augmentation
        dialog = minisel(parent_window=parent_window)
        result = dialog.exec_()
        augment = dialog.get_parameters()
        return (augment, result == QDialog.Accepted)

    def predict_output_mode_changed(self):
        # print('inside')
        # print(self.sender())
        # TODO if the model requires several inputs then update the nb of inputs --> TODO
        pass


if __name__ == '__main__':
    # just for a test

    app = QApplication(sys.argv)
    augment, ok = minisel.getDataAndParameters(parent_window=None)

    print(augment, ok)

    if ok:
        print("continue")
    else:
        print('abort')


    sys.exit(0)
