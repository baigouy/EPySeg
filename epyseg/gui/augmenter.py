import json

from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QGridLayout, QComboBox, QLabel, \
    QDoubleSpinBox, QDialog, QDialogButtonBox, QPushButton, QToolTip

from epyseg.deeplearning.augmentation.generators.data import DataGenerator
from epyseg.deeplearning.docs.doc2html import markdown_file_to_html
from epyseg.dialogs.opensave import openDirectoryDialog
import sys
from PyQt5 import QtCore


class DataAugmentationGUI(QDialog):

    def __init__(self, parent_window=None):
        super().__init__(parent=parent_window)
        self.initUI()

    def initUI(self):
        layout = QGridLayout()
        layout.setColumnStretch(0, 90)
        layout.setColumnStretch(1, 10)
        layout.setColumnStretch(2, 2)
        layout.setHorizontalSpacing(3)
        layout.setVerticalSpacing(3)

        types_of_augmentations = list(DataGenerator.augmentation_types_and_ranges.keys())

        normalization_type_label = QLabel('type (None = raw data)')
        layout.addWidget(normalization_type_label, 0, 0)

        self.augmentation = QComboBox(objectName='augmentation')
        for type in types_of_augmentations:
            self.augmentation.addItem(type)

        self.augmentation.currentTextChanged.connect(self.on_combobox_changed)

        layout.addWidget(self.augmentation, 0, 1)

        self.rate_label = QLabel('range/rate')
        self.rate_label.setEnabled(False)
        layout.addWidget(self.rate_label, 1, 0)

        self.rate_spin = QDoubleSpinBox()

        self.rate_spin.setRange(0., 1.)
        self.rate_spin.setSingleStep(0.01)
        self.rate_spin.setValue(0.15)
        self.rate_spin.setEnabled(False)

        layout.addWidget(self.rate_spin, 1, 1)

        # help for pre-processing
        # help_ico = QIcon.fromTheme('help-contents')
        self.help_button_in_data_aug = QPushButton('?', None)
        bt_width = self.help_button_in_data_aug.fontMetrics().boundingRect(self.help_button_in_data_aug.text()).width() + 7
        self.help_button_in_data_aug.setMaximumWidth(bt_width * 2)
        self.help_button_in_data_aug.clicked.connect(self.show_tip)

        layout.addWidget(self.help_button_in_data_aug, 0, 2, 2, 1)

        # OK and Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, 2,0,1,3)

        self.setLayout(layout)

    def show_tip(self):
        QToolTip.showText(self.sender().mapToGlobal(QPoint(30, 20)), markdown_file_to_html('data_augmentation.md'))
        # QToolTip.showText(self.sender().mapToGlobal(QPoint(30, 20)), "unknown button")

    def on_combobox_changed(self):
        default_value = DataGenerator.augmentation_types_and_ranges[self.augmentation.currentText()]
        show = default_value is not None
        self.rate_label.setEnabled(show)
        self.rate_spin.setEnabled(show)
        if show:
            self.rate_spin.setRange(default_value[0], default_value[1])
            self.rate_spin.setValue(default_value[2])

    def open_folder(self):
        self.output_folder = openDirectoryDialog(parent_window=self.parent_window)
        if self.output_folder is not None:
            self.path.setText(self.output_folder)

    def text(self):
        return self.path.text()

    def augment(self):
        type = self.augmentation.currentText()
        if type == 'None':
            return json.dumps({'type': None})
        else:
            if not self.rate_spin.isEnabled():
                return json.dumps({'type': type})
            # no need to send value if 0
            if self.rate_spin.value() == 0:
                return json.dumps({'type': None})
            return json.dumps({'type':type, 'value':self.rate_spin.value()})

    @staticmethod
    def getAugmentation(parent=None):
        dialog = DataAugmentationGUI(parent)
        result = dialog.exec_()
        augment = dialog.augment()
        return (augment, result == QDialog.Accepted)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    augment, ok = DataAugmentationGUI.getAugmentation()
    print(augment, ok)
    sys.exit(0)
