#!/usr/bin/python
# -'''- coding: utf-8 -'''-

import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QPlainTextEdit, QVBoxLayout, QApplication, QDialogButtonBox, QSpinBox, QLabel, \
    QCheckBox


# shall return the cutoff for bond length and also the minimal cell area
# and later on maybe some additional parameters
class FinishAllDialog(QDialog):

    def __init__(self, parent=None, title=None):
        super(FinishAllDialog, self).__init__(parent)
        if title is not None:
            self.setWindowTitle(title)

        label = QLabel(
            'Bond cut-off\n(bonds below the chosen value,\nwill be considered as vertices,\nthis only affects the number of neighbors for a cell)')
        self.four_way_cutoff = QSpinBox()
        self.four_way_cutoff.setSingleStep(1)
        self.four_way_cutoff.setRange(1, 1000000)  # 1_000_000 makes no sense but anyway
        self.four_way_cutoff.setValue(2)
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.four_way_cutoff)

        label2 = QLabel(
            'Cell area cut-off\n(all cells with a cytoplsamic area in pixels\nbelow the chosen value will be removed from the segmentation mask\nNB: this is irreversible)')
        self.min_cell_area_spin = QSpinBox()
        self.min_cell_area_spin.setSingleStep(1)
        self.min_cell_area_spin.setRange(1, 1000000)  # 1_000_000 makes no sense but anyway
        self.min_cell_area_spin.setValue(10)
        layout.addWidget(label2)
        layout.addWidget(self.min_cell_area_spin)

        self.polarity_check = QCheckBox('Measure polarity (Slower)')
        self.polarity_check.setChecked(False)
        layout.addWidget(self.polarity_check)

        self.check_3D = QCheckBox('3D measurements (Slow and requires a "height_map.tif" file)')
        layout.addWidget(self.check_3D)

        # or keep MT and deactivate progress
        self.multi_threading = QCheckBox('Multi Threading Enabled (faster but more memory, not recommended work in progress)')
        self.multi_threading.setChecked(False)
        layout.addWidget(self.multi_threading)

        # OK and Cancel buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        # TODO add an ok or cancel button
        self.setLayout(layout)

    @staticmethod
    def get_values(parent=None, title=None):
        dialog = FinishAllDialog(parent=parent, title=title)
        result = dialog.exec_()
        four_way_cutoff = dialog.four_way_cutoff.value()
        cell_area_cutoff = dialog.min_cell_area_spin.value()
        measure_polarity = dialog.polarity_check.isChecked()
        measure_3D = dialog.check_3D.isChecked()
        multi_threading = dialog.multi_threading.isChecked()
        values = [cell_area_cutoff, four_way_cutoff, measure_polarity, measure_3D, multi_threading]
        return (values, result == QDialog.Accepted)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    parameters, ok = FinishAllDialog.get_values(title="Options")
    # form.show()
    # text, ok = app.exec_()
    # print(form.get_value())

    # text, ok = QInputDialog.getText(self, 'Text Input Dialog', 'Enter your SQl command:')
    if ok:
        # self.le1.setText(str(text))
        print(parameters)
    else:
        pass

    sys.exit(0)
