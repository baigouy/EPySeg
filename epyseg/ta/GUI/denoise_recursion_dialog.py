#!/usr/bin/python
# -'''- coding: utf-8 -'''-

import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QPlainTextEdit, QVBoxLayout, QApplication, QDialogButtonBox, QSpinBox, QLabel


class DenoiseRecursionDialog(QDialog):

    def __init__(self, parent=None, title=None):
        super(DenoiseRecursionDialog, self).__init__(parent)
        if title is not None:
            self.setWindowTitle(title)

        label = QLabel('How many times should the denoiser module be run on the image?\n(Ideally keep this number to 1 and increase only when the output is noisy)\n(typically 4 or 5 is a good value for very noisy images)')
        self.recursion_spinner = QSpinBox()
        self.recursion_spinner.setSingleStep(1)
        self.recursion_spinner.setRange(1, 20)  # 1_000_000 makes no sense but anyway
        self.recursion_spinner.setValue(1)
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.recursion_spinner)
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
    def get_value(parent=None, title=None):
        dialog = DenoiseRecursionDialog(parent=parent, title=title)
        result = dialog.exec_()
        values = dialog.recursion_spinner.value()
        return (values, result == QDialog.Accepted)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    recursion, ok = DenoiseRecursionDialog.get_value(title="Denoiser options")
    # form.show()
    # text, ok = app.exec_()
    # print(form.get_value())

    # text, ok = QInputDialog.getText(self, 'Text Input Dialog', 'Enter your SQl command:')

    if ok:
        # self.le1.setText(str(text))
        print(recursion)
    else:
        pass

    sys.exit(0)