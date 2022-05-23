#!/usr/bin/python
# -'''- coding: utf-8 -'''-

import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QPlainTextEdit, QVBoxLayout, QApplication, QDialogButtonBox


class QPlainTextInputDialog(QDialog):

    def __init__(self, parent=None, default_text=None, title=None):
        super(QPlainTextInputDialog, self).__init__(parent)
        if default_text is None:
            default_text = ''
        if title is not None:
            self.setWindowTitle(title)

        self.edit = QPlainTextEdit(default_text)
        layout = QVBoxLayout()
        layout.addWidget(self.edit)
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
    def get_text(parent=None, default_text=None, title=None):
        dialog = QPlainTextInputDialog(parent=parent,default_text=default_text, title=title)
        result = dialog.exec_()
        values = dialog.edit.toPlainText()
        return (values, result == QDialog.Accepted)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    text, ok = QPlainTextInputDialog.get_text(default_text="SELECT * FROM vertices_2D",title="Please type an SQL command")
    # form.show()
    # text, ok = app.exec_()
    # print(form.get_text())

    # text, ok = QInputDialog.getText(self, 'Text Input Dialog', 'Enter your SQl command:')

    if ok:
        # self.le1.setText(str(text))
        print(text)
    else:
        pass

    sys.exit(0)