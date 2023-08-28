import os
from epyseg.settings.global_settings import set_UI  # set the UI to be used by qtpy
set_UI()
import sys
from qtpy import QtCore
from qtpy.QtWidgets import QDialog, QPlainTextEdit, QVBoxLayout, QApplication, QDialogButtonBox, QSpinBox, QLabel, \
    QLineEdit, QComboBox
import re

class SQL_column_name_and_type(QDialog):
    def __init__(self, parent=None, title=None, existing_column_name=None):
        super(SQL_column_name_and_type, self).__init__(parent)
        if title is not None:
            self.setWindowTitle(title)

        self.existing_column_name = existing_column_name
        if isinstance(self.existing_column_name, list):
            self.existing_column_name = [name.lower() for name in self.existing_column_name]

        layout = QVBoxLayout()

        label = QLabel('Please enter a column name that is not in the table.\nThe name should contain no space')
        layout.addWidget(label)

        self.column_name = QLineEdit()
        self.column_name.textChanged.connect(self.column_name_changed)
        layout.addWidget(self.column_name)

        self.type = QComboBox()
        self.type.addItem('REAL')
        self.type.addItem('INTEGER')
        self.type.addItem('TEXT')
        self.type.addItem('BLOB')
        layout.addWidget(self.type)

        # OK and Cancel buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def column_name_changed(self):
        """
        Handle the column name text change event.
        Enables or disables the OK button based on the validity of the column name.

        """
        if self.is_text_valid():
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        else:
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def is_text_valid(self):
        """
        Check if the entered column name is valid.
        The name should not exist in the existing column names and should adhere to the SQL column name rules.

        Returns:
            bool: True if the column name is valid, False otherwise.

        """
        if self.existing_column_name is not None:
            if self.column_name.text().lower() in self.existing_column_name:
                return False

        if re.findall(r'^[a-zA-Z_][a-zA-Z0-9_]*$', self.column_name.text()):
            return True
        return False

    @staticmethod
    def get_value(parent=None, title=None, existing_column_name=None):
        """
        Static method to display the SQL column name and type dialog and retrieve the entered values.

        Args:
            parent (QWidget): The parent widget.
            title (str): The title of the dialog.
            existing_column_name (list): A list of existing column names.

        Returns:
            tuple: A tuple containing the entered column name and selected type, and a boolean indicating if OK was clicked.

        """
        dialog = SQL_column_name_and_type(parent=parent, title=title, existing_column_name=existing_column_name)
        result = dialog.exec_()
        values = [dialog.column_name.text(), dialog.type.currentText()]
        return (values, result == QDialog.Accepted)


if __name__ == '__main__':
    if False:
        # Code for testing column name validity
        import sys

        txt = 'this_is_a_test'

        r1 = re.findall(r'^[a-zA-Z_][a-zA-Z0-9_]*$', txt)
        if r1:
            print(True)
        else:
            print(False)
        print(r1)

        sys.exit(0)

    app = QApplication(sys.argv)
    colnmame_n_type, ok = SQL_column_name_and_type.get_value(title="New Column Name", existing_column_name=['time'])

    if ok:
        print(colnmame_n_type)
    else:
        pass

    sys.exit(0)
