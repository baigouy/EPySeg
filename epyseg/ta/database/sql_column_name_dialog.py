# enter a SQL col name --> if name is not valid then forget about it...
# maybe also need enter the type of the new column --> ????


import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QPlainTextEdit, QVBoxLayout, QApplication, QDialogButtonBox, QSpinBox, QLabel, \
    QLineEdit, QComboBox
import re

# NULL. The value is a NULL value.
# INTEGER. The value is a signed integer, stored in 1, 2, 3, 4, 6, or 8 bytes depending on the magnitude of the value.
# REAL. The value is a floating point value, stored as an 8-byte IEEE floating point number.
# TEXT. The value is a text string, stored using the database encoding (UTF-8, UTF-16BE or UTF-16LE).
# BLOB

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

        self.buttons.button( QDialogButtonBox.Ok).setEnabled(False)

        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def column_name_changed(self):
        if self.is_text_valid():
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        else:
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)


    # TODO --> need also check that the column does not exist...
    def is_text_valid(self):
        # print(self.column_name.text().isalnum()) # not good because excludes _

        # print(self.column_name.text().lower(), self.existing_column_name, self.column_name.text().lower() in self.existing_column_name)

        if self.existing_column_name is not None:
            if self.column_name.text().lower() in self.existing_column_name:
                return False

        # this is a regex to check whether the column name is ok
        if re.findall(r'^[a-zA-Z_][a-zA-Z0-9_]*$', self.column_name.text()):
            return True
        return False

    @staticmethod
    def get_value(parent=None, title=None, existing_column_name=None):
        dialog = SQL_column_name_and_type(parent=parent, title=title, existing_column_name=existing_column_name)
        result = dialog.exec_()
        values = [dialog.column_name.text(), dialog.type.currentText()]
        return (values, result == QDialog.Accepted)

if __name__ == '__main__':

    if False:
        import sys

        # txt = 'thuis is a test'
        txt = 'thuis_is_a_test'
        # txt = 'thuis_is.a_test'
        # txt = 'thuis_is a_test'
        # txt = 'thuis_is99a_test'
        # txt = ' '
        # txt = ''
        # if ''.join(e for e in string if e.isalnum())
        # if txt.isalnum():
        #     print(True)
        # else:
        #     print(False)

        # r1 = re.findall(r'[^\s]+',txt)
        # regex for valid sql column name
        r1 = re.findall(r'^[a-zA-Z_][a-zA-Z0-9_]*$',txt)
        if r1:
            print(True)
        else:
            print(False)
        print(r1)


        # p = re.compile(r'[^\s]+')
        # p = re.compile('\S+')
        # p = re.compile(r'[A-Za-z0-9 _.,!"/$]*')
        # print(p.match('thuis is a test'))
        # if p.match('thuis is a test'):
        #     print(True)
        # else:
        #     print(False)


        sys.exit(0)


    app = QApplication(sys.argv)
    colnmame_n_type, ok = SQL_column_name_and_type.get_value(title="New Column Name", existing_column_name=['time'])
    # form.show()
    # text, ok = app.exec_()
    # print(form.get_value())

    # text, ok = QInputDialog.getText(self, 'Text Input Dialog', 'Enter your SQl command:')

    if ok:
        # self.le1.setText(str(text))
        print(colnmame_n_type)
    else:
        pass

    sys.exit(0)