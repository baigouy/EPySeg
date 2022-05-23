import sys


#https://doc.qt.io/qtforpython/PySide2/QtWidgets/QDialogButtonBox.html
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QPushButton, QWidget, QHBoxLayout, QLabel, \
    QDoubleSpinBox, QCheckBox, QSpinBox, QApplication


class TrackingDialog(QDialog):

    def __init__(self, parent=None):
        super(TrackingDialog, self).__init__(parent)

        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self.setupUI()
        self.setWindowTitle('Tracking options')

        # OK and Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

    def setupUI(self):
        main_panel = QWidget()
        main_panel.setLayout(QVBoxLayout())
        # main_panel.layout().addWidget(QLabel('Weak Blur'))

        self.b1 = QCheckBox("Recursive error removal (loses less cells but algorithm becomes much slower)") # NB I may also ask for the nb of recursions
        self.b1.setChecked(False)
        main_panel.layout().addWidget(self.b1)

        self.c1 = QCheckBox("Warp using mermaid maps if available (tuto coming soon)")
        self.c1.setChecked(True)
        main_panel.layout().addWidget(self.c1)

        # do it by default as it makes a lot of sense
        # self.d1 = QCheckBox("Pre register")
        # self.d1.setChecked(True)
        # main_panel.layout().addWidget(self.d1)

        self.layout().addWidget(main_panel)

    # get user values for cropping the image
    def values(self):
        return (self.b1.isChecked(), self.c1.isChecked())

    # static method to create the dialog and return (values, accepted)
    @staticmethod
    def getValues(parent=None, preview_enabled=False):
        dialog = TrackingDialog(parent=parent)
        result = dialog.exec_()
        values = dialog.values()
        return (values, result)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    values, ok = TrackingDialog.getValues()

    print(ok)

    if ok:
        print('just do preview')
    else:
        print('nothing todo')

    print(values, ok)
