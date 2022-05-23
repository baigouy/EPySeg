import sys


#https://doc.qt.io/qtforpython/PySide2/QtWidgets/QDialogButtonBox.html
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QPushButton, QWidget, QHBoxLayout, QLabel, \
    QDoubleSpinBox, QCheckBox, QSpinBox, QApplication


class WshedDialog(QDialog):

    def __init__(self, parent=None):
        super(WshedDialog, self).__init__(parent)

        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self.setupUI()

        # OK and Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        preview_button = QPushButton('Preview')
        preview_button.clicked.connect(self.preview)
        buttons.addButton(preview_button, QDialogButtonBox.ApplyRole)

        layout.addWidget(buttons)

    def setupUI(self):
        main_panel = QWidget()
        main_panel.setLayout(QHBoxLayout())
        main_panel.layout().addWidget(QLabel('Weak Blur'))

        self.weak_blur_spin = QDoubleSpinBox()
        self.weak_blur_spin.setSingleStep(0.1)
        self.weak_blur_spin.setRange(0, 100)
        self.weak_blur_spin.setValue(1)
        main_panel.layout().addWidget(self.weak_blur_spin)

        self.b1 = QCheckBox("Dual blur")
        self.b1.setChecked(True)
        self.b1.stateChanged.connect(self.biblur)
        main_panel.layout().addWidget(self.b1)

        self.strong_blur_label = QLabel('Strong Blur (optional*)')
        main_panel.layout().addWidget(self.strong_blur_label)

        self.strong_blur_spin = QDoubleSpinBox()
        self.strong_blur_spin.setSingleStep(0.1)
        self.strong_blur_spin.setRange(0, 200)
        self.strong_blur_spin.setValue(3)
        main_panel.layout().addWidget(self.strong_blur_spin)


        min_cell_size_label = QLabel('Exclude cells below:')
        main_panel.layout().addWidget(min_cell_size_label)
        self.min_cell_size_spin = QSpinBox()
        self.min_cell_size_spin.setSingleStep(1)
        self.min_cell_size_spin.setRange(0, 10000)
        self.min_cell_size_spin.setValue(10)
        main_panel.layout().addWidget(self.min_cell_size_spin)
        px_label = QLabel('px')
        main_panel.layout().addWidget(px_label)


        self.layout().addWidget(main_panel)

        # override TAGenericDialog methods
        # Dialog.no_color = self.preview # need override
        # Dialog.values = self.values


    def biblur(self):
        self.strong_blur_spin.setEnabled(self.b1.isChecked())
        self.strong_blur_label.setEnabled(self.b1.isChecked())

    # get user values for cropping the image
    def values(self):
        weak = self.weak_blur_spin.value()
        strong = self.strong_blur_spin.value()
        if not self.b1.isChecked():
            strong = None
        area_exclusion =   self.min_cell_size_spin.value()

        # then even simplify if max and min are unchanged --> if min is 0 set it to None, etc...
        return (weak, strong, area_exclusion)

    # override this if needed
    def preview(self):
        self.done(QDialogButtonBox.Apply)

    # static method to create the dialog and return (values, accepted)
    @staticmethod
    def getValues(parent=None, preview_enabled=False):
        dialog = WshedDialog(parent=parent)
        result = dialog.exec_()
        values = dialog.values()
        return (values, result)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    values, ok = WshedDialog.getValues()

    if ok == QDialogButtonBox.Apply:
        print('just do preview')
    elif ok == QDialog.Accepted:
        print('process all')
    else:
        print('nothing todo')
    # print(values, ok)
