from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QSpinBox, QDialog, QDialogButtonBox, QCheckBox, QFrame
import sys
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()


class DefineROI(QDialog):

    def __init__(self, parent_window=None, x1=0, y1=0, x2=0, y2=0):
        super().__init__(parent=parent_window)
        self.setWindowTitle('Define your ROI')
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.initUI()

    def initUI(self):
        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setColumnStretch(0, 25)
        layout.setColumnStretch(1, 75)

        self.random_checkbox = QCheckBox('Random ROI (random crop with defined width and height)', objectName='random_checkbox')
        self.random_checkbox.setChecked(False)

        labelX1 = QLabel('x1')
        self.x1_spinner = QSpinBox(objectName='x1_spinner')
        self.x1_spinner.setSingleStep(1)
        self.x1_spinner.setRange(0, 1_000_000)
        if self.x1 is not None:
            self.x1_spinner.setValue(self.x1)
        else:
            self.random_checkbox.setChecked(True)

        labelY1 = QLabel('y1')
        self.y1_spinner = QSpinBox(objectName='y1_spinner')
        self.y1_spinner.setSingleStep(1)
        self.y1_spinner.setRange(0, 1_000_000)
        if self.y1 is not None:
            self.y1_spinner.setValue(self.y1)
        else:
            self.random_checkbox.setChecked(True)

        self.labelX2 = QLabel('x2')
        self.x2_spinner = QSpinBox(objectName='x2_spinner')
        self.x2_spinner.setSingleStep(1)
        self.x2_spinner.setRange(0, 1_000_000)
        self.x2_spinner.setValue(self.x2)

        self.labelY2 = QLabel('y2')
        self.y2_spinner = QSpinBox(objectName='y2_spinner')
        self.y2_spinner.setSingleStep(1)
        self.y2_spinner.setRange(0, 1_000_000)
        self.y2_spinner.setValue(self.y2)

        self.random_checkbox.stateChanged.connect(self._change_mode)
        if self.random_checkbox.isChecked():
            self._change_mode()

        # line separator
        line_sep = QFrame()
        line_sep.setFrameShape(QFrame.HLine)
        line_sep.setFrameShadow(QFrame.Sunken)

        # self.setGeometry(QRect(0, 0, self.prev_width, self.prev_height))
        # self.setFixedSize(self.size())

        layout.addWidget(self.random_checkbox, 0, 0, 1, 2)
        layout.addWidget(labelX1, 1, 0)
        layout.addWidget(self.x1_spinner, 1, 1)
        layout.addWidget(labelY1, 2, 0)
        layout.addWidget(self.y1_spinner, 2, 1)
        layout.addWidget(self.labelX2, 3, 0)
        layout.addWidget(self.x2_spinner, 3, 1)
        layout.addWidget(self.labelY2, 4, 0)
        layout.addWidget(self.y2_spinner, 4, 1)
        layout.addWidget(line_sep, 5, 0, 1, 2)

        self.setLayout(layout)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox, 6, 0, 1, 2)

        self.setGeometry(QRect(0, 0, 300, 150))
        # self.setFixedSize(self.size())

    def _change_mode(self):
        if self.random_checkbox.isChecked():
            self.labelX2.setText('width')
            self.labelY2.setText('height')
            self.x1_spinner.setEnabled(False)
            self.y1_spinner.setEnabled(False)
        else:
            self.labelX2.setText('x2')
            self.labelY2.setText('y2')
            self.x1_spinner.setEnabled(True)
            self.y1_spinner.setEnabled(True)

    def getCrop(self):
        # TODO maybe add controls and return only if valid
        # Should I sort things too
        if self.x1_spinner.isEnabled():
            x1 = self.x1_spinner.value()
        else:
            x1 = None
        if self.y1_spinner.isEnabled():
            y1 = self.y1_spinner.value()
        else:
            y1 = None
        x2 = self.x2_spinner.value()
        y2 = self.y2_spinner.value()

        if x1 == x2:
            logger.error('x1=x2, ROI width is null --> ignoring ROI')
            return None
        if y1 == y2:
            logger.error('y1=y2, ROI height is null --> ignoring ROI')
            return None

        if x1 is not None:
            if x1 > x2:
                x1, x2 = x2, x1
        else:
            if x2 == 0:
                logger.error('ROI width is null --> ignoring ROI')
                return None
        if y1 is not None:
            if y1 > y2:
                y1, y2 = y2, y1
        else:
            if y2 == 0:
                logger.error('ROI height is null --> ignoring ROI')
                return None

        return x1, y1, x2, y2

    @staticmethod
    def getDataAndParameters(parent_window=None, x1=0, y1=0, x2=0, y2=0):  #
        dialog = DefineROI(parent_window=parent_window, x1=x1, y1=y1, x2=x2, y2=y2)

        result = dialog.exec_()
        if result:
            roi = dialog.getCrop()
        else:
            return None, False
        return (roi, result == QDialog.Accepted)


if __name__ == '__main__':
    # just for a test
    app = QApplication(sys.argv)
    # ex = DefineROI(x1=120, y1=26, x2=16, y2=300)
    # ex = DefineROI(x1=120, y1=26, x2=120, y2=300)
    # ex = crop_or_preview(preview_only=True)
    # img = Img('/home/aigouy/mon_prog/Python/Deep_learning/unet/data/membrane/test/11.png')
    # img = Img('/home/aigouy/mon_prog/Python/Deep_learning/unet/data/membrane/test/122.png')
    # img = Img('/home/aigouy/mon_prog/Python/data/3D_bicolor_ovipo.tif')
    # img = Img('/home/aigouy/mon_prog/Python/data/Image11.lsm')
    # img = Img('/home/aigouy/mon_prog/Python/data/lion.jpeg')
    # img = Img('/home/aigouy/mon_prog/Python/data/epi_test.png')
    # ex.set_image(None)

    # ROI, ok = DefineROI.getDataAndParameters(x1=120, y1=26, x2=120, y2=300)
    ROI, ok = DefineROI.getDataAndParameters(x1=None, y1=None, x2=128, y2=256)

    if ok:
        print(ROI)

    # ex.show()
    # app.exec_()
    sys.exit(0)
