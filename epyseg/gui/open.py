from PyQt5.QtCore import QSize
from natsort import natsorted
import glob
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QPushButton, QWidget, QLineEdit, QApplication, QGridLayout, QStyle, QLabel
from epyseg.dialogs.opensave import Open_Save_dialogs
import sys
import os

class QLineEditDND(QLineEdit):

    def __init__(self, title, parent, tip_text=None):
        super().__init__(title, parent)
        if tip_text is not None:
            self.setPlaceholderText(tip_text)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        # TODO test on osX to see if that works
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    # TODO handle several files if needed
    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            self.setText(e.mimeData().urls()[0].toLocalFile())

class OpenFileOrFolderWidget(QWidget):

    def __init__(self, parent_window=None, add_timer_to_changetext=False, show_ok_or_not_icon=False, label_text=None,
                 is_file=False, extensions="All Files (*);;", show_size=False, tip_text=None):
        super().__init__(parent=parent_window)
        self.ok_ico = self.style().standardIcon(QStyle.SP_DialogYesButton).pixmap(QSize(12, 12))
        self.not_ok_ico = self.style().standardIcon(QStyle.SP_DialogNoButton).pixmap(QSize(12, 12))
        self.show_ok_or_not_icon = show_ok_or_not_icon
        self.parent_window = parent_window
        self.add_timer_to_changetext = add_timer_to_changetext
        self.label_text = label_text
        self.is_file = is_file
        self.tip_text = tip_text
        # self.is_file_or_folder = is_file_or_folder
        self.extensions = extensions
        self.show_size = show_size
        self.initUI()

    def initUI(self):
        layout = QGridLayout()
        size_begin = 80
        size_label = 0
        size_end = 20
        size_warning = 0
        size_input_size = 0

        if self.label_text is not None:
            size_label = 10
            size_begin -= size_label

        if self.show_ok_or_not_icon:
            size_warning = 3
            size_begin -= size_warning

        if self.show_size:
            size_input_size = 5
            size_begin -= size_input_size

        layout.setColumnStretch(0, size_label)
        layout.setColumnStretch(1, size_begin)
        layout.setColumnStretch(2, size_end)
        layout.setColumnStretch(3, size_warning)
        layout.setColumnStretch(4, size_input_size)
        layout.setHorizontalSpacing(3)
        layout.setVerticalSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.path = QLineEditDND('', self, tip_text=self.tip_text)

        if self.add_timer_to_changetext:
            from PyQt5.QtCore import QSize, QTimer
            timer = QTimer()
            timer.setSingleShot(True)
            self.path.textChanged.connect(lambda: timer.start(600))
            timer.timeout.connect(self.finalize_text_change)
        self.path.setDragEnabled(True)

        open_ico = QIcon.fromTheme("folder-open")
        open_button = QPushButton(open_ico, "Open", self)
        # bt_width = open_button.fontMetrics().boundingRect(open_button.text()).width() + 30
        # open_button.setMaximumWidth(bt_width)
        # if self.is_file_or_folder:
        #     open_button.clicked.connect(self.open_file_or_folder)
        # el
        if self.is_file:
            open_button.clicked.connect(self.open_file)
        else:
            open_button.clicked.connect(self.open_folder)

        self.ok_or_not_ico = QLabel('')
        self.ok_or_not_ico.setPixmap(self.not_ok_ico)

        if self.show_ok_or_not_icon:
            layout.addWidget(self.ok_or_not_ico, 0, 3)

        if self.label_text is not None:
            layout.addWidget(QLabel(self.label_text), 0, 0)

        self.size_label = QLabel('0000')

        if self.show_size:
            layout.addWidget(self.size_label, 0, 4)

        layout.addWidget(self.path, 0, 1)
        layout.addWidget(open_button, 0, 2)
        self.setLayout(layout)

    def open_folder(self):
        if os.path.isdir(self.path.text()):
            self.output_file_or_folder = Open_Save_dialogs().openDirectoryDialog(parent_window=self.parent_window)
        else:
            self.output_file_or_folder = Open_Save_dialogs().openDirectoryDialog(parent_window=self.parent_window,
                                                                                 path=self.path.text())
        if self.output_file_or_folder is not None:
            self.path.setText(self.output_file_or_folder)

    def open_file(self):
        if os.path.isfile(self.path.text()):
            self.output_file_or_folder = Open_Save_dialogs().openFileNameDialog(parent_window=self.parent_window,
                                                                                extensions=self.extensions)
        else:
            self.output_file_or_folder = Open_Save_dialogs().openFileNameDialog(parent_window=self.parent_window,
                                                                                extensions=self.extensions,
                                                                                path=self.path.text())
        if self.output_file_or_folder is not None:
            self.path.setText(self.output_file_or_folder)

    def text(self):
        if self.path.text().strip() == '':
            return None
        return self.path.text()

    def finalize_text_change(self):
        print('in', self.path.text())

    def set_icon_ok(self, ok):
        if not self.show_ok_or_not_icon:
            return
        if ok:
            self.ok_or_not_ico.setPixmap(self.ok_ico)
        else:
            self.ok_or_not_ico.setPixmap(self.not_ok_ico)

    def set_size(self, size):
        self.size_label.setText(size)

    def get_list_using_glob(self):
        try:
            filenames = [file for file in glob.glob(self.path.text())]
            filenames = natsorted(filenames)  # human-like sorting of file names
            return filenames
        except:
            return None

if __name__ == '__main__':
    # just for a test
    app = QApplication(sys.argv)
    ex = OpenFileOrFolderWidget(parent_window=None)
    ex.show()
    app.exec_()
