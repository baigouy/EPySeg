import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
from os.path import expanduser
import os

class Open_Save_dialogs():

    def openFileNameDialog(self, parent_window=None, extensions="Supported Files (*.jpg *.tif *.png);;All Files (*)",
                           path=expanduser('~')):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        if not os.path.exists(path):
            path = expanduser('~')
        fileName, _ = QFileDialog.getOpenFileName(parent_window, "QFileDialog.getOpenFileName()", path,
                                                  extensions, options=options)
        return fileName

    def openFileNamesDialog(self, parent_window=None, extensions="Supported Files (*.jpg *.tif *.png);;All Files (*)",
                            path=expanduser('~')):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        if not os.path.exists(path):
            path = expanduser('~')
        files, _ = QFileDialog.getOpenFileNames(parent_window, "QFileDialog.getOpenFileNames()", path,
                                                extensions, options=options)
        return files

    def saveFileDialog(self, parent_window=None, path=expanduser('~')):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        if not os.path.exists(path):
            path = expanduser('~')
        fileName, _ = QFileDialog.getSaveFileName(parent_window, "QFileDialog.getSaveFileName()", path,
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        return fileName

    def openDirectoryDialog(self, parent_window=None, path=expanduser('~')):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        if not os.path.exists(path):
            path = expanduser('~')
        folderName = QFileDialog.getExistingDirectory(parent_window, "Select a Directory", path,
                                                      options=options)
        if folderName is not None:
            if not folderName.strip():
                return None
            if not folderName.endswith("/") and not folderName.endswith("\\"):
                folderName += '/'
        return folderName

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Open_Save_dialogs()
    sys.exit(0)