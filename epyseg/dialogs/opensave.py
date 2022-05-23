import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
from os.path import expanduser
import os


# class Open_Save_dialogs():

def openFileNameDialog(parent_window=None, extensions="Supported Files (*.jpg *.tif *.png);;All Files (*)",
                       path=expanduser('~')):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    if not os.path.exists(path):
        path = expanduser('~')
    fileName, _ = QFileDialog.getOpenFileName(parent_window, "Select a File", path, extensions, options=options)
    return fileName

def openFileNamesDialog(parent_window=None, extensions="Supported Files (*.jpg *.tif *.png);;All Files (*)",
                        path=expanduser('~')):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    if not os.path.exists(path):
        path = expanduser('~')
    files, _ = QFileDialog.getOpenFileNames(parent_window, "Select Files", path, extensions, options=options)
    return files

# not so easy to do in fact
# def openSingleFileOrDirectoryDialog(self, parent_window=None, extensions="Supported Files (*.jpg *.tif *.png);;All Files (*)",
#                        path=expanduser('~')):
#     setFileMode(QFileDialog::Directory | QFileDialog::ExistingFiles)
#     options = QFileDialog.Options()
#     options |= QFileDialog.DontUseNativeDialog
#     if not os.path.exists(path):
#         path = expanduser('~')
#     fileName, _ = QFileDialog.getOpenFileName(parent_window, "QFileDialog.getOpenFileName()", path,
#                                               extensions, options=options)


# create file if does not exist and delete if name varies --> maybe do this as an option...
def saveFileDialog(parent_window=None, path=expanduser('~'),  extensions="All Files (*);;Text Files (*.txt)", default_ext=None):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    try:
        # nice hack to allow to specify a default name even if the file does not exist as long as the parent folder exists
        if not os.path.exists(path) and not os.path.exists(os.path.dirname(path)):
            path = expanduser('~')
    except:
        path = expanduser('~')
    fd = QFileDialog(parent_window, "Save a File", path, extensions, options=options)
    if default_ext is not None:
        fd.setDefaultSuffix(default_ext)
    fd.setAcceptMode(QFileDialog.AcceptSave)
    selected = fd.exec()
    if selected:
        fileName = fd.selectedFiles()[0]
        return fileName
    else:
        return

def openDirectoryDialog(parent_window=None, path=expanduser('~')):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    if not os.path.exists(path):
        path = expanduser('~')
    folderName = QFileDialog.getExistingDirectory(parent_window, "Select a Directory", path, options=options)
    if folderName is not None:
        if not folderName.strip():
            return None
        if not folderName.endswith("/") and not folderName.endswith("\\"):
            folderName += '/'
    return folderName

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # ex = Open_Save_dialogs()
    filename = saveFileDialog(default_ext='.tif')
    print(filename)
    sys.exit(0)