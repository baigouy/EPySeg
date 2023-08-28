import os
from epyseg.settings.global_settings import set_UI  # set the UI to be used py qtpy
set_UI()
import sys
from qtpy.QtWidgets import QApplication, QFileDialog
from os.path import expanduser


def openFileNameDialog(parent_window=None, extensions="Supported Files (*.jpg *.tif *.png);;All Files (*)",
                       path=expanduser('~')):
    """
    Opens a file dialog to select a single file.

    Args:
        parent_window (object): Parent window object. Default is None.
        extensions (str): File extension filters for the dialog. Default is "Supported Files (*.jpg *.tif *.png);;All Files (*)".
        path (str): Initial path for the dialog. Default is user's home directory.

    Returns:
        str: Selected file name.

    # Examples:
    #     >>> filename = openFileNameDialog()
    """
    if not os.path.exists(path):
        path = expanduser('~')
    fileName, _ = QFileDialog.getOpenFileName(parent_window, "Select a File", path, extensions,
                                              options=QFileDialog.DontUseNativeDialog)
    return fileName


def openFileNamesDialog(parent_window=None, extensions="Supported Files (*.jpg *.tif *.png);;All Files (*)",
                        path=expanduser('~')):
    """
    Opens a file dialog to select multiple files.

    Args:
        parent_window (object): Parent window object. Default is None.
        extensions (str): File extension filters for the dialog. Default is "Supported Files (*.jpg *.tif *.png);;All Files (*)".
        path (str): Initial path for the dialog. Default is user's home directory.

    Returns:
        list: List of selected file names.

    # Examples:
    #     >>> files = openFileNamesDialog()
    """
    if not os.path.exists(path):
        path = expanduser('~')
    files, _ = QFileDialog.getOpenFileNames(parent_window, "Select Files", path, extensions,
                                            options=QFileDialog.DontUseNativeDialog)
    return files


def saveFileDialog(parent_window=None, path=expanduser('~'), extensions="All Files (*);;Text Files (*.txt)",
                    default_ext=None):
    """
    Opens a file dialog to save a file.

    Args:
        parent_window (object): Parent window object. Default is None.
        path (str): Initial path for the dialog. Default is user's home directory.
        extensions (str): File extension filters for the dialog. Default is "All Files (*);;Text Files (*.txt)".
        default_ext (str): Default file extension. Default is None.

    Returns:
        str: Selected file name to save.

    # Examples:
    #     >>> filename = saveFileDialog(default_ext='.tif')
    """
    try:
        if not os.path.exists(path) and not os.path.exists(os.path.dirname(path)):
            path = expanduser('~')
    except:
        path = expanduser('~')
    fd = QFileDialog(parent_window, "Save a File", path, extensions, options=QFileDialog.DontUseNativeDialog)
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
    """
    Opens a dialog to select a directory.

    Args:
        parent_window (object): Parent window object. Default is None.
        path (str): Initial path for the dialog. Default is user's home directory.

    Returns:
        str: Selected directory path.

    # Examples:
    #     >>> folder = openDirectoryDialog()
    """
    if not os.path.exists(path):
        path = expanduser('~')
    folderName = QFileDialog.getExistingDirectory(parent_window, "Select a Directory", path,
                                                  options=QFileDialog.DontUseNativeDialog)
    if folderName is not None:
        if not folderName.strip():
            return None
        if not folderName.endswith("/") and not folderName.endswith("\\"):
            folderName += '/'
    return folderName


if __name__ == '__main__':
    app = QApplication(sys.argv)
    filename = saveFileDialog(default_ext='.tif')
    print(filename)
    sys.exit(0)
