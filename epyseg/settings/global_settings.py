# this class should contain all the global settings
import os

# default_UI = 'pyqt6' # 'non existing'
default_UI = 'pyqt6' #'pyqt6' #'pyqt6'#'pyside6' #'pyside2' #'pyqt5' # 'non existing'
# force = False

def default_qtpy_UI():
    # {'pyqt5': 'PyQt5', 'pyside2': 'PySide2', 'pyqt6': 'PyQt6', 'pyside6': 'PySide6'}
    global default_UI
    if default_UI not in list_UIs():
        default_UI = list_UIs()[0]
    return default_UI

def list_UIs():
    from qtpy import API_NAMES
    # return {'pyqt5': 'PyQt5', 'pyside2': 'PySide2', 'pyqt6': 'PyQt6', 'pyside6': 'PySide6'}.keys()
    return list(API_NAMES.keys())

def set_UI(ignore_if_already_set=True, UI=default_UI):
    if ignore_if_already_set and 'QT_API' in os.environ:
        # print('QT_API set to:'+os.environ['QT_API'] + '--> ignoring')
        return
    # print('before',os.environ['QT_API'])
    # if force:
    #     if UI is None:
    #         UI=default_UI
    # raise Exception # never called --> so where is the fucking bug
    if UI is None:
        UI = default_qtpy_UI()
    os.environ['QT_API'] = UI
    # print('chosen UI', UI)
    try:
        from qtpy.QtWidgets import QWidget
    except:
        # print('UI not found --> rolling back to default')
        # just fort safety, if pyqt6 is not found --> get back to default
        del os.environ['QT_API']

def print_default_qtpy_UI_really_in_use():
    try:
        UI_defined = None
        from qtpy.QtCore import PYQT_VERSION_STR
        try:
            UI_defined =os.environ['QT_API']
        except:
            pass
        print(UI_defined +' v'+PYQT_VERSION_STR)
    except:
        pass

if __name__ == '__main__':
    # import os
    # os.environ['QT_API']='tata'
    # print(os.environ['QT_API'])

    # os.environ['QT_API']='pyqt5'

    # default_UI='test'
    # # print(default_qtpy_UI())
    # set_UI()
    # print(os.environ['QT_API'])
    # set_UI()
    # print(os.environ['QT_API'])
    # print(list_UIs())
    #
    # print_default_qtpy_UI_really_in_use()

    # os.environ['QT_API'] = 'pyqt5'
    set_UI()
    print_default_qtpy_UI_really_in_use()
