# this class should contain all the global settings
import os

# default_UI = 'pyqt6' # 'non existing'
default_UI = 'pyqt6' #'pyqt6' #'pyqt6'#'pyside6' #'pyside2' #'pyqt5' # 'non existing'
# force = False

def default_qtpy_UI():
    """
    Returns the default QtPy UI.

    Returns:
        str: The default QtPy UI.

    Examples:
        >>> default_qtpy_UI()
        'pyqt6'

    """
    global default_UI
    if default_UI not in list_UIs():
        default_UI = list_UIs()[0]
    return default_UI

def list_UIs():
    """
    Returns a list of available UI options for QtPy.

    Returns:
        list: The list of available UI options.

    Examples:
        >>> list_UIs()
        ['pyqt5', 'pyside2', 'pyqt6', 'pyside6']

    """
    from qtpy import API_NAMES
    return list(API_NAMES.keys())

def set_UI(ignore_if_already_set=True, UI=default_UI):
    """
    Sets the QtPy UI environment variable.

    Args:
        ignore_if_already_set (bool, optional): Flag to ignore if the UI is already set. Defaults to True.
        UI (str, optional): The UI to set. Defaults to default_UI.

    Examples:
        >>> set_UI()
        >>> print(os.environ['QT_API'])
        pyqt5

    """
    if ignore_if_already_set and 'QT_API' in os.environ:
        return

    if UI is None:
        UI = default_qtpy_UI()
    os.environ['QT_API'] = UI

def print_default_qtpy_UI_really_in_use():
    """
    Prints the default QtPy UI and the version in use.

    Examples:
        >>> print_default_qtpy_UI_really_in_use()
        pyqt5 v5.14.0

    """
    try:
        UI_defined = None
        from qtpy.QtCore import PYQT_VERSION_STR
        try:
            UI_defined = os.environ['QT_API']
        except:
            pass
        print(UI_defined + ' v' + PYQT_VERSION_STR)
    except:
        pass

def force_qtpy_to_use_user_specified(bool=True):
    """
    Forces QtPy to use the user-specified UI.

    Args:
        bool (bool, optional): Flag to enable or disable forcing QtPy to use the user-specified UI. Defaults to True.

    """
    if bool:
        os.environ['FORCE_QT_API'] = '1'
    else:
        try:
            del os.environ['FORCE_QT_API']
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
