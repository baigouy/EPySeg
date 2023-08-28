import os
from epyseg.settings.global_settings import set_UI
set_UI()
from qtpy.QtCore import QTimer


class Blinker:
    """
    Utility class for blinking components by changing their background color to red temporarily.

    Methods:
        blink(component): Blinks the given component or a list of components.
        reset_stylesheet(component): Resets the stylesheet of the given component or a list of components.

    """

    def __init__(self):
        pass

    def blink(self, component):
        """
        Blinks the given component or a list of components by changing their background color to red temporarily.

        Args:
            component (QWidget or list[QWidget]): The component or list of components to be blinked.

        """
        if isinstance(component, list):
            for comp in component:
                comp.setStyleSheet("background-color: red;")
        else:
            component.setStyleSheet("background-color: red;")

        self.current_timer = QTimer()
        self.current_timer.timeout.connect(lambda: self.reset_stylesheet(component))
        self.current_timer.setSingleShot(True)
        self.current_timer.start(3000)

    def reset_stylesheet(self, component):
        """
        Resets the stylesheet of the given component or a list of components, restoring the original appearance.

        Args:
            component (QWidget or list[QWidget]): The component or list of components to reset the stylesheet.

        """
        if isinstance(component, list):
            for comp in component:
                comp.setStyleSheet('')
        else:
            component.setStyleSheet('')


if __name__ == '__main__':
    blinker = Blinker()
    # blinker.blink()
