from PyQt5.QtCore import QTimer


class Blinker:

    def __init__(self):
        pass

    def blink(self, component):
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
        if isinstance(component, list):
            for comp in component:
                comp.setStyleSheet('')
        else:
            component.setStyleSheet('')

if __name__ == '__main__':
    blinker = Blinker()
    # blinker.blink()
