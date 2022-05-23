# pip install PyQtWebEngine or  'PyQtWebEngine==5.13.0'  + 'PyQt5==5.13.0' # version need be in sync with pyt otherwise there is a bug
from markdown import markdown  # nb not all markdown is supported, but it's not so bad...
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QVBoxLayout, QTabWidget
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
import os


class PyQT_markdown(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.tabs = QTabWidget(self)
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.resize(300, 512)

    def set_markdown_from_file(self, filepath, title=None):
        new_tab = QWidget()
        if title is None:
            title = os.path.basename(filepath)
            title = os.path.splitext(title)[0]
            title = title.replace('_', ' ')
            title = title.title()
        self.tabs.addTab(new_tab, title)
        view = QWebEngineView()
        self._set_markdown_file(view, filepath)
        new_tab.layout = QVBoxLayout()
        new_tab.layout.addWidget(view)
        new_tab.setLayout(new_tab.layout)
        self.tabs.update()

    def _set_markdown_file(self, view, filepath):
        html = ''
        with open(filepath, 'r') as file:
            data = file.read()
            html = markdown(data)
        # print(html)
        # html = '<html><body>this is a test</body></html>'
        view.setHtml(html)


if __name__ == '__main__':
    import sys

    # nothing is displayed --> WHY

    app = QApplication(sys.argv)
    w = PyQT_markdown()
    w.show()
    w.set_markdown_from_file("../deeplearning/docs/getting_started.md")
    w.set_markdown_from_file("../deeplearning/docs/getting_started2.md")
    sys.exit(app.exec_())
