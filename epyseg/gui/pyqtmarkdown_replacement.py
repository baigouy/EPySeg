from qtpy.QtWidgets import QApplication, QMainWindow, QTextEdit
from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QTextCursor, QDesktopServices

URL_prefix = 'https://github.com/baigouy/EPySeg/blob/master/epyseg/deeplearning/docs/'
class ClickableTextEdit(QTextEdit):
    def __init__(self, parent=None, help_URLs=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setTextInteractionFlags(Qt.TextBrowserInteraction)
        if help_URLs is not None:
            html_content = 'Please click any link below to access the corresponding online help:<BR><BR>'
            for title, link in help_URLs.items():
                html_content += "<a href='" + URL_prefix + link + "'>" + title + "</a><BR><BR>"

        self.setHtml(html_content)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        cursor = self.cursorForPosition(event.position().toPoint())
        cursor.select(QTextCursor.WordUnderCursor)

        selected_text = None
        if not cursor.isNull():
            selected_text = cursor.charFormat().anchorHref()

        if selected_text:
            if selected_text.startswith("http://") or selected_text.startswith("https://"):
                try:
                    QDesktopServices.openUrl(QUrl(selected_text))
                except:
                    import webbrowser
                    webbrowser.open(selected_text)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # just a demo
        help_URLs = {'predict using pre-trained network': 'getting_started2.md',
                     'build and train a custom network': 'getting_started.md',
                     'further train the EPySeg model on your data': 'getting_started3.md',
                     'load a pre-trained model': 'pretrained_model.md',
                     'Build a model from scratch': 'model.md',
                     'Load a model': 'load_model.md',
                     'Train': 'train.md',
                     'Predict': 'predict.md',
                     'Training dataset parameters': 'preprocessing.md',
                     'Data augmentation': 'data_augmentation.md'}

        self.text_edit = ClickableTextEdit(help_URLs)
        self.setCentralWidget(self.text_edit)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()