from PyQt5 import QtCore
from PyQt5.QtCore import QPointF, QRect
from PyQt5.QtGui import QTextDocument, QTextOption
from PyQt5.QtGui import QPainter, QImage, QColor, QFont
import sys
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from epyseg.draw.shapes.rect2d import Rect2D
# log errors
from epyseg.tools.logger import TA_logger
logger = TA_logger()

class TAText2D(Rect2D):

    # TODO add bg to it so that it can be drawn
    def __init__(self, text=None, doc=None, opacity=1., *args, **kwargs):
        if doc is not None and isinstance(doc, QTextDocument):
            self.doc = doc
            self.doc.setDocumentMargin(0)  # important so that the square is properly placed
        else:
            self.doc = QTextDocument()
            self.doc.setDocumentMargin(0) # important so that the square is properly placed

            textOption = self.doc.defaultTextOption()
            textOption.setWrapMode(QTextOption.NoWrap)
            self.doc.setDefaultTextOption(textOption)

            if text is not None:
                self.doc.setHtml(text)
        self.isSet = True
        self.doc.adjustSize()
        size = self.getSize()
        super(TAText2D, self).__init__(0, 0, size.width(), size.height())
        self.opacity = opacity

    def set_opacity(self, opacity):
        self.opacity = opacity

    def setText(self, html):
        self.doc.setHtml(html)
        self.doc.adjustSize()
        size = self.getSize()
        self.setWidth(size.width(), size.height())

    def setDoc(self, doc):
        self.doc = doc
        self.doc.setDocumentMargin(0)
        self.doc.adjustSize()
        size = self.getSize()
        self.setWidth(size.width(), size.height())

    def draw(self, painter):
      painter.save()
      painter.setOpacity(self.opacity)
      painter.translate(self.x(), self.y())
      self.doc.drawContents(painter)
      painter.restore()
      # maybe activate this upon debug
      # painter.save()
      # painter.setPen(QtCore.Qt.red)
      # painter.drawRect(self)
      # painter.restore()

    def boundingRect(self):
        return self

    def getSize(self):
        return self.doc.size()

    def getWidth(self):
        return self.boundingRect().width()

    def getHeight(self):
        return self.boundingRect().height()

    def setText(self, text):
        self.doc.setHtml(text)
        size = self.size()
        self.setWidth(size.width(), size.height())

    def set_P1(self, *args):
        if not args:
            logger.error("no coordinate set...")
            return
        if len(args) == 1:
            self.moveTo(args[0].x(), args[0].y())
        else:
            self.moveTo(QPointF(args[0], args[1]))

    def get_P1(self):
        return QPointF(self.x(), self.y())

    def getPlainText(self):
        return self.doc.toPlainText()

    def getHtmlText(self):
        return self.doc.toHtml()

if __name__ == '__main__':
    # this could be a pb ...
    app = QApplication(sys.argv)# IMPORTANT KEEP !!!!!!!!!!!

    # window = MyWidget()
    # window.show()

    # ça marche car ça override la couleur par defaut du truc
    # c'est parfait et 2000X plus facile que ds java --> cool

    html = '<!DOCTYPE html>            <html>          <font color=red>     <head>                  <title>Font Face</title>               </head>               <body>       <font face = "Symbol" size = "5">Symbol</font><br />              <font face = "Times New Roman" size = "5">Times New Roman</font><br />                  <font face = "Verdana" size = "5">Verdana</font><br />                  <font face = "Comic sans MS" size =" 5">Comic Sans MS</font><br />                  <font face = "WildWest" size = "5">WildWest</font><br />                  <font face = "Bedrock" size = "5">Bedrock</font><br />               </body>            </html>'

    # html = "<font color=blue size=24>this is a test<sup>2</sup><br></font><font color=green size=12>continued<sub>1</sub><br></font><font color=white size=12>test greek <font face='Symbol' size=32>a</font> another &alpha;<font face='Arial' color='Orange'>I am a sentence!</font>"
    text = TAText2D(html)

    # hexagon.append(QPointF(10, 20))
    print(text)

    # print(hexagon.translate(10, 20)) # why none ???
    # translate and so on can all be saved...

    image = QImage('./../data/handCorrection.png')
    # image = QImage(QSize(400, 300), QImage.Format_RGB32)
    painter = QPainter()
    painter.begin(image)
    # painter.setOpacity(0.3);
    painter.drawImage(0, 0, image)
    painter.setPen(QtCore.Qt.blue)
    text.opacity = 0.7
    painter.translate(10, 20)
    painter.setPen(QColor(168, 34, 3))

    text.draw(painter) # ça marche pourrait overloader ça avec du svg

    painter.drawRect(text)# try to draw the bounds

    # painter.setPen(QtCore.Qt.green)
    # painter.setFont(QFont('SansSerif', 50))


    painter.setFont(QFont('Decorative', 10))
    # painter.drawText(256, 256, "this is a test")

# nothing works it just doesn't draw for unknown reason ????
#     painter.drawText(QRect(60,60,256,256), Qt.AlignCenter, "this is a test")

    painter.setPen(QtGui.QColor(200, 0, 0))
    # painter.drawText(20, 20, "MetaGenerator") # fait planter le soft --> pkoi exit(139) ...
    painter.drawText(QRect(60,60,256,256), Qt.AlignCenter, "Text centerd in the drawing area")

    # painter.drawText(QRect(100, 100, 200, 100), "Text you want to draw...");
    print('here')
    painter.end()

    # image = QImage(QSize(400, 300), QImage::Format_RGB32);
    # QPainter
    # painter( & image);
    # painter.setBrush(QBrush(Qt::green));
    # painter.fillRect(QRectF(0, 0, 400, 300), Qt::green);
    # painter.fillRect(QRectF(100, 100, 200, 100), Qt::white);
    # painter.setPen(QPen(Qt::black));


    # painter.save()
    # painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
    # painter.eraseRect(r)
    # painter.restore()
    print('saving', './../trash/test_pyQT_draw_text.png')
    image.save('./../trash/test_pyQT_draw_text.png', "PNG")


    # split text and find bounding rect of the stuff --> so that it is well positioned
    # or do everything in svg and just show what's needed ???

    #pas mal TODO faire une classe drawsmthg qui dessine n'importe quelle forme que l'on lui passe avec des parametres de couleur, transparence, ...

    # tt marche aps mal ça va très vite
    sys.exit(0)
