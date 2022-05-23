from PyQt5.QtCore import QRect, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget

from epyseg.draw.shapes.rect2d import Rect2D
from epyseg.draw.shapes.square2d import Square2D
from epyseg.draw.widgets.vectorial import VectorialDrawPane
from PyQt5.QtWidgets import qApp, QMenu, QApplication
from PyQt5 import QtCore, QtGui

from epyseg.img import toQimage
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()

class Createpaintwidget(QWidget):

    def __init__(self):
        super().__init__()
        self.vdp = VectorialDrawPane(active=False) #, demo=True
        self.image = None
        self.imageDraw = None
        self.cursor = None
        self.maskVisible = True
        self.scale = 1.0
        self.drawing = False
        self.brushSize = 3
        self._clear_size = 30
        self.drawColor = QtGui.QColor(QtCore.Qt.red) # blue green cyan
        self.eraseColor = QtGui.QColor(QtCore.Qt.black)
        self.cursorColor = QtGui.QColor(QtCore.Qt.green)
        self.lastPoint = QtCore.QPoint()
        self.change = False
        # KEEP IMPORTANT required to track mouse even when not clicked
        self.setMouseTracking(True)  # KEEP IMPORTANT
        self.scrollArea = None
        self.statusBar = None

    def setImage(self, img):
        if img is None:
            self.image = None
            self.imageDraw = None
            self.update()
            return
        else:
            self.image = toQimage(img) #.getQimage() # bug is here

            # self.image = QPixmap(100,200).toImage()
        width = self.image.size().width()
        height = self.image.size().height()
        top = self.geometry().x()
        left = self.geometry().y()
        self.setGeometry(top, left, width*self.scale, height*self.scale)
        self.imageDraw = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
        self.imageDraw.fill(QtCore.Qt.transparent)
        self.cursor = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
        self.cursor.fill(QtCore.Qt.transparent)
        self.update()

    def mousePressEvent(self, event):
        if not self.hasMouseTracking():
            return
        self.clickCount = 1
        if self.vdp.active:
            self.vdp.mousePressEvent(event)
            self.update()
            return

        if event.buttons() == QtCore.Qt.LeftButton or event.buttons() == QtCore.Qt.RightButton:
            self.drawing = True
            zoom_corrected_pos = event.pos() / self.scale
            self.lastPoint = zoom_corrected_pos
            self.drawOnImage(event)

    def mouseMoveEvent(self, event):
        if not self.hasMouseTracking():
            return
        # print('in mouse move', self.hasMouseTracking(), self.drawing, self.vdp.active)
        if self.statusBar:
            zoom_corrected_pos = event.pos() / self.scale
            self.statusBar.showMessage('x=' + str(zoom_corrected_pos.x()) + ' y=' + str(
                zoom_corrected_pos.y()))
        if self.vdp.active:
            self.vdp.mouseMoveEvent(event)
            region = self.scrollArea.widget().visibleRegion()
            self.update(region)
            return
        self.drawOnImage(event)

    def drawOnImage(self, event):
        zoom_corrected_pos = event.pos() / self.scale
        if self.drawing and (event.buttons() == QtCore.Qt.LeftButton or event.buttons() == QtCore.Qt.RightButton):
            # now drawing or erasing over the image
            painter = QtGui.QPainter(self.imageDraw)
            if event.buttons() == QtCore.Qt.LeftButton:
                painter.setPen(QtGui.QPen(self.drawColor, self.brushSize, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                          QtCore.Qt.RoundJoin))
            else:
                painter.setPen(QtGui.QPen(self.eraseColor, self.brushSize, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                          QtCore.Qt.RoundJoin))
            if self.lastPoint != zoom_corrected_pos:
                painter.drawLine(self.lastPoint, zoom_corrected_pos)
            else:
                # if zero length line then draw point instead
                painter.drawPoint(zoom_corrected_pos)
            painter.end()

        # Drawing the cursor TODO add boolean to ask if drawing cursor should be shown
        painter = QtGui.QPainter(self.cursor)
        # We erase previous pointer
        r = QtCore.QRect(QtCore.QPoint(), self._clear_size * QtCore.QSize() * self.brushSize)
        painter.save()
        r.moveCenter(self.lastPoint)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        painter.eraseRect(r)
        painter.restore()
        # draw the new one
        painter.setPen(QtGui.QPen(self.cursorColor, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                  QtCore.Qt.RoundJoin))
        painter.drawEllipse(zoom_corrected_pos, self.brushSize / 2.,
                            self.brushSize / 2.)
        painter.end()
        region = self.scrollArea.widget().visibleRegion()
        self.update(region)

        # required to erase mouse pointer
        self.lastPoint = zoom_corrected_pos

    def mouseReleaseEvent(self, event):
        if not self.hasMouseTracking():
            return
        if self.vdp.active:
            self.vdp.mouseReleaseEvent(event)
            self.update()  # required to update drawing
            return
        if event.button == QtCore.Qt.LeftButton:
            self.drawing = False
        if self.clickCount == 1:
            QTimer.singleShot(QApplication.instance().doubleClickInterval(),
                              self.updateButtonCount)

    # adds context/right click menu but only in vectorial mode
    def contextMenuEvent(self, event):
        if not self.vdp.active:
            return
        cmenu = QMenu(self)
        newAct = cmenu.addAction("New")
        opnAct = cmenu.addAction("Open")
        quitAct = cmenu.addAction("Quit")
        action = cmenu.exec_(self.mapToGlobal(event.pos()))
        if action == quitAct:
            qApp.quit()

    def updateButtonCount(self):
        self.clickCount = 1

    def mouseDoubleClickEvent(self, event):
        self.clickCount = 2
        self.vdp.mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        canvasPainter = QtGui.QPainter(self)
        # the scrollpane visible region
        visibleRegion = self.scrollArea.widget().visibleRegion()
        # the corresponding rect
        visibleRect = visibleRegion.boundingRect()
        # the visibleRect taking zoom into account
        scaledVisibleRect = QRect(visibleRect.x() / self.scale, visibleRect.y() / self.scale,
                                  visibleRect.width() / self.scale, visibleRect.height() / self.scale)
        if self.image is None:
            canvasPainter.eraseRect(visibleRect)
            canvasPainter.end()
            return

        canvasPainter.drawImage(visibleRect, self.image, scaledVisibleRect)
        if not self.vdp.active and self.maskVisible:
            canvasPainter.drawImage(visibleRect, self.imageDraw, scaledVisibleRect)
            # should draw the cursor
        canvasPainter.drawImage(visibleRect, self.cursor, scaledVisibleRect)

        if self.vdp.active:
            self.vdp.paintEvent(canvasPainter, scaledVisibleRect)
        canvasPainter.end()
