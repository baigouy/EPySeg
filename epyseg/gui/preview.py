from PyQt5.QtCore import QRect, Qt, QRectF
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout, QScrollArea
from epyseg.draw.shapes.square2d import Square2D
from epyseg.draw.widgets.paint import Createpaintwidget
from epyseg.img import Img
from epyseg.draw.shapes.rect2d import Rect2D
import sys

# in fact that is maybe already what I want!!!
# but I may also want to draw on it with a pen --> should have everything

class crop_or_preview(QWidget):

    def __init__(self, parent_window=None, preview_only=False):
        super().__init__(parent=parent_window)
        self.scale = 1.0
        self.x1 = self.x2 = self.y1 = self.y2 = None
        self.preview_only = preview_only
        self.initUI()

    def initUI(self):
        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.paint = Createpaintwidget()

        self.paint.vdp.active = True
        self.paint.vdp.drawing_mode = True
        if not self.preview_only:
            self.paint.vdp.shape_to_draw = Rect2D

        self.scrollArea = QScrollArea()
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scrollArea.setWidget(self.paint)
        self.paint.scrollArea = self.scrollArea
        self.setMouseTracking(not self.preview_only)
        self.paint.setMouseTracking(not self.preview_only)  # KEEP IMPORTANT
        self.paint.mouseMoveEvent = self.mouseMoveEvent
        self.paint.mousePressEvent = self.mousePressEvent
        self.paint.mouseReleaseEvent = self.mouseReleaseEvent

        self.prev_width = 192
        self.prev_height = 192

        self.scrollArea.setGeometry(QRect(0, 0, self.prev_width, self.prev_height))
        self.setGeometry(QRect(0, 0, self.prev_width, self.prev_height))
        self.setFixedSize(self.size())
        layout.addWidget(self.scrollArea)

        self.setLayout(layout)

    def set_image(self, img):
        self.paint.vdp.shapes.clear()
        self.paint.setImage(img)  # bug is here
        if img is None:
            self.paint.scale = self.scale = self.paint.vdp.scale = 1.
        else:
            max_size = min(self.prev_width / img.get_width(), self.prev_height / img.get_height())
            self.paint.scale = self.scale = self.paint.vdp.scale = max_size

        if self.paint.image is not None:
            self.paint.resize(self.scale * self.paint.image.size())
            self.scrollArea.resize(self.scale * self.paint.image.size())

    def mousePressEvent(self, event):
        self.paint.vdp.shapes.clear()
        if self.paint.vdp.active:
            self.paint.vdp.mousePressEvent(event)
            if self.paint.vdp.currently_drawn_shape is not None:
                self.paint.vdp.currently_drawn_shape.stroke = 3 / self.scale
            self.update()

    def mouseMoveEvent(self, event):
        self.paint.vdp.mouseMoveEvent(event)
        region = self.scrollArea.widget().visibleRegion()
        self.paint.update(region)

    def mouseReleaseEvent(self, event):
        if self.paint.vdp.active:
            try:
                self.paint.vdp.mouseReleaseEvent(event)
                self.update()  # required to update drawing
                self.update_ROI()
            except:
                pass

    def set_square_ROI(self, bool):
        if bool:
            self.paint.vdp.shape_to_draw = Square2D
        else:
            self.paint.vdp.shape_to_draw = Rect2D

    def setRoi(self, x1, y1, x2, y2):
        # if x1 is None and y1 is None and x2 is None and y2 is None:
        #     self.paint.vdp.shapes.clear()
        #     self.paint.update()
        #     self.update()  # required to update drawing
        #     self.update_ROI()
        #     return
        if x1 != x2 and y1 != y2:
            # TODO add controls for size of ROI --> TODO but ok for now

            self.paint.vdp.shapes.clear()
            if x1 is not None and y1 is not None:
                rect2d = Rect2D(x1, y1, x2 - x1, y2 - y1)
                rect2d.stroke = 3 / self.scale
                self.paint.vdp.shapes.append(rect2d)
                self.paint.vdp.currently_drawn_shape = None
                self.paint.update()
                self.update()  # required to update drawing
                self.update_ROI()
            else:
                self.x1 = x1
                self.x2 = x2
                self.y1 = y1
                self.y2 = y2
                self.paint.update()
                self.update()

            # self.paint.vdp.update()
            # self.paint.vdp.currently_drawn_shape.stroke = 3 / self.scale
            # self.x1 = x1
            # self.x2 = x2
            # self.y1 = y1
            # self.y2 = y2


    def update_ROI(self):
        try:
            rect = self.paint.vdp.shapes[0]
            x1 = rect.x()
            y1 = rect.y()
            x2 = rect.x() + rect.width()
            y2 = rect.y() + rect.height()

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0

            if x2 < 0:
                x2 = 0
            if y2 < 0:
                y2 = 0

            if rect.width() >= self.paint.image.size().width():
                x2 = self.paint.image.size().width()
            if rect.height() >= self.paint.image.size().height():
                y2 = self.paint.image.size().height()

            if x1 > x2:
                tmp = x2
                x2 = x1
                x1 = tmp

            if y1 > y2:
                tmp = y2
                y2 = y1
                y1 = tmp

            if x1 == x2:
                x1 = x2 = None
            if y1 == y2:
                y1 = y2 = None

            self.x1 = x1
            self.x2 = x2
            self.y1 = y1
            self.y2 = y2
        except:
            self.x1 = self.x2 = self.y1 = self.y2 = None

    def get_crop_parameters(self):
        if self.paint.vdp.shapes:
            self.update_ROI()
        if self.x1 is None:
            if self.x2 is not None:
                return {'x1': self.x1, 'y1': self.y1, 'w': int(self.x2), 'h': int(self.y2)}
            else:
                return None
        return {'x1': int(self.x1), 'y1': int(self.y1), 'x2': int(self.x2), 'y2': int(self.y2)}

if __name__ == '__main__':

    # ok in  fact that is already a great popup window --> can I further improve it ???

    # just for a test
    app = QApplication(sys.argv)
    ex = crop_or_preview()
    # ex = crop_or_preview(preview_only=True)
    # img = Img('/home/aigouy/mon_prog/Python/Deep_learning/unet/data/membrane/test/11.png')
    # img = Img('/home/aigouy/mon_prog/Python/Deep_learning/unet/data/membrane/test/122.png')
    # img = Img('/home/aigouy/mon_prog/Python/data/3D_bicolor_ovipo.tif')
    # img = Img('/home/aigouy/mon_prog/Python/data/Image11.lsm')
    # img = Img('/home/aigouy/mon_prog/Python/data/lion.jpeg')
    # img = Img('/home/aigouy/mon_prog/Python/data/epi_test.png')
    img = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini10_fake_swaps/focused_Series012.png')
    ex.set_image(img)

    # test = QRectF(None, None, 128, 128)

    ex.setRoi(None, None, 128, 256)
    # ex.set_image(None)
    ex.show()
    app.exec_()

    print(ex.get_crop_parameters())
