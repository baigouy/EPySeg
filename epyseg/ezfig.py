# TODO add support for scale bars allow saving or serialization --> how can I do that ??? save as a series of commands so that it can always be recreated from scratch from source image, allow export as SVF or various formats with DPI scaling ???
# todo CAN i USE css to change font and apply style would be so much simpler and better than current stuff...

import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSize, QRect
from PyQt5.QtGui import QPainter
from PyQt5.QtSvg import QSvgGenerator

from epyseg.draw.shapes.polygon2d import Polygon2D
from epyseg.draw.shapes.line2d import Line2D
from epyseg.draw.shapes.rect2d import Rect2D
from epyseg.draw.shapes.square2d import Square2D
from epyseg.draw.shapes.ellipse2d import Ellipse2D
from epyseg.draw.shapes.circle2d import Circle2D
from epyseg.draw.shapes.point2d import Point2D
from epyseg.draw.shapes.polyline2d import PolyLine2D
from epyseg.draw.shapes.image2d import Image2D
from epyseg.figure.column import Column
# logging
from epyseg.draw.shapes.txt2d import TAText2D
from epyseg.figure.row import Row
from epyseg.tools.logger import TA_logger

logger = TA_logger()

class MyWidget(QtWidgets.QWidget):

    def __init__(self, parent=None, demo=False):
        QtWidgets.QWidget.__init__(self, parent)
        self.shapes_to_draw = []
        self.lastPoint = None
        if demo:
            self.shapes_to_draw.append(Polygon2D(0, 0, 10, 0, 10, 20, 0, 20, 0, 0, color=0x00FF00))
            self.shapes_to_draw.append(
                Polygon2D(100, 100, 110, 100, 110, 120, 10, 120, 100, 100, color=0x0000FF, fill_color=0x00FFFF,
                          stroke=2))
            self.shapes_to_draw.append(Line2D(0, 0, 110, 100, color=0xFF0000, stroke=3))
            self.shapes_to_draw.append(Rect2D(200, 150, 250, 100, stroke=10))
            self.shapes_to_draw.append(Ellipse2D(0, 50, 600, 200, stroke=3))
            self.shapes_to_draw.append(Circle2D(150, 300, 30, color=0xFF0000))
            self.shapes_to_draw.append(PolyLine2D(10, 10, 20, 10, 20, 30, 40, 30, color=0xFF0000, stroke=2))
            self.shapes_to_draw.append(PolyLine2D(10, 10, 20, 10, 20, 30, 40, 30, color=0xFF0000, stroke=2))
            self.shapes_to_draw.append(Point2D(128, 128, color=0xFF0000, stroke=6))
            self.shapes_to_draw.append(Point2D(128, 128, color=0x00FF00, stroke=1))
            self.shapes_to_draw.append(Point2D(10, 10, color=0x000000, stroke=6))

            self.shapes_to_draw.append(Rect2D(0, 0, 512, 512, color=0xFF00FF, stroke=6))
            img0 = Image2D('/media/D/Sample_images/sample_images_PA/trash_test_mem/counter/00.png')
            # img0 = Image2D('D:/dataset1/unseen/100708_png06.png')
            # size is not always respected and that is gonna be a pb but probably size would be ok for journals as they would not want more than 14 pt size ????

            # ci dessous la font size marche mais je comprend rien ... pkoi ça marche et l'autre marche pas les " et ' ou \' ont l'air clef...
            # in fact does not work font is super small
            # img0.setLetter(TAText2D(text='<font face="Comic Sans Ms" size=\'12\' color=yellow ><font size=`\'12\'>this is a <br>test</font></font>'))
            # img0.setLetter(TAText2D(text='<font face="Comic Sans Ms" color=yellow ><font size=12 >this is a <br>test</font></font>'))
            # img0.setLetter(TAText2D("<p style='font-size: large; font-color: yellow;'><b>Serial Number:</b></p> "))
            # a l'air de marcher mais à tester
            # img0.setLetter(TAText2D('<html><body><p><font face="verdana" color="yellow" size="2000">font_face = "verdana"font_color = "green"font_size = 3</font></html>'))

            # try that https://www.learnpyqt.com/examples/megasolid-idiom-rich-text-editor/
            # img0.setLetter(TAText2D(text='<html><font face="times" size=3 color=yellow>test</font></html>'))
            # img0.setLetter(TAText2D(text="<p style='font-size: 12pt; font-style: italic; font-weight: bold; color: yellow; text-align: center;'> <u>Don't miss it</u></p><p style='font-size: 12pt; font-style: italic; font-weight: bold; color: yellow; text-align: center;'> <u>Don't miss it</u></p>"))

            # this is really a one liner but a bit complex to do I find
            # chaque format different doit etre dans un span different --> facile
            # ça marche mais voir comment faire ça
            img0.setLetter(TAText2D(
                text='<p style="text-align:left;color: yellow">This text is left aligned <span style="float:right;font-style: italic;font-size: 8pt;"> This text is right aligned </span><span style="float:right;font-size: 4pt;color:red"> This text is another text </span></p>'))

            img1 = Image2D('/media/D/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
            # img1 = Image2D('D:/dataset1/unseen/focused_Series012.png')
            # img1.setLetter(TAText2D(text="<font face='Comic Sans Ms' size=16 color='blue' >this is a <br>test</font>"))
            # ça ça marche vraiment en fait --> use css to write my text instead of that

            # ça ça a l'air de marcher --> pas trop mal en fait du coup
            # ça a l'air de marcher maintenant --> could use that and do a converter for ezfig ???
            # img1.setLetter(TAText2D(text="<span style='font-size: 12pt; font-style: italic; font-weight: bold; color: yellow; paddind: 20px; text-align: center;'> <u>Don't miss it</u></span><span style='font-size: 4pt; font-style: italic; font-weight: bold; color: #00FF00; paddind: 3px; text-align: right;'> <u>test2</u></span>"))

            # TODO need remove <meta name="qrichtext" content="1" /> from the stuff otherwise alignment is not ok... TODO --> should I offer a change to that ??? maybe not
            test_text = '''
            </style></head><body style=" font-family:'Comic Sans MS'; font-size:22pt; font-weight:400; font-style:normal;">
            <p style="color:#00ff00;"><span style=" color:#ff0000;">toto</span><br />tu<span style=" vertical-align:super;">tu</span></p>
            '''
            img1.setLetter(TAText2D(text=test_text))
            # background-color: orange;
            # span div et p donnent la meme chose par contre c'est sur deux lignes
            # display:inline; float:left # to display as the same line .... --> does that work html to svg
            # https://stackoverflow.com/questions/10451445/two-div-blocks-on-same-line --> same line for two divs

            img2 = Image2D('/media/D/Sample_images/sample_images_PA/trash_test_mem/counter/02.png')
            img3 = Image2D('/media/D/Sample_images/sample_images_PA/trash_test_mem/counter/03.png')
            img4 = Image2D('/media/D/Sample_images/sample_images_PA/trash_test_mem/counter/04.png')
            img5 = Image2D('/media/D/Sample_images/sample_images_PA/trash_test_mem/counter/05.png')
            img6 = Image2D('/media/D/Sample_images/sample_images_PA/trash_test_mem/counter/06.png')
            img7 = Image2D('/media/D/Sample_images/sample_images_PA/trash_test_mem/counter/07.png')
            img8 = Image2D('/media/D/Sample_images/sample_images_PA/trash_test_mem/counter/08.png')
            img9 = Image2D('/media/D/Sample_images/sample_images_PA/trash_test_mem/counter/09.png')
            img10 = Image2D('/media/D/Sample_images/sample_images_PA/trash_test_mem/counter/10.png')

            # img10 = Image2D('/media/D/Sample_images/sample_images_PA/trash_test_mem/counter/10.png')
            # img2 = Image2D('D:/dataset1/unseen/100708_png06.png')
            # img3 = Image2D('D:/dataset1/unseen/100708_png06.png')
            # img4 = Image2D('D:/dataset1/unseen/100708_png06.png')
            # img5 = Image2D('D:/dataset1/unseen/100708_png06.png')
            # img6 = Image2D('D:/dataset1/unseen/focused_Series012.png')
            # img7 = Image2D('D:/dataset1/unseen/100708_png06.png')
            # img8 = Image2D('D:/dataset1/unseen/100708_png06.png')
            # img9 = Image2D('D:/dataset1/unseen/100708_png06.png')
            # img10 = Image2D('D:/dataset1/unseen/focused_Series012.png')

            # is row really different from a panel ??? probably not that different
            # row = img1 + img2

            # self.shapes_to_draw.append(row)
            # self.shapes_to_draw.append(row)

            # pkoi ça creerait pas plutot un panel
            # au lieu de creer une row faut creer des trucs
            # row2 = img4 + img5
            # fig = row / row2
            # fig = col(row, row2, width=512)# ça c'est ok
            # self.shapes_to_draw.append(fig)

            # TODO add image swapping and other changes and also implement sticky pos --> just need store initial pos

            # print(len(row))
            # for img in row:
            #     print(img.boundingRect())

            # fig.setToWidth(512) # bug is really here I do miss something but what and why

            # print('rows', len(fig))
            # for r in fig:
            #     print('bounding rect', r.boundingRect())
            #     print('cols in row', len(r))

            # self.shapes_to_draw.append(fig)

            # peut etre si rien n'est mis juste faire une row avec un panel
            row1 = Row(img0, img1, img2)  # , img6, #, nCols=3, nRows=2

            # marche pas en fait car un truc ne prend pas en charge les figs

            # ça marche donc en fait tt peut etre un panel en fait

            col1 = Column(img4, img5, img6)  # , img6, img6, nCols=3, nRows=2,

            col2 = Column(img3, img7)

            # col1+=col2
            col1 /= col2
            # col1+=img3

            # print('mega begin', panel2.nCols, panel2.nRows, panel2.orientation, len(panel2.images), type(panel2), panel2.boundingRect())
            print('mega begin', len(col1.images), type(col1), col1.boundingRect())

            row2 = Row(img8, img9)

            # row1+=row2
            row1 /= row2
            # row1+= img7

            # all seems fine now

            # panel = Panel(img0)# , img1, img2, img3)  # , img6, #, nCols=3, nRows=2

            # # marche pas en fait car un truc ne prend pas en charge les figs
            #
            # # ça marche donc en fait tt peut etre un panel en fait
            #
            # panel2 = Panel(img4, img5)  # ,

            # panel2.setToWidth(256)
            # panel3.setToWidth(256)
            # panel.setToWidth(256)

            print(type(col1))

            # tt marche
            # should I put align top left or right...

            col1.setToHeight(512)
            # panel3.setToWidth(512)
            row1.setToWidth(512)

            # can I add self to any of the stuff --> check --> if plus --> adds if divide --> stack it --> good idea and quite simple

            # fig = col(panel,panel2,panel3)

            # panel2+=panel
            # print(type(panel2))

            # panel2.setToHeight(512)
            # panel2.setToWidth(512)

            # all seems fine now --> see how I can fix things

            # panel2+=panel3 # bug here cause does not resize the panel properly
            # print('mega final', panel2.nCols, panel2.nRows, panel2.orientation, len(panel2.images), type(panel2), panel2.boundingRect(), panel.boundingRect())
            print('mega final', len(col1.images), type(col1), col1.boundingRect(), row1.boundingRect())

            # on dirait que tt marche

            # maintenant ça marche mais reessayer qd meme
            # panel2.setToWidth(256) # seems still a very modest bug somewhere incompressible height and therefore ratio is complex to calculate for width with current stuff --> see how I can do --> should I ignore incompressible within stuff --> most likely yes... and should set its width and height irrespective of that
            # panel2.setToWidth(512) # seems still a very modest bug somewhere incompressible height and therefore ratio is complex to calculate for width with current stuff --> see how I can do --> should I ignore incompressible within stuff --> most likely yes... and should set its width and height irrespective of that

            # panel2.setToHeight(1024) #marche pas --> à fixer
            # panel2.setToHeight(128) # marche pas --> faut craiment le coder en fait --> voir comment je peux faire ça

            # marche pas non plus en fait --> bug qq part
            # panel2.setToHeight(82.65128578548527) # marche pas --> faut craiment le coder en fait --> voir comment je peux faire ça

            # panel += img7
            # panel -= img0
            # panel -= img1
            # panel -= img10
            # self.shapes_to_draw.append(panel)
            # panel2.set_P1(256, 300)

            # panel2.set_P1(512,0)
            # panel3.set_P1(1024, 0)
            # self.shapes_to_draw.append(panel2)

            self.shapes_to_draw.append(col1)
            self.shapes_to_draw.append(row1)
            # self.shapes_to_draw.append(fig)

        # panel2 | panel
        # panel2 | panel # for swapping panels
        # panel | panel2

        # panel << img3

        # ça ne marche pas pkoi

        # img3 >> panel # does not work --> need implement it in my image2D

        # panel >> img3

        # cannot be dragged --> is it because self.is_set
        # row.packX()
        # row.packY() # ça marche mais finaliser le truc pr que ça soit encore plus simple à gerer
        # img.setP1(10,10) #translate it --> cool
        # self.shapes_to_draw.append(img1)
        # self.shapes_to_draw.append(img2)
        # self.shapes_to_draw.append(img3)

        self.shapes_to_draw.append(Square2D(300, 260, 250, stroke=3))

        self.selected_shape = None

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.save()
        for shape in self.shapes_to_draw:
            shape.drawAndFill(painter)
        painter.restore()

    def cm_to_inch(self, size_in_cm):
        return size_in_cm / 2.54

    def scaling_factor_to_achieve_DPI(self, desired_dpi):
        return desired_dpi / 72

    # TODO maybe get bounding box size in px by measuring the real bounding box this is especially important for raster images so that everything is really saved...
    SVG_INKSCAPE = 96
    SVG_ILLUSTRATOR = 72

    def save(self, path, filetype=None, title=None, description=None, svg_dpi=SVG_INKSCAPE):
        if path is None or not isinstance(path, str):
            logger.error('please provide a valide path to save the image "' + str(path) + '"')
            return
        if filetype is None:
            if path.lower().endswith('.svg'):
                filetype = 'svg'
            else:
                filetype = 'raster'
        dpi = 72  # 300 # inkscape 96 ? check for illustrator --> check

        if filetype == 'svg':
            generator = QSvgGenerator()
            generator.setFileName(path)
            if svg_dpi == self.SVG_ILLUSTRATOR:
                generator.setSize(QSize(595, 842))
                generator.setViewBox(QRect(0, 0, 595, 842))
            else:
                generator.setSize(QSize(794, 1123))
                generator.setViewBox(QRect(0, 0, 794, 1123))

            if title is not None and isinstance(title, str):
                generator.setTitle(title)
            if description is not None and isinstance(description, str):
                generator.setDescription(description)
            generator.setResolution(
                svg_dpi)  # fixes issues in inkscape of pt size --> 72 pr illustrator and 96 pr inkscape but need change size

            painter = QPainter(generator)

            print(generator.title(), generator.heightMM(), generator.height(), generator.widthMM(),
                  generator.resolution(), generator.description(), generator.logicalDpiX())
        else:
            scaling_factor_dpi = 1
            scaling_factor_dpi = self.scaling_factor_to_achieve_DPI(300)
            image = QtGui.QImage(
                QSize(self.cm_to_inch(21) * dpi * scaling_factor_dpi, self.cm_to_inch(29.7) * dpi * scaling_factor_dpi),
                QtGui.QImage.Format_RGB32)
            painter = QPainter(image)  # see what happens in case of rounding of pixels
            painter.scale(scaling_factor_dpi, scaling_factor_dpi)
        painter.setRenderHint(
            QPainter.HighQualityAntialiasing)  # to improve rendering #Antialiasing otherwise or nothing
        self.paint(painter)
        painter.end()
        if filetype != 'svg':
            image.save(path)

    def paint(self, painter):
        painter.save()
        for shape in self.shapes_to_draw:
            shape.drawAndFill(painter)
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

            # check if a shape is selected and only move that
            for shape in reversed(self.shapes_to_draw):
                if shape.contains(self.lastPoint):
                    logger.debug('you clicked shape:' + str(shape))
                    self.selected_shape = shape
                    return

    def mouseMoveEvent(self, event):
        if self.selected_shape is not None:
            self.selected_shape.translate(event.pos() - self.lastPoint)
        self.lastPoint = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.selected_shape = None
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = False


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = MyWidget(demo=True)
    widget.show()


    # all seems to work now --> just finalize things
    # maybe do a panel and allow cols and rows and also allow same size because in some cases people may want that --> offer an option that is fill and
    # TO SAVE AS SVG
    if False:
        widget.save('/media/D/Sample_images/sample_images_svg/out2.svg')
        # TODO make it also paint to a raster just to see what it gives

    # TO SAVE AS RASTER --> quite good
    if False:
        # Tif, jpg and png are supported --> this is more than enouch for now
        # self.paintToFile('/media/D/Sample_images/sample_images_svg/out2.png')
        # self.paintToFile('/media/D/Sample_images/sample_images_svg/out2.tif')
        widget.save('/media/D/Sample_images/sample_images_svg/out2.jpg')
    sys.exit(app.exec_())
