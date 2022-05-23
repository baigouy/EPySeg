# almost all ok but need also

# what should I now do --> implement the fusion visually with mini images --> keep the unique ID of the object and get its AR and get its --> that would be great and let the user decide
# would need draw the hints over the image --> label cells with a unique ID --> see how I can do that
# TODO implement a visual cropper with a rect that one can draw over the image --> should be easy to do and is also very useful...
# find a smart way to label things
# maybe from objects --> label parents by their order in the list and label progeny with the nb of the progeny and so recursively if it contains more than one progeny --> not so easy todo
# the pb is that unique ID will otherwise be complex using has or uuid or alternatively I could retrotranslate uniqueIDs to some simple yet evolving local hashtag code and display it on the image --> think about it
# how to do the preview or just copy rects that has the shape of the object and create a preview rapidly
# see how I can do that
# not so easy
# also not easy to have panels with same size
# also see how to do the external labels in a way that it is not too painful
# the squares can have 'text' and 'img' depending on their content for the preview --> think about that!!!
# or for the labels --> do a duplicate of a row or a column but then link the images --> or link it to cols or row --> think about it --> not that easy for left panels especially if they should span sevral rowsz5qix=
# implement clones and try another tracking algo
# see how I should do that
# if an image is selected --> offer a swap
# if the same type of object is selected --> offer a swap
# otherwise offer / or _ --> easy to explain with text by the way if I can label the objects
# TODO create a list of content that is either a normal list or a browsable list or a tree --> maybe useful and easy to handle -> can also do the drag and drop at this level because all three levels are there --> this is another idea
# TODO --> think about it and decide
# could also give colors to objects/images --> label them with a color and show fusion based on colors --> the color can be derived from the hash (for exemple using the hash as a seed for random)
# think about retrotranslation of things
# for big objects --> put name on the center of it for small objects --> see how to do
# otherwise do and allow cancel and redo --> need clone all objects for that but can be useful (but a lot of mem especiallt with stacks)... or need save them and reload them --> can also be slow and painful
# get the rects and color them all
# --> fairly easy cause can get them iterably the rect --> and clone the object with just a clone of rects of the same --> then apply to that
# --> think about it can I fake it ? so that I don't have to really reload the image --> probably not
# think of How I can do the preview --> the colored rects are cool --> they could behave the same as images 2D but just have a color --> in a way i could also use them with empty sutff
# try get a simple version of the image with rects --> TODO --> just need the rects of the shape and not the intermediate ones --> could make sense


# TODO --> start to add the various options

# TODO --> faire des cloneurs

# try
# all is fairly good already but need improve plots a lot, if possible I guess yes --> can easily make things in a complex way
# finalize all
# almost there --> à tester
# move to front/send to back --> TODO

# TODO --> need remove object from their parent
# also needs check if object is contained in a parent and need find the parent if so
# only take objects that are in the list in fact

# done --> mouse click can handle zoom
# next step is what now???

# TODO if escape is pressed then I should remove the selection --> set it to none --> important

# maybe do both the selection and the autochange of tab so that selection is easy to handle for the user --> if reclicks on the same --> change the selection --> maybe this is the simplest solution and really the optimal thing

# selection in EZF --> see how I can handle that better

#    public String getSelection() {
#         if (isSelectionNull()) {
#             return null;
#         }
#         if (isSelectionFigure()) {
#             return "Figure";
#         }
#         if (isSelectionRow()) {
#             return "Row";
#         }
#         if (isSelectionMontage()) {
#             return "Panel";
#         }
#         if (isSelectionImage()) {
#             return "image";
#         }
#         if (isSelectionArrayList()) {
#             return "ArrayList";
#         }
#         return null;
#     }
#
#     public boolean isSelectionFigure() {
#         return (selected_shape_or_group instanceof Figure);
#     }
#
#     public boolean isSelectionRow() {
#         return (selected_shape_or_group instanceof Row);
#     }
#
#     public boolean isSelectionMontage() {
#         return (selected_shape_or_group instanceof Montage);
#     }
#
#     public boolean isSelectionNull() {
#         return (selected_shape_or_group == null);
#     }
#
#     public boolean isSelectionImage() {
#         return (selected_shape_or_group instanceof MyImage2D);
#     }
#
#     public boolean isSelectionArrayList() {
#         return (selected_shape_or_group instanceof ArrayList);
#     }
#
#     public boolean isSelectionArrayOfFigures() {
#         if (!isSelectionArrayList()) {
#             return false;
#         }
#         return ((ArrayList) selected_shape_or_group).get(0) instanceof Figure;
#     }
#
#     public boolean isSelectionArrayOfRows() {
#         if (!isSelectionArrayList()) {
#             return false;
#         }
#         return ((ArrayList) selected_shape_or_group).get(0) instanceof Row;
#     }
#
#     public boolean isSelectionArrayOfMontage() {
#         if (!isSelectionArrayList()) {
#             return false;
#         }
#         return ((ArrayList) selected_shape_or_group).get(0) instanceof Montage;
#     }
#
#     public boolean isSelectionArrayOfImages() {
#         if (!isSelectionArrayList()) {
#             return false;
#         }
#         return ((ArrayList) selected_shape_or_group).get(0) instanceof MyImage2D;
#     }

# TODO add a scrollbar around it and also allow to zoom --> pretty much a clone of the paint function --> TODO
# TODO handle menus with right click
# TODO add shortcuts and define a small GUI that would handle things in a much simpler way!!!
# if image is not 2D --> ask what to do with it in order to make it 2D and maybe return the corresponding script for the image
# --> maybe it's a good idea
# à tester
# aussi permettre des right clicks pr avoir pleins d'options et peut etre montrer le content d'un panel
# maybe also faire une queue
# maybe faire un organize qui se base sur du texte pour organiser les images
# par exemple pack les images ligne par ligne
# eg:
# test1.tif test2.tif test3.tif --> creates a row with three images
# test1.tif\ntest2.tif\ntest3.tif --> creates a column with those 3 images
# if just the word ROW --> open a dialog to pick images
# also support plots and themes --> TODO
# maybe distinguish deco from other things, e.g. original strings can be always kept and the edited ones can be kept too --> avoid permanently losing data
# TODO define my own save format and add a menu --> would be super useful
# either do this in this GUI or in another --> think about it ???

# TODO mix this with VectorialDrawPane2 to get the best of the two worlds then remove one of them and all other clones such as VectorialDrawPane (take care it's used in epyseg...')

# finir GUI ou pas ??? et aussi permettre de changer forme
# essayer de tout pythonizer au max pr eviter pbs
# voir comment permettre des rotations de formes comme dans EZF et aussi des rotations d'images

# TODO faire un GUI et aussi exporter les images dans ipython --> TODO
# faire un menu etc...
# faire que l'on puisse montrer ou non le menu de telle sorte que l'object puisse aussi servir de viewer en fait je dois déjà avoir ça je pense --> juste faire de la copie et autoriser le DND

# vector graphics ont l'air de marcher mais faudrait vraiment checker ça en détail
# essayer plus de trucs

# ça remarche enfin ça change bcp de choses mais au moins c'est plus constant maintenant et les object sont moins complexes et facilement dessinables sans translation si besoin --> plus facil ea gérer
# maintenant faire pareil avec les vectorial stuff --> TODO

# voir comment je fais dans l'autre ???

'''
here is how to do rotation of shapes --> keep them centered on the stuff of interest --> that is what I want indeed
painter.save();
painter.translate(width/2, height/2);
painter.rotate(textRotation);
painter.drawText(boundingRect, alignmentHorizontal | alignmentVertical, QString(format).arg(progress).arg(rangeMax));
painter.restore();
'''

# TODO en fait pr que le svg marche il faut faire un scaling de tout sinon ça ne marche pas --> des pbs d'alignement --> reflechir à pkoi...

# scale bar seems ok but need careful check somewhere

# TODO check scale bar length but may be ok or a small scaling factor somewhere --> if so need fix it
# TODO MEGA BIG bug in scale when change DPI --> need be fixed somewhere --> need a scale parameter in imaghes and see how to implement it and it needs to propagate to the scale bar --> see how

# allow to draw change size of objects etc...
# and ask if in free floating mode or in packed mode --> TODO
# add a scale bar etc...
# see how to handle fonts and have a default font unless changed by the user --> TODO otherwise apply style
# TODO maybe detect if selected a letter and allow edit it in fact that may be doable with the selection --> can further loop inside --> that may even make things simpler to ues and handle
# TODO add support for rotation, etc ... --> get inspiration from other stuff
# should I allow infinite texts or just as in the other --> maybe just as in the other
# TODO allow to create a label bar from two objects --> must span all of these objects --> good idea and simple to implement and to do and dimension should be the common one or ignore it
# need font support ???
# how can I change font from a formatted text already --> create a doc and change font in it but what about special chars --> is there a way to ignore ??? --> think about it...

# TODO allow the GUI to load locally under jupyter as napari does (unfortunately will not work on colab though...) but still useful
# allow cropping rotation or shear of the image --> TODO
# see how easy it is

# all seems fine now --> just support svg
# MEGA TODO should I ask for bg color when saving as raster ???
# MEGA TODO trim raster image to real size if smaller than page size --> no need to save things that take extra space for nothing... and take up a lot of hard drive space

# TODO add support for scale bars allow saving or serialization --> how can I do that ??? save as a series of commands so that it can always be recreated from scratch from source image, allow export as SVF or various formats with DPI scaling ???
# todo CAN I use css to change font and apply style would be so much simpler and better than current stuff...
# TODO check saving is really saving a svg object --> TODO

# see what else do I need and what I can do
# see what I have in the stuff
# should I allow a table like stuff ??? I think it's not useful as they can be
# do a ROI stuff --> allow to draw annotations on the image
# allow lettering outside --> see how ?


# TODO faire un draw selection --> TODO

import sys
import traceback
import copy # used to clone class instances

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSize, QRect, QRectF, QPoint
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtSvg import QSvgGenerator
from PyQt5.QtWidgets import QMenu

from epyseg.draw.shapes.freehand2d import Freehand2D
from epyseg.draw.shapes.scalebar import ScaleBar
from epyseg.draw.shapes.vectorgraphics2d import VectorGraphics2D
from epyseg.draw.shapes.polygon2d import Polygon2D
from epyseg.draw.shapes.line2d import Line2D
from epyseg.draw.shapes.rect2d import Rect2D
from epyseg.draw.shapes.square2d import Square2D
from epyseg.draw.shapes.ellipse2d import Ellipse2D
from epyseg.draw.shapes.circle2d import Circle2D
from epyseg.draw.shapes.point2d import Point2D
from epyseg.draw.shapes.polyline2d import PolyLine2D
from epyseg.draw.shapes.image2d import Image2D
from epyseg.figure.alignment import packX
from epyseg.figure.column import Column
from epyseg.draw.shapes.txt2d import TAText2D
from epyseg.figure.fig_tools import get_master_bounds, get_master_bounds2
from epyseg.figure.row import Row
import os.path
from epyseg.tools.logger import TA_logger  # logging

# en fait pas mal et faire autour un GUI avec un menu qui contient ça
# voir dans TA comment j'ai fait ça...

# can I use that to plot with TA the vectors on the image --> in a way that's a bit redundant compared to the quiver of matplotlib, but there are a lot of things that are hard to handle in matloplib and it's not vectorial...

logger = TA_logger()

# can I use this also to render offline a drawing on an image --> try it maybe
# how do I know the size of this stuff ??? --> need update its size according to its content and probably the zoom --> TODO

class MyWidget(QtWidgets.QWidget):

    def __init__(self, parent=None, resizable=True):
        QtWidgets.QWidget.__init__(self, parent)
        self.shapes_to_draw = []
        self.lastPoint = None
        self.resizable = resizable
        self.is_sticky = False
        self._stored_shape_n_its_coords = {}

        self.dragging = False
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
        self.scale = 1.
        self.selected_shape = None
        # if True:
        #     # add right click options
        #     self.contextMenuEvent()

    def set_scale(self, scale):
        if scale < 0.01:
            scale = 0.01
        if scale > 50:
            scale = 50
        self.scale = scale

    # a right click menu
    # TODO --> do a more realistic menu
    def contextMenuEvent(self, event):
        contextMenu = QMenu(self)
        newAct = contextMenu.addAction("New")
        openAct = contextMenu.addAction("Open")
        quitAct = contextMenu.addAction("Quit")
        move_to_front = contextMenu.addAction("Move To Front")
        # copy_ = contextMenu.addAction("Duplicate") # ça ne marche pas
        send_to_back = contextMenu.addAction("Send To Back")
        action = contextMenu.exec_(self.mapToGlobal(event.pos()))
        if action == quitAct:
            self.close()
        if action == move_to_front:
            self.move_to_front()
        if action == send_to_back:
            self.send_to_back()

        # marche pas --> il faut hardcoder la copie comme dans java --> TODO
        # if action == copy_:
        #     self.copy()
        # ça marche --> est-ce que je peux recupérer l'object sous la souris ??? pr faire des actions differentes en fonction
        # maybe offer different options for different levels in the same menu --> gain of time and no need for cycling the selection

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.save()
        # handles zoom
        if self.scale != 1:
            painter.scale(self.scale, self.scale)
        for shape in self.shapes_to_draw:
            shape.draw(painter)
        self.draw_selection(painter)
        painter.restore()

    # TODO --> see how to do that ???
    # the other possibility is to draw from within the object --> can be draw its skeleton by the way --> would allow to handle more shapes
    # see how I can handle selection groups --> see how I was doing in EZF by the way because it was really working well!!
    def draw_selection(self, painter):
        if self.selected_shape is not None:
            # could also color according to shapes
            rect = None
            # the rect is this one
            # in some cases I need to have access to the original object size --> need keep it but the best is that bounds mimic current object shape
            if isinstance(self.selected_shape, QRectF):
                # this is a rect2D --> I can therefore easily draw it --> TODO
                rect = self.selected_shape.boundingRect()
            else:
                if isinstance(self.selected_shape, list):
                    # print('master rect')
                    rect = get_master_bounds2(self.selected_shape)

            # print(rect)
            if rect is not None:
                painter.setPen(QColor(255, 0, 0))
                painter.drawRect(rect)
            else:
                print('shape not supported yet for drawing selection')

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
                filetype = os.path.splitext(path)[1]
        # dpi = 72  # 300 # inkscape 96 ? check for illustrator --> check

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

            # print(generator.title(), generator.heightMM(), generator.height(), generator.widthMM(),
            #       generator.resolution(), generator.description(), generator.logicalDpiX())
        else:
            scaling_factor_dpi = 1
            scaling_factor_dpi = self.scaling_factor_to_achieve_DPI(300)

            # in fact take actual page size ??? multiplied by factor
            # just take real image size instead

            # image = QtGui.QImage(QSize(self.cm_to_inch(21) * dpi * scaling_factor_dpi, self.cm_to_inch(29.7) * dpi * scaling_factor_dpi), QtGui.QImage.Format_RGBA8888) # minor change to support alpha # QtGui.QImage.Format_RGB32)

            # NB THE FOLLOWING LINES CREATE A WEIRD ERROR WITH WEIRD PIXELS DRAWN some sort of lines NO CLUE WHY
            img_bounds = self.updateBounds()
            image = QtGui.QImage(
                QSize(img_bounds.width() * scaling_factor_dpi, img_bounds.height() * scaling_factor_dpi),
                QtGui.QImage.Format_RGBA8888)  # minor change to support alpha # QtGui.QImage.Format_RGB32)
            # print('size at dpi',QSize(img_bounds.width() * scaling_factor_dpi, img_bounds.height()* scaling_factor_dpi))
            # QSize(self.cm_to_inch(0.02646 * img_bounds.width())
            # self.cm_to_inch(0.02646 * img_bounds.height())
            # need convert pixels to inches
            # is there a rounding error

            # force white bg for non jpg
            try:
                # print(filetype.lower())
                # the tif and png file formats support alpha
                if not filetype.lower() == '.png' and not filetype.lower() == '.tif' and not filetype.lower() == '.tiff':
                    image.fill(QColor.fromRgbF(1, 1, 1))
                else:
                    # image.fill(QColor.fromRgbF(1, 1, 1, alpha=1))
                    # image.fill(QColor.fromRgbF(1, 1, 1, alpha=1))
                    # TODO KEEP in fact image need BE FILLED WITH TRANSPARENT OTHERWISE GETS WEIRD DRAWING ERRORS
                    # TODO KEEP SEE https://stackoverflow.com/questions/13464627/qt-empty-transparent-qimage-has-noise
                    # image.fill(qRgba(0, 0, 0, 0))
                    image.fill(QColor.fromRgbF(0, 0, 0, 0))
            except:
                pass
            painter = QPainter(image)  # see what happens in case of rounding of pixels
            # painter.begin()
            painter.scale(scaling_factor_dpi, scaling_factor_dpi)
        painter.setRenderHint(QPainter.HighQualityAntialiasing)  # to improve rendering quality
        self.paint(painter)
        painter.end()
        if filetype != 'svg':
            image.save(path)

    # make it sticky and allow restore or not on drag failed and depending on the mode used
    def sticky(self, mode):
        if not self.is_sticky:
            return
        if mode == 'INIT':
            # store coords of objects and reintegrate them
            self._stored_shape_n_its_coords = {}
            if self.selected_shape is not None:
                if isinstance(self.selected_shape, list):
                    for shp in self.selected_shape:
                        self._stored_shape_n_its_coords[shp] = shp.get_P1()
                else:
                    # cannot be added because unhashable... -> how can I do that
                    self._stored_shape_n_its_coords[self.selected_shape] = self.selected_shape.get_P1()
        else:
            # restore coords
            if self._stored_shape_n_its_coords:
                # restore coords of selection
                for shp, pos in self._stored_shape_n_its_coords.items():
                    shp.set_P1(pos)
                # update
                self.update()
                self.update_size()



        # print('bob')

    def paint(self, painter):
        painter.save()
        for shape in self.shapes_to_draw:
            shape.draw(painter)
        painter.restore()

    def update_size(self):
        if not self.resizable:
            return
        bounds = self.updateBounds()
        # print(bounds)
        size = QSize(bounds.width() * self.scale + 2, bounds.height() * self.scale + 2)
        # print("-->",size)
        self.resize(size)

    # in fact upon
    # do not override this because it's needed actually
    def updateBounds(self):
        return get_master_bounds(self.shapes_to_draw)
        # # loop over shape to get bounds to be able to draw an image
        # bounds = QRectF()
        # max_width = 0
        # max_height = 0
        # for shape in self.shapes_to_draw:
        #    rect = shape.boundingRect()
        #    max_width = max(max_width, rect.x()+rect.width())
        #    max_height = max(max_height, rect.y()+rect.height())
        # bounds.setWidth(max_width)
        # bounds.setHeight(max_height)
        #
        #
        # # print(bounds.width())
        # # width = self.image.size().width()
        # # height = self.image.size().height()
        # # top = self.geometry().x()
        # # left = self.geometry().y()
        # # self.setGeometry(top, left, width*self.scale, height*self.scale)
        # # self.setGeometry(top, left, max_width*self.scale, max_height*self.scale)
        #
        # # looks like I have a big bug
        # # en fait la geom c'est nul vaudrait mieux changer la size ???
        # # or maybe put it as fixed size --> think about it but no big deal
        # # but need also implement scale
        #
        # # in fact whenever I add an element to it I need to call this function --> so I need to go through a function and not access the set of objects directly
        # # self.setGeometry(top, left, max_width*10, max_height*10) # TODO implement scale # very dirty hack but allows to have the complete image be drawn --> see the proper way to do that but I guess I should fit the max of the scrollarea visible region or content and set it to that # VERY DIRTY BUT OK FOR NOW MEGA TODO IMPROVE THAT
        # # self.setMinimumWidth(max_width)
        # # self.setMinimumWidth(max_height)
        #
        # # how can I update this when the user is scrolling or changing the panel size
        # # maybe the size of this stuff should be that of scrollarea if visible region is bigger --> TODO
        #
        # # becomes 0 --> why ???
        # # print(top, left, left-max_width, top-max_height)
        # # print(self.geometry())
        #
        # # need also connect it to the scrollable --> TODO
        #
        # return bounds

    def is_ctrl_modifier(self):
        modifier = self.get_modifier()
        # if modifiers == (QtCore.Qt.ControlModifier |
        #               QtCore.Qt.ShiftModifier):
        # if modifiers == QtCore.Qt.ShiftModifier:
        if modifier == QtCore.Qt.ControlModifier:
            return True
        return False

    def get_modifier(self):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        return modifiers

    # TODO detect also ctrl + click
    # the other possibility is to use a different shortcut for going inside the object or double  click to further edit it --> would allow to ease selection and then when done the stuff is over
    # or do the same as previously because that works and allows a lot of things to be done

    # --> ok
    # if None --> either it is last or
    def get_inner_at_coord(self, shape_to_screen, last_click):
        if not shape_to_screen.contains(last_click):
            return None
        try:
            # object is not iterable --> skip and move to next
            some_object_iterator = iter(shape_to_screen)
        except TypeError as te:
            # we have reached a non iterable object --> this is therefore the last level object --> ignore
            return shape_to_screen
        # if object is in object --> need iter it too --> see how I can do that
        # implement the clones
        # try new tracking algo
        for shp in shape_to_screen:
            # can I use intersect to be sure the ovelap is ok
            # if not shp.boundingRect().intersects(current_sel.boundingRect()):
            if not shp.boundingRect().contains(last_click):
                continue
            if shp.boundingRect().contains(last_click):
                # c'est ici il faut retourner seulement si
                # need further check this
                return shp
        # nothing found --> return None
        return None


    # il y a un big bug --> surtout quand image is big
    def get_all_inner_objects_at_position(self, shape_to_screen, last_click):
        shapes_and_progeny = [shape_to_screen]
        if not self.is_iterable(shape_to_screen):
            return shapes_and_progeny
        else:
            inner = shape_to_screen
            while True:
                if not self.is_iterable(inner):
                    return shapes_and_progeny
                else:
                    inner = self.get_first_encounter_at_click(inner, last_click)
                    shapes_and_progeny.append(inner)
        return shapes_and_progeny


    def get_first_encounter_at_click(self, shape_to_screen, last_click):
        for shape in shape_to_screen:
            if shape.boundingRect().contains(last_click):
                return shape
        return None




    # I somehow need check if the object contains the object --> because can be a problem for overlaping shapes --> see how to handle that

    # def get_progeny(self):
    #
    # def shape_contains_another(self, shape_to_check, cur_sel):
    #     # if current selection is not contained in shape, then
    #
    #     return False
    #
    # def get_inner_object_at(self, shape_to_screen, last_click):
    #     try:
    #         # object is not iterable --> skip and move to next
    #         some_object_iterator = iter(shape_to_screen)
    #     except TypeError as te:
    #         # we have reached a non iterable object --> this is therefore the last level object --> ignore
    #         return shape_to_screen
    #     for shp in shape_to_screen:
    #
    #
    #
    def is_iterable(self, shape_to_screen):
        try:
            # object is not iterable --> skip and move to next
            some_object_iterator = iter(shape_to_screen)
            return True
        except TypeError as te:
            # we have reached a non iterable object --> this is therefore the last level object --> ignore
            return False
        return False


    # simpler way is to get all possible inners and act according to result

    # need find the level of an object and go deeper!!!
    # try if parent --> maybe while is parent --> take the object
    # alternatively match all possible objects below and if this is the last level then skip...
    def get_inner_shape(self, shape_to_screen, current_sel, last_click):
        if not shape_to_screen.boundingRect().intersects(current_sel.boundingRect()):
            return None
        # need find if object is lower level
        lower_level_object = shape_to_screen
        while True:
            old_object = lower_level_object
            lower_level_object = self.get_inner_at_coord(lower_level_object, last_click)
            if old_object == lower_level_object:
                return lower_level_object
            # nothing below --> ignore
            if lower_level_object is None:
                # nothing found --> need move to the next object
                return None
            if lower_level_object == current_sel:
                lower_level_object = self.get_inner_at_coord(lower_level_object, last_click)
                # the current selected object is the selected object --> need select the parent
                # if lower_level_object == current_sel:
                    # try going deeper or return current object
                if lower_level_object is None:
                    # the object has never been found!!! --> return None --> the new object is not a child of the current sel
                    return None
                else:
                    # need check if object is contained in object and if so
                    return lower_level_object
            else:
                # if not parent need check if object is contained otherwise return the parent
                # we found a potential click --> as long as object is not found in then keep it
                # if object is not contained in lower_object then return object otherwise go deeper the last level where the object it
                # if object is at a lower level then loop
                check_deeper = lower_level_object
                while True:
                    old_chck = check_deeper
                    check_deeper = self.get_inner_at_coord(check_deeper, last_click)
                    if check_deeper == current_sel:
                        return self.get_inner_at_coord(check_deeper, last_click)
                    if check_deeper is None:
                        break
                    if old_chck == check_deeper:
                        # return lower_level_object
                        break

                return lower_level_object

        # ça a l'air de marcher --> check if something was dragged before or not
        # try:
        #     # object is not iterable --> skip and move to next
        #     some_object_iterator = iter(shape_to_screen)
        # except TypeError as te:
        #     return None
        # # if object is in object --> need iter it too --> see how I can do that
        # # implement the clones
        # # try new tracking algo
        # for shp in shape_to_screen:
        #     # can I use intersect to be sure the ovelap is ok
        #     # if not shp.boundingRect().intersects(current_sel.boundingRect()):
        #     if not shp.boundingRect().contains(last_click):
        #         continue
        #     if shp.boundingRect().contains(last_click):
        #         # c'est ici il faut retourner seulement si
        #         # need further check this
        #         bckup_shp = shp
        #
        #         return shp
        # # nothing found --> return None
        return None
        # can be done recursively from outside --> how to always find the level below an object or parent if lowest level
        # if objects are equal --> try go deeper else still need make sure if it is a subset or


    # nb shall I get selection based on scale --> probably yes --> TODO implement that
    def mousePressEvent(self, event):
        self.dragging = False
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
            self.lastPoint.setX(self.lastPoint.x() / self.scale)
            self.lastPoint.setY(self.lastPoint.y() / self.scale)

            # print('before', self.selected_shape)
            # check if a shape is selected and only move that
            for shape in reversed(self.shapes_to_draw):
                if shape.contains(self.lastPoint):
                    # logger.debug('you clicked shape:' + str(shape))
                    if self.selected_shape is None or not self.is_ctrl_modifier():
                        # selection has not changed --> ignoring changes
                        if isinstance(self.selected_shape, list) and shape in self.selected_shape:
                            return
                        if isinstance(self.selected_shape, list):
                            # check that click is not in master rect of selection
                            if get_master_bounds2(self.selected_shape).contains(self.lastPoint):
                                return
                        # check if an object is in
                        elements_below_click = self.get_all_inner_objects_at_position(shape, self.lastPoint)
                        print('elements_below_click',elements_below_click)
                        if elements_below_click and self.selected_shape in elements_below_click: # or if is inside --> need pool it out
                            # try:
                            #     some_object_iterator = iter(shape)
                            # except TypeError as te:
                            #     continue
                            # go deeper
                            print('going deeper in shape crappy still')
                            # for elm in shape:
                            #     # if shape is itslef iterable I need check further in until can't go deeper and then stop --> see how to do that --> do that in a nested way
                            #     # only do things if the shapes overlap otherwise skip
                            #     # if inner is selected need make it sticky
                            #     if elm.boundingRect().contains(self.lastPoint):
                            #         shape = elm # in fcat this is selection
                            #         self.selected_shape = shape # when I select it is moved but why is that so...
                            #         return # must not allow fusion when shape is contained within

                            # or maybe use double click to go deeper --> would simplify a lot my code in fact !!! and makes total sense in fact --> see how that is implemented

                            # print('$'*20, )
                            # selection = [type(obj) for obj in elements_below_click]
                            # print(selection)

                            if self.selected_shape in elements_below_click:
                                idx = elements_below_click.index(self.selected_shape)
                                if idx == len(elements_below_click)-1:
                                    self.selected_shape = elements_below_click[0]  # when I select it is moved but why is that so...
                                    return
                                else:
                                    self.selected_shape =elements_below_click[idx+1]
                                    return

                            # inner_shape = self.get_inner_shape(shape, self.selected_shape, self.lastPoint)
                            # print('#' * 20, type(shape), type(self.selected_shape), type(inner_shape))
                            # if inner_shape is not None:
                            #     self.selected_shape = inner_shape  # when I select it is moved but why is that so...
                            #     return # must not allow fusion when shape is contained within

                            # if there is overlap need deep check
                            # easy to check if in if they have overlapping rects and also with

                            # if sh

                            # try:
                            #     some_object_iterator = iter(shape)
                            # except TypeError as te:
                            #     # print(shape, 'is not iterable')
                            #     # if object is not iterable --> need loop again
                            #     self.selected_shape = shape

                        else:
                            self.selected_shape = shape
                        self.sticky('INIT')
                        return
                    else:
                        # print('in here')
                        self.append_to_selection(shape)
                        print(self.selected_shape)
                        self.sticky('INIT')
                        return

    def append_to_selection(self, shape):
        if shape is None:
            return
        if isinstance(self.selected_shape, list):
            # do not add the shape more than once to avoid weird issues
            if shape not in self.selected_shape:
                self.selected_shape.append(shape)
            else:
                self.selected_shape.remove(shape)
                if not self.selected_shape:
                    self.selected_shape = None
        else:
            if shape == self.selected_shape:
                self.selected_shape = None
                return
            self.selected_shape = [self.selected_shape, shape]
        # self.selected_shape = list(set(self.selected_shape)) # remove dupes

    def remove_sel(self):
        update_required = self.selected_shape is not None
        self.selected_shape = None
        if update_required:
            self.update()

    def mouseMoveEvent(self, event):
        if self.selected_shape is not None:
            self.dragging = True
            if isinstance(self.selected_shape, list):
                for element in self.selected_shape:
                    trans = event.pos()
                    trans.setX(trans.x() / self.scale)
                    trans.setY(trans.y() / self.scale)
                    element.translate(trans - self.lastPoint)
            else:
                trans = event.pos()
                trans.setX(trans.x() / self.scale)
                trans.setY(trans.y() / self.scale)
                self.selected_shape.translate(trans - self.lastPoint)
            # need update bounds of the panel
            # self.updateBounds()
            self.update_size()
        self.lastPoint = event.pos()
        self.lastPoint.setX(self.lastPoint.x() / self.scale)
        self.lastPoint.setY(self.lastPoint.y() / self.scale)
        self.update()

    def get_shape_at_coord(self, coords, ignore_cur_sel=True):
        # this makes sure the returned object is in the list
        # need remove both objects from the list and return one new object instead
        for shape in reversed(self.shapes_to_draw):
            if shape.contains(coords):
                if ignore_cur_sel and (shape == self.selected_shape or (
                        isinstance(self.selected_shape, list) and shape in self.selected_shape)):
                    continue
                else:
                    return shape

    def mouseReleaseEvent(self, event):
        # self.selected_shape = None # we don't want to do that
        if event.button() == QtCore.Qt.LeftButton:

            # TODO also handle drag and drop of elements
            # and make sticky or not
            # TODO

            # get the dragged stuff and the dropped stuff and take action
            # get element dropped over --> the first

            if self.drawing:
                coords = event.pos()
                coords.setX(self.lastPoint.x() / self.scale)
                coords.setY(self.lastPoint.y() / self.scale)
                shape_dropped_over = self.get_shape_at_coord(coords, ignore_cur_sel=True)

                if shape_dropped_over is None:
                    self.sticky('RESTORE')

                print('shape dropped over ', type(shape_dropped_over))
                print('shape dragged ', type(self.selected_shape))
                # if isinstance(self.selected_shape, list):
                #     print('shape dragged 0 ',type(self.selected_shape[0]))
                #
                #     # dirty try to fuse them
                #     try:
                #         # " maybe ignore too complex fusions"
                #         shape_dropped_over |= self.selected_shape[0]
                #         if self.self.selected_shape[0] in self.shapes_to_draw:
                #             self.shapes_to_draw.remove(self.self.selected_shape[0])
                #         print('fusion successful')
                #         # upon success --> remove sel and update fig and figure size too or update sel to the fusion ???
                #         self.selected_shape = None
                #         self.update_size()
                #
                #     except:
                #         traceback.print_exc()
                #         print('fusion failed')
                # else:
                # dirty try to fuse them

                # see how to really do that and make sure all is ok ...
                # see how to handle all the dnds --> maybe smartest is a tree --> with row or col and its content that is itself a row or a col --> not bad in fact --> would also make it easy to move or change components from one to another
                # TODO maybe allow convert row to panel --> quite easy to do most likely in fact
                # make text objects like any other objects so that they can span an entire figure or not --> good idea and simpler to handle in fact!!!
                if self.dragging and not isinstance(self.selected_shape, list):
                    if self.is_image_type(shape_dropped_over) and self.is_image_type(self.selected_shape):
                        try:
                            # do not do fusion for now
                            # raise Exception
                            # would be great to have a preview of all options possible --> could color differently the different types of fusion maybe or ask the user to select what to fuse it too
                            # maybe panels are useful too????
                            print('fusion of', type(shape_dropped_over), type(self.selected_shape))
                            # try also the bayesian tracking algorithm and the other tracking algos

                            # need remove the two shapes from the list and add the new one
                            if shape_dropped_over in self.shapes_to_draw:
                                self.shapes_to_draw.remove(shape_dropped_over)

                            # need check that the shape is an image type for both the target and the parent --> TODO
                            shape_dropped_over |= self.selected_shape
                            # need remove all the parents from the list
                            if self.selected_shape in self.shapes_to_draw:
                                self.shapes_to_draw.remove(self.selected_shape)

                            # if shape_dropped_over in self.shapes_to_draw:
                            # self.shapes_to_draw.remove(shape_dropped_over)

                            print('fusion successful')
                            # upon success --> remove sel and update fig and figure size too
                            self.selected_shape = None
                        except:
                            traceback.print_exc()
                            print('fusion failed')
                            self.sticky('RESTORE')
                        finally:
                            if shape_dropped_over is not None:
                                self.shapes_to_draw.append(shape_dropped_over)
                            self.update_size()
                    else:
                        self.sticky('RESTORE')

                else:
                    print("can't handle groups for fusion...")
                    self.sticky('RESTORE')

            # print('content of self.shapes_to_draw --> ', self.shapes_to_draw) # see how to detect all the things that need be removed

            # depending on dropped shape --> allow or not an action to occur!!!
            # see also how to handle plots --> TODO



            self.drawing = False
            self.update()
            self.dragging = False

    def is_image_type(self, object):
        if isinstance(object, Row):
            return True
        if isinstance(object, Column):
            return True
        if isinstance(object, Image2D):
            return True
        return False

    # this is ok but I may need to add buttons for it too...
    def send_to_back(self):
        if self.selected_shape is None:
            return
        if isinstance(self.selected_shape,list):
            for sel in self.selected_shape:
                # need change order in the list
                self.shapes_to_draw.remove(sel)
                self.shapes_to_draw.insert(0,sel)
        else:
            self.shapes_to_draw.remove(self.selected_shape)
            self.shapes_to_draw.insert(0, self.selected_shape)
        self.update()

    def move_to_front(self):
        if self.selected_shape is None:
            return
        if isinstance(self.selected_shape,list):
            for sel in self.selected_shape:
                # need change order in the list
                self.shapes_to_draw.remove(sel)
                self.shapes_to_draw.append(sel)
        else:
            self.shapes_to_draw.remove(self.selected_shape)
            self.shapes_to_draw.append(self.selected_shape)
        self.update()

    # ne marche pas --> faut tout hardcoder je pense
    # https://stackoverflow.com/questions/56542287/attempt-to-pickle-unknown-type-while-creating-a-deepcopy
    # https://stackoverflow.com/questions/10618956/copy-deepcopy-raises-typeerror-on-objects-with-self-defined-new-method/10622689#10622689
    def copy(self):
        if self.selected_shape is None:
            return
        if isinstance(self.selected_shape, list):
            for sel in self.selected_shape:
                cp = copy.deepcopy(sel)
                cp.translate(25, 25)
                self.shapes_to_draw.append(cp)
        else:
            # NB deep copy does not seem to work...
            cp = copy.deepcopy(self.selected_shape)
            cp.translate(25,25)
            self.shapes_to_draw.append(cp)
        self.update()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = MyWidget(resizable=False)

    demo = True
    if demo:
        shapes_to_draw = []
        shapes_to_draw.append(Polygon2D(0, 0, 10, 0, 10, 20, 0, 20, 0, 0, color=0x00FF00))
        shapes_to_draw.append(
            Polygon2D(100, 100, 110, 100, 110, 120, 10, 120, 100, 100, color=0x0000FF, fill_color=0x00FFFF,
                      stroke=2))
        shapes_to_draw.append(Line2D(0, 0, 110, 100, color=0xFF0000, stroke=3))
        shapes_to_draw.append(Rect2D(200, 150, 250, 100, stroke=10, fill_color=0xFF0000))
        shapes_to_draw.append(Ellipse2D(0, 50, 600, 200, stroke=3))
        shapes_to_draw.append(Circle2D(150, 300, 30, color=0xFF0000, fill_color=0x00FFFF))
        shapes_to_draw.append(PolyLine2D(10, 10, 20, 10, 20, 30, 40, 30, color=0xFF0000, stroke=2))
        shapes_to_draw.append(PolyLine2D(10, 10, 20, 10, 20, 30, 40, 30, color=0xFF0000, stroke=2))
        shapes_to_draw.append(Point2D(128, 128, color=0xFF0000, fill_color=0x00FFFF, stroke=0.65))
        shapes_to_draw.append(Point2D(128, 128, color=0x00FF00, stroke=0.65))
        shapes_to_draw.append(Point2D(10, 10, color=0x000000, fill_color=0x00FFFF, stroke=3))

        shapes_to_draw.append(Rect2D(0, 0, 512, 512, color=0xFF00FF, stroke=6))
        img0 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/00.png')

        inset = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
        inset2 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
        inset3 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
        # inset.setToHeight(32)
        # check inset

        scale_bar = ScaleBar(30, '<font color="#FF00FF">10µm</font>')
        scale_bar.set_P1(0, 0)
        # scale_bar.set_scale(self.get_scale())
        # # scale_bar.set_P1(self.get_P1().x()+extra_space, self.get_P1().y()+extra_space)

        # can I add things to an image in a smart way

        img0.add_object(scale_bar, Image2D.TOP_LEFT)
        # scale_bar0 = ScaleBar(30, '<font color="#FF00FF">10µm</font>')
        # scale_bar0.set_P1(0, 0)
        # img0.add_object(scale_bar0, Image2D.TOP_LEFT)
        img0.add_object(inset3, Image2D.TOP_LEFT)

        # all seems fine and could even add insets to it --> not so hard I guess
        # check

        # see how to handle insets --> in a way they can be classical images and one should just determine what proportion of the parent image width they should occupy -->
        # need a boolean or need set fraction of orig --> jut one variable
        # maybe also need draw a square around it of a given size --> see how to do that ???

        # img0.add_object(TAText2D(text='<font color="#FF0000">top right</font>'), Image2D.TOP_RIGHT)
        # img0.add_object(TAText2D(text='<font color="#FF0000">top right2</font>'), Image2D.TOP_RIGHT)
        # img0.add_object(TAText2D(text='<font color="#FF0000">top right3</font>'), Image2D.TOP_RIGHT)

        img0.add_object(inset,
                        Image2D.BOTTOM_RIGHT)  # ça marche meme avec des insets mais faudrait controler la taille des trucs... --> TODO
        # img0.add_object(inset2, Image2D.BOTTOM_RIGHT)  # ça marche meme avec des insets mais faudrait controler la taille des trucs... --> TODO
        # img0.add_object(TAText2D(text='<font color="#FF0000">bottom right</font>'), Image2D.BOTTOM_RIGHT)
        # img0.add_object(TAText2D(text='<font color="#FF0000">bottom right2</font>'), Image2D.BOTTOM_RIGHT)
        img0.add_object(TAText2D(text='<font color="#FF0000">bottom right3</font>'), Image2D.BOTTOM_RIGHT)

        # ask whether a border should be drawn for the inset or not ??? and ask for its width...

        # ça a l'air de marcher mais voir comment faire pour gérer

        # img0.add_object(TAText2D(text='<font color="#FF0000">bottom left1</font>'), Image2D.BOTTOM_LEFT)
        # img0.add_object(TAText2D(text='<font color="#FF0000">bottom left2</font>'), Image2D.BOTTOM_LEFT)
        # img0.add_object(TAText2D(text='<font color="#FF0000">bottom left3</font>'), Image2D.BOTTOM_LEFT)

        # seems to work --> just finalize things up...

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
        # img0.setLettering(TAText2D(
        #     text='<p style="text-align:left;color: yellow">This text is left aligned <span style="float:right;font-style: italic;font-size: 8pt;"> This text is right aligned </span><span style="float:right;font-size: 4pt;color:red"> This text is another text </span></p>'))
        img0.setLettering('<font color="red">A</font>')
        # letter
        img0.annotation.append(Rect2D(88, 88, 200, 200, stroke=3, color=0xFF00FF))
        img0.annotation.append(Ellipse2D(88, 88, 200, 200, stroke=3, color=0x00FF00))
        img0.annotation.append(Circle2D(33, 33, 200, stroke=3, color=0x0000FF))
        img0.annotation.append(Line2D(33, 33, 88, 88, stroke=3, color=0x0000FF))
        img0.annotation.append(Freehand2D(10, 10, 20, 10, 20, 30, 288, 30, color=0xFFFF00, stroke=3))
        # img0.annotation.append(PolyLine2D(10, 10, 20, 10, 20, 30, 288, 30, color=0xFFFF00, stroke=3))
        img0.annotation.append(Point2D(128, 128, color=0xFFFF00, stroke=6))
        # everything seems to work but do check

        img1 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
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
        img1.setLettering(TAText2D(text=test_text))

        # background-color: orange;
        # span div et p donnent la meme chose par contre c'est sur deux lignes
        # display:inline; float:left # to display as the same line .... --> does that work html to svg
        # https://stackoverflow.com/questions/10451445/two-div-blocks-on-same-line --> same line for two divs

        img2 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/02.png')

        # crop is functional again but again a packing error
        img2.crop(left=60)
        img2.crop(right=30)
        img2.crop(bottom=90)
        img2.crop(top=60)
        # img2.crop(all=0) # reset crop
        # img2.crop(top=0) # reset crop --> seems ok
        # now seems ok --> see how to do that with figures/vector graphics ...
        # img2.crop(right=60)
        # img2.crop(bottom=60)
        img3 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/03.png')
        img4 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/04.png')
        img5 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/05.png')
        img6 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/06.png')
        img7 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/07.png')
        img8 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/08.png')

        # reference point is the original image and stroke should be constant irrespective of zoom --> most likely need the scaling factor too there
        # reference size is also the underlying original image --> TODO
        # img8.annotation.append(Line2D(0, 0, 110, 100, color=0xFF0000, stroke=3))
        img8.annotation.append(Rect2D(60, 60, 100, 100, stroke=20, color=0xFF00FF))
        # need make the scale rese
        # img8.annotation.append(Ellipse2D(0, 50, 600, 200, stroke=3))

        img9 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/09.png')
        img10 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/10.png')
        # Data for plotting
        import numpy as np
        import matplotlib.pyplot as plt

        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)

        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(xlabel='time (s)', ylabel='voltage (mV)',
               title='About as simple as it gets, folks')
        ax.grid()

        # plt.show()
        # fig.savefig("test.png")
        # plt.show()
        # ça marche --> voici deux examples de shapes

        # first graph test --> TODO improve that
        graph2d = VectorGraphics2D(fig)
        graph2d.crop(all=20)  # not great neither

        vectorGraphics = VectorGraphics2D('/E/Sample_images/sample_images_svg/cartman.svg')

        # nb cropping marche en raster mais pas en svg output --> besoin de faire un masque d'ecretage --> pourrait aussi dessiner un rectangle de la meme taille de facon à le faire

        # TODO KEEP unfortunately cropping does not work when saved as svg but works when saved as raster...
        vectorGraphics.crop(left=10, right=30, top=10, bottom=10)
        animatedVectorGraphics = VectorGraphics2D('/E/Sample_images/sample_images_svg/animated.svg')

        # bug cause shears the stuff --> would need crop the other dimension too to maintain AR
        animatedVectorGraphics.crop(left=30)  # , top=20, bottom=20

        # self.shapes_to_draw.append(graph2d)

        # img10 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/10.png')
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
        row1 = Row(img0, img1, img2, graph2d,
                   animatedVectorGraphics)  # , graph2d, animatedVectorGraphics  # , img6, #, nCols=3, nRows=2 #le animated marche mais faut dragger le bord de l'image mais pas mal qd meme
        # see how I should handle size of graphs but I'm almost there

        # marche pas en fait car un truc ne prend pas en charge les figs
        # ça marche donc en fait tt peut etre un panel en fait

        col1 = Column(img4, img5, img6, vectorGraphics)  # , vectorGraphics# , img6, img6, nCols=3, nRows=2,
        col2 = Column(img3, img7, img10)
        #
        # col1.setLettering('<font color="#FFFFFF">A</font>')

        # col1+=col2
        col1 /= col2
        # col1+=img3

        # print('mega begin', panel2.nCols, panel2.nRows, panel2.orientation, len(panel2.images), type(panel2), panel2.boundingRect())
        # print('mega begin', len(col1.images), type(col1), col1.boundingRect())

        # ok need increment and need see how to change the font of the stuff and bg color and fg color --> TODO but ok for now
        row2 = Row(img8, img9)
        row2.setLettering('<font color="#FFFFFF">a</font>')

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

        # print(type(col1))

        # tt marche
        # should I put align top left or right...
        # should I put align top left or right...

        col1.setToHeight(512)  # creates big bugs now
        # panel3.setToWidth(512)
        row1.setToWidth(512)

        # c'est ici qu'il y a un pb de taille --> à fixer en fait --> TODO
        print('size row1', row1)

        # row1.setLettering('<font color="#FF00FF">B</font>')
        # row1.setLettering(' ') # remove letters
        # row1.setToWidth(1024)
        # ça a l'air de marcher...

        # it now seems ok
        # from epyseg.figure.alignment import alignRight, alignLeft, alignTop, alignBottom, alignCenterH, alignCenterV

        # alignLeft(row1, col1)
        # alignRight(row1, col1)
        # alignTop(row1, col1)
        # alignBottom(row1, col1)
        # alignCenterH(row1, col1)
        # alignCenterV(row1, col1)

        # can I add self to any of the stuff --> check --> if plus --> adds if divide --> stack it --> good idea and quite simple

        # fig = col(panel,panel2,panel3)

        # panel2+=panel
        # print(type(panel2))

        # panel2.setToHeight(512)
        # panel2.setToWidth(512)

        # all seems fine now --> see how I can fix things

        # panel2+=panel3 # bug here cause does not resize the panel properly
        # print('mega final', panel2.nCols, panel2.nRows, panel2.orientation, len(panel2.images), type(panel2), panel2.boundingRect(), panel.boundingRect())
        # print('mega final', len(col1.images), type(col1), col1.boundingRect(), row1.boundingRect())

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

        shapes_to_draw.append(col1)
        shapes_to_draw.append(row1)

        # big bug marche pas
        # packX(3, None, *[img0, img1, img2])  # ça marche presque de nouveau

        # print(img0.boundingRect(), img1.boundingRect(), img2.boundingRect())
        #
        # self.shapes_to_draw.append(img0)
        # self.shapes_to_draw.append(img1)
        # self.shapes_to_draw.append(img2)

        img4.setLettering('<font color="#0000FF">Z</font>')  # ça marche mais voir comment faire en fait
        # self.shapes_to_draw.append(fig)

        shapes_to_draw.append(Square2D(300, 260, 250, stroke=3))
        widget.shapes_to_draw = shapes_to_draw

    widget.show()

    # all seems to work now --> just finalize things
    # maybe do a panel and allow cols and rows and also allow same size because in some cases people may want that --> offer an option that is fill and
    # TO SAVE AS SVG

    if False:
        # if True:
        widget.save('/E/Sample_images/sample_images_svg/out2.svg')
        # TODO make it also paint to a raster just to see what it gives

        # TO SAVE AS RASTER --> quite good
        # Tif, jpg and png are supported --> this is more than enouch for now
        # self.paintToFile('/E/Sample_images/sample_images_svg/out2.png')
        # self.paintToFile('/E/Sample_images/sample_images_svg/out2.tif')
        # widget.save('/E/Sample_images/sample_images_svg/out2.jpg')
        # widget.save('/E/Sample_images/sample_images_svg/out2.png')
        widget.save('/E/Sample_images/sample_images_svg/out2.jpg')
        widget.save('/E/Sample_images/sample_images_svg/out2.png')
        widget.save('/E/Sample_images/sample_images_svg/out2.tif')  # now has noise
        # first save is ok then gets weird lines

    # --> vraiment pas mal mais voir si je peux tt faire et pkoi si slow

    sys.exit(app.exec_())
