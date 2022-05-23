# DEPRECATED!!!!!!!!!!!!!!!!!!!! use row and cols instead


# deprecated cause not very useful cause can be changed easily



# test of all then try to publish again
# see how to handle scale bars...





# https://docs.python.org/2/library/operator.html
# can I have fractions also in panels to really create complex stuff --> maybe think about it though --> no it's probably not a good idea
# pb is if I add a panel to another it will be unpacked --> is that what I want ??? probably not I just want it to be added as one single image I guess

# finalement est ce qu'un vertical row ou horizontal row suffirait pas c'est un peu comme un qvboxlayou et un qhboxlayout
# puis je tout faire avec ça ??? peut etre en fait


# vbox
# hbox
# fig serait un container de vbox et de hbox en fait --> puise-je tt faire avec ça peut etre
# ai-je besoin d'un panel en fait puisqu'un panel est juste un arrangement de vbox et de hbox
# le seul avantage c'est d'organiser les images en fait en 3*2 ou alike --> peut etre remettre row mais vraiment l'améliorer et garder le panel
# en fait le panel serait 2D sinon mieux vaut utiliser une vbox ou une hbox
# puis-je aussi faire un systeme de grid ??? pas si facile avec des images


from epyseg.draw.shapes.rect2d import Rect2D
from PyQt5.QtCore import QPointF, QRectF

from epyseg.figure.fig_tools import preview
from epyseg.figure.row import Row
# import math
# logger
from epyseg.tools.logger import TA_logger

# en fait

# TODO controler les set2width and settoheight de ce truc afin d'etre sur qu'ils fonctionnent car je ne pense pas que ce soit le cas en fait

logger = TA_logger(logging_level=TA_logger.DEBUG)


# can have a 1D and 2D with horizontal or vertical stuff
# TODO really try
# ça marche un panel peut contenir autant de panels que necessaire --> tt sera panel dans ce truc
# pourrait implement un sticky mode qui se comporte comme une figure ds EZF et un floating mode ou tout flotte
# should I sketch content for fusion --> draw how it would look like if done

# ce truc serait dedie au 2D en fait et le reste se ferait par des trucs differents
# could be called row and col and allow them to contain panels

# figure serait le master container --> en fait je garde la structure
# d'avant'

# pkoi ne pas tt restaurer alors


class Panel(Rect2D):



    # orientation='horizontal' orientation='vertical'
    def __init__(self, *args, nCols=None, nRows=None, space=3,
                 orientation='horizontal'):  # replace orientation by horizontal or vertical instead of with or height just set fixed dimension size
        # self.width = 512 # KEEP # already exists in Qrect so not need to set it elsewhere !!!!!!
        # self.height = 0
        # super(Panel, self).__init__(0, 0, 0, 0)
        self.space = space
        self.orientation = orientation

        if nCols is None and nRows is None:
            # by default panels behave as 1D panel either being a row or a column --> quite cool
            # should do the job ???
            # try it
            if len(args) > 0:
                # self.nCols = self.nRows = round(math.sqrt(len(args)))
                # TODO need suppress this behaviour if I want to get rid of the row stuff to replace everything by panels --> would be simpler in a way
                # this way I can select single objects
                # print(self.nCols, round(self.nCols))
                if self.orientation == 'horizontal':
                    self.nCols = len(args)
                    self.nRows = 1
                else:
                    self.nRows = len(args)
                    self.nCols = 1
            else:
                self.nCols = self.nRows = 0
        else:
            self.nCols = nCols
            self.nRows = nRows
        if self.nRows is None or self.nRows == 0:
            self.nRows = 1
        if self.nCols is None or self.nCols == 0:
            self.nCols = 1
        nb_images = self.nCols * self.nRows
        if len(args) > nb_images:
            logger.error('too few images incrementing nb of rows to compensate')
            while (self.nCols * self.nRows < len(args)):
                # add more rows so that it can fit
                self.nRows += 1

        # store all images in there and create the appropriate figure with rows and cols specified
        self.images = []
        if args:
            for arg in args:
                if isinstance(arg, str):
                    # TODO need check image is not null --> file was found really
                    self.images.append(Image2D(arg))
                elif isinstance(arg, Image2D):
                    self.images.append(arg)
                else:
                    logger.error('unsupported input for panel ' + str(arg))
        super(Rect2D, self).__init__()
        self.isSet = True  # required to allow dragging
        self.createTable()

    def __iter__(self):
        ''' Returns the Iterator object '''
        self._index = -1
        return self

    def __next__(self):
        ''' Returns the next image or panel in the row '''
        self._index += 1
        if self._index < len(self.images):
            return self.images[self._index]
        # End of Iteration
        raise StopIteration

    def __len__(self):
        if self.images is None:
            return 0
        return len(self.images)

    def __add__(self, other):
        # if isinstance(other, Image2D):
        self.images.append(other)
        self.createTable()
        # behaviour change to replace everything by panels if possible
        # elif isinstance(other, Panel):
        #     print(self.images)
        #     print(len(self.images))
        #     print(other.images)
        #     print(len(other.images))
        #     self.images = self.images + other.images
        #     self.createTable()
        # NEW BEHAVIOUR
        # elif isinstance(other, Panel):
        #     print('adding panel to panel')
        #     self.images.append(other)
        #     self.createTable()
        #     print(self.images)
        #     print(len(self.images))
        #     print(other.images)
        #     print(len(other.images))
        #     self.images = self.images + other.images
        #     self.createTable()
        return self

    def __sub__(self, other):
        if other in self.images:
            self.images.remove(other)
        self.createTable()
        return self

    def __lshift__(self, other):
        # move image left with respect to self

        if isinstance(other, Image2D):
            # swap the position of the two images and repack
            if other in self.images:
                pos = self.images.index(other)
                if pos - 1 >= 0:
                    self.images[pos - 1], self.images[pos] = self.images[pos], self.images[pos - 1]
                else:
                    return self
                self.createTable()
                return self
        else:
            logger.error('not implemented yet swapping two objects ' + str(type(other)))

    __ilshift__ = __lshift__

    def __rshift__(self, other):
        # move image left with respect to self
        if isinstance(other, Image2D):
            if other in self.images:
                pos = self.images.index(other)
                if pos + 1 < len(self.images):
                    self.images[pos + 1], self.images[pos] = self.images[pos], self.images[pos + 1]
                else:
                    return self
                self.createTable()
                return self
        else:
            logger.error('not implemented yet swapping two objects ' + str(type(other)))

    __irshift__ = __rshift__

    def __or__(self, other):
        if isinstance(other, Panel):
            pos = self.get_P1().toPoint()
            other_pos = other.get_P1().toPoint()
            self.set_P1(other_pos)
            other.set_P1(pos)
        else:
            logger.error('swapping not implemented yet for ' + str(type(other)))

    def is_empty(self):
        if self.images is None or not self.images:
            return True
        return False

    def setToHeight(self, height):
        # self.height = height
        self.setHeight(height)
        self.setWidth(0)
        self.createTable()

    # TODO handle differently width and height ???
    # pb it does also change height which it shouldn't --> I have a bug somewhere in the computations --> the nb of cols is incorrect I need to get incompressible space for all content stuff and not only outside
    def setToWidth(self, width):
        # self.width = width
        self.setWidth(width)
        self.setHeight(0)
        self.createTable()

    def set_ori(self, *args):
        self.set_P1(*args)

    def set_P1(self, *args):
        curP1 = self.get_P1()
        Rect2D.set_P1(self, *args)
        newP1 = self.get_P1()
        # need do a translation of all its content images
        for img in self:
            img.translate(newP1.x() - curP1.x(), newP1.y() - curP1.y())

        self.updateBoudingRect()

    def translate(self, *args):
        if len(args) == 1:
            point = args[0]
            QRectF.translate(self, point.x(), point.y())
            for img in self:
                img.translate(point.x(), point.y())
        else:
            QRectF.translate(self, args[0], args[1])
            for img in self:
                img.translate(QPointF(args[0], args[1]))

    def draw(self, painter, draw=True):
        if draw:
            painter.save()
        if draw:
            for img in self:
                img.draw(painter, draw=draw)
            painter.restore()

    def fill(self, painter, draw=True):
        if self.fill_color is None:
            return
        if draw:
            painter.save()
        if draw:
            for img in self:
                img.fill(painter, draw=draw)
            painter.restore()

        # TODO pb will draw the shape twice.... ---> because painter drawpolygon is called twice

    def drawAndFill(self, painter):
        painter.save()
        self.draw(painter, draw=True)
        painter.restore()


    # need handle incompressibility here in fact otherwise bug
    def createTable_neo(self):
        if len(self) == 0:
            return
        if self.nRows == 0:
            self.nRows = 1
        if self.nCols == 0:
            self.nCols = 1

        # print('nb images ' + str(len(self)) + ' nRows:' + str(self.nRows) + ' nCols:' + str(self.nCols))
        if self.nCols * self.nRows < len(self):
            while self.nCols * self.nRows < len(self):
                self.nRows += 1

        # hack to make it linear then compute more complex stuff otherwise if it is really 2D
        self.nRows = 1
        self.nCols = len(self)
        # for img in self:
        counter = 0

        # maybe bug is just because of not properly handling set to with and set to height

        # why the hell do I have a figure in panel really need to recode that
        # from epyseg.figure.row import Column
        # fig = col(space=self.space)

        fig = []

        # TODO recode that and do not go through rows for that
        # if it is None

        # en fait aligner en y les images suffirait puis les packer en x puis faire le retaillage

        # for img in self.images:
        #     print('type', type(img))

        for i in range(self.nRows):
            row = []
            # last_size = None
            for j in range(self.nCols):
                if i * self.nCols + j < len(self):
                    img = self.images[i * self.nCols + j]
                    # last_size = img.boundingRect()
                    # print('last_size',last_size)
                    # if row is None:
                    #     row = Column(img)
                    # else:
                    #     row += img
                    row.append(img)
                # else:
                # not enough images so adding empty images with same size as last one
                # if last_size is not None:
                #     logger.debug('not enough images so adding empty images with same size as last entry')
                #     row += Image2D(width=last_size.width(), height=last_size.height())
            if row is not None and not len(row) == 0:
                from epyseg.draw.widgets.image_organizer import packX
                # packing seems correct then need align top
                packX(row, self.space) # ça ça marche
                from epyseg.draw.widgets.image_organizer import alignTop
                alignTop(row) # ça ça marche
                # seems ok too

                print('#' * 20)
                for c in row:
                    print(c.boundingRect())
                print('#' * 20)
                # fig += row
                fig.append(row)
        # if len(fig) == 0:

        print('fig len', len(fig))  # --> ok
        #     logger.error('empty panel created there has to be an error somewhere')

        from epyseg.draw.widgets.image_organizer import packY
        # not necessarily that by the way
        # packY(fig, space=self.space)
        from epyseg.draw.widgets.image_organizer import sameHeight

        # required

        print('brfore same height', self.boundingRect())

        sameHeight(self.images, space=self.space) # bug here
        print('arfter same height', self.boundingRect())

        from epyseg.draw.widgets.image_group import group
        if self.width() != 0:
            group(*self.images).setToWidth(self.width())
        if self.height() !=0:
            group(*self.images).setToHeight(self.height())
        # sameHeight(, space=self.space)  # see that # ça n'a pas l'air de marcher
        for r in self.images:
            print('bounding rect after same height', r.boundingRect(), len(self.images))

        # should also update bounding rect of all the content in a way

        # for r in fig:
            # print('type', type(r))

            # group(*r).setToWidth(512)

        packY(fig, space=self.space)

        # print('width here',self.width())
        # on dirait que ça marche mais faudrait distinguer la width de la desired width...
        # if self.width()!=0:
        #     fig.setToWidth(self.width()) # or settoheight by the way
        self.updateBoudingRect()

    # TODO recode that then I'll be done
    # really weird code --> can I make it better
    def createTable(self):

        # print('create table called', len(self))
        if len(self) == 0:
            return
        if self.nRows == 0:
            self.nRows = 1
        if self.nCols == 0:
            self.nCols = 1
        print('nb images ' + str(len(self)) + ' nRows:' + str(self.nRows) + ' nCols:' + str(self.nCols))
        if self.nCols * self.nRows < len(self):
            while self.nCols * self.nRows < len(self):
                self.nRows += 1
        # for img in self:
        counter = 0

        # maybe bug is just because of not properly handling set to with and set to height

        # why the hell do I have a figure in panel really need to recode that
        from epyseg.draw.widgets.col import col
        fig = col(space=self.space)
        self.rows=[]

        # TODO recode that and do not go through rows for that
        # if it is None
        for i in range(self.nRows):
            row = None
            last_size = None
            for j in range(self.nCols):
                if i * self.nCols + j < len(self):
                    img = self.images[i * self.nCols + j]
                    last_size = img.boundingRect()
                    print('last_size', last_size)
                    if row is None:
                        row = Row(img)
                    else:
                        row += img
                else:
                    # not enough images so adding empty images with same size as last one
                    if last_size is not None:
                        logger.debug('not enough images so adding empty images with same size as last entry')
                        row += Image2D(width=last_size.width(), height=last_size.height())
            if row is not None and not row.isEmpty():
                row.packX(self.space)
                self.rows.append(row)
                fig += row

        if len(fig) == 0:
            logger.error('empty panel created there has to be an error somewhere')

        # not necessarily that by the way
        fig.packY(space=self.space)

        print('width here', self.width())
        # on dirait que ça marche mais faudrait distinguer la width de la desired width...
        if self.width() != 0:
            fig.setToWidth(self.width())  # or settoheight by the way
        self.updateBoudingRect()
        # unique = len(self) == 1

    def updateBoudingRect(self):
        '''updates the image bounding rect depending on content'''
        x = None
        y = None
        x2 = None
        y2 = None
        for img in self:
            topLeft = img.get_P1()
            if x is None:
                x = topLeft.x()
            if y is None:
                y = topLeft.y()
            x = min(topLeft.x(), x)
            y = min(topLeft.y(), y)
            if x2 is None:
                x2 = topLeft.x() + img.boundingRect().width()
            if y2 is None:
                y2 = topLeft.y() + img.boundingRect().height()
            x2 = max(topLeft.x() + img.boundingRect().width(), x2)
            y2 = max(topLeft.y() + img.boundingRect().height(), y2)

        self.setX(x)
        self.setY(y)
        self.setWidth(x2 - x)
        self.setHeight(y2 - y)

    # @return the block incompressible width
    # def getIncompressibleWidth(self):
    #     # in fact need get nb of cols a
    #     extra_space = (self.nCols - 1.) * self.space
    #     extra_space += self.getExtraIncompressibleWidth()
    #     return extra_space

    # def getExtraIncompressibleWidth(self):
    #     extra_space = 0
    #     return extra_space
    #
    # # @return the block incompressible height
    # def getIncompressibleHeight(self):
    #     # in fact need get nb of cols a
    #     extra_space = (self.nRows - 1.) * self.space
    #     extra_space += self.getExtraIncompressibleHeight()
    #     return extra_space
    #
    # def getExtraIncompressibleHeight(self):
    #     extra_space = 0
    #     return extra_space

if __name__ == '__main__':
    from epyseg.draw.shapes.image2d import Image2D  # KEEP Really required to avoid circular imports

    img0 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/00.png')
    # img0.setLetter(TAText2D(
    #     text='<p style="text-align:left;color: yellow">This text is left aligned <span style="float:right;font-style: italic;font-size: 8pt;"> This text is right aligned </span><span style="float:right;font-size: 4pt;color:red"> This text is another text </span></p>'))

    img1 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
    # img1 = Image2D('D:/dataset1/unseen/focused_Series012.png')
    # img1.setLetter(TAText2D(text="<font face='Comic Sans Ms' size=16 color='blue' >this is a <br>test</font>"))
    # ça ça marche vraiment en fait --> use css to write my text instead of that

    # ça ça a l'air de marcher --> pas trop mal en fait du coup
    # ça a l'air de marcher maintenant --> could use that and do a converter for ezfig ???
    # img1.setLetter(TAText2D(text="<span style='font-size: 12pt; font-style: italic; font-weight: bold; color: yellow; paddind: 20px; text-align: center;'> <u>Don't miss it</u></span><span style='font-size: 4pt; font-style: italic; font-weight: bold; color: #00FF00; paddind: 3px; text-align: right;'> <u>test2</u></span>"))

    # TODO need remove <meta name="qrichtext" content="1" /> from the stuff otherwise alignment is not ok... TODO --> should I offer a change to that ??? maybe not
    # test_text = '''
    #       </style></head><body style=" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;">
    #       <p style="color:#00ff00;"><span style=" color:#ff0000;">toto</span><br />tu<span style=" vertical-align:super;">tu</span></p>
    #       '''
    # img1.setLetter(TAText2D(text=test_text))
    # background-color: orange;
    # span div et p donnent la meme chose par contre c'est sur deux lignes
    # display:inline; float:left # to display as the same line .... --> does that work html to svg
    # https://stackoverflow.com/questions/10451445/two-div-blocks-on-same-line --> same line for two divs

    img2 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/02.png')
    img3 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/03.png')
    img4 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/04.png')
    img5 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/05.png')
    img6 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/06.png')
    img7 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/07.png')
    img8 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/08.png')
    img9 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/09.png')
    img10 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/10.png')

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
    panel = Panel(img0) #, img1, img2, img3)  # , img6, #, nCols=3, nRows=2

    # marche pas en fait car un truc ne prend pas en charge les figs

    # ça marche donc en fait tt peut etre un panel en fait

    panel2 = Panel(img4)#, img5, img6, img7)  # , img6, img6, nCols=3, nRows=2,

    # panel3 = Panel(img8, img9)

    print(type(panel2))
    panel2 += panel
    print(type(panel2))
    # panel2+=panel3 # bug here cause does not resize the panel properly
    # print(panel2.nCols, panel2.nRows, panel2.orientation, len(panel2.images), type(panel2), panel2.boundingRect())

    panel2.setToWidth(128)

    # img1 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/00.png')
    # img2 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/01.png')
    # img3 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/02.png')
    # img4 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/03.png')

    # panel = Panel(img1, img2)
    print('last', panel2.boundingRect(), panel2.boundingRect().height())

    # if True:
    #     import sys
    #
    #     sys.exit(0)

    # result = img1 + img2
    #
    # print(result)
    # print(len(result.images))

    #    result += img3
    #    result += img4

    #    print(result)
    #    print(len(result.images)) # ça ça marche

    #    if True:
    #        sys.exit(0)

    # img3.pop()

    # row2 = img3 + img4
    # row2 = row(img3, img4)

    # print(row2)
    # print(row2.images)  # why 4 images here ....
    # print('row2', len(row2.images))  # pas bon trop d'images --> pkoi ????

    # print(Image2D)
    #
    # fig = result + row2
    # print(fig)
    # print(len(fig.images))
    #
    # count = 0
    # for img in fig.images:
    #     print(count, img.filename)
    #     count += 1
    #
    # final_result = fig - img4
    # print(final_result)
    # print(final_result.images)
    # print(len(final_result.images))
    #
    # ça marche

    # final_result = img4 + fig  # ça ne marche pas
    # print(final_result)
    # print(final_result.images)
    # print(len(final_result.images))

    # si je divide une row alors ça ajoute une colonne à la figure --> return 2 rows in a figure

    img5 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/04.png')
    img6 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/05.png')

    fig = img5 / img6  # --> could create a panel vertical containing two stuff

    print(fig)
    # print(fig.cols)
    # print(len(fig.cols))

    final_fig = fig - img5

    # print(final_fig)
    # print(final_fig.rows)
    # print(len(final_fig.rows))

    # final_fig += row2

    # print(final_fig)
    # print(final_fig.rows)
    # print(len(final_fig.rows))

    img8 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/06.png')
    img9 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/07.png')
    img10 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/08.png')
    img11 = Image2D('/E/Sample_images/sample_images_PA/trash_test_mem/counter/09.png')

    # fig = (img8 + img9) / (img10 + img11)
    # print(fig)
    # print(fig.rows)
    # print(len(fig.rows))
    # print('r1', len(fig.rows[0].images))
    # print('r2', len(fig.rows[1].images))

    # nb there are big size bugs here in terms of size --> not the same size but maybe this is what I want ????
    columns = Panel(img8, img9, img10, img11, nCols=2)
    columns.setToWidth(300)

    preview(columns)