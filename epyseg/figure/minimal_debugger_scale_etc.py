# minimal bug to get rid of size issues
# shall I assume all rects are rects or shall I allow rotation
# still need assume everything is packed
# --> think about that ???
from epyseg.draw.shapes.image2d import Image2D
from epyseg.figure.alignment import packX
from epyseg.figure.column import Column
from epyseg.figure.fig_tools import preview, get_common_scaling_factor
from epyseg.figure.row import Row
from timeit import default_timer as timer


img1 = Image2D('/E/Sample_images/sample_images_EZF/counter/00.png')
img2 = Image2D('/E/Sample_images/sample_images_EZF/counter/01.png')
img3 = Image2D('/E/Sample_images/sample_images_EZF/counter/02.png')
img4 = Image2D('/E/Sample_images/sample_images_EZF/counter/03.png')
img5 = Image2D('/E/Sample_images/sample_images_EZF/counter/04.png')
img6 = Image2D('/E/Sample_images/sample_images_EZF/counter/05.png')
img7 = Image2D('/E/Sample_images/sample_images_EZF/counter/06.png')
img8 = Image2D('/E/Sample_images/sample_images_EZF/counter/07.png')
img9 = Image2D('/E/Sample_images/sample_images_EZF/counter/08.png')
img10 = Image2D('/E/Sample_images/sample_images_EZF/counter/09.png')

if False:
    row1 = Row(img1, img2)
    row2 = Row(img3, img4, img5)

    print(row1)
    print(row2)

    print(type(row1))

    row1 /= row2

    print(row1)

    print(type(row1))


    # size is ok if no set to width
    row1.setToWidth(512) # --> indeed size is not ok there --> some update is not made properly in there



    #
    for elm in row1:
        print("inner size", elm) # the size of these things is indeed 512 --> why isn't then the parent up to date ????


    print(row1)

    # row1.updateBoudingRect() # ça a marche --> pkoi marche pas n haut alors
    #
    # print(row1)


    preview(row1)


# test the new resizing algo that is so much smarter and faster than the older one
if True:
    row1 = Row(img1, img2, img3, img4, img5)
    # row2 = Row(img6, img7, img8)


    col1 = Column(img6, img7, img8)

    print(row1) # PyQt5.QtCore.QRectF(0.0, 0.0, 1269.6600575972946, 352.0)
    for elm in row1:
        print("inner size0", elm)


    if False:
        # row1.sameHeight(3)
        # print(row1)
        sizes = []
        for elm in row1:
            sizes.append(elm.width())
            # print("inner size", elm)



        # new way to compute size --> much smarter than previous way --> start replace everywhere
        # sizes
        scaling = get_common_scaling_factor(sizes, 512,3)
        print('scaling, ',scaling)
        # need set to scale
        # need

        for elm in  row1:
            # elm.setWidth(scaling*elm.width())
            # elm.setHeight(scaling*elm.height())
            elm.scale*=scaling



        # pb there is a bug and the image and stuff don't match definitely my new code is better and smarter
        # this does noyt work but why ????
        # need pack them in x
        packX(3,None, *row1.images)
        row1.updateBoudingRect()

        # why is the width an integer ??? --> do I do mistakes
        # really need recode the changes in size because that is way smarter!!!
        for elm in row1:
            print("inner size2", elm)

        print('--> row1',row1) # --> ça marche --> essayer donc de faire ça car tres smart et fast en fait

        # preview(row1)


    start_loop = timer()

    print('end set to width', timer() - start_loop) # 0.006189999054186046




    # row1|=col1
    # row1|=col1 # --> still a row --> where is the bug then ???
    col1/=row1 # --> still a row --> where is the bug then ???
    row1 = col1
    print(type(row1))

    row1.setToWidth(512)

    # for elm in row1:
    #     print("inner size", elm)

    print(row1)

    preview(row1)