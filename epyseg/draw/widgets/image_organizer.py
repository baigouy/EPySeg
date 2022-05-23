

# TODO recode all from scratch rather than waisting time fixing it
# do a row kind of element that aligns vertically


# ça marche aussi maintenant reste plus qu'à faire le set width stuff set to width
def sameHeight(things_to_align, space):

    print('len(things_to_align)', len(things_to_align))
    if space is None:
        space = 0
    max_height = None
    # try:
    #     iter(things_to_align)
    for img in things_to_align:

        if max_height is None:
            max_height = img.boundingRect().height()
        max_height = max(img.boundingRect().height(), max_height)
        print('TUTU', img, type(img))
        print('TUTA', img, img.boundingRect(), img.boundingRect().height)
        print('TOTO', img, img.boundingRect(), img.boundingRect().height())


    # why not looping till the end --> only gets one
    count = 0
    for img in things_to_align:
        print(len(things_to_align), count)
        count+=1
        print('setting to height', max_height)
        print('before 2', type(img))
        img.setToHeight(max_height)
        print('after 2', type(img), img.boundingRect())

    print('max_height', max_height)

    packX(things_to_align,space)
    alignTop(things_to_align,updateBounds=False)
    # things_to_align.updateBoudingRect()

def packX(things_to_align, space=3):
    last_x = 0
    last_y = 0

    for i in range(len(things_to_align)):
        img = things_to_align[i]
        if i != 0:
            last_x += space
        img.set_P1(last_x, img.get_P1().y())
        last_x = img.boundingRect().x() + img.boundingRect().width()

    # things_to_align.updateBoudingRect()

def packY(things_to_align, space=3):
    last_x = 0
    last_y = 0


    print('1',things_to_align)


    for i in range(len(things_to_align)):
        img = things_to_align[i]
        print('2',img)
        if isinstance(img, list):
            from epyseg.draw.widgets.image_group import group
            img = group(*img)
        # should I create a row of objects in fact I need a group and move everything with that --> TODO --> maybe
        if i != 0:
            last_y += space

        print('img', type(img))

        img.set_P1(img.get_P1().x(), last_y)
        # get all the bounding boxes and pack them with desired space in between
        # get first point and last point in x
        x = img.boundingRect().x()
        y = img.boundingRect().y()
        last_x = img.boundingRect().x() + img.boundingRect().width()
        last_y = img.boundingRect().y() + img.boundingRect().height()
    # things_to_align.updateBoudingRect()



# Align vectorial objects to the top
# in fact should align in y
def alignTop(things_to_align, updateBounds=False):
    first_left = None
    for img in things_to_align:
        cur_pos = img.get_P1()
        if first_left is None:
            first_left = cur_pos
        img.set_P1(cur_pos.x(), first_left.y())
    # if updateBounds:
    #     things_to_align.updateMasterRect()
