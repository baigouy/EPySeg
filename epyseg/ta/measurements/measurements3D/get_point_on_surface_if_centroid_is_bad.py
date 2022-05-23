# https://gis.stackexchange.com/questions/76498/how-is-st-pointonsurface-calculated # really great and what I'm looking for
# https://geocompr.robinlovelace.net/geometric-operations.html#centroids

# this sounds like a smart and nice method --> see how fast this is to implement that

# can I do a fast version of it that relies on vertices to find this point
# scan line by line the shape maybe using bounding box and take the center of the longest continuous line as the centroid --> maybe that is a really good idea

# the following code is based on the idea described here https://gis.stackexchange.com/questions/76498/how-is-st-pointonsurface-calculated
from math import ceil

from epyseg.img import Img
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage import measure
from skimage.draw import disk
from timeit import default_timer as timer


# that seems to work but do some cleaning and the only issue is that if cell is too close to border, such as a fake cell surrounding the image then it will be the first line --> and it would be easy

# TODO also always use that for tracking as this should be much better than things I was using before and should prevent issues
__DEBUG__ = False



# TODO clean code
# this point will by definition always be inside the cell even in concave shapes

# the other possibility is the ultimate erosion but this will be extremely slow and resource consuming; I guess
# inspired by https://gis.stackexchange.com/questions/76498/how-is-st-pointonsurface-calculated but then I deviated from it
def point_on_surface(region, labels, prefer_centroid_if_inside_the_cell=True):
    #  A tuple of the bounding box's start coordinates for each dimension,
    #         followed by the end coordinates for each dimension



    # print(region.bbox)

    # TODO --> maybe ceil the centroid to get a better centroid point (pb can be outside of the image although it is very unlikely (in fact it is not possible --> can ceil)

    cell_id = labels[region.coords[0][0], region.coords[0][1]]

    if prefer_centroid_if_inside_the_cell:
        centroid = region.centroid

        # TODO replace by rounding maybe some day but ok for now

        # if labels[int(ceil(centroid[0])), int(ceil(centroid[1]))]==cell_id:
        #     return (int(ceil(centroid[0])), int(ceil(centroid[1])))
        if labels[int(round(centroid[0])), int(round(centroid[1]))]==cell_id:
                return (int(round(centroid[0])), int(round(centroid[1])))

    if __DEBUG__:
        print(cell_id)

    max_line_length = 0
    # line_length_counter = 0
    point_on_surface = (region.coords[0][0], region.coords[0][1])

    # how do I get the
    # get the centroid of the region
    # --> ended ok
    cell_encountered = False
    first_encounter_x = None
    line_length_counter = 0
    # there is a bug

    for iii in range(region.bbox[0], region.bbox[2]):
        # start_coord_x = region.bbox[1]
        # end_coord_x = region.bbox[3]
    # if line_by_line:
        cell_encountered = False
        first_encounter_x = None
        line_length_counter=0 # if I don't do set it to 0 here, I'll get some sort of center of mass effect which is maybe desired or not but if I do that this is very different from the idea above maybe ask for mode 1 or 2
        for jjj in range(region.bbox[1], region.bbox[3]):

            if labels[iii, jjj] == cell_id:
                # print(cell_encountered, iii, jjj, line_length_counter)
                if not cell_encountered:
                    if __DEBUG__:
                        print('first encounter', iii, jjj)
                    first_encounter_x = jjj
                    cell_encountered = True
                line_length_counter += 1
                # print('1',line_length_counter, first_encounter_x, jjj) # --> inded
            else:
                if __DEBUG__:
                    if line_length_counter>0:
                        print('last encounter', iii, jjj, line_length_counter)
                if line_length_counter >= max_line_length and first_encounter_x is not None:
                    max_line_length = line_length_counter
                    point_on_surface = (
                        iii, int(first_encounter_x + (jjj - first_encounter_x) / 2))  # get the centroid of the line
                    if __DEBUG__:
                        print('2',region.bbox, max_line_length, point_on_surface, first_encounter_x, cell_encountered, iii, jjj)

                first_encounter_x = None  # reset first encounter
                cell_encountered = False  # reset first encounter
                line_length_counter = 0
                # else:
                #     line_length_counter = 0
                #     cell_encountered = False  # reset first encounter
                #     first_encounter_x = None  # reset first encounter

        # if line_by_line:
        # need reset also at the end of every loop if it has not found something
        # nb replacing this by >= changes a bit the stuff --> that does make sense especially for cells at the border!!!
        if line_length_counter > max_line_length and first_encounter_x is not None: # en fait c'est mieux comme ça
            max_line_length = line_length_counter
            point_on_surface = (
                iii, int(first_encounter_x + (jjj - first_encounter_x) / 2))  # get the centroid of the line
            if __DEBUG__:
                print('2', region.bbox, max_line_length, point_on_surface, first_encounter_x, cell_encountered, iii, jjj)


    # if line_length_counter > max_line_length and first_encounter_x is not None: # en fait c'est mieux comme ça
    #     max_line_length = line_length_counter
    #     point_on_surface = (
    #         iii, int(first_encounter_x + (jjj - first_encounter_x) / 2))  # get the centroid of the line
    #
    #     if __DEBUG__:
    #         print('2', region.bbox, max_line_length, point_on_surface, first_encounter_x, cell_encountered, iii, jjj)
    if __DEBUG__:
        print(region.bbox)
    return point_on_surface

if __name__ == '__main__':

    start = timer()

    # get a good test image here just to test the stuff
    cells = \
    Img('/E/Sample_images/sample_images_PA/trash_test_mem/full_wing_multi/early_stages/FocStich_RGB005/handCorrection.png')[
        ..., 0]
    labels = measure.label(cells, connectivity=1, background=255)  # FOUR_CONNECTED
    # plt.imshow(labels)
    # plt.show()


    # c'est bon ça a l'air de bien marche en fait
    for region in regionprops(labels):
        if region.area > 60000:
            # get its centroid
            pt_on_surface = point_on_surface(region, labels)
            print(pt_on_surface)

            # compare computed position to the calculated position
            centroid = region.centroid
            x, y = disk(centroid, 30, shape=labels.shape)
            # prefer centroid if inside
            labels[x, y] = 30000

            # labels[pt_on_surface[0], pt_on_surface[1]] = 6000
            # for xx, yy in zip(x, y):
            x, y = disk(pt_on_surface, 30, shape=labels.shape)
            labels[x, y] = 25000

            # ax.add_patch(circ)

    # --> 5 secs --> not that bad I guess because the image is huge...

    duration = timer() - start
    print(duration)

    plt.imshow(labels)
    plt.show()
