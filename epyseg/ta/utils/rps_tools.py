import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from skimage.draw import line
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
import collections
import matplotlib.pyplot as plt
from skimage.measure._regionprops import RegionProperties
import math
from timeit import default_timer as timer
from numba import jit, njit

__DEBUG__ = False  # for pavlidis code -> stores and prints decisions and directions

# pavlidis coords all clockwise and properly ordered!
# from personal.pyTA.tsts.px_ordering import order_points_new
# from personal.pyTA.tsts.px_ordering3 import sort_pixels_clowkise2

# maybe stop if visiting the first pixel for the third time

pavlidis_front_up = [[-1, -1],  # px up left
                     [-1, 0],  # px up
                     [-1, 1]]  # px up right

pavlidis_front_left = [[1, -1],  # px lower left
                       [0, -1],  # left
                       [-1, -1]]  # upper left

pavlidis_front_bottom = [[1, 1],  # px lower right
                         [1, 0],  # px bottom center
                         [1, -1]]  # px lower left

pavlidis_front_right = [[-1, 1],  # px upper right
                        [0, 1],  # right
                        [1, 1]]  # px lower right

pavlidis_lwr_pixels_up =[[1, -1],  # px lower left
                     [0, -1],  # left
                     [1, 1]]  # px lower right

pavlidis_lwr_pixels_0 =[[1, -1],  # px lower left
                     [0, -1],  # left
                     [1, 1]]  # px lower right

pavlidis_lwr_pixels_90 =[[-1, -1],  # px upper left
                     [-1, 0],  # up
                     [1, -1]]  # px lower left

pavlidis_lwr_pixels_180 =[[-1, 1],  # px upper right
                     [-1, -1],  # upper left
                     [0, 1]]  # px right

pavlidis_lwr_pixels_270 =[[-1, 1],  # px upper right
                     [1, 0],  # bottom
                     [1, 1]]  # px lower right

starting_pavlidis = [pavlidis_lwr_pixels_0, pavlidis_lwr_pixels_90, pavlidis_lwr_pixels_180, pavlidis_lwr_pixels_270]


# KEEP I could even make this better using slice since it's anyway lines!!! --> IN FACT THIS IS NOT A GOOD IDEA as if outside of bonds it will crash and that also makes it hard to use for regions!
# orientation_pavlidis = np.asarray([pavlidis_front_up, pavlidis_front_right, pavlidis_front_bottom, pavlidis_front_left])
orientation_pavlidis = [pavlidis_front_up, pavlidis_front_right, pavlidis_front_bottom, pavlidis_front_left]


def __check_pixel_pavlidis(img, y, x, color_of_interest=None):
    if isinstance(img, np.ndarray):
        return img[y, x] == color_of_interest
    if isinstance(img, list):
        return [y, x] in img




# the njit version is super slow and does not bring much --> skip it, it is however required to do measurements such as perimeter and alike
# TODO do a njit version of pavlidis --> would be much faster this time
@njit
def pavlidis2(img, start_y=None, start_x=None, closed_contour=False, color_of_interest=None,
             auto_detect_extreme_points_for_region_upon_missing_start_coords=True):
    # based on http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/theo.html

    coords = []
    # debug = []
    # debug.append('test')
    # debug.clear()
    # if __DEBUG__:




    # if not isinstance(img, RegionProperties):



    if start_x >= img.shape[1] or start_y >= img.shape[0] or start_x < 0 or start_y < 0:
        print('error coordinates outside the image')
        return coords
    if color_of_interest is None:
        color_of_interest = img[start_y, start_x]
    if img[start_y, start_x] != color_of_interest:
        print('error starting coordinates outside the object')
        return coords
    # else:
    #     # get regionprops coords
    #     img = img.coords
    #     if start_x is None or start_y is None:
    #         if not auto_detect_extreme_points_for_region_upon_missing_start_coords:
    #             start_y, start_x = img[0][0], img[0][1]
    #         else:
    #             # to be on the safe side get one of the most distant extreme points as a starting point
    #             feret_1, feret_2 = get_feret_from_points(extreme_points(img))
    #             start_y = feret_1[0]
    #             start_x = feret_1[1]
    #     else:
    #         if not __check_pixel_pavlidis(img, start_y, start_x):
    #             print('error coordinates outside the image')  # convert to logger error
    #             return coords
    #     # convert region props to list for convenience
    #     img = img.tolist()

    cur_coord_x = start_x
    cur_coord_y = start_y
    coords.append((start_y, start_x))

    # if __DEBUG__:
    #     # pt = str((start_y, start_x))
    #     # print('pt', pt)
    #     # debug.append(pt)
    #     print('__DEBUG__',(start_y, start_x))

    look_direction = 0
    counter = 0
    no_success_counter = 0

    # count nb of orientation change without success and if too many --> quit!


    # TODO need check the entry point is valid and fits with the data
    # shall I even more recode pavlidis --> how to avoid infinite loops

    nb_of_encounters_of_first_pixel = 0

    # if outside of the image need stop, in fact need implement the count of the nb of changes in direction

    # f you rotate 3 times without finding any black pixels, this means that you are standing on an isolated pixel
    while (True):  # danger infinite loop # maybe I should limit that somehow
        counter += 1
        success = False
        no_success_counter = 0

        if no_success_counter >= 3:
            return coords # on an isolated pixel

        if nb_of_encounters_of_first_pixel>=3:
            return coords

        if cur_coord_x == start_x and cur_coord_y == start_y:
            nb_of_encounters_of_first_pixel+=1

        # need replace by two loops
        # ppp=-1
        # for shift in orientation_pavlidis[look_direction]:

        # print()
        # MEGA NB THERE IS AN INFINITE LOOP BUG HERE
        # print('tutu', orientation_pavlidis[look_direction], orientation_pavlidis[look_direction].shape[0])
        print('look_direction',look_direction, counter)
        for ppp in range(orientation_pavlidis[look_direction].shape[0]+1):
            shift = orientation_pavlidis[look_direction][ppp]
            print('shift',shift, coords) #  shift [          160760960 7595447239221182464] --> bug here
            # ppp+=1
            # check pixels in front of the current pixel



            coords_to_test_x = cur_coord_x + int(shift[1])
            coords_to_test_y = cur_coord_y + int(shift[0])



            print(coords_to_test_x, coords_to_test_y)

            # prevent infinite loop for non closed contours
            # if not closed_contour:  # THIS PART OF THE CODE IS NOT VERY SMART AND CAN LEAD TO ERRORS IF START POINTS ARE NOT PROPERLY DEFINED (OK FOR NOW THOUGH!)
            # required for debug and to avoid infinite loops
            if True:
                if (coords_to_test_y, coords_to_test_x) in coords:
                    # if __DEBUG__:
                    #     # debug.append('point already encountered --> quitting')
                    #     # print(debug)
                    #     print('__DEBUG__','point already encountered --> quitting')
                    return coords

            # if isinstance(img, np.ndarray):
            if coords_to_test_y >= img.shape[0] or coords_to_test_x >= img.shape[
                1] or coords_to_test_y < 0 or coords_to_test_x < 0:
                continue

            # if img[coords_to_test_y, coords_to_test_x] == color_of_interest:
            # if __check_pixel_pavlidis(img, coords_to_test_y, coords_to_test_x, color_of_interest):
            if img[coords_to_test_y, coords_to_test_x] == color_of_interest:

                cur_coord_y = coords_to_test_y
                cur_coord_x = coords_to_test_x

                if cur_coord_x == start_x and cur_coord_y == start_y:
                    # if __DEBUG__:
                    #     # print(debug)
                    #     print('__DEBUG__','end')
                    return coords
                coords.append((cur_coord_y, cur_coord_x))
                # if __DEBUG__:
                #     # debug.append((cur_coord_y, cur_coord_x))
                #     print('__DEBUG__', (cur_coord_y, cur_coord_x))

                # if P1 is the pixel then need change orientation
                if ppp == 0:
                    look_direction -= 1
                    if look_direction < 0:
                        look_direction = 3
                    # if __DEBUG__:
                    #     # debug.append('successP1dir' + str(look_direction))
                    #     print('__DEBUG__', 'successP1dir' + str(look_direction))
                break
        if not success:
            # if __DEBUG__:
            #     # debug.append('faileddirection' + str(look_direction))
            #     print('__DEBUG__','faileddirection' + str(look_direction))
            # no valid pixel found --> rotate view

            # I would need to restart the loop ???

            look_direction += 1
            if look_direction > 3:
                look_direction = 0
            if no_success_counter >= 4:
                # if __DEBUG__:
                #     # print(debug)
                #     print('__DEBUG__','end')
                return coords
            no_success_counter += 1


    #     if __DEBUG__:
    #         # debug.append('direction' + str(look_direction))
    #         print('__DEBUG__', 'direction' + str(look_direction))
    # if __DEBUG__:
    #     print('__DEBUG__', 'end')
    return coords





# URGENT TODO check if it is really faster than the graph sklearn method or not especially for bonds and perimeter --> check??? at least it is definitely useful for filled shapes
# TODO maybe auto add an extreme point for np.ndarray images too --> easy --> use maybe np.argwhere or alike !!!
def pavlidis(img, start_y=None, start_x=None, closed_contour=False, color_of_interest=None,
             auto_detect_extreme_points_for_region_upon_missing_start_coords=True, starting_orientation = 0):

    # __DEBUG__ = True

    # print('__DEBUG__',__DEBUG__)

    # based on http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/theo.html

    coords = []
    if __DEBUG__:
        debug = []

    if not isinstance(img, RegionProperties):
        # print('not reg prop')
        if start_x >= img.shape[1] or start_y >= img.shape[0] or start_x < 0 or start_y < 0:
            print('error coordinates outside the image')
            return coords
        if color_of_interest is None:
            color_of_interest = img[start_y, start_x]
        if img[start_y, start_x] != color_of_interest:
            print('error starting coordinates outside the object')
            return coords
        # if isolated --> return
        # check color of all neighbs of ntry point and skip if alone
    else:
        # get regionprops coords
        # print('reg prop')
        img = img.coords
        if start_x is None or start_y is None:
            if not auto_detect_extreme_points_for_region_upon_missing_start_coords:
                start_y, start_x = img[0][0], img[0][1]
            else:
                # to be on the safe side get one of the most distant extreme points as a starting point
                # maybe best is to take a pavlidis optimal start maybe instead --> and maybe faster ...
                feret_1, feret_2 = get_feret_from_points(extreme_points(img))
                start_y = feret_1[0]
                start_x = feret_1[1]
        else:
            # img = img.tolist()
            if not __check_pixel_pavlidis(img.tolist(), start_y, start_x):
                # print('type', type(img))

                print('error coordinates outside the image')  # convert to logger error
                return coords
        # if len(img[0]) == 1:
        #     return [start_y, start_x]
        # convert region props to list for convenience
        img = img.tolist()

    # print(len(img))
    # if len(img)==1:
        # just one point --> nothing to sort --> return
        # return [start_y, start_x]

    cur_coord_x = start_x
    cur_coord_y = start_y
    coords.append((start_y, start_x))

    if __DEBUG__:
        debug.append((start_y, start_x))

    look_direction = 0
    if starting_orientation!=0:
        look_direction = starting_orientation

    counter = 0
    no_success_counter = 0

    old_coords_x = -1
    old_coords_y = -1

    infinite_loop_counter = 0

    limit = None
    if isinstance(img, list):
        limit =  10 * len(img)

    # count nb of orientation change without success and if too many --> quit!
    while (True):  # danger infinite loop # maybe I should limit that somehow
        counter += 1
        success = False
        no_success_counter = 0 # useless cause always reset

        if limit is not  None:
            if counter>limit:
                print('error infinite loop Pavlidis --> breaking')
                return coords
        # print('counter', counter, no_success_counter, infinite_loop_counter, old_coords_x, old_coords_y, cur_coord_x, cur_coord_y)

        # print(cur_coord_x, cur_coord_y)
        if infinite_loop_counter>100:
            if __DEBUG__:
                print('debug', debug)
            return coords

        if old_coords_y == cur_coord_y and old_coords_x== cur_coord_x:
            infinite_loop_counter+=1

        else:
            old_coords_y = cur_coord_y
            old_coords_x = cur_coord_x
            # there was a big bug --> I wass never resetting the infinite loop counter which does not make sense for big bonds
            infinite_loop_counter=0

        for ppp, shift in enumerate(orientation_pavlidis[look_direction]):
            # check pixels in front of the current pixel
            coords_to_test_x = cur_coord_x + shift[1]
            coords_to_test_y = cur_coord_y + shift[0]

            if counter > 1:
                if coords_to_test_x == start_x and coords_to_test_y == start_y:
                    if __DEBUG__:
                        print('debug', debug)
                    # we encountered the first pixel again --> return stuff
                    return coords

            # prevent infinite loop for non closed contours
            if not closed_contour:  # THIS PART OF THE CODE IS NOT VERY SMART AND CAN LEAD TO ERRORS IF START POINTS ARE NOT PROPERLY DEFINED (OK FOR NOW THOUGH!)
                if (coords_to_test_y, coords_to_test_x) in coords:
                    if __DEBUG__:
                        debug.append('point already encountered --> quitting')
                        print('debug', debug)
                    return coords

            if isinstance(img, np.ndarray):
                if coords_to_test_y >= img.shape[0] or coords_to_test_x >= img.shape[1] or coords_to_test_y < 0 or coords_to_test_x < 0:
                    continue

            # if img[coords_to_test_y, coords_to_test_x] == color_of_interest:
            if __check_pixel_pavlidis(img, coords_to_test_y, coords_to_test_x, color_of_interest):

                cur_coord_y = coords_to_test_y
                cur_coord_x = coords_to_test_x

                if cur_coord_x == start_x and cur_coord_y == start_y:
                    if __DEBUG__:
                        print('debug', debug)
                    return coords
                coords.append((cur_coord_y, cur_coord_x))
                if __DEBUG__:
                    debug.append((cur_coord_y, cur_coord_x))

                # if P1 is the pixel then need change orientation
                if ppp == 0:
                    look_direction -= 1
                    if look_direction < 0:
                        look_direction = 3
                    if __DEBUG__:
                        debug.append('successP1dir' + str(look_direction))
                success = True
                break
        if not success:
            # print('failed', no_success_counter)
            if __DEBUG__:
                debug.append('faileddirection' + str(look_direction))
            # no valid pixel found --> rotate view
            look_direction += 1
            if look_direction > 3:
                look_direction = 0
            if no_success_counter >= 4:
                if __DEBUG__:
                    print('debug', debug)
                return coords
            no_success_counter += 1
        if __DEBUG__:
            debug.append('direction' + str(look_direction))
    if __DEBUG__:
        print('debug', debug)
    return coords


# can plot images or regions (regionprops) and corresponding pavlidis coords
def __plot_pavlidis(img, coords, skip_plot=True):
    if isinstance(img, RegionProperties):
        # convert region to image
        # print(region.bbox)
        region = img
        tmp_img = np.zeros((max(region.bbox[2], region.bbox[0]) + 1,
                            max(region.bbox[1], region.bbox[3]) + 1))
        # KEEP TOP MEGA TOP TIP TIPTOP TIP TOP fill numpy array using region coords
        tmp_img[region.coords[:, 0], region.coords[:, 1]] = 255
        img = tmp_img

        # print('img.shape',img.shape) # fixed

    # check if dupes
    print(coords)
    output = np.zeros_like(img)
    if contains_duplicate_coordinates(coords):
        print('DUPLICATED POINTS DETECTED')
        print(find_duplicate_coordinates(coords))
        coords = remove_duplicate_coordinates(coords)
        print('deduplicated coords', coords)
        print('dist between extreme points after deduplication', dist2D(coords[0], coords[-1]),
              dist2D(coords[0], coords[-1]) <= math.sqrt(2))
        # TODO TRY FIX IT AND SEE IF THAT WORKS ???
    else:
        print('no dupes')

    for val, (y, x) in enumerate(coords):
        # print(coords)
        output[y, x] = val + 1

    # img[img == 255]=0

    if not skip_plot:
        plt.imshow(output)
        plt.show()

    output[output != 0] = 255
    if not (output == img).all():
        print('MISSING PIXELS' * 20)
        missed = np.copy(img)
        missed[output != 0] = 128
        if not skip_plot:
            plt.imshow(missed)
            plt.show()
    else:
        print('FULL MATCH')


# returns extreme points from region.coords
def extreme_points(coords, return_array=True):
    tmp_coords= coords


    # print('coords',coords)
    if isinstance(coords, tuple):
        tmp_coords=np.asarray(list(zip(coords[0].tolist(), coords[1].tolist())))
        # tmp_coords = np.asarray(list(zip(coords)))
        # print('coords', coords, tmp_coords)

    top = tuple(tmp_coords[tmp_coords[..., 1].argmin()])
    bottom = tuple(tmp_coords[tmp_coords[..., 1].argmax()])
    left = tuple(tmp_coords[tmp_coords[..., 0].argmin()])
    right = tuple(tmp_coords[tmp_coords[..., 0].argmax()])
    if not return_array:
        return top, bottom, left, right
    else:
        return [top, bottom, left, right]


# returns the most distant points in a set of points
def get_feret_from_points(contour_or_extreme_points):
    C = cdist(contour_or_extreme_points, contour_or_extreme_points)
    furthest_points = np.where(C == C.max())
    feret_1 = contour_or_extreme_points[furthest_points[0][0]]
    feret_2 = contour_or_extreme_points[furthest_points[1][0]]
    return feret_1, feret_2


# it is a pixel sorting algo and a very smart alternative to the pavlidis algo for lines or perimeter (especially because it is insensitive to holes in the connectivity and insensitive to the choice of the start point for non closed shapes)
# THIS IS MUCH FASTER AND POWERFUL THAN PAVLIDIS --> RATHER USE THAT FOR CONTOURS AND FOR BONDS (JUST USE PAVLIDIS FOR FILLED SHAPES)!!!
# pb can have big jumps in it --> may need rotate the result array --> TODO URGENT --> DO THAT ESPECIALLY IF CONTOUR IS LONG
# compute dist to first or last or between last and first and detect closest and do roll of the array so that all is ok after that --> TODO
# RELY ON PAVLIDIS FOR NOW!!!
def nearest_neighbor_ordering(coords, remove_dupes=False):
    # largely based on https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
    if not isinstance(coords, np.ndarray):
        coords = np.asarray(coords)

    # create a graph to connect each point to its two nearest neighbors
    clf = NearestNeighbors(n_neighbors=2).fit(coords)
    # sparse matrix where each row is a node
    G = clf.kneighbors_graph()
    # construct a graph from the sparse matrix
    T = nx.from_scipy_sparse_matrix(G)
    order = list(nx.dfs_preorder_nodes(T, 0))
    coords = coords[order]

    # remove dupes if user wants to
    if remove_dupes and contains_duplicate_coordinates(coords):
        no_dupes = remove_duplicate_coordinates(coords.tolist())
        coords = np.asarray(no_dupes)

    return coords


def find_duplicate_coordinates(coords):
    return [item for item, count in collections.Counter(coords).items() if count > 1]


def contains_duplicate_coordinates(coords):
    if not isinstance(coords, list):
        coords = coords.tolist()
    try:
        if len(set(coords)) != len(coords):
            return True
    except:
        tmp = {tuple(coords) for coords in coords}
        if len(tmp) != len(coords):
            return True
    return False


def remove_duplicate_coordinates(coords):
    fixed_coords = []
    [fixed_coords.append(item) for item in coords if item not in fixed_coords]
    return fixed_coords


def dist2D(pt1, pt2):
    return math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


# perimeter_pixels should be an (N, 2 or 3) np.ndarray
# This is really what I wanted to have
def compute_perimeter(perimeter_pixels):
    # for iii in range(0,len(perimeter_pixels[0])-1):
    # pairwise distance
    # D = np.sqrt(((perimeter_pixels[:, :, None] - perimeter_pixels[:, :, None].T) ** 2).sum(1))

    # d = np.diff(perimeter_pixels, axis=0)
    # consecutive_distances = np.hypot(d[:, 0], d[:, 1])
    # print(consecutive_distances)
    # consecutive_distances = np.sqrt((d ** 2).sum(axis=1))
    # print(consecutive_distances)
    # nb need compute distance between extremities too if they are in close contact --> TODO

    length = compute_distance_between_consecutive_points(perimeter_pixels).sum()
    # print(length)
    return length


def compute_distance_between_consecutive_points(oredered_pixels):
    if isinstance(oredered_pixels, list):
        oredered_pixels = np.asarray(oredered_pixels)
    d = np.diff(oredered_pixels, axis=0)
    # Equivalent to ``sqrt(x1**2 + x2**2)``, element-wise.  If `x1` or
    #     `x2` is scalar_like (i.e., unambiguously cast-able to a scalar type),
    #     it is broadcast for use with each element of the other argument.
    #     (See Examples)
    # consecutive_distances = np.hypot(d[:, 0], d[:, 1])
    consecutive_distances = np.sqrt((d ** 2).sum(axis=1))  # smarter than hypot cause can also easily support 3D !!!
    return consecutive_distances


# nb maybe force 2D to check if points are really continuous!!!
# does that work in 3D --> need think a bit about it or fake convert to 2D --> good idea!!!
def is_distance_continuous(ordered_distances):
    if isinstance(ordered_distances, list):
        ordered_distances = np.asarray(ordered_distances)
    if len(ordered_distances.shape) == 2:
        ordered_distances = compute_distance_between_consecutive_points(ordered_distances)
    max_dist_pos = np.argmax(ordered_distances)
    max_dist = ordered_distances[max_dist_pos]

    # print('max_dist, max_dist_pos',max_dist, max_dist_pos)
    # if autoroll and max_dist>math.sqrt(2):
    #     np.roll(ordered_distances)
    if max_dist <= math.sqrt(2):
        return True
    else:
        return max_dist, max_dist_pos


# from http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/theo.html
# b) Go to the source of the problem; namely, the choice of the start pixel
# There is an important restriction concerning the direction in which you enter the start pixel. Basically, you have to enter the start pixel such that when you're standing on it, the pixel adjacent to you from the left is white. The reason for imposing such a restriction is:
# since you always consider the 3 pixels in front of you in a certain order, you'll tend to miss a boundary pixel lying directly to the left of the start pixel in certain patterns.
# need check LWR pixels are black, may need to change orientation though if nothing is found --> a bit complex but doable still


# NB THAT DOES NOT WORK FOR OPENED SHAPES --> THE SHAPE REALLY NEED BE CLOSED TO WORK!!!

# with shape [('circle', ((2, 3), (15, 16)))] --> error optimal pavlidis not found, returning first pixel and inifinite loop --> need a control --> but most likely this shape cannot exist...
# 2 15 optimal pavlidis start (2, 15)
# You actually can choose ANY black boundary pixel to be your start pixel as long as when you're initially standing on it, your left adjacent pixel is NOT black. --> in fact easy then

def find_pavlidis_optimal_orientation(pixel_coordinates, start_coords=None):

    pixel_coordinates = pixel_coordinates.tolist()
    print(pixel_coordinates)
    if start_coords is not None:
        y=start_coords[0]
        x=start_coords[1]
        if [y-1, x] in pixel_coordinates:
            return 0
        if [y, x+1] in pixel_coordinates:
            return 1
        if [y+1, x] in pixel_coordinates:
            return 2
        if [y, x-1] in pixel_coordinates:
            return 3

    for y, x in pixel_coordinates:
        print(y,x)
        if [y-1, x] in pixel_coordinates:
            return 0
        if [y, x+1] in pixel_coordinates:
            return 1
        if [y+1, x] in pixel_coordinates:
            return 2
        if [y, x-1] in pixel_coordinates:
            return 3
    return None


def find_pavlidis_optimal_start2(pixel_coordinates):
    # px_coords =pixel_coordinates.tolist()
    pixel_coordinates = pixel_coordinates.tolist()

    for y, x in pixel_coordinates:
        # You actually can choose ANY black boundary pixel to be your start pixel as long as when you're initially standing on it, your left adjacent pixel is NOT black. --> in fact easy then
        # nb the left adjacent pixel is the upper left pixel in fact --> no in fact it's left
        # if (y-1, x-1) not in pixel_coordinates:
        # if (y-1, x-1) not in pixel_coordinates and (y-1,x-0) in pixel_coordinates:
        if [y-1, x] not in pixel_coordinates:
            return (y,x)
    return None


def find_pavlidis_optimal_start(pixel_coordinates):
    # print(pixel_coordinates)
    # px_coords = np.vstack(pixel_coordinates)
    # print(px_coords)
    # print(px_coords.shape)

    # px_coords = list(zip(pixel_coordinates[0].tolist(),pixel_coordinates[1].tolist()))
    px_coords = pixel_coordinates.tolist()
    # print('px_coords',px_coords)
    # px_coords = np.asarray(px_coords)
    for y,x in px_coords:
        # print(y,x)
        # in fact if in another orientation I need to set the entry so it will not work --> just loop for one direction then
        for orientation, pavlidis_shift in enumerate(starting_pavlidis):

            # print(orientation, '--', pavlidis_shift)
            # pavlidis_shift = starting_pavlidis[0]
            # check that all lwr neighbs of the starting pixel are black/non colored the same as the pavlidis cell
            counter_white = 0
            for shift in pavlidis_shift:
                # print('g',(y + shift[0], x + shift[1]) in px_coords)
                if (y + shift[0], x + shift[1]) in px_coords:
                    counter_white+=1

                # print((y-0, x-0) in px_coords)
            if counter_white !=0:
                print('counter_white',counter_white)
            if counter_white == 3:
                return (y,x), orientation


    print('error optimal pavlidis not found, returning None')
    return None, 0
    # return pixel_coordinates[0][0], pixel_coordinates[1][0]


# fairly good --> keep pavlidis like that, hope it will not be too slow...

if __name__ == '__main__':

    if True:
        # TODO finalize my sorting code --> TODO
        # generate random shapes

        # [('triangle', ((12, 41), (6, 39)))] --> with this traingle the pavlidis is the contour algo that works the best !!!

        #  NB with this traingle that really does not work!!!! -->
        # [('triangle', ((29, 51), (4, 29)))]
        # True
        # [[30 16]
        #  [29 16]]

        import skimage.draw
        # img, labels = skimage.draw.random_shapes((32, 32), max_shapes=3)
        img, lbs = skimage.draw.random_shapes((64, 64), max_shapes=1)

        img = img[..., 0]
        img[img == 255] = 0
        img[img != 0] = 255
        skip_plot = False

        # single point image
        # check for single pixel sorting --> TODO
        # debug for one pixel wide pavlidis -> KEEP
        # img = np.zeros_like(img)
        # img[int(img.shape[0]/2), int(img.shape[1]/2)] = 255

        # np.nonzero(x)
        # plt.imshow(img)
        # plt.show()


        first_pixel = np.nonzero(img)  # np.argwhere(img!=255)#np.nonzero(img)
        print(lbs)  # NB THERE ARE DUPLICATED POINTS WITH TRIANGLES --> IS THERE A SMART WAY TO HANDLE DUPLICATED POINTS --> AND OR TO RECONNECT THE MISSING PARTS --> MAYBE BUT THINK ABOUT IT!!!
        # NB SHALL I DO A NO REPICK !!! BY CUTTING POINTS ALREADY TAKEN ?????

        # need get the contour just otherwsie does not work --> cheat !!!

        lb = label(img, connectivity=1, background=0)
        rps = regionprops(lb)


        first_pixel = np.nonzero(img)  # np.argwhere(img!=255)#np.nonzero(img)

        # could loop over those pixels to find a pavlidis compatible entry

        # pb here is I need check orientation
        # neo_start, orientation = find_pavlidis_optimal_start(first_pixel) # faisl for [('triangle', ((58, 60), (54, 56)))] also for [('circle', ((3, 4), (20, 21)))]
        # print(first_pixel[0][0], first_pixel[1][0], 'optimal pavlidis start', neo_start)


        start = timer()


        # simple algo that fails for triangles but ok in fact # maybe a better way would be to run it several times and instead of removing dupes to take the longest pixel repeat
        sortedpx = pavlidis(img, first_pixel[0][0], first_pixel[1][0], True)
        # sortedpx = pavlidis(img, neo_start[0], neo_start[1], True) # --> [('circle', ((23, 24), (1, 2)))] infinite loop with neo start --> need check if begining is encountered twice and return if that is True

        # ça marche pr les triangles mais plante des fois pr les cercles --> implement the no start twice algo
        # if neo_start is None:
        #     neo_start,_ = get_feret_from_points(extreme_points(first_pixel))
        #     plt.imshow(lb)
        #     plt.show()
        #     print('corrected start',neo_start)

        # [('circle', ((3, 4), (20, 21)))] --> infinite loop why --> in fact these are the bounds --> if no empty pixel in the shape then stuff will not work --> must not compute the pavlidis because may enter an infinite loop !!!
        # somehow would need a control for that!!!
        # can there be bugs if no data is generated through

        # if there is a single point just return directly coords as there is really nothing todo

        # [('ellipse', ((24, 41), (47, 58)))] --> bug in detection of the contour (but probably an impossible shape --> somehow need to do controls, still but ok

        # sortedpx = pavlidis(img, neo_start[0], neo_start[1], True, starting_orientation=orientation) # evn when distance > sqrt 2 the order of pixels is actually still correct!!! # --> [('circle', ((23, 24), (1, 2)))] infinite loop with neo start --> need check if begining is encountered twice and return if that is True
        # sortedpx = pavlidis(img, first_pixel[0][0], first_pixel[0][1], True) # evn when distance > sqrt 2 the order of pixels is actually still correct!!! # --> [('circle', ((23, 24), (1, 2)))] infinite loop with neo start --> need check if begining is encountered twice and return if that is True
        duration = timer() - start

        print('end pavlidis 1')


        # --> infinite loop unfortunately...
        #fails for [('rectangle', ((42, 54), (25, 29)))]
        # start = timer()
        # sortedpx2 = pavlidis2(img, first_pixel[0][0], first_pixel[1][0], True) # --> infinite loop for [('rectangle', ((40, 49), (13, 53)))]
        # duration2 = timer() - start

        # print('pvalidis time difference', duration, duration2, sortedpx, sortedpx2)


        __plot_pavlidis(img, sortedpx, skip_plot=False)
        img[img!=0]=0

        sortedpx = np.asarray(sortedpx)


        img[sortedpx[:,0], sortedpx[:,1]]=255
        # img[sortedpx]=255


        lb = label(img, connectivity=1, background=0)
        rps = regionprops(lb)

        # errors should not happen with a proper entry point for pavlidis --> maybe check that --> auto would do the job


        # NB none of these new algos works as nicely to sort pixels as does the pavlidis algo --> stick to that for now
        # test of other algos compared to pavlidis --> nothing works as nicely
        # they are immediately stuck at the entry in the case of triangles
        # both work for squares though and classical shapes --> if px count is ok then maybe I can rely on them and if not ok then roll back to pavlidis --> just to gain speed --> INDEED TRY THAT!!!
        # none of the two work for ellipses neither but not clear why --> are these algos somehow just four connected --> check them deeply


        # crap and if only one pixel --> a bug
        # sorted_pixels = sort_pixels_clowkise2(rps[0].coords)
        # print(is_distance_continuous(sorted_pixels))


        # mount a talk
        # __plot_pavlidis(img, sorted_pixels, skip_plot=False)

        # sorted_pixels = nearest_neighbor_ordering(rps[0].coords) # for small triangles it can crash ValueError: Expected n_neighbors <= n_samples,  but n_samples = 2, n_neighbors = 3 --> put it in a try loop if used
        # __plot_pavlidis(img, sorted_pixels, skip_plot=False)

        # qsdqsqsdsqdqsdqsdsqd

    if False:
        array = np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        # array=np.asarray([[0,0],[1,1],[2,2]])
        print(array)
        print(array.shape)
        print(np.roll(array, -1, axis=-1))
        print(np.roll(array, -1, axis=0))  # --> this is what I wwnat in fact
        print(np.roll(array, 1, axis=0))  # --> this is what I wwnat in fact
        # print(np.roll(array, -1))

        line_test = np.asarray(
            [[255, 256], [255, 257], [255, 258], [255, 259], [255, 260], [255, 261], [255, 262], [255, 263], [255, 264],
             [255, 265], [255, 266], [255, 267], [255, 268], [255, 269], [255, 270], [255, 271], [255, 272], [255, 273],
             [255, 274], [255, 275], [255, 276], [255, 277], [255, 278], [255, 279], [255, 280], [255, 281], [255, 282],
             [255, 283], [255, 284], [255, 285], [255, 286], [255, 287], [255, 288], [255, 289], [255, 290], [255, 291],
             [255, 292], [255, 293], [255, 294], [255, 295], [255, 296], [255, 297], [255, 298], [255, 299], [255, 300],
             [255, 301], [255, 302], [255, 303], [255, 304], [255, 305], [255, 306], [255, 307], [255, 308], [255, 309],
             [255, 310], [255, 311], [255, 312], [255, 313], [255, 314], [255, 315], [255, 316], [255, 317], [255, 318],
             [255, 319], [255, 320], [255, 321], [255, 322], [255, 323], [255, 324], [255, 325], [255, 326], [255, 327],
             [255, 328], [255, 329], [255, 330], [255, 331], [255, 332], [255, 333], [255, 334], [255, 335], [255, 336],
             [255, 337], [255, 338], [255, 339], [255, 340], [255, 341], [255, 342], [255, 343], [255, 344], [255, 345],
             [255, 346], [255, 347], [255, 348], [255, 349], [255, 350], [255, 351], [255, 352], [255, 353], [255, 354],
             [255, 355], [255, 356], [255, 357], [255, 358], [255, 359], [255, 360], [255, 361], [255, 362], [255, 363],
             [255, 364], [255, 365], [255, 366], [255, 367], [255, 368], [255, 369], [255, 370], [255, 371], [255, 372],
             [255, 373], [255, 374], [255, 375], [255, 376], [255, 377], [255, 378], [255, 379], [255, 380], [255, 381],
             [255, 382], [255, 383], [255, 384], [255, 385], [255, 386], [255, 387], [255, 388], [255, 389], [255, 390],
             [255, 391], [255, 392], [255, 393], [255, 394], [255, 395], [255, 396], [255, 397], [255, 398], [255, 399],
             [255, 400], [255, 401], [255, 402], [255, 403], [255, 404], [255, 405], [255, 406], [255, 407], [255, 408],
             [255, 409], [255, 410], [255, 411], [255, 412], [255, 413], [255, 414], [255, 415], [255, 416], [255, 417],
             [255, 418], [255, 419], [255, 420], [255, 421], [255, 422], [255, 423], [255, 424], [255, 425], [255, 426],
             [255, 427], [255, 428], [255, 429], [255, 430], [255, 431], [255, 432], [255, 433], [255, 434], [255, 435],
             [255, 436], [255, 437], [255, 438], [255, 439], [255, 440], [255, 441], [255, 442], [255, 443], [255, 444],
             [255, 445], [255, 446], [255, 447], [255, 448], [255, 449], [255, 450], [255, 451], [255, 452], [255, 453],
             [255, 454], [255, 455], [255, 456], [255, 457], [255, 458], [255, 459], [255, 460], [255, 461], [255, 462],
             [255, 463], [255, 464], [255, 465], [255, 466], [255, 467], [255, 468], [255, 469], [255, 470], [255, 471],
             [255, 472], [255, 473], [255, 474], [255, 475], [255, 476], [255, 477], [255, 478], [255, 479], [255, 480],
             [255, 481], [255, 482], [255, 483], [255, 484], [255, 485], [255, 486], [255, 487], [255, 488], [255, 489],
             [255, 490], [255, 491], [255, 492], [255, 493], [255, 494], [255, 495], [255, 496], [255, 497], [255, 498],
             [255, 499], [255, 500], [255, 501], [255, 502], [255, 503], [255, 504], [255, 505], [255, 506], [255, 507],
             [255, 508], [255, 509], [255, 510], [255, 511], [256, 255], [256, 254], [256, 253], [256, 252], [256, 251],
             [256, 250], [256, 249], [256, 248], [256, 247], [256, 246], [256, 245], [256, 244], [256, 243], [256, 242],
             [256, 241], [256, 240], [256, 239], [256, 238], [256, 237], [256, 236], [256, 235], [256, 234], [256, 233],
             [256, 232], [256, 231], [256, 230], [256, 229], [256, 228], [256, 227], [256, 226], [256, 225], [256, 224],
             [256, 223], [256, 222], [256, 221], [256, 220], [256, 219], [256, 218], [256, 217], [256, 216], [256, 215],
             [256, 214], [256, 213], [256, 212], [256, 211], [256, 210], [256, 209], [256, 208], [256, 207], [256, 206],
             [256, 205], [256, 204], [256, 203], [256, 202], [256, 201], [256, 200], [256, 199], [256, 198], [256, 197],
             [256, 196], [256, 195], [256, 194], [256, 193], [256, 192], [256, 191], [256, 190], [256, 189], [256, 188],
             [256, 187], [256, 186], [256, 185], [256, 184], [256, 183], [256, 182], [256, 181], [256, 180], [256, 179],
             [256, 178], [256, 177], [256, 176], [256, 175], [256, 174], [256, 173], [256, 172], [256, 171], [256, 170],
             [256, 169], [256, 168], [256, 167], [256, 166], [256, 165], [256, 164], [256, 163], [256, 162], [256, 161],
             [256, 160], [256, 159], [256, 158], [256, 157], [256, 156], [256, 155], [256, 154], [256, 153], [256, 152],
             [256, 151], [256, 150], [256, 149], [256, 148], [256, 147], [256, 146], [256, 145], [256, 144], [256, 143],
             [256, 142], [256, 141], [256, 140], [256, 139], [256, 138], [256, 137], [256, 136], [256, 135], [256, 134],
             [256, 133], [256, 132], [256, 131], [256, 130], [256, 129], [256, 128], [256, 127], [256, 126], [256, 125],
             [256, 124], [256, 123], [256, 122], [256, 121], [256, 120], [256, 119], [256, 118], [256, 117], [256, 116],
             [256, 115], [256, 114], [256, 113], [256, 112], [256, 111], [256, 110], [256, 109], [256, 108], [256, 107],
             [256, 106], [256, 105], [256, 104], [256, 103], [256, 102], [256, 101], [256, 100], [256, 99], [256, 98],
             [256, 97], [256, 96], [256, 95], [256, 94], [256, 93], [256, 92], [256, 91], [256, 90], [256, 89],
             [256, 88], [256, 87], [256, 86], [256, 85], [256, 84], [256, 83], [256, 82], [256, 81], [256, 80],
             [256, 79], [256, 78], [256, 77], [256, 76], [256, 75], [256, 74], [256, 73], [256, 72], [256, 71],
             [256, 70], [256, 69], [256, 68], [256, 67], [256, 66], [256, 65], [256, 64], [256, 63], [256, 62],
             [256, 61], [256, 60], [256, 59], [256, 58], [256, 57], [256, 56], [256, 55], [256, 54], [256, 53],
             [256, 52], [256, 51], [256, 50], [256, 49], [256, 48], [256, 47], [256, 46], [256, 45], [256, 44],
             [256, 43], [256, 42], [256, 41], [256, 40], [256, 39], [256, 38], [256, 37], [256, 36], [256, 35],
             [256, 34], [256, 33], [256, 32], [256, 31], [256, 30], [256, 29], [256, 28], [256, 27], [256, 26],
             [256, 25], [256, 24], [256, 23], [256, 22], [256, 21], [256, 20], [256, 19], [256, 18], [256, 17],
             [256, 16], [256, 15], [256, 14], [256, 13], [256, 12], [256, 11], [256, 10], [256, 9], [256, 8], [256, 7],
             [256, 6], [256, 5], [256, 4], [256, 3], [256, 2], [256, 0], [256, 1]])

        print(line_test[255])
        print(line_test.shape[0])
        print(np.roll(line_test, -255 - 1, axis=0))

        # en fait c'est juste pas possible de corriger car le ligne est tt simplement inverser --> faudrait inverser l'ordre complet de la ligne --> reflechir à comment faire en fait --> pas si simple --> TODO

        import sys

        sys.exit(0)

    # test for nearest neighbor ordering
    if True:
        coords = [(4, 30), (5, 30), (6, 31), (7, 31), (8, 32), (9, 32), (10, 33), (11, 33), (12, 34), (13, 34),
                  (14, 35),
                  (15, 36), (15, 35), (15, 34), (15, 33), (15, 32), (15, 31), (15, 30), (15, 29), (15, 28), (15, 27),
                  (15, 26), (15, 25), (15, 24), (14, 25), (13, 26), (12, 26), (11, 27), (10, 27), (9, 28), (8, 28),
                  (7, 29),
                  (6, 29), (5, 30)]
        print(coords)

        # remove dupes
        # coords = list(set(coords))

        # print(coords.shape)
        nn = nearest_neighbor_ordering(coords, remove_dupes=True)

        print(nn)
        print(nn.shape)

        # print('ordered coords', no_dupes)
        print('dist extremities', dist2D(nn[0], nn[-1]),
              dist2D(nn[0], nn[-1]) > math.sqrt(2))  # still dist > sqrt 2 --> impossible to manage for such shape

    # ça marche
    # test for ordering of pixels of bond like structures --> should always work I think
    if True:
        img = np.zeros((512, 512))
        rr, cc = line(256, 0, 255, img.shape[1] - 1)

        img[rr, cc] = 255

        labels = label(img, connectivity=2, background=0)
        rps = regionprops(labels)

        # in the middle there is
        # 255 511
        # 256 0 --> pixels are scanned line by line and not ordered --> length of the line need be calculated after a pavlidis algo --> practically see how I can do that in a smart way based on ROIs

        for region in rps:
            # that is a lot of code and can all be put and stored directly in the pavlidis --> TODO

            # TODO try a plot of the region
            # maybe get bounds and plot within them --> should be easy in fact

            print(region.bbox)
            # create an image for the bbox

            # tmp_img = np.zeros((abs(region.bbox[2]-region.bbox[0]),abs(region.bbox[1]-region.bbox[3])))
            # tmp_img = np.zeros((max(region.bbox[2],region.bbox[0])+1,max(region.bbox[1],region.bbox[3])+1)) # then easy to plt my object in it
            # print(tmp_img.shape)
            # print(region.coords.shape)
            # # fill numpy array using region coords TOP MEGA TOP TIP TIPTOP TIP TOP
            # tmp_img[region.coords[:,0], region.coords[:,1]] = 255 # so cool another very easy way to plot coords --> but is that fast or slow ??? --> IF FAST COULD REPLACE SO MANY OF MY CALLS !!!
            # tmp_img[np.split(region.coords, axis=-1)] = 255 # there must be a way to do it like that too!!

            # create an image
            # then try to plot in it
            # plt.imshow(tmp_img)
            # plt.show()

            # quite good by the way
            # smart is maybe to try to use as a seed of of the most distant points

            # pavlidis for region --> allow __plot_pavlidis to plot this and maybe add checks for the stuff
            # add pavlidis

            # --> really not worth it for contours --> rather use nn instead
            # total pavlidis 18.360223728988785
            # total nn 2.1951778780203313
            # start_pavlidis = timer()
            # for i in range(100):


            # optimal_pavlidis_entry = find_pavlidis_optimal_start(region.coords) --> this unfortunately does not work for non closed stuff --> need the stuff really be closed otherwise take an extremity is required...
            # print('optimal_pavlidis_entry', optimal_pavlidis_entry)
            sorted_coords = pavlidis(region)  # --> marche pas si pas à une extremite en fait --> in that case I May need to start from the extreme point --> in that case would that always work
            # sorted_coords = pavlidis(region, optimal_pavlidis_entry[0], optimal_pavlidis_entry[1])  # --> marche pas si pas à une extremite en fait --> in that case I May need to start from the extreme point --> in that case would that always work
            # pb est que je peux avoir des jumps --> faudrait faire une rotation du truc pr y arriver
            # sorted_coords = nearest_neighbor_ordering(region.coords)  # --> marche pas si pas à une extremite en fait --> in that case I May need to start from the extreme point --> in that case would that always work

            print('mega test of all', np.asarray(sorted_coords).shape)
            compute_perimeter(sorted_coords)

            print('is_distance_continuous pavlidis', is_distance_continuous(sorted_coords))
            print('is_distance_continuous nn', is_distance_continuous(nearest_neighbor_ordering(
                region.coords)))  # could return the coord of the max -- give me the roll to apply # --> TODO but maybe ok in fact
            output = is_distance_continuous(nearest_neighbor_ordering(region.coords))

            # orderd_yet_another = order_points_new(region.coords)
            # seems to work well too --> but try with different shapes
            start_pavlidis = timer()

            # for i in range(1):
            #     orderd_yet_another = sort_pixels_clowkise2(region.coords)
            # print('total other', timer() - start_pavlidis) # total other 0.2618692510004621 --> this is super fast --> if that works then I would

            # print('is_distance_continuous another', is_distance_continuous(orderd_yet_another))
            #
            # __plot_pavlidis(region, orderd_yet_another, skip_plot=False)

            try:
                if len(output) == 2:
                    max_dist, max_dist_pos = output

                    uncorrected = nearest_neighbor_ordering(region.coords)

                    print(uncorrected[max_dist_pos])  # --> 255, 511 --> ok in fact

                    # cannot correct because would need to invert a big chunk of the ordering --> literally reverse half of the array

                    # or I would need signs maybe vectors

                    # corrected = np.roll(uncorrected, -max_dist_pos, axis=0) # I HAVE A BUG IN MY ROLL --> FIX IT!!!
                    # corrected = np.roll(uncorrected, -(len(uncorrected)-max_dist_pos), axis=0) # I HAVE A BUG IN MY ROLL --> FIX IT!!!
                    # corrected = np.roll(uncorrected, max_dist_pos, axis=0) # I HAVE A BUG IN MY ROLL --> FIX IT!!!
                    # # , axis = 0
                    # print('uncorrected', uncorrected.tolist())
                    #
                    # # ça ne marche pas le roll --> comment le faire en fait
                    # print('corrected', corrected.tolist())
                    # print('is_distance_continuous nn',
                    #       is_distance_continuous(corrected))

                    __plot_pavlidis(region, uncorrected, skip_plot=False)
            except:
                # if iscontinuous --> ignore and do not try to roll
                pass

            # crappy_test = 23
            # print(len(crappy_test))

            # print('total pavlidis', timer()-start_pavlidis)

            # start_pavlidis = timer()
            # for i in range(100):
            #     sorted_coords = nearest_neighbor_ordering(
            #         region.coords)  # --> marche pas si pas à une extremite en fait --> in that case I May need to start from the extreme point --> in that case would that always work
            # print('total nn', timer() - start_pavlidis)

            # ça marche et c'est assez simple en fait
            __plot_pavlidis(region, sorted_coords, skip_plot=False)

            # shall I add checks in pavlidis
            #

            # print(sorted_coords)
            print(len(sorted_coords), region.coords.shape[0])

            # bug ????
            # print(sorted_coords)
            # if still some missing points then can use the other stuff

        plt.imshow(img)
        plt.show()

        import sys

        sys.exit(0)
        # almost ok now just try to plot stuff

    # test pavlidis on several shapes
    if True:
        start_time = timer()
        img = np.zeros((5, 5), dtype=np.uint8)
        start = (1, 1)
        extent = (3, 3)
        rr, cc = rectangle(start, extent=extent, shape=img.shape)
        img[rr, cc] = 255
        # img
        # array([[0, 0, 0, 0, 0],
        #        [0, 1, 1, 1, 0],
        #        [0, 1, 1, 1, 0],
        #        [0, 1, 1, 1, 0],
        #        [0, 0, 0, 0, 0]], dtype=uint8)
        # img =

        # false
        skip_plot = True

        print(pavlidis(img, 0, 0, True, color_of_interest=255))
        print(pavlidis(img, 4, 4, True, color_of_interest=255))
        print(pavlidis(img, 120, 120, True, color_of_interest=255))
        print(pavlidis(img, -1, 2, True, color_of_interest=255))
        # ok
        __plot_pavlidis(img, pavlidis(img, 1, 1, True),
                        skip_plot=skip_plot)  # expected [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1)]
        __plot_pavlidis(img, pavlidis(img, 1, 2, True), skip_plot=skip_plot)  # expected

        # ça a l'air de marcher et c'est facile --> test speed

        img = np.zeros((2048, 2048), dtype=np.uint8)
        start = (1, 1)
        extent = (2046, 2046)
        rr, cc = rectangle(start, extent=extent, shape=img.shape)
        img[rr, cc] = 255
        # img
        # array([[0, 0, 0, 0, 0],
        #        [0, 1, 1, 1, 0],
        #        [0, 1, 1, 1, 0],
        #        [0, 1, 1, 1, 0],
        #        [0, 0, 0, 0, 0]], dtype=uint8)
        # img =

        __plot_pavlidis(img, pavlidis(img, 1, 1, True),
                        skip_plot=skip_plot)  # expected [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1)]
        __plot_pavlidis(img, pavlidis(img, 1, 2, True), skip_plot=skip_plot)  # expected

        # print(len(pavlidis(img, 1, 1, True)))

        print("done 1", timer() - start_time)  # 1000 gigantic sorted contours -->

        # for i in range(100):
        #     print(pavlidis(img, 1, 1))  # expected [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1)]
        #     print(pavlidis(img, 1, 2))  # expected
        #
        # # tt a l'air ok --> juste voir le resultat et tt plotter
        # print("done 100", timer() - start_time) # 1000 gigantic sorted contours --> 14.264636867999798s --> probably not that bad in fact --> if still too slow then code it in Cython --> TODO

        # try with a circle too --> TODO

        from skimage.draw import polygon_perimeter

        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc = polygon_perimeter([5, -1, 5, 10], [-1, 5, 11, 5], shape=img.shape, clip=True)
        img[rr, cc] = 255
        # img
        # array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        #        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        #        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        #        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        #        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
        #        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)

        # print('in polygon')
        __plot_pavlidis(img, pavlidis(img, 4, 0, True), skip_plot=skip_plot)  # on a polygon
        # print('done')

        import numpy as np
        from skimage.draw import bezier_curve

        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc = bezier_curve(1, 5, 5, -2, 8, 8, 2)
        img[rr, cc] = 255

        print('in bezier')
        # maybe blacken the pixel and or restore it in the end???
        __plot_pavlidis(img, pavlidis(img, 4, 1, False),
                        skip_plot=skip_plot)  # on a bezier --> not closed --> if takes same point again --> then get out --> will be incomple

        print('img[2,6]', img[2, 6])  # there is a bug cause 2,6 = 0 and should never be in

        # il y a un bug ici qui le fait remonter --> marche pas --> maybe record all
        __plot_pavlidis(img, pavlidis(img, 1, 5, False),
                        skip_plot=skip_plot)  # in fact I really need to do things this way !!! and really need check contour or disconnect after me --> no cluse# to get it right --> need start from a vertex # can be set to True if starting from a vertex --> in fact that is even required to get the stuff done properly otherwise pxs are missing # NB THE NO REPICK IS PROBABLY NOT A SMART IDEA
        __plot_pavlidis(img, pavlidis(img, 8, 8, False), skip_plot=skip_plot)

        # --> there is a bug with duplicated pixels in here--> WHY
        print(
            'done bezier')  # --> pb as no repick is set the stuff always goes back and forth --> need a no repick just to be sure --> if countour is not connected --> it is needed
        # img
        # array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        #        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

        # NB IL ME MANQUE DES CHANGEMENTS D'orientation sur P1 --> OK now !!!

        # ça marche sur des trucs comme ça

        print("final", timer() - start_time)  # 1000 gigantic sorted contours -->

        # all is ok in fact

        from skimage.draw import circle_perimeter

        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc = circle_perimeter(4, 4, 3)
        img[rr, cc] = 255
        # img
        # array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        #        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        #        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        #        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        #        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        #        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        #        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

        print(img[1, 3])
        __plot_pavlidis(img, pavlidis(img, 1, 3, False), skip_plot=skip_plot)

        from skimage.draw import circle_perimeter_aa

        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc, _ = circle_perimeter_aa(4, 4, 3)
        img[rr, cc] = 255
        # img
        # array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        #        [  0,   0,  60, 211, 255, 211,  60,   0,   0,   0],
        #        [  0,  60, 194,  43,   0,  43, 194,  60,   0,   0],
        #        [  0, 211,  43,   0,   0,   0,  43, 211,   0,   0],
        #        [  0, 255,   0,   0,   0,   0,   0, 255,   0,   0],
        #        [  0, 211,  43,   0,   0,   0,  43, 211,   0,   0],
        #        [  0,  60, 194,  43,   0,  43, 194,  60,   0,   0],
        #        [  0,   0,  60, 211, 255, 211,  60,   0,   0,   0],
        #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        #        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)

        __plot_pavlidis(img, pavlidis(img, 1, 2, True), skip_plot=skip_plot)
        __plot_pavlidis(img, pavlidis(img, 1, 3, True), skip_plot=skip_plot)
        # that really seems to work now!!!!

        from skimage.draw import disk

        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc = disk((4, 4), 5)
        img[rr, cc] = 255
        # img
        # array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        #        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        #        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        #        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

        __plot_pavlidis(img, pavlidis(img, 0, 2, True), skip_plot=skip_plot)

        from skimage.draw import ellipse

        img = np.zeros((10, 12), dtype=np.uint8)
        rr, cc = ellipse(5, 6, 3, 5, rotation=np.deg2rad(30))
        img[rr, cc] = 255
        # img
        # array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        #        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        #        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        #        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        #        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        #        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

        __plot_pavlidis(img, pavlidis(img, 2, 6, True), skip_plot=skip_plot)

        from skimage.draw import ellipse_perimeter

        rr, cc = ellipse_perimeter(2, 3, 4, 5)
        img = np.zeros((9, 12), dtype=np.uint8)
        img[rr, cc] = 255
        # img
        # array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        #        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        #        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        #        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=uint8)
        __plot_pavlidis(img, pavlidis(img, 0, 7, False), skip_plot=skip_plot)
        __plot_pavlidis(img, pavlidis(img, 8, 0, False), skip_plot=skip_plot)

        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc = ellipse_perimeter(5, 5, 3, 4)
        img[rr, cc] = 255
        # img
        # array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        #        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        #        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        #        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

        __plot_pavlidis(img, pavlidis(img, 4, 1, True), skip_plot=skip_plot)

        # this algo works exactly as I desire --> quite good!!!
        from skimage.draw import line

        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc = line(1, 1, 8, 8)
        img[rr, cc] = 255
        # img
        # array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

        __plot_pavlidis(img, pavlidis(img, 1, 1, False), skip_plot=skip_plot)
        __plot_pavlidis(img, pavlidis(img, 8, 8, False), skip_plot=skip_plot)

        from skimage.draw import line_aa

        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc, val = line_aa(1, 1, 8, 8)
        img[rr, cc] = 255
        # img
        # array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 255, 74, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 74, 255, 74, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 74, 255, 74, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 74, 255, 74, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 74, 255, 74, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 74, 255, 74, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 74, 255, 74, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 74, 255, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

        __plot_pavlidis(img, pavlidis(img, 1, 1, False),
                        skip_plot=skip_plot)  # ça marche du tonnerre c'est vraiment exactement le contour et juste le contour --> parfait!!!
        __plot_pavlidis(img, pavlidis(img, 8, 8, False),
                        skip_plot=skip_plot)  # ici il y a une petite erreur mais je pense ok en fait --> pr eviter erreur faudrait partir avec un orientation differente --> TODO see if I can find a fix otherwise ok for now -> check if this is one of the exception for the starting pixels --> not so bad in fact and I dunno how I could prevent that

        from skimage.draw import polygon

        img = np.zeros((10, 10), dtype=np.uint8)
        r = np.array([1, 2, 8])
        c = np.array([1, 7, 4])
        rr, cc = polygon(r, c)
        img[rr, cc] = 255
        # img
        # array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        #        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        #        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        #        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

        __plot_pavlidis(img, pavlidis(img, 1, 1, True),
                        skip_plot=skip_plot)  # this one has a bug but it's a case I'll never face --> would have to choose a better start
        __plot_pavlidis(img, pavlidis(img, 8, 4, True),
                        skip_plot=skip_plot)  # this one has a bug but it's a case I'll never face --> would have to choose a better start # bug here too --> gives duplicated pixels
        __plot_pavlidis(img, pavlidis(img, 4, 3, True),
                        skip_plot=skip_plot)  # I think there is anyway no way of getting a clockwise sorted contour for this shape OR ?

        from skimage.draw import polygon_perimeter

        img = np.zeros((10, 10), dtype=np.uint8)
        rr, cc = polygon_perimeter([5, -1, 5, 10],
                                   [-1, 5, 11, 5],
                                   shape=img.shape, clip=True)
        img[rr, cc] = 255
        # img
        # array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        #        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        #        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        #        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        #        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
        #        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)

        __plot_pavlidis(img, pavlidis(img, 0, 4, True), skip_plot=skip_plot)

        # generate random shapes
        import skimage.draw

        # img, labels = skimage.draw.random_shapes((32, 32), max_shapes=3)
        img, labels = skimage.draw.random_shapes((64, 64), max_shapes=1)

        img = img[..., 0]
        img[img == 255] = 0
        img[img != 0] = 255
        skip_plot = False
        # np.nonzero(x)
        # plt.imshow(img)
        # plt.show()

        first_pixel = np.nonzero(img)  # np.argwhere(img!=255)#np.nonzero(img)
        print(
            labels)  # NB THERE ARE DUPLICATED POINTS WITH TRIANGLES --> IS THERE A SMART WAY TO HANDLE DUPLICATED POINTS --> AND OR TO RECONNECT THE MISSING PARTS --> MAYBE BUT THINK ABOUT IT!!!
        # NB SHALL I DO A NO REPICK !!! BY CUTTING POINTS ALREADY TAKEN ?????

        __plot_pavlidis(img, pavlidis(img, first_pixel[0][0], first_pixel[1][0], True),
                        skip_plot=skip_plot)  # maybe will be a pb with 1 px wide cells --> test it and see if there is a fix for that or not ???
        # it really seems to work well!!!

        # coords of duplicated triangle (31, 47)
        # [(30, 47), (31, 47), (32, 48), (33, 48), (34, 49), (35, 49), (36, 50), (37, 51), (38, 51), (39, 52), (40, 52),
        #  (41, 53), (42, 54), (43, 54), (44, 55), (45, 55), (46, 56), (47, 56), (48, 57), (49, 58), (50, 58), (51, 59),
        #  (52, 59), (53, 60), (54, 61), (54, 60), (54, 59), (54, 58), (54, 57), (54, 56), (54, 55), (54, 54), (54, 53),
        #  (54, 52), (54, 51), (54, 50), (54, 49), (54, 48), (54, 47), (54, 46), (54, 45), (54, 44), (54, 43), (54, 42),
        #  (54, 41), (54, 40), (54, 39), (54, 38), (54, 37), (54, 36), (54, 35), (54, 34), (53, 35), (52, 36), (51, 36),
        #  (50, 37), (49, 37), (48, 38), (47, 38), (46, 39), (45, 39), (44, 40), (43, 40), (42, 41), (41, 42), (40, 42),
        #  (39, 43), (38, 43), (37, 44), (36, 44), (35, 45), (34, 45), (33, 46), (32, 46), (31, 47)]

        # another duplicated triangle --> is there an easy fix ???
        # [(37, 28), (38, 28), (39, 29), (40, 29), (41, 30), (42, 30), (43, 31), (44, 31), (45, 32), (46, 33), (47, 33),
        #  (48, 34), (49, 34), (50, 35), (51, 35), (52, 36), (53, 37), (53, 36), (53, 35), (53, 34), (53, 33), (53, 32),
        #  (53, 31), (53, 30), (53, 29), (53, 28), (53, 27), (53, 26), (53, 25), (53, 24), (53, 23), (53, 22), (53, 21),
        #  (53, 20), (53, 19), (52, 20), (51, 21), (50, 21), (49, 22), (48, 22), (47, 23), (46, 23), (45, 24), (44, 25),
        #  (43, 25), (42, 26), (41, 26), (40, 27), (39, 27), (38, 28)]
        # DUPLICATED
        # POINTS
        # DETECTED

        # almost all done --> just need fix all soon --> TODO

        # [('triangle', ((16, 31), (0, 17)))]
        # [(16, 8), (17, 8), (18, 9), (19, 9), (20, 10), (21, 10), (22, 11), (23, 12), (24, 12), (25, 13), (26, 13), (27, 14),
        #  (28, 14), (29, 15), (30, 16), (30, 15), (30, 14), (30, 13), (30, 12), (30, 11), (30, 10), (30, 9), (30, 8),
        #  (30, 7), (30, 6), (30, 5), (30, 4), (30, 3), (30, 2), (30, 1), (30, 0), (29, 1), (28, 2), (27, 2), (26, 3),
        #  (25, 3), (24, 4), (23, 4), (22, 5), (21, 6), (20, 6), (19, 7), (18, 7), (17, 8)]

        # [('triangle', ((38, 41), (52, 55)))]
        # 38 53 255 color_of_interest
        # [(38, 53), (39, 53), (40, 54), (40, 53), (40, 52), (39, 53)]
        # DUPLICATED POINTS DETECTED
        # [(39, 53)]

        # maybe remove last point/dupe to get rid of the error --> would that work ???

        # deduplicated
        # coords[(29, 50), (30, 50), (31, 51), (31, 50), (31, 49)]

        # nb mini ellipse equivalent to a single px area cell has no error in contour and no dupes --> VERY GOOD
        # [('ellipse', ((21, 24), (0, 3)))]
        # [(21, 1), (22, 2), (23, 1), (22, 0)]
        # no dupes

        # MEGA TODO: NB DEDUPLICATION DID NOT WORK  --> distance between end points > sqrt 2 --> is there any way I can fix duplication in a smarter way maybe by 2D distance between adjacent points --> so that I minimize it ???? --> THINK ABOUT THAT!!!
        # but in fact there is no way to get a perfect contour from this OR --> maybe yes upon removal of pixels futher away than sqrt 2 --> THINK ABOUT IT AND TEST THAT!!!!
        # nb if I remove 4,30 and one 5,30 then it WOULD WORK --> I WOULD GET A CLOSED CONTOUR THAT IS MINIMAL --> THINK ABOUT THE FASTEST WAY TO DO THAT
        # [('triangle', ((4, 16), (24, 37)))]
        # 4 30 255 color_of_interest
        # [(4, 30), (5, 30), (6, 31), (7, 31), (8, 32), (9, 32), (10, 33), (11, 33), (12, 34), (13, 34), (14, 35), (15, 36), (15, 35), (15, 34), (15, 33), (15, 32), (15, 31), (15, 30), (15, 29), (15, 28), (15, 27), (15, 26), (15, 25), (15, 24), (14, 25), (13, 26), (12, 26), (11, 27), (10, 27), (9, 28), (8, 28), (7, 29), (6, 29), (5, 30)]
        # DUPLICATED POINTS DETECTED
        # [(5, 30)]
        # deduplicated coords [(4, 30), (5, 30), (6, 31), (7, 31), (8, 32), (9, 32), (10, 33), (11, 33), (12, 34), (13, 34), (14, 35), (15, 36), (15, 35), (15, 34), (15, 33), (15, 32), (15, 31), (15, 30), (15, 29), (15, 28), (15, 27), (15, 26), (15, 25), (15, 24), (14, 25), (13, 26), (12, 26), (11, 27), (10, 27), (9, 28), (8, 28), (7, 29), (6, 29)]
        # dist between extreme points after deduplication 2.23606797749979 False
