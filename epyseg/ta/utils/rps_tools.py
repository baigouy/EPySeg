import numpy as np
import networkx as nx
from skimage.data._binary_blobs import binary_blobs
from sklearn.neighbors import NearestNeighbors
from skimage.draw import line
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
import collections
import matplotlib.pyplot as plt
from skimage.measure._regionprops import RegionProperties
import math
from timeit import default_timer as timer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from numba import jit, njit
__DEBUG__ = False  # for pavlidis code -> stores and prints decisions and directions


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

orientation_pavlidis = [pavlidis_front_up, pavlidis_front_right, pavlidis_front_bottom, pavlidis_front_left]


def __check_pixel_pavlidis(img, y, x, color_of_interest=None):
    """
    Check if a pixel in the image matches the specified color of interest.

    Args:
        img (np.ndarray or list): Image or region properties.
        y (int): Y-coordinate of the pixel.
        x (int): X-coordinate of the pixel.
        color_of_interest: Color value to compare against.

    Returns:
        bool: True if the pixel matches the color of interest, False otherwise.
    """
    if isinstance(img, np.ndarray):
        return img[y, x] == color_of_interest
    if isinstance(img, list):
        return [y, x] in img


def get_feret_from_region(region):
    """
    Calculate the Feret diameter (major axis) of a region.

    Args:
        region: Region properties.

    Returns:
        tuple: Coordinates of the two endpoints of the major axis.
    """
    major_axis_length = region.major_axis_length
    orientation = region.orientation
    centroid = region.centroid

    x0 = centroid[1] - np.sin(orientation) * major_axis_length / 2
    y0 = centroid[0] - np.cos(orientation) * major_axis_length / 2
    x1 = centroid[1] + np.sin(orientation) * major_axis_length / 2
    y1 = centroid[0] + np.cos(orientation) * major_axis_length / 2

    return (y0,x0),(y1,x1)

@njit
def pavlidis2(img, start_y=None, start_x=None, closed_contour=False, color_of_interest=None,
             auto_detect_extreme_points_for_region_upon_missing_start_coords=True):
    """
    Traces a contour in an image using the Pavlidis algorithm.

    Args:
        img (ndarray): The input image.
        start_y (int): The starting y-coordinate for the contour tracing. Default is None.
        start_x (int): The starting x-coordinate for the contour tracing. Default is None.
        closed_contour (bool): Flag indicating whether the contour is closed. Default is False.
        color_of_interest: The color value of interest for contour tracing. Default is None.
        auto_detect_extreme_points_for_region_upon_missing_start_coords (bool): Flag indicating whether
            to automatically detect extreme points for the region if start coordinates are missing.
            Default is True.

    Returns:
        list: A list of (y, x) coordinates representing the traced contour.

    # Examples:
    #     >>> img = np.array([[0, 0, 0, 0, 0],
    #     ...                 [0, 1, 1, 1, 0],
    #     ...                 [0, 1, 0, 1, 0],
    #     ...                 [0, 1, 1, 1, 0],
    #     ...                 [0, 0, 0, 0, 0]])
    #     >>> contour = pavlidis2(img, start_y=2, start_x=1, closed_contour=True)
    #     >>> print(contour)
    #     [(2, 1), (1, 2), (1, 3), (2, 4), (3, 3), (3, 2)]
    #
    #     >>> img = np.array([[0, 0, 0, 0, 0],
    #     ...                 [0, 1, 1, 1, 0],
    #     ...                 [0, 1, 0, 1, 0],
    #     ...                 [0, 1, 1, 1, 0],
    #     ...                 [0, 0, 0, 0, 0]])
    #     >>> contour = pavlidis2(img, start_y=2, start_x=1, closed_contour=False)
    #     >>> print(contour)
    #     [(2, 1), (1, 2), (1, 3), (2, 4), (3, 3)]
    """

    coords = []

    if start_x >= img.shape[1] or start_y >= img.shape[0] or start_x < 0 or start_y < 0:
        print('error coordinates outside the image')
        return coords

    if color_of_interest is None:
        color_of_interest = img[start_y, start_x]

    if img[start_y, start_x] != color_of_interest:
        print('error starting coordinates outside the object')
        return coords

    cur_coord_x = start_x
    cur_coord_y = start_y
    coords.append((start_y, start_x))

    look_direction = 0
    counter = 0
    no_success_counter = 0
    nb_of_encounters_of_first_pixel = 0

    while True:
        counter += 1
        success = False
        no_success_counter = 0

        if no_success_counter >= 3:
            return coords

        if nb_of_encounters_of_first_pixel >= 3:
            return coords

        if cur_coord_x == start_x and cur_coord_y == start_y:
            nb_of_encounters_of_first_pixel += 1

        print('look_direction', look_direction, counter)

        for ppp in range(orientation_pavlidis[look_direction].shape[0]+1):
            shift = orientation_pavlidis[look_direction][ppp]
            print('shift', shift, coords)

            coords_to_test_x = cur_coord_x + int(shift[1])
            coords_to_test_y = cur_coord_y + int(shift[0])

            print(coords_to_test_x, coords_to_test_y)

            if True:  # Avoiding infinite loop for non-closed contours
                if (coords_to_test_y, coords_to_test_x) in coords:
                    return coords

            if coords_to_test_y >= img.shape[0] or coords_to_test_x >= img.shape[1] or \
                    coords_to_test_y < 0 or coords_to_test_x < 0:
                continue

            if img[coords_to_test_y, coords_to_test_x] == color_of_interest:
                cur_coord_y = coords_to_test_y
                cur_coord_x = coords_to_test_x

                if cur_coord_x == start_x and cur_coord_y == start_y:
                    return coords
                coords.append((cur_coord_y, cur_coord_x))

                if ppp == 0:
                    look_direction -= 1
                    if look_direction < 0:
                        look_direction = 3
                break

        if not success:
            look_direction += 1
            if look_direction > 3:
                look_direction = 0

            if no_success_counter >= 4:
                return coords
            no_success_counter += 1

    return coords


def pavlidis(img, start_y=None, start_x=None, closed_contour=False, color_of_interest=None,
             auto_detect_extreme_points_for_region_upon_missing_start_coords=True, starting_orientation=0, early_stop=None):
    """
    Traces a contour in an image using the Pavlidis algorithm.

    Args:
        img: The input image.
        start_y: The starting y-coordinate for the contour tracing. Default is None.
        start_x: The starting x-coordinate for the contour tracing. Default is None.
        closed_contour: Flag indicating whether the contour is closed. Default is False.
        color_of_interest: The color value of interest for contour tracing. Default is None.
        auto_detect_extreme_points_for_region_upon_missing_start_coords: Flag indicating whether
            to automatically detect extreme points for the region if start coordinates are missing.
            Default is True.
        starting_orientation: The starting orientation for contour tracing. Default is 0.
        early_stop: The number of pixels to trace before stopping. Default is None.

    Returns:
        list: A list of (y, x) coordinates representing the traced contour.

    Examples:
        >>> img = np.array([[0, 0, 0, 0, 0],
        ...                 [0, 1, 1, 1, 0],
        ...                 [0, 1, 0, 1, 0],
        ...                 [0, 1, 1, 1, 0],
        ...                 [0, 0, 0, 0, 0]])
        >>> contour = pavlidis(img, start_y=1, start_x=1, closed_contour=True)
        >>> print(contour)
        [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1)]

        >>> img = np.array([[0, 0, 0, 0, 0],
        ...                 [0, 1, 1, 1, 0],
        ...                 [0, 1, 0, 1, 0],
        ...                 [0, 1, 1, 1, 0],
        ...                 [0, 0, 0, 0, 0]])
        >>> contour = pavlidis(img, start_y=3, start_x=1, closed_contour=True)
        >>> print(contour)
        [(3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2)]
    """
    coords = []

    if not isinstance(img, RegionProperties):
        if start_x is not None and start_y is not None:
            if start_x >= img.shape[1] or start_y >= img.shape[0] or start_x < 0 or start_y < 0:
                print('error coordinates outside the image')
                return coords
            if color_of_interest is None:
                color_of_interest = img[start_y, start_x]
            if img[start_y, start_x] != color_of_interest:
                print('error starting coordinates outside the object')
                return coords
        else:
            feret_1, feret_2 = get_feret_from_points(extreme_points(img))
            start_y = feret_1[0]
            start_x = feret_1[1]
            img = img.tolist()
    else:
        img = img.coords
        if start_x is None or start_y is None:
            if not auto_detect_extreme_points_for_region_upon_missing_start_coords:
                start_y, start_x = img[0][0], img[0][1]
            else:
                feret_1, feret_2 = get_feret_from_points(extreme_points(img))
                start_y = feret_1[0]
                start_x = feret_1[1]
        else:
            if not __check_pixel_pavlidis(img.tolist(), start_y, start_x):
                print('error coordinates outside the image')
                return coords
        img = img.tolist()

    cur_coord_x = start_x
    cur_coord_y = start_y
    coords.append((start_y, start_x))

    look_direction = 0
    if starting_orientation != 0:
        look_direction = starting_orientation

    counter = 0
    no_success_counter = 0

    old_coords_x = -1
    old_coords_y = -1

    infinite_loop_counter = 0

    limit = None
    if isinstance(img, list):
        limit = 10 * len(img)

    while True:
        counter += 1
        success = False
        no_success_counter = 0

        if early_stop is not None and len(coords) >= early_stop:
            return coords

        if limit is not None:
            if counter > limit:
                print('error infinite loop Pavlidis --> breaking')
                return coords

        if infinite_loop_counter > 100:
            return coords

        if old_coords_y == cur_coord_y and old_coords_x == cur_coord_x:
            infinite_loop_counter += 1
        else:
            old_coords_y = cur_coord_y
            old_coords_x = cur_coord_x
            infinite_loop_counter = 0

        for ppp, shift in enumerate(orientation_pavlidis[look_direction]):
            coords_to_test_x = cur_coord_x + shift[1]
            coords_to_test_y = cur_coord_y + shift[0]

            if counter > 1:
                if coords_to_test_x == start_x and coords_to_test_y == start_y:
                    return coords

            if not closed_contour:
                if (coords_to_test_y, coords_to_test_x) in coords:
                    return coords

            if isinstance(img, np.ndarray):
                if coords_to_test_y >= img.shape[0] or coords_to_test_x >= img.shape[1] or \
                        coords_to_test_y < 0 or coords_to_test_x < 0:
                    continue

            if __check_pixel_pavlidis(img, coords_to_test_y, coords_to_test_x, color_of_interest):
                cur_coord_y = coords_to_test_y
                cur_coord_x = coords_to_test_x

                if cur_coord_x == start_x and cur_coord_y == start_y:
                    return coords
                coords.append((cur_coord_y, cur_coord_x))

                if ppp == 0:
                    look_direction -= 1
                    if look_direction < 0:
                        look_direction = 3
                success = True
                break

        if not success:
            look_direction += 1
            if look_direction > 3:
                look_direction = 0
            if no_success_counter >= 4:
                return coords
            no_success_counter += 1

    return coords


def __plot_pavlidis(img, coords, skip_plot=True):
    """
    Plot the image or regions (regionprops) with corresponding Pavlidis coordinates.

    Args:
        img: Image or RegionProperties object.
        coords: List of coordinates.
        skip_plot: Boolean value indicating whether to skip plotting the image.

    Returns:
        None
    """

    if isinstance(img, RegionProperties):
        region = img
        tmp_img = np.zeros((max(region.bbox[2], region.bbox[0]) + 1,
                            max(region.bbox[1], region.bbox[3]) + 1))
        tmp_img[region.coords[:, 0], region.coords[:, 1]] = 255
        img = tmp_img

    output = np.zeros_like(img)

    if contains_duplicate_coordinates(coords):
        coords = remove_duplicate_coordinates(coords)

    for val, (y, x) in enumerate(coords):
        output[y, x] = val + 1

    if not skip_plot:
        plt.imshow(output)
        plt.show()

    output[output != 0] = 255

    if not (output == img).all():
        missed = np.copy(img)
        missed[output != 0] = 128
        if not skip_plot:
            plt.imshow(missed)
            plt.show()
    else:
        print('FULL MATCH')


def extreme_points(coords, return_array=True):
    """
    Find the extreme points from a set of coordinates.

    Args:
        coords: List of coordinates.
        return_array: Boolean value indicating whether to return the extreme points as a list or individual tuples.

    Returns:
        List of extreme points [(top), (bottom), (left), (right)] if return_array is True, else returns individual tuples.
    """
    tmp_coords = coords

    if isinstance(coords, tuple):
        tmp_coords = np.asarray(list(zip(coords[0].tolist(), coords[1].tolist())))

    top = tuple(tmp_coords[tmp_coords[..., 1].argmin()])
    bottom = tuple(tmp_coords[tmp_coords[..., 1].argmax()])
    left = tuple(tmp_coords[tmp_coords[..., 0].argmin()])
    right = tuple(tmp_coords[tmp_coords[..., 0].argmax()])

    if not return_array:
        return top, bottom, left, right
    else:
        return [top, bottom, left, right]

# returns the most distant points in a set of points
# nb I found a bug in this code in some cases --> needs a fix --> see /home/aigouy/mon_prog/Python/epyseg_pkg/personal/wing/ferret_error.tif
# no clue why though
# the feret is wrong with values ((612, 457), (612, 458), (612, 459), (612, 460), (612, 461), (612, 462), (612, 463), (612, 464), (613, 441), (613, 442), (613, 443), (613, 444), (613, 445), (613, 446), (613, 447), (613, 448), (613, 449), (613, 450), (613, 451), (613, 452), (613, 453), (613, 454), (613, 455), (613, 456), (614, 426), (614, 427), (614, 428), (614, 429), (614, 430), (614, 431), (614, 432), (614, 433), (614, 434), (614, 435), (614, 436), (614, 437), (614, 438), (614, 439), (614, 440), (615, 411), (615, 412), (615, 413), (615, 414), (615, 415), (615, 416), (615, 417), (615, 418), (615, 419), (615, 420), (615, 421), (615, 422), (615, 423), (615, 424), (615, 425), (616, 395), (616, 396), (616, 397), (616, 398), (616, 399), (616, 400), (616, 401), (616, 402), (616, 403), (616, 404), (616, 405), (616, 406), (616, 407), (616, 408), (616, 409), (616, 410), (617, 380), (617, 381), (617, 382), (617, 383), (617, 384), (617, 385), (617, 386), (617, 387), (617, 388), (617, 389), (617, 390), (617, 391), (617, 392), (617, 393), (617, 394), (617, 499), (617, 500), (617, 501), (617, 502), (617, 503), (617, 504), (617, 505), (617, 506), (617, 507), (617, 508), (617, 509), (617, 510), (617, 511), (617, 512), (617, 513), (617, 514), (617, 515), (617, 516), (618, 372), (618, 373), (618, 374), (618, 375), (618, 376), (618, 377), (618, 378), (618, 379), (618, 473), (618, 474), (618, 475), (618, 476), (618, 477), (618, 478), (618, 479), (618, 480), (618, 481), (618, 482), (618, 483), (618, 484), (618, 485), (618, 486), (618, 487), (618, 488), (618, 489), (618, 490), (618, 491), (618, 492), (618, 493), (618, 494), (618, 495), (618, 496), (618, 497), (618, 498), (618, 517), (618, 518), (618, 519), (618, 520), (618, 521), (618, 522), (618, 523), (618, 524), (618, 525), (618, 526), (619, 374), (619, 375), (619, 447), (619, 448), (619, 449), (619, 450), (619, 451), (619, 452), (619, 453), (619, 454), (619, 455), (619, 456), (619, 457), (619, 458), (619, 459), (619, 460), (619, 461), (619, 462), (619, 463), (619, 464), (619, 465), (619, 466), (619, 467), (619, 468), (619, 469), (619, 470), (619, 471), (619, 472), (619, 527), (619, 528), (619, 529), (619, 530), (619, 531), (619, 532), (619, 533), (619, 534), (619, 535), (619, 536), (620, 376), (620, 377), (620, 378), (620, 421), (620, 422), (620, 423), (620, 424), (620, 425), (620, 426), (620, 427), (620, 428), (620, 429), (620, 430), (620, 431), (620, 432), (620, 433), (620, 434), (620, 435), (620, 436), (620, 437), (620, 438), (620, 439), (620, 440), (620, 441), (620, 442), (620, 443), (620, 444), (620, 445), (620, 446), (620, 537), (620, 538), (620, 539), (620, 540), (620, 541), (620, 542), (620, 543), (620, 544), (620, 545), (620, 546), (621, 379), (621, 380), (621, 395), (621, 396), (621, 397), (621, 398), (621, 399), (621, 400), (621, 401), (621, 402), (621, 403), (621, 404), (621, 405), (621, 406), (621, 407), (621, 408), (621, 409), (621, 410), (621, 411), (621, 412), (621, 413), (621, 414), (621, 415), (621, 416), (621, 417), (621, 418), (621, 419), (621, 420), (621, 547), (621, 548), (621, 549), (621, 550), (621, 551), (621, 552), (621, 553), (621, 554), (621, 555), (621, 556), (622, 381), (622, 382), (622, 383), (622, 384), (622, 385), (622, 386), (622, 387), (622, 388), (622, 389), (622, 390), (622, 391), (622, 392), (622, 393), (622, 394), (622, 557), (622, 558), (622, 559), (622, 560), (622, 561), (622, 562), (622, 563), (622, 564), (622, 565), (622, 566), (623, 567), (623, 568), (623, 569), (623, 570), (623, 571), (623, 572), (623, 573), (623, 574), (623, 575), (623, 576), (624, 577), (624, 578), (624, 579), (624, 580), (624, 581), (624, 582), (624, 583), (624, 584), (624, 585), (624, 586), (625, 587), (625, 588), (625, 589), (625, 590), (625, 591), (625, 592), (625, 593), (625, 594), (625, 595), (625, 596), (626, 597), (626, 598), (626, 599), (626, 600), (626, 601), (626, 602), (626, 603), (626, 604), (626, 605), (626, 606), (627, 607), (627, 608), (627, 609), (627, 610), (627, 611), (627, 612), (627, 613), (627, 614), (627, 615), (627, 616), (628, 617), (628, 618), (628, 619), (628, 620), (628, 621), (628, 622), (629, 623), (629, 624), (629, 625), (630, 626), (630, 627), (630, 628), (631, 629), (631, 630), (631, 631), (632, 632), (632, 633), (632, 634), (633, 635), (633, 636), (633, 637), (634, 638), (634, 639), (634, 640), (635, 641), (635, 642), (635, 643), (636, 644), (636, 645), (636, 646), (637, 647), (637, 648), (637, 649), (638, 650), (638, 651), (638, 652), (639, 653), (639, 654), (639, 655), (640, 656), (640, 657), (640, 658), (641, 659), (641, 660), (641, 661), (642, 662), (642, 663), (643, 664), (643, 665), (643, 666), (644, 667), (644, 668), (644, 669), (645, 670), (645, 671), (645, 672), (645, 819), (645, 820), (645, 821), (645, 822), (645, 823), (645, 824), (645, 825), (645, 826), (645, 827), (645, 828), (645, 829), (645, 830), (645, 831), (645, 832), (645, 833), (645, 834), (645, 835), (645, 836), (645, 837), (645, 838), (645, 839), (645, 840), (645, 841), (645, 842), (645, 843), (645, 844), (645, 845), (645, 846), (645, 847), (645, 848), (645, 849), (645, 850), (645, 851), (645, 852), (645, 853), (645, 854), (645, 855), (645, 856), (645, 857), (645, 858), (645, 859), (645, 860), (645, 861), (645, 862), (645, 863), (645, 864), (645, 865), (645, 866), (645, 867), (645, 868), (645, 869), (645, 870), (645, 871), (645, 872), (645, 873), (645, 874), (645, 875), (645, 876), (645, 877), (645, 878), (645, 879), (645, 880), (645, 881), (646, 673), (646, 674), (646, 675), (646, 750), (646, 751), (646, 752), (646, 753), (646, 754), (646, 755), (646, 756), (646, 757), (646, 758), (646, 759), (646, 760), (646, 761), (646, 762), (646, 763), (646, 764), (646, 765), (646, 766), (646, 767), (646, 768), (646, 769), (646, 770), (646, 771), (646, 772), (646, 773), (646, 774), (646, 775), (646, 776), (646, 777), (646, 778), (646, 779), (646, 780), (646, 781), (646, 782), (646, 783), (646, 784), (646, 785), (646, 786), (646, 787), (646, 788), (646, 789), (646, 790), (646, 791), (646, 792), (646, 793), (646, 794), (646, 795), (646, 796), (646, 797), (646, 798), (646, 799), (646, 800), (646, 801), (646, 802), (646, 803), (646, 804), (646, 805), (646, 806), (646, 807), (646, 808), (646, 809), (646, 810), (646, 811), (646, 812), (646, 813), (646, 814), (646, 815), (646, 816), (646, 817), (646, 818), (647, 676), (647, 677), (647, 678), (647, 745), (647, 746), (647, 747), (647, 748), (647, 749), (648, 679), (648, 680), (648, 681), (648, 740), (648, 741), (648, 742), (648, 743), (648, 744), (649, 682), (649, 683), (649, 684), (649, 736), (649, 737), (649, 738), (649, 739), (650, 685), (650, 686), (650, 687), (650, 731), (650, 732), (650, 733), (650, 734), (650, 735), (651, 688), (651, 689), (651, 690), (651, 726), (651, 727), (651, 728), (651, 729), (651, 730), (652, 691), (652, 692), (652, 693), (652, 721), (652, 722), (652, 723), (652, 724), (652, 725), (653, 694), (653, 695), (653, 696), (653, 716), (653, 717), (653, 718), (653, 719), (653, 720), (654, 697), (654, 698), (654, 699), (654, 712), (654, 713), (654, 714), (654, 715), (655, 700), (655, 701), (655, 702), (655, 707), (655, 708), (655, 709), (655, 710), (655, 711), (656, 703), (656, 704), (656, 705), (656, 706))
# BIG BUG THE SOLUTION MAY NOT BE UNIQUE --> NEED A FIX WHEN POSSIBLE IN THE CASE OF BONDS!!--> see example just above and the rps_tools test class

def get_feret_from_points(contour_or_extreme_points):
    """
    Calculate the Feret diameter (maximum distance) between two points from a given set of contour or extreme points.

    Args:
        contour_or_extreme_points: List or array-like object containing contour or extreme points.

    Returns:
        Tuple of two points representing the Feret diameter.

    # Examples:
    #     >>> contour_points = [(10, 20), (30, 40), (50, 60), (70, 80)]
    #     >>> extreme_points(contour_points, return_array=True)
    #
    #     # Calculate Feret diameter from contour points
    #     >>> feret_contour = get_feret_from_points(contour_points)
    #     >>> print('Feret Diameter (Contour Points):', feret_contour)
    #
    #     # Calculate Feret diameter from extreme points
    #     >>> feret_extreme = get_feret_from_points(extreme_points)
    #     >>> print('Feret Diameter (Extreme Points):', feret_extreme)
    """
    C = cdist(contour_or_extreme_points, contour_or_extreme_points)
    furthest_points = np.where(C == C.max())
    feret_1 = contour_or_extreme_points[furthest_points[0][0]]
    feret_2 = contour_or_extreme_points[furthest_points[0][1]]
    return feret_1, feret_2


# it is a pixel sorting algo and a very smart alternative to the pavlidis algo for lines or perimeter (especially because it is insensitive to holes in the connectivity and insensitive to the choice of the start point for non closed shapes)
# THIS IS MUCH FASTER AND POWERFUL THAN PAVLIDIS --> RATHER USE THAT FOR CONTOURS AND FOR BONDS (JUST USE PAVLIDIS FOR FILLED SHAPES)!!!
# pb can have big jumps in it --> may need rotate the result array --> TODO URGENT --> DO THAT ESPECIALLY IF CONTOUR IS LONG
# compute dist to first or last or between last and first and detect closest and do roll of the array so that all is ok after that --> TODO
# RELY ON PAVLIDIS FOR NOW!!!

def nearest_neighbor_ordering(coords, remove_dupes=False):
    """
    Sort a set of coordinates to form a continuous line using the nearest neighbor algorithm.

    Args:
        coords (array-like): List or array-like object containing the coordinates.
        remove_dupes (bool): Flag indicating whether to remove duplicate coordinates (default: False).

    Returns:
        Sorted coordinates forming a continuous line.

    # Examples:
    #     >>> coords = [(0, 0), (1, 1), (2, 2), (3, 3)]
    #     >>> ordered_coords = nearest_neighbor_ordering(coords)
    #     >>> print('Ordered Coordinates:', ordered_coords)
    #
    #     >>> coords = [(0, 0), (1, 1), (2, 2), (1, 1)]
    #     >>> ordered_coords = nearest_neighbor_ordering(coords, remove_dupes=True)
    #     >>> print('Ordered Coordinates (No Dupes):', ordered_coords)
    """
    if not isinstance(coords, np.ndarray):
        coords = np.asarray(coords)

    clf = NearestNeighbors(n_neighbors=2).fit(coords)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)
    order = list(nx.dfs_preorder_nodes(T, 0))
    coords = coords[order]

    if remove_dupes and contains_duplicate_coordinates(coords):
        no_dupes = remove_duplicate_coordinates(coords.tolist())
        coords = np.asarray(no_dupes)

    return coords


def find_duplicate_coordinates(coords):
    """
    Find duplicate coordinates from a given list.

    Args:
        coords (array-like): List or array-like object containing the coordinates.

    Returns:
        List of duplicate coordinates.
    """
    return [item for item, count in collections.Counter(coords).items() if count > 1]


def contains_duplicate_coordinates(coords):
    """
    Check if a list contains duplicate coordinates.

    Args:
        coords (array-like): List or array-like object containing the coordinates.

    Returns:
        True if duplicate coordinates are found, False otherwise.
    """
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
    """
    Remove duplicate coordinates from a list.

    Args:
        coords (array-like): List or array-like object containing the coordinates.

    Returns:
        List of coordinates without duplicates.
    """
    fixed_coords = []
    [fixed_coords.append(item) for item in coords if item not in fixed_coords]
    return fixed_coords


def dist2D(pt1, pt2):
    """
    Calculate the 2D Euclidean distance between two points.

    Args:
        pt1 (tuple): First point (x1, y1).
        pt2 (tuple): Second point (x2, y2).

    Returns:
        Euclidean distance between the two points.
    """
    return math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def compute_perimeter(perimeter_pixels):
    """
    Compute the perimeter length using a set of pixels.

    Args:
        perimeter_pixels (np.ndarray): Array of shape (N, 2 or 3) containing the perimeter pixels.

    Returns:
        Perimeter length.
    """
    length = compute_distance_between_consecutive_points(perimeter_pixels).sum()
    return length

def associate_rps_to_rps_using_centroid_distance(primary_rps, rps_to_be_associated, no_repick=False):
    # MEGA TODO needs be greatly be improved but the idea is there and it's much faster than computing the overlap score
    from personal.coccinelle_ladybug.HashableRegionProps import HashableRegionProps
    from epyseg.ta.measurements.TAmeasures import distance_between_points
    output={}
    # detected_arena = []
    for primary_region in primary_rps:
        # primary_region = (HashableRegionProps) primary_region
        output[ (HashableRegionProps)(primary_region)]=[]
        for region in rps_to_be_associated:
            # if compute_coords_overlap_between_0_and_1(region.coords, primary_region.coords)!=0:
            #     output[(HashableRegionProps)(primary_region)].append(region)
            if distance_between_points(primary_region.centroid, region.centroid)<6:
                    # detected_arena.append(region)
                    output[(HashableRegionProps)(primary_region)].append(region)
                    if no_repick:
                        break
        # if no_repick:
        #     rps_to_be_associated = [region for region in rps_to_be_associated if region not in detected_arena]
    return output

def associate_rps_to_rps_tmp(primary_rps, rps_to_be_associated, no_repick=False):
    # THIS CODE IS ALSO NOT GREAT BECAUSE I SHOULD BREAK WHEN I FIND OVERLAP IF NO REPICK
    # in fact I should maximize the overlap to do this better
    from personal.coccinelle_ladybug.HashableRegionProps import HashableRegionProps
    from personal.geom.tools import compute_coords_overlap_between_0_and_1
    output={}
    # detected_arena = []
    for primary_region in primary_rps:
        # primary_region = (HashableRegionProps) primary_region
        output[ (HashableRegionProps)(primary_region)]=[]
        for region in rps_to_be_associated:
            if compute_coords_overlap_between_0_and_1(region.coords, primary_region.coords)!=0: # REALLY NOT GREAT I SHOULD TAKE THE MOST OVERLAPPING REGION
                output[(HashableRegionProps)(primary_region)].append(region)
                if no_repick:
                    break
                # if no_repick:
                #     detected_arena.append(region)
        # if no_repick:
        #     rps_to_be_associated = [region for region in rps_to_be_associated if region not in detected_arena]
    return output


def associate_rps_to_rps(primary_rps, rps_to_be_associated, no_repick=False): # THE COOL THING ABOUT THIS CODE IS THAT IT CAN ASSOCAITE SEVERAL SMALL RPS TO A BIG ONE --> WHICH IS USEFUL FOR ARENA MATCHING
    # THIS CODE IS ALSO NOT GREAT BECAUSE I SHOULD BREAK WHEN I FIND OVERLAP IF NO REPICK
    # in fact I should maximize the overlap to do this better
    from personal.coccinelle_ladybug.HashableRegionProps import HashableRegionProps
    from personal.geom.tools import compute_coords_overlap_between_0_and_1
    output={}
    detected_arena = []
    for primary_region in primary_rps:
        # primary_region = (HashableRegionProps) primary_region
        output[ (HashableRegionProps)(primary_region)]=[]
        for region in rps_to_be_associated:
            if compute_coords_overlap_between_0_and_1(region.coords, primary_region.coords)!=0: # REALLY NOT GREAT I SHOULD TAKE THE MOST OVERLAPPING REGION
                output[(HashableRegionProps)(primary_region)].append(region)
                if no_repick:
                    detected_arena.append(region)
        if no_repick:
            rps_to_be_associated = [region for region in rps_to_be_associated if region not in detected_arena]
    return output



def associate_rps_to_rps2(primary_rps, rps_to_be_associated, no_repick=False):
    from personal.coccinelle_ladybug.HashableRegionProps import HashableRegionProps
    from personal.geom.tools import compute_coords_overlap_between_0_and_1
    output={}
    detected_arena = []
    for primary_region in primary_rps:
        # primary_region = (HashableRegionProps) primary_region
        output[ primary_region]=[]
        for region in rps_to_be_associated:
            if compute_coords_overlap_between_0_and_1(region.coords, primary_region)!=0:
                output[primary_region].append(region)
                if no_repick:
                    detected_arena.append(region)
        if no_repick:
            rps_to_be_associated = [region for region in rps_to_be_associated if region not in detected_arena]
    return output

def sort_rps_by_feature_using_its_name(rps, feature_name, feature_element=None, return_index=False):
    '''
    sorts rps by feature (area, centroid, ...) and feature index (i.e first element/axis of centroid)
    can return the sorted features or an index
    '''
    sorted_list = sorted(enumerate(rps), key=lambda p: getattr(p[1], feature_name) if feature_element is None else getattr(p[1], feature_name)[feature_element])
    if return_index:
        return sorted_list
    else:
        return [region for idx, region in sorted_list]


def get_combined_bbox(regions):
    # Initialize variables for the combined bounding box
    min_row = float('inf')
    min_col = float('inf')
    max_row = 0
    max_col = 0

    # Iterate over the regions to find the combined bounding box
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        min_row = min(min_row, minr)
        min_col = min(min_col, minc)
        max_row = max(max_row, maxr)
        max_col = max(max_col, maxc)

    # Return the combined bounding box as a tuple
    combined_bbox = (min_row, min_col, max_row, max_col)
    return combined_bbox

def get_rps_features_by_name(rps, feature_name, feature_element=None):
    """
    Returns a list of region property features with the specified name, optionally returning only a specific element
    of tuple features.

    Parameters
    ----------
    rps : list of skimage.measure._regionprops._RegionProperties
        A list of region properties objects obtained from skimage.measure.regionprops.
    feature_name : str
        The name of the feature to extract from each region properties object.
    feature_element : int or str, optional
        The index or key of the element to extract from tuple features. Defaults to None.

    Returns
    -------
    list
        A list of features with the specified name, optionally containing only the specified element of tuple features.

    Examples
    --------
    >>> # Get region properties and feature values for each region
    >>> props = get_sample_rps()
    >>> print('area_values', get_rps_features_by_name(props, 'area'))
    area_values [42, 36, 45, 4, 3, 6, 3, 43, 32, 76, 2, 2, 35, 3, 104, 2, 4, 1, 37, 4, 5, 1, 1, 6, 3, 230, 1, 2, 2, 48, 5, 6, 7, 128, 40, 2, 1, 1, 6, 7, 1, 1, 8, 7, 7, 223, 3, 2, 82, 1, 5, 55, 3, 184, 1, 2, 28, 1, 38, 1]
    >>> print('centroid_x_values', get_rps_features_by_name(props, 'centroid', 0))
    centroid_x_values [1.5714285714285714, 1.6111111111111112, 5.133333333333334, 3.5, 5.333333333333333, 7.833333333333333, 10.333333333333334, 15.093023255813954, 15.90625, 21.55263157894737, 18.5, 18.5, 23.914285714285715, 24.333333333333332, 31.375, 26.5, 32.75, 33.0, 37.0, 35.25, 36.0, 38.0, 40.0, 41.5, 48.333333333333336, 56.582608695652176, 48.0, 52.0, 57.5, 60.354166666666664, 60.2, 64.83333333333333, 64.71428571428571, 77.359375, 68.5, 67.0, 75.0, 75.0, 78.16666666666667, 82.28571428571429, 81.0, 82.0, 84.0, 85.71428571428571, 88.0, 103.24215246636771, 92.33333333333333, 92.0, 98.28048780487805, 96.0, 100.0, 105.0, 101.66666666666667, 112.59782608695652, 106.0, 109.5, 117.5, 117.0, 124.5, 123.0]
    """

    # Get feature values for each region
    features = [getattr(region, feature_name) for region in rps]

    # If feature is a tuple, extract specified element
    if feature_element is not None:
        features = [feature[feature_element] for feature in features]

    return features


def get_sample_rps():
    """
    Returns region properties for connected regions in a binary blobs image.

    Returns
    -------
    list of skimage.measure._regionprops._RegionProperties
        A list of region properties objects obtained from skimage.measure.regionprops on a binary blobs image.

    Examples
    --------
    >>> # Get region properties for connected regions in a binary blobs image
    >>> print(len(get_sample_rps()))
    60
    """

    # Load binary blobs image with 3 blobs
    image = binary_blobs(length=128, blob_size_fraction=0.1, n_dim=2, volume_fraction=0.1, seed=1)

    # Label connected regions in image
    label_image = label(image)

    # Calculate regionprops for each labeled region
    props = regionprops(label_image)
    return props

def cluster_regionprops_by_feature(rps, feature_for_clustering, feature_element=None):

    # sorted_list = sorted(enumerate(rps), key=lambda p: p[1].centroid[SORTING_AXIS])
    sorted_list = sort_rps_by_feature_using_its_name(rps, feature_for_clustering, feature_element=feature_element, return_index=True)
    # sorted_indices = [i for i, _ in sorted_list]
    objects = [object for _, object in sorted_list]

    # Sort the regionprops based on their centroids Y position
    # objects = sorted(rps, key=lambda p: p.centroid[0])
    # Sort the list and get a list of tuples containing the original index and element

    # Extract the sorted indices from the list of tuples
    # sorted_indices = [i for i, _ in sorted_list]

    # indices_comparison = [elm.index()]

    # X = np.array([[p.centroid[0]] for p in objects])
    #
    # print('X', X.shape)

    # Determine the optimal number of clusters using the silhouette score
    X = np.array(get_rps_features_by_name(objects, feature_for_clustering, feature_element=feature_element))
    if len(X.shape) ==1:
        X = X.reshape(-1, 1)

    # best_k=-1
    silhouette_scores = []
    K = range(2, len(np.unique(X))) # max nb of clusters is actually the max nb of unique elements in X



    # print('K',K)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(X) # quick fix change in KMeans
        labels = kmeans.labels_
        # if len(labels)==X.shape[0]:
        #     best_k = 1
        # else:
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)


    # if best_k==-1:
    best_k = K[np.argmax(silhouette_scores)]

    # print('best_k',best_k)

    if best_k != 1:
        # Cluster the rows based on the y-coordinates of the centroids
        kmeans = KMeans(n_clusters=best_k, random_state=0,n_init='auto').fit(X)
        labels = kmeans.labels_
    else:
        labels = [0 for _ in objects]

    # print('labels', len(labels))

    # Create a new list with the elements of `my_list` in the sorted order
    # reordered_labels = [labels[i] for i in sorted_indices]


    # maybe return labels with
    # now group every ROI in each group --> TODO
    groups_n_rps = {}
    for iii,label in enumerate(labels):
        if label in groups_n_rps:
            regions = groups_n_rps[label]
        else:
            regions = []
        regions.append(objects[iii])
        groups_n_rps[label] = regions

    return groups_n_rps



def compute_distance_between_consecutive_points(ordered_pixels):
    """
    Compute the distance between consecutive points.

    Args:
        ordered_pixels (array-like): List or array-like object containing the ordered pixels.

    Returns:
        Array of distances between consecutive points.
    """
    if isinstance(ordered_pixels, list):
        ordered_pixels = np.asarray(ordered_pixels)
    d = np.diff(ordered_pixels, axis=0)
    consecutive_distances = np.sqrt((d ** 2).sum(axis=1))
    return consecutive_distances


def is_distance_continuous(ordered_distances):
    """
    Check if a set of ordered distances is continuous.

    Args:
        ordered_distances (array-like): List or array-like object containing the ordered distances.

    Returns:
        True if the distances are continuous, otherwise returns the maximum distance and its position.
    """
    if isinstance(ordered_distances, list):
        ordered_distances = np.asarray(ordered_distances)
    if len(ordered_distances.shape) == 2:
        ordered_distances = compute_distance_between_consecutive_points(ordered_distances)
    max_dist_pos = np.argmax(ordered_distances)
    max_dist = ordered_distances[max_dist_pos]

    if max_dist <= math.sqrt(2):
        return True
    else:
        return max_dist, max_dist_pos

def find_pavlidis_optimal_orientation(pixel_coordinates, start_coords=None):
    """
    Find the optimal orientation for the Pavlidis algorithm based on the pixel coordinates.

    Args:
        pixel_coordinates (np.ndarray): Array of shape (N, 2) containing the pixel coordinates.
        start_coords (tuple): Tuple containing the start coordinates (default: None).

    Returns:
        Optimal orientation value (0, 1, 2, or 3).

    Notes:
        The optimal orientation is chosen such that when standing on the start pixel, the left adjacent pixel is white.
    #
    # Examples:
    #     >>> pixel_coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    #     >>> orientation = find_pavlidis_optimal_orientation(pixel_coords, start_coords=(1, 1))
    #     >>> print('Optimal Orientation:', orientation)
    #
    #     >>> pixel_coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    #     >>> orientation = find_pavlidis_optimal_orientation(pixel_coords)
    #     >>> print('Optimal Orientation:', orientation)
    """
    pixel_coordinates = pixel_coordinates.tolist()
    if start_coords is not None:
        y = start_coords[0]
        x = start_coords[1]
        if [y - 1, x] in pixel_coordinates:
            return 0
        if [y, x + 1] in pixel_coordinates:
            return 1
        if [y + 1, x] in pixel_coordinates:
            return 2
        if [y, x - 1] in pixel_coordinates:
            return 3

    for y, x in pixel_coordinates:
        if [y - 1, x] in pixel_coordinates:
            return 0
        if [y, x + 1] in pixel_coordinates:
            return 1
        if [y + 1, x] in pixel_coordinates:
            return 2
        if [y, x - 1] in pixel_coordinates:
            return 3
    return None


def find_pavlidis_optimal_start2(pixel_coordinates):
    """
    Find the optimal start pixel for the Pavlidis algorithm based on the pixel coordinates.

    Args:
        pixel_coordinates (np.ndarray): Array of shape (N, 2) containing the pixel coordinates.

    Returns:
        Optimal start pixel coordinates.

    Notes:
        The optimal start pixel is chosen such that when initially standing on it, the left adjacent pixel is not black.

    # Examples:
    #     >>> pixel_coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    #     >>> start_coords = find_pavlidis_optimal_start2(pixel_coords)
    #     >>> print('Optimal Start Coordinates:', start_coords)
    """
    pixel_coordinates = pixel_coordinates.tolist()

    for y, x in pixel_coordinates:
        if [y - 1, x] not in pixel_coordinates:
            return y, x
    return None


def find_pavlidis_optimal_start(pixel_coordinates):
    """
    Find the optimal start pixel and orientation for the Pavlidis algorithm based on the pixel coordinates.

    Args:
        pixel_coordinates (np.ndarray): Array of shape (N, 2) containing the pixel coordinates.

    Returns:
        Optimal start pixel coordinates and orientation.

    # Examples:
    #     >>> pixel_coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    #     >>> start_coords, orientation = find_pavlidis_optimal_start(pixel_coords)
    #     >>> print('Optimal Start Coordinates:', start_coords)
    #     >>> print('Optimal Orientation:', orientation)
    """
    px_coords = pixel_coordinates.tolist()

    for y, x in px_coords:
        for orientation, pavlidis_shift in enumerate(starting_pavlidis):
            counter_white = 0
            for shift in pavlidis_shift:
                if (y + shift[0], x + shift[1]) in px_coords:
                    counter_white += 1

            if counter_white != 0:
                print('counter_white', counter_white)
            if counter_white == 3:
                return (y, x), orientation

    print('Error: Optimal pavlidis not found. Returning None')
    return None, 0


def order_regions_by_y_axis(region1, region2):
    # Assuming region1 and region2 are instances of regionprops

    # Calculate y-axis centroids
    centroid_y1 = region1.centroid[0]
    centroid_y2 = region2.centroid[0]

    # Compare based on y-axis centroid
    if centroid_y1 < centroid_y2:
        return region1, region2
    else:
        return region2, region1


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
