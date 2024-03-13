# TODO finish clean the code manually too many comments

import os
from epyseg.settings.global_settings import set_UI # set the UI to be used py qtpy
set_UI()# TODO --> get started

import traceback
import matplotlib.pyplot as plt
from qtpy.QtWidgets import QApplication
from numba import njit, jit
from epyseg.img import Img, RGB_to_int24, int24_to_RGB
from epyseg.ta.tracking.rapid_detection_of_vertices import neighbors8
from epyseg.tools.early_stopper_class import early_stop
from epyseg.utils.loadlist import loadlist
# from personal.pyTA.tracking.rapid_detection_of_vertices import _cantor_pairing
from epyseg.ta.colors.colorgen import r_g_b_from_rgb, r_g_b_to_rgb
from epyseg.ta.GUI.multi_image_display import ImgDisplayWindow
from epyseg.ta.luts.lut_minimal_test import apply_lut, matplotlib_to_TA
from epyseg.ta.tracking.tools import get_n_files_from_list, smart_name_parser
from timeit import default_timer as timer
import numpy as np
from skimage.measure import label, regionprops
import operator

# TODO --> really test the signatures of njit just to see

USE_NUMBA = True  # ça a pas l'air de marcher ou je comprend pas --> check --> maybe check nopython in jit
FIX_SWAPS = True

@njit
def get_vx_neighbors(vertices_coords, RGB24img):
    """
    Get the neighbors of each vertex.

    Args:
        vertices_coords (numpy.ndarray): Array of vertex coordinates.
        RGB24img (numpy.ndarray): RGB image.

    Returns:
        list: List of neighbor colors for each vertex.

    """
    neighbors = []

    for iii in range(vertices_coords.shape[0]):
        i = int(vertices_coords[iii, 1])
        j = int(vertices_coords[iii, 0])

        neighbours8 = RGB24img[j - 1:j + 2, i - 1:i + 2]
        neighbs = []
        for neighbor in neighbours8.ravel():
            if neighbor != 0xFFFFFF:
                neighbs.append(neighbor)
        neighbors.append(neighbs)

    return neighbors


def find_best_matches_both_ways(img_t_cur, img_t_another, translation=None):
    """
    Find the best matches between two images in both directions.

    Args:
        img_t_cur (numpy.ndarray): Image at time t_cur.
        img_t_another (numpy.ndarray): Image at another time.
        translation (None, optional): Translation parameters. Defaults to None.

    Returns:
        dict: Best matches between colors in img_t_cur and img_t_another.

    """
    match_t_cur_in_t_prev = {}
    match_t_prev_in_t_cur = {}

    for j in range(1, img_t_cur.shape[0] - 1):
        for i in range(1, img_t_cur.shape[1] - 1):
            cur_col = img_t_cur[j, i]
            if cur_col != 0xFFFFFF:
                col_in_other_image = img_t_another[j, i]
                if col_in_other_image != 0xFFFFFF:
                    if cur_col in match_t_cur_in_t_prev.keys():
                        match_t_cur_in_t_prev[cur_col].append(col_in_other_image)
                    else:
                        match_t_cur_in_t_prev[cur_col] = [col_in_other_image]

    # Rest of the code is not provided, please complete it if needed

    return match_t_cur_in_t_prev
    # print(match_t_cur_in_t_prev)


@njit
def find_vertices(RGB24img, return_vertices=True, return_bonds=False):
    """
    Find vertices in the RGB image.

    Args:
        RGB24img (numpy.ndarray): RGB image.
        return_vertices (bool, optional): Whether to return vertices. Defaults to True.
        return_bonds (bool, optional): Whether to return bonds. Defaults to False.

    Returns:
        numpy.ndarray or tuple: Vertices or vertices and bonds.

    """
    vertices = [(0, 0)]
    vertices.clear()
    bonds = np.zeros_like(RGB24img)

    for j in range(1, RGB24img.shape[0] - 1):
        for i in range(1, RGB24img.shape[1] - 1):
            if RGB24img[j, i] == 0xFFFFFF:
                neighbours8 = RGB24img[j - 1:j + 2, i - 1:i + 2]
                neighbour_count = np.unique(neighbours8).size

                if neighbour_count >= 4:
                    vertices.append((j, i))

                if neighbour_count == 3:
                    id1 = None
                    id2 = None
                    for neighbour in neighbours8.ravel():
                        if neighbour != 0xFFFFFF:
                            if id1 is None:
                                id1 = neighbour
                            elif id2 is None and neighbour != id1:
                                id2 = neighbour
                                break

                    id = _cantor_pairing(id1, id2)  # Bug in the mapping or int32 conversion, ignore for now

                    bonds[j, i] = id

    if return_bonds:
        return bonds.astype(np.uint64)

    return np.asarray(vertices).astype(np.uint64)


@njit
def find_vertices2(RGB24img):
    """
    Find vertices in the RGB image.

    Args:
        RGB24img (numpy.ndarray): RGB image.

    Returns:
        numpy.ndarray: Vertices with white pixels marked as 255.

    """
    vertices = np.zeros_like(RGB24img)

    for j in range(1, RGB24img.shape[0] - 1):
        for i in range(1, RGB24img.shape[1] - 1):
            if RGB24img[j, i] == 0xFFFFFF:
                neighbours8 = RGB24img[j - 1:j + 2, i - 1:i + 2]
                neighbour_count = len(np.unique(neighbours8))

                if neighbour_count >= 4:
                    vertices[j, i] = 255

    return vertices


@njit
def _cantor_pairing2(id1, id2, _sort_points=True, cantor_mapping=None):
    """
    Perform Cantor pairing of two IDs.

    Args:
        id1: First ID.
        id2: Second ID.
        _sort_points (bool, optional): Whether to sort the points. Defaults to True.
        cantor_mapping (dict, optional): Mapping for Cantor IDs. Defaults to None.

    Returns:
        int: Cantor pairing result.

    """
    x = id1
    y = id2
    if _sort_points and id1 < id2:
        x = id2
        y = id1
    cantor_id = (((x + y) * (x + y + 1.) / 2.) + y)
    if cantor_mapping is not None:
        if cantor_id in cantor_mapping:
            return cantor_mapping[cantor_id]
        else:
            simple_id = len(cantor_mapping) + 1
            cantor_mapping[cantor_id] = simple_id
            return simple_id
    return cantor_id


@njit
def _cantor_pairing(id1, id2, _sort_points=True):
    """
    Perform Cantor pairing of two IDs.

    Args:
        id1: First ID.
        id2: Second ID.
        _sort_points (bool, optional): Whether to sort the points. Defaults to True.

    Returns:
        int: Cantor pairing result.

    """
    x = id1
    y = id2
    if _sort_points and id1 < id2:
        x = id2
        y = id1
    return int(((x + y) * (x + y + 1) / 2) + y)


@njit
def associate_cell_to_its_neighbors2(vertices_coords, RGB24img, bg_color=0xFFFFFF):
    """
    Associate cells to their neighbors based on vertices.

    Args:
        vertices_coords (numpy.ndarray): Coordinates of the vertices.
        RGB24img (numpy.ndarray): RGB image.
        bg_color (int, optional): Background color. Defaults to 0xFFFFFF.

    Returns:
        list: List of pairs representing the neighbors.

    """
    pairs = [(0, 0)]
    pairs.clear()

    for iii in range(vertices_coords.shape[0]):
        i = int(vertices_coords[iii, 1])
        j = int(vertices_coords[iii, 0])

        min_y = j - 1
        max_y = j + 2
        min_x = i - 1
        max_x = i + 2
        min_y = min_y if min_y >= 0 else 0
        max_y = max_y if max_y < RGB24img.shape[0] else RGB24img.shape[0]
        min_x = min_x if min_x >= 0 else 0
        max_x = max_x if max_x < RGB24img.shape[1] else RGB24img.shape[1]
        neighbours8 = RGB24img[min_y:max_y, min_x:max_x]

        neighbours82 = np.unique(neighbours8.ravel())

        for iii, id1 in enumerate(neighbours82):
            if id1 != bg_color:
                for jjj in range(iii + 1, len(neighbours82)):
                    id2 = neighbours82[jjj]
                    if id2 != bg_color:
                        pairs.append((id1, id2))
                        pairs.append((id2, id1))

    return pairs


@njit
def associate_cell_to_its_neighbors(RGB24img):
    """
    Associate cells to their neighbors.

    Args:
        RGB24img (numpy.ndarray): RGB image.

    Returns:
        list: List of pairs representing the neighbors.

    """
    pairs = [(0, 0)]
    pairs.clear()

    for j in range(1, RGB24img.shape[0] - 1):
        for i in range(1, RGB24img.shape[1] - 1):
            if RGB24img[j, i] == 0xFFFFFF:
                neighbours8 = RGB24img[j - 1:j + 2, i - 1:i + 2]
                neighbour_count = len(np.unique(neighbours8))

                if neighbour_count == 3:
                    id1 = None
                    id2 = None
                    for neighbour in neighbours8.ravel():
                        if neighbour != 0xFFFFFF:
                            if id1 is None:
                                id1 = neighbour
                            elif id2 is None and neighbour != id1:
                                id2 = neighbour
                                break

                    pairs.append((id1, id2))
                    pairs.append((id2, id1))

    return pairs

@njit
def find_bonds(RGB24img):
    bonds = np.zeros_like(RGB24img)

    for j in range(1, RGB24img.shape[0] - 1):
        for i in range(1, RGB24img.shape[1] - 1):
            if RGB24img[j, i] == 0xFFFFFF:
                neighbours8 = RGB24img[j - 1:j + 2, i - 1:i + 2]
                neighbour_count = len(np.unique(neighbours8))

                if neighbour_count == 3:
                    id1 = None
                    id2 = None
                    for neighbour in neighbours8.ravel():
                        if neighbour != 0xFFFFFF:
                            if id1 is None:
                                id1 = neighbour
                            elif id2 is None and neighbour != id1:
                                id2 = neighbour
                                break

                    id = _cantor_pairing(id1, id2)
                    bonds[j, i] = id

    return bonds


def find_neighbors_to_check(cell_of_interest_in_cur, cells_and_their_neighbors_t_cur,
                  cells_and_their_neighbors_t_other):
    """
    Finds the neighbors to check for potential swaps or assignments.

    Args:
        cell_of_interest_in_cur: The cell of interest in the current time frame.
        cells_and_their_neighbors_t_cur (dict): Dictionary mapping cells to their neighbors in the current time frame.
        cells_and_their_neighbors_t_other (dict): Dictionary mapping cells to their neighbors in another time frame.

    Returns:
        tuple: A tuple containing two lists - possible_swaps and possible_assignments.

    """
    neighbs_to_check = []
    neighbs_of_cell_of_interest_in_t_cur = cells_and_their_neighbors_t_cur[cell_of_interest_in_cur]

    for neighb_of_cell_of_interest_in_t_cur in neighbs_of_cell_of_interest_in_t_cur:
        if neighb_of_cell_of_interest_in_t_cur in cells_and_their_neighbors_t_other.keys():
            neighbours_of_neighbs_in_t_other = cells_and_their_neighbors_t_other[neighb_of_cell_of_interest_in_t_cur]
            for neighbour_of_neighbs_in_t_other in neighbours_of_neighbs_in_t_other:
                neighbs_to_check.append(neighbour_of_neighbs_in_t_other)

    neighbs_to_check.extend(neighbs_of_cell_of_interest_in_t_cur)
    neighbs_to_check = set(neighbs_to_check)
    if cell_of_interest_in_cur in neighbs_to_check:
        neighbs_to_check.remove(cell_of_interest_in_cur)
    neighbs_to_check = list(neighbs_to_check)

    possible_swaps = [id for id in neighbs_to_check if id in cells_and_their_neighbors_t_cur.keys()]
    possible_assignments = [id for id in neighbs_to_check if id not in cells_and_their_neighbors_t_cur.keys()]

    return possible_swaps, possible_assignments


@jit(nopython=True)
def find_first(item, vec):
    """
    Finds the index of the first occurrence of an item in a vector.

    Args:
        item: The item to find.
        vec: The vector to search.

    Returns:
        int: The index of the first occurrence of the item, or -1 if not found.

    """
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1

def plot_n_images_in_line(*args, title=None):
    """
    Plots multiple images in a single line.

    Args:
        *args: Variable length argument list of images to plot.
        title (str): Optional title for the plot.

    """
    nb_images = len(args)
    f, axarr = plt.subplots(ncols=nb_images, sharex=True, sharey=True)

    for iii, img in enumerate(args):
        axarr[iii].imshow(img)

    f.subplots_adjust(0, 0, 1, 1)
    plt.tight_layout()

    full_screen = True
    if full_screen:
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

    hide_white_space = True
    if hide_white_space:
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    if title is not None:
        plt.title(title)

    plt.show()


def associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_cur):
    """
    Associates cells to their neighbor IDs in a dictionary.

    Args:
        cells_and_their_neighbors_cur (dict): Dictionary mapping cells to their neighbor IDs.

    Returns:
        dict: Dictionary mapping cells to lists of their neighbor IDs.

    """
    cells_and_their_neighbs = {}

    for id, neighb in cells_and_their_neighbors_cur:
        if id in cells_and_their_neighbs:
            cells_and_their_neighbs[id].append(neighb)
        else:
            cells_and_their_neighbs[id] = [neighb]

    return cells_and_their_neighbs


overseg = 'overseg'
underseg = 'underseg'
division = 'division'
death = 'death'
gone = 'gone'  # can be out of focus and in that case should really not be labelled as death # same as death but unlabeled as a solution --> is that useful or not (can be a delaminating cell, practically it's the same as death isn't it --> do I need to complicate things ??? --> probably not)
new = 'new'
swapped = 'swapped'
lost_track = 'lost_track'
ignore = 'ignore'  # cell is perfect no action required
# NB DIVISION COULD BE TRANSFORMED INTO INTERCALATING IF A DIVISION PARTNER CANNOT BE FOUND OR IF AREA IS VERY LOW INITIALLY FOR THE NEW CELL --> IN FACT I HAVE NEW SO I DON'T NEED INTERCALATING --> BECAUSE IT'S REDUNDANT

fates = [overseg, underseg, division, death, new, swapped, lost_track,
         ignore]  # can there be other fates # there must be rules depending on the stuff


def take_decision(cell_of_interest_in_cur, cells_and_their_neighbors_minus_1, cells_and_their_neighbors_cur,
                  cells_and_their_neighbors_plus_1, cells_present_in_t_cur_but_absent_in_t_minus_1_score,
                  cells_present_in_t_cur_but_absent_in_t_plus_1_score, sensitivity=0.75,
                  assume_previous_image_is_GT=True):
    """
    Takes a decision about a cell based on its presence in different frames and neighbor information.

    Args:
        cell_of_interest_in_cur: Cell of interest in the current frame.
        cells_and_their_neighbors_minus_1: Dictionary mapping cells to their neighbors in the previous frame.
        cells_and_their_neighbors_cur: Dictionary mapping cells to their neighbors in the current frame.
        cells_and_their_neighbors_plus_1: Dictionary mapping cells to their neighbors in the next frame.
        cells_present_in_t_cur_but_absent_in_t_minus_1_score: Scores indicating the presence of cells in the current
            frame but absent in the previous frame.
        cells_present_in_t_cur_but_absent_in_t_plus_1_score: Scores indicating the presence of cells in the current
            frame but absent in the next frame.
        sensitivity (float): Sensitivity threshold for considering neighbor changes significant. Default is 0.75.
        assume_previous_image_is_GT (bool): Flag indicating whether to assume the previous image as the ground truth.
            Default is True.

    Returns:
        list: List of possible decisions for the cell.

    """
    # TODO add lost_track in some cases

    # case 1: cell is missing in the previous frame
    if cell_of_interest_in_cur not in cells_and_their_neighbors_minus_1.keys():
        if assume_previous_image_is_GT:
            # check if missing cell is in the next frame
            if cell_of_interest_in_cur in cells_and_their_neighbors_plus_1.keys():
                # assume it is a division
                return ['division', 'new']
            else:
                return ['overseg']
        else:
            possibilities = ['underseg-1', 'division', 'overseg', 'new']
            if cell_of_interest_in_cur not in cells_and_their_neighbors_plus_1.keys():
                possibilities.remove('division')
                possibilities.remove('new')
            return possibilities

    # case 2: cell is missing in the next frame
    if cell_of_interest_in_cur not in cells_and_their_neighbors_plus_1.keys():
        if cell_of_interest_in_cur in cells_and_their_neighbors_minus_1.keys():
            # cell was present before, likely a dying cell or an overseg
            return ['death', 'gone', 'underseg+1']
        else:
            return ['overseg']

    # cell is present in all frames, but its score is suboptimal
    # check if score is 1 in both previous and next frames, ignore the cell
    if cells_present_in_t_cur_but_absent_in_t_minus_1_score[cell_of_interest_in_cur] == 1 and \
            cells_present_in_t_cur_but_absent_in_t_plus_1_score[cell_of_interest_in_cur] == 1:
        return ['ignore']

    if sensitivity is not None:
        if cells_present_in_t_cur_but_absent_in_t_minus_1_score[cell_of_interest_in_cur] >= sensitivity and \
                cells_present_in_t_cur_but_absent_in_t_plus_1_score[cell_of_interest_in_cur] >= sensitivity:
            # cell most likely has changed one or two neighbors, no reason to worry
            return ['ignore']

    # now some of those cells may be swapped, fix them

    # cell score is below the threshold, needs attention
    # if score has significantly decreased compared to the previous frame, may be an underseg or swapping
    # check for missing or gained cells in the previous and next frames to determine if it's overseg or not
    # not easy to determine, center on the problematic cells

    # case 3: missing cell in the current frame, absent in the previous frame (which is GT)
    # either a division or an overseg
    # if the cell is also absent in the next frame, it's for sure an overseg
    # offer the possibility to discard
    # offer several possibilities according to rules

    pass


def get_cells_in_image_n_fisrt_pixel(RGB24_img, return_flat_idx=False):
    """
    Retrieves the cells in the image and their first pixel indices.

    Args:
        RGB24_img (ndarray): RGB image.
        return_flat_idx (bool): Flag indicating whether to return the flat indices or 2D indices. Default is False.

    Returns:
        tuple: A tuple containing unique cell values and their corresponding indices.

    """
    u, indices = np.unique(RGB24_img, return_index=True)

    if not return_flat_idx:
        indices = np.unravel_index(indices, RGB24_img.shape)

    return u, indices


def apply_color_to_labels(lab_t_cur, map_tracks_n_label_t_cur):
    """
    Applies color mapping to the label image.

    Args:
        lab_t_cur (ndarray): Label image.
        map_tracks_n_label_t_cur (dict): Mapping of tracks to label values.

    Returns:
        ndarray: Updated label image with applied color mapping.

    """
    tracked_cells = np.zeros_like(lab_t_cur)

    for k, v in map_tracks_n_label_t_cur:
        tracked_cells[lab_t_cur == v] = k

    return tracked_cells

def map_track_id_to_label(first_pixels_t_cur, track_t_cur, lab_t_cur):
    """
    Maps the track IDs to their corresponding labels based on the first pixels of each cell.

    Args:
        first_pixels_t_cur (tuple): Tuple of arrays representing the first pixels of each cell.
        track_t_cur (ndarray): Array representing the tracks of cells.
        lab_t_cur (ndarray): Array representing the labels of cells.

    Returns:
        ndarray: Array containing the mapping of track IDs to labels.
    """
    map = []
    for j, i in zip(*first_pixels_t_cur):
        map.append((track_t_cur[j, i], lab_t_cur[j, i]))
    return np.asarray(map)


def compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_t_other,
                           cells_present_in_t_cur_but_absent_in_t_other=None,
                           cells_present_in_t_cur_but_absent_in_t_other_score=None, score_plotting_image=None,
                           track_t_cur=None):
    """
    Computes the neighbor score between cells in the current frame and cells in another frame.

    Args:
        cells_and_their_neighbors_cur (dict): Dictionary containing the cells in the current frame and their neighbors.
        cells_and_their_neighbors_t_other (dict): Dictionary containing the cells in another frame and their neighbors.
        cells_present_in_t_cur_but_absent_in_t_other (list, optional): List to store cells present in the current frame but absent in the other frame. Defaults to None.
        cells_present_in_t_cur_but_absent_in_t_other_score (dict, optional): Dictionary to store the neighbor scores of cells present in the current frame but absent in the other frame. Defaults to None.
        score_plotting_image (ndarray, optional): Array to plot the computed scores. Defaults to None.
        track_t_cur (ndarray, optional): Array representing the tracks of cells in the current frame. Defaults to None.

    Returns:
        float: Sum of the neighbor scores.
    """
    sum_score = 0
    for cell, neigbs in cells_and_their_neighbors_cur.items():
        if cell in cells_and_their_neighbors_t_other.keys():
            neigbsb = cells_and_their_neighbors_t_other[cell]
            score = pairwise_score(neigbs, neigbsb)
            sum_score += score
            if cells_present_in_t_cur_but_absent_in_t_other_score is not None:
                cells_present_in_t_cur_but_absent_in_t_other_score[cell] = score
            if score_plotting_image is not None and track_t_cur is not None:
                score_plotting_image[track_t_cur == cell] = score
        else:
            if cells_present_in_t_cur_but_absent_in_t_other is not None:
                cells_present_in_t_cur_but_absent_in_t_other.append(cell)
            if cells_present_in_t_cur_but_absent_in_t_other_score is not None:
                cells_present_in_t_cur_but_absent_in_t_other_score[cell] = 0
    return sum_score


def pairwise_score(neigbs, neigbsb):
    """
    Computes the pairwise score between two sets of neighbors.

    Args:
        neigbs (list): List of neighbors for the first set of cells.
        neigbsb (list): List of neighbors for the second set of cells.

    Returns:
        float: Pairwise score between the two sets of neighbors.
    """
    score = (len(set(neigbs) & set(neigbsb)) * 2) / (len(neigbs) + len(neigbsb))
    return score


def get_cells_and_their_neighbors_from_image(img, vertices=None, bg_color=0xFFFFFF):
    """
    Retrieves the cells and their neighbors from an image.

    Args:
        img (ndarray): Array representing the image.
        vertices (ndarray, optional): Array containing the vertices of the cells. Defaults to None.
        bg_color (int, optional): Background color of the image. Defaults to 0xFFFFFF.

    Returns:
        dict: Dictionary containing the cells and their neighbors.
    """
    if len(img.shape) == 3:
        tmp = RGB_to_int24(img)
    else:
        tmp = img

    if vertices is None:
        vertices = np.stack(np.where(tmp == bg_color), axis=1)

    cells_and_their_neighbors = np.unique(
        np.asarray(associate_cell_to_its_neighbors2(vertices, tmp, bg_color=bg_color)), axis=0)
    cells_and_their_neighbors = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors)
    return cells_and_their_neighbors


def detect_divisions(cells_and_their_neighbors_cur, cells_and_their_neighbors_t_minus_1,
                     cells_and_their_neighbors_t_plus_1=None):
    """
    Detects cell divisions based on the cells and their neighbors in different frames.

    Args:
        cells_and_their_neighbors_cur (dict or ndarray): Dictionary or array containing the cells and their neighbors in the current frame.
        cells_and_their_neighbors_t_minus_1 (dict or ndarray): Dictionary or array containing the cells and their neighbors in the previous frame.
        cells_and_their_neighbors_t_plus_1 (dict or ndarray, optional): Dictionary or array containing the cells and their neighbors in the next frame. Defaults to None.

    Returns:
        dict: Dictionary containing the dividing pairs of cells.
    """
    dividing_pairs = {}

    if isinstance(cells_and_their_neighbors_cur, np.ndarray):
        cells_and_their_neighbors_cur = get_cells_and_their_neighbors_from_image(cells_and_their_neighbors_cur)
    if isinstance(cells_and_their_neighbors_t_minus_1, np.ndarray):
        cells_and_their_neighbors_t_minus_1 = get_cells_and_their_neighbors_from_image(
            cells_and_their_neighbors_t_minus_1)
    if isinstance(cells_and_their_neighbors_t_plus_1, np.ndarray):
        cells_and_their_neighbors_t_plus_1 = get_cells_and_their_neighbors_from_image(
            cells_and_their_neighbors_t_plus_1)

    new_cells_in_t_cur = set(cells_and_their_neighbors_cur.keys()) - set(cells_and_their_neighbors_t_minus_1.keys())
    if cells_and_their_neighbors_t_plus_1 is not None:
        new_cells_in_t_cur = new_cells_in_t_cur & set(cells_and_their_neighbors_t_plus_1.keys())

    for cell in new_cells_in_t_cur:
        max_score = 0
        neighbs_current_cell = cells_and_their_neighbors_cur[cell]
        for possible_sister in neighbs_current_cell:
            if possible_sister not in cells_and_their_neighbors_t_minus_1.keys():
                continue
            possible_sister_neighbors = cells_and_their_neighbors_t_minus_1[possible_sister]
            merged_set = list(possible_sister_neighbors)
            merged_set.extend(neighbs_current_cell)
            merged_set = list(set(merged_set))
            neighbs_in_previous_image = cells_and_their_neighbors_t_minus_1[possible_sister]
            score = pairwise_score(neighbs_in_previous_image, merged_set)
            if score > max_score:
                max_score = score
                dividing_pairs[cell] = possible_sister

    return dividing_pairs

def get_border_cells_plus_one(cells_n_their_neighbors, border_cells, forbidden_ids=[0xFFFFFF], remove_border_cells=False):
    """
    Retrieves the border cells plus one layer of neighbors.

    Args:
        cells_n_their_neighbors (dict): Dictionary containing the cells and their neighbors.
        border_cells (list): List of border cells.
        forbidden_ids (list, optional): List of forbidden cell IDs. Defaults to [0xFFFFFF].
        remove_border_cells (bool, optional): Whether to remove border cells from the result. Defaults to False.

    Returns:
        list: List of border cells plus one layer of neighbors.
    """
    border_cells_plus_one = []
    for border_cell in border_cells:
        try:
            border_cells_plus_one.extend(cells_n_their_neighbors[border_cell])
        except:
            pass
    if forbidden_ids is not None:
        border_cells_plus_one = list(set(border_cells_plus_one)-set(forbidden_ids))
    else:
        border_cells_plus_one = list(set(border_cells_plus_one))
    if remove_border_cells:
        border_cells_plus_one = list(set(border_cells_plus_one)-set(border_cells))
    return border_cells_plus_one


def plot_anything_a_la_TA(cell_id, list_of_cells_to_plot, color=0xFF0000, keep_mask=True):
    """
    Generates an image where specific cells are colored.

    Args:
        cell_id (ndarray): Array representing the cell IDs.
        list_of_cells_to_plot (list): List of cells to be colored.
        color (int, optional): Color to be used for coloring the cells. Defaults to 0xFF0000.
        keep_mask (bool, optional): Whether to keep the mask (cell ID 0xFFFFFF) in the generated image. Defaults to True.

    Returns:
        ndarray: Array representing the generated image.
    """
    new_img = np.zeros_like(cell_id)
    for cell in list_of_cells_to_plot:
        new_img[cell_id == cell] = color
    if keep_mask:
        new_img[cell_id == 0xFFFFFF] = 0xFFFFFF
    return new_img


@njit
def get_border_cells(img, bg_color=None):
    """
    Retrieves the border cells from an image.

    Args:
        img (ndarray): Array representing the image.
        bg_color (int, optional): Background color of the image. Defaults to None.

    Returns:
        list: List of border cells.
    """
    border_cells = []
    for j in range(img.shape[0]):
        border_cells.append(img[j, 0])
        border_cells.append(img[j, 1])
        border_cells.append(img[j, img.shape[1]-1])
        border_cells.append(img[j, img.shape[1]-2])

    for i in range(img.shape[1]):
        border_cells.append(img[0, i])
        border_cells.append(img[1, i])
        border_cells.append(img[img.shape[0]-1, i])
        border_cells.append(img[img.shape[0]-2, i])

    border_cells = list(set(border_cells))
    if bg_color is not None:
        if bg_color in border_cells:
            border_cells.remove(bg_color)
    return border_cells


def plot_dividing_cells_a_la_TA(img, dividing_pairs, plot_cell_outline=True,
                                exclude_cells_bigger_than_percent_of_image=20,
                                exclude_cells_with_size_difference_superior=3,
                                remove_divisions_involving_a_border_cell=True):
    """
    Generates an image highlighting cell divisions based on the provided dividing pairs.

    Args:
        img (ndarray): Array representing the image.
        dividing_pairs (dict): Dictionary containing the dividing pairs of cells.
        plot_cell_outline (bool, optional): Whether to plot the outline of cells. Defaults to True.
        exclude_cells_bigger_than_percent_of_image (int, optional): Percentage of the image area above which cells are excluded. Defaults to 20.
        exclude_cells_with_size_difference_superior (int, optional): Size difference ratio above which cells are excluded. Defaults to 3.
        remove_divisions_involving_a_border_cell (bool, optional): Whether to remove divisions involving a border cell. Defaults to True.

    Returns:
        ndarray: Array representing the image with highlighted cell divisions.
    """
    if len(img.shape) == 3:
        img = RGB_to_int24(img)

    cell_divisions = np.zeros_like(img)
    img_area = cell_divisions.shape[0] * cell_divisions.shape[1]

    border_cells = None
    if remove_divisions_involving_a_border_cell:
        border_cells = get_border_cells(img)

    div_counter = 0
    for sister1, sister2 in dividing_pairs.items():

        if remove_divisions_involving_a_border_cell:
            if sister1 in border_cells or sister2 in border_cells:
                continue

        if exclude_cells_bigger_than_percent_of_image is not None and exclude_cells_with_size_difference_superior > 0:
            sister_1_size = cell_divisions[img == sister1].size
            sister_2_size = cell_divisions[img == sister2].size
            if (sister_1_size + sister_2_size) / img_area >= exclude_cells_with_size_difference_superior / 100.:
                continue
        if exclude_cells_with_size_difference_superior is not None and exclude_cells_with_size_difference_superior != 0:
            size_ratio = cell_divisions[img == sister1].size / cell_divisions[img == sister2].size

            if size_ratio < 1:
                size_ratio = 1. / size_ratio

            if size_ratio > exclude_cells_with_size_difference_superior:
                continue

        cell_divisions[img == sister1] = div_counter + 1
        cell_divisions[img == sister2] = div_counter + 1

        div_counter += 1

    if plot_cell_outline:
        cell_divisions[img == 0xFFFFFF] = 0xFFFFFF

    return cell_divisions

def optimize_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_t_other, current_cells_n_score,
                   map_tracks_n_label_t_cur, threshold=0.5):
    # if shift is validated I need change the mappings --> either swap cells or swap their values --> the pb is that this will change the relationship to the rps --> need be smart but ok
    # convert map_tracks_n_label_t_cur to a dict then convert it back to array after all changes are applied

    tmp_map_tracks_n_label_t_cur = {k: v for k, v in map_tracks_n_label_t_cur}

    # print('tmp_map_tracks_n_label_t_cur', tmp_map_tracks_n_label_t_cur)  # --> ok mais ensuite perd tt --> why???
    # print(len(tmp_map_tracks_n_label_t_cur))

    # sort current_cells_n_score values and keys from min to max until a threshold try to optimize score
    # find if any other pair improves score !!! and if so do the swap and recompute the score
    # probably need update all in the lists too --> TODO
    # or do not do recursively --> check that later --> first try find and see if score improves
    possible_swapping_pairs = {}
    possible_assignment = {}

    # nb maybe compute the score here --> no need to do it outside then --> simpler
    scores_sorted_ascending = dict(sorted(current_cells_n_score.items(), key=operator.itemgetter(1)))
    low_scores_to_check = [cell for cell in scores_sorted_ascending.keys() if
                           scores_sorted_ascending[cell] <= threshold]

    # print('scores_sorted_descending', scores_sorted_ascending)
    # print('scores_sorted_descending', [r_g_b_from_rgb(cell) for cell in scores_sorted_ascending.keys()])
    # # if r_g_b_to_rgb(252, 78, 252) in scores_sorted_ascending.keys():
    # #     print('score ', scores_sorted_ascending[r_g_b_to_rgb(252, 78, 252)], r_g_b_to_rgb(252, 78, 252))
    # # else:
    # #     print('score not in')
    # print('low_scores_to_check', low_scores_to_check)
    # print('in stuff to chekc', r_g_b_to_rgb(156, 248, 184) in low_scores_to_check,
    #       r_g_b_to_rgb(156, 248, 184))  # false --> this cell has a score of 1.0 --> how is it possible ???
    # print('in stuff to chekc', r_g_b_to_rgb(66, 162, 164) in low_scores_to_check,
    #       r_g_b_to_rgb(66, 162, 164))  # true --> score 0.6 --> ok
    # print('in stuff to chekc2', r_g_b_to_rgb(137, 134, 15) in low_scores_to_check,
    #       r_g_b_to_rgb(137, 134, 15))  # true --> score 0.6 --> ok

    # try:
    #     print(r_g_b_to_rgb(156, 248, 184) in cells_and_their_neighbors_cur,
    #           r_g_b_to_rgb(156, 248, 184) in cells_and_their_neighbors_t_other)
    #     print('manually calculated score', pairwise_score(cells_and_their_neighbors_cur[r_g_b_to_rgb(156, 248, 184)],
    #                                                       cells_and_their_neighbors_t_other[
    #                                                           r_g_b_to_rgb(156, 248, 184)]),
    #           cells_and_their_neighbors_cur[r_g_b_to_rgb(156, 248, 184)],
    #           cells_and_their_neighbors_t_other[r_g_b_to_rgb(156, 248, 184)])
    #     print('manually calculated score2', pairwise_score(cells_and_their_neighbors_cur[r_g_b_to_rgb(66, 162, 164)],
    #                                                        cells_and_their_neighbors_t_other[
    #                                                            r_g_b_to_rgb(156, 248, 184)]),
    #           cells_and_their_neighbors_cur[r_g_b_to_rgb(66, 162, 164)],
    #           cells_and_their_neighbors_t_other[r_g_b_to_rgb(156, 248, 184)])
    #     print('manually calculated score3', pairwise_score(cells_and_their_neighbors_cur[r_g_b_to_rgb(156, 248, 184)],
    #                                                        cells_and_their_neighbors_t_other[
    #                                                            r_g_b_to_rgb(66, 162, 164)]),
    #           cells_and_their_neighbors_cur[r_g_b_to_rgb(156, 248, 184)],
    #           cells_and_their_neighbors_t_other[r_g_b_to_rgb(66, 162, 164)])
    # except:
    #     print('missing cell')
    #     pass

    # low_scores_to_check[980445, 4479535, 9829855, 10240303, 10369245, 10652374, 11560212, 16222777, 16404369] --> a l'air de marcher
    # loop those over the entire list

    # possible swap 16703757 980445
    # possible swap 16703757 4479535
    # possible swap 16703757 9829855
    # possible swap None 10240303
    # possible swap None 10369245
    # possible swap 16703757 10652374
    # possible swap None 11560212
    # possible swap 16703757 16222777
    # possible swap 16703757 16404369

    # shall I store all possible swaps and then apply them in the end
    # to improve the glocal score should improve

    # it is in the stuff to check but never makes it to the end --> WHY

    # possible assignment peut aussi detecter les paires en division--> très bonne idée

    #  TODO also check all possible swaps between pairs of cells with low score non 0 that would optimize global score
    # pb is score if they share neighbs --> individual score not better because need take into account all
    # maybe do brute force scoring for low cells once just to see

    # brute force swap --> create all swapping pairs non zero and check them

    for cell in low_scores_to_check:
        current_score = current_cells_n_score[cell]
        # print(current_score)
        # TODO en fait c'est impossible car le score est tjrs inferieur à 1 sauf si l'utilisateur veut vraiment mettre 1 --> remove that or keep it to force analyze the whole image ??? --> I guess remove it
        if current_score >= 1.0:
            continue
        final_cell_to_swap = None

        # stuff below is a bit less good with best score... --> seems not a good idea but check that I have no bug some day
        # best_score = 0.

        if cell in cells_and_their_neighbors_t_other.keys():
            neigbsb = cells_and_their_neighbors_t_other[cell]

            # PB IS THAT I SHOULD ASSUME SCORE WITH ASSUMING BETTER SWAP
            # look in neighborhood if there is a better match
            # nb I could add its own neighbors by the way --> no need to only take those of the neighbor in prev --> need to check if it is a swap or an
            for cell_to_swap_with, neigbs in cells_and_their_neighbors_cur.items():
                if cell_to_swap_with == cell:
                    continue

                # (len(list(set(neigbs) & set(neigbsb))) * 2) / (len(neigbsb) + len(neigbs))
                score = pairwise_score(neigbs, neigbsb)

                # if cell_to_swap_with == 4367012 and cell == 10287288:
                #     print('in da sgit', score, neigbs, neigbsb, current_score)

                if score >= current_score:
                    current_score = score
                    final_cell_to_swap = cell_to_swap_with
                # if score >= best_score:
                #     best_score = score
                #     final_cell_to_swap = cell_to_swap_with
            if final_cell_to_swap is not None:
            # if final_cell_to_swap is not None and best_score>0.:
            #     print('possible swap', cell, final_cell_to_swap, current_score, r_g_b_from_rgb(cell),                      r_g_b_from_rgb(final_cell_to_swap))  # indeed a perfect swap
                possible_swapping_pairs[cell] = final_cell_to_swap
        else:
            # same code in fact -->
            # la cellule n'est pas dans other --> verif si le swap est  dans celui là sinon assigner le fate sinon creer une nouvelle id unique qui n'est ni dans l'un ni dans l'autre
            # possible_assignment
            # find any cell id that would share the max of the neighbors --> loop over all or loop over all that have low score and look for better

            # probably only need loop over low scores --> most of the cells will not have that then!!!
            # IN FCAT ONLY NEED BE DONE FOR NEW CELLS --> HUGE GAIN OF TIME!!!!
            # for cellb, neigbsb in cells_and_their_neighbors_t_other.items():
            # neigbsb = cells_and_their_neighbors_t_other[cell]
            neigbsb = cells_and_their_neighbors_cur[cell]

            for cell_to_swap_with, neigbs in cells_and_their_neighbors_t_other.items():
                if cell_to_swap_with == cell:
                    continue
                score = pairwise_score(neigbs,
                                       neigbsb)  # (len(list(set(neigbs) & set(neigbsb))) * 2) / (len(neigbsb) + len(neigbs))
                if score > current_score:
                    current_score = score
                    final_cell_to_swap = cell_to_swap_with
                # if score > best_score:
                #     best_score = score
                #     final_cell_to_swap = cell_to_swap_with
            if final_cell_to_swap is not None:
            # if final_cell_to_swap is not None and best_score > 0.:
            #     print('possible assignment', cell, final_cell_to_swap, current_score, r_g_b_from_rgb(cell),                      r_g_b_from_rgb(final_cell_to_swap), current_score)  # indeed a perfect swap
                possible_assignment[cell] = final_cell_to_swap
            # pass

    # check if any of the lost cells in current would give a better score than a cell in the current because if that is the case --> offer assignment
    missing_cells_from_prev = set(cells_and_their_neighbors_t_other.keys()) - set(cells_and_their_neighbors_cur.keys())

    # print('in stuff to chekc2b', r_g_b_to_rgb(137, 134, 15) in missing_cells_from_prev,          r_g_b_to_rgb(137, 134, 15), r_g_b_to_rgb(44, 140, 93))  # true --> score 0.6 --> ok
    if True:
        for cell in missing_cells_from_prev:
            best_score = 0
            # comme c'est du brute force ça rallonge pas mal le processus, pr finalement très peu de cellules --> is that really worth it ???
            final_cell_to_swap = None
            neigbsb = cells_and_their_neighbors_t_other[cell]
            for cell_to_swap_with, neigbs in cells_and_their_neighbors_cur.items():
                current_score = current_cells_n_score[cell_to_swap_with]
                if current_score >= 1.:
                    continue

                score = pairwise_score(neigbs,
                                       neigbsb)  # (len(list(set(neigbs) & set(neigbsb))) * 2) / (len(neigbsb) + len(neigbs))

                if score >= current_score:
                    # current_score = score
                    if score >= best_score:
                        best_score = score
                        final_cell_to_swap = cell_to_swap_with
            if final_cell_to_swap is not None:
                # print('possible assignment2', cell, final_cell_to_swap, current_score, r_g_b_from_rgb(cell),                      r_g_b_from_rgb(final_cell_to_swap), current_score)  # indeed a perfect swap
                possible_assignment[final_cell_to_swap] = cell
    # if any of these cells gives a better score than the current offer a swap

    # this cell never enters here although there has to be an equal cell maybe
    # print('inside swaps and so on', r_g_b_to_rgb(156, 248, 184) in possible_assignment.keys(),  r_g_b_to_rgb(156, 248, 184) in possible_assignment.values(),  r_g_b_to_rgb(156, 248, 184) in possible_swapping_pairs.keys(), r_g_b_to_rgb(156, 248, 184) in possible_swapping_pairs.values())

    # the two cells never enter in this !!!
    # print('inside swaps and so on', r_g_b_to_rgb(156, 248, 184) in possible_assignment.keys(),
    #       r_g_b_to_rgb(156, 248, 184) in possible_assignment.values(),
    #       r_g_b_to_rgb(156, 248, 184) in possible_swapping_pairs.keys(),
    #       r_g_b_to_rgb(156, 248, 184) in possible_swapping_pairs.values())
    # print('inside swaps and so on', r_g_b_to_rgb(66, 162, 164) in possible_assignment.keys(),
    #       r_g_b_to_rgb(66, 162, 164) in possible_assignment.values(),
    #       r_g_b_to_rgb(66, 162, 164) in possible_swapping_pairs.keys(),
    #       r_g_b_to_rgb(66, 162, 164) in possible_swapping_pairs.values())

    # maybe store al the possible swaps and apply them and try
    # low scores could even contain new cells
    # score TODO finalize that
    # print('total score after optimization', sum(cells_present_in_t_cur_but_absent_in_t_minus_1_score.values()))

    # get the dict and copy it

    import copy
    cp = copy.deepcopy(cells_and_their_neighbors_cur)
    initial_score = sum(current_cells_n_score.values())
    # new_score= initial_score

    print('initila score', initial_score)

    final_validated_changes = {}

    # I need treat the swapping and assignment differently!!!

    # in fact if the cell is present in the image --> real swapping and can be added otherwise it's an assignment --> do things differently
    # need further filter

    # FILTER AGAIN assignments
    # AN ASSIGNMENT SHOULD HAVE ONLY ONE CELL IN THE IMAGE

    # filtered_assignments = {k : v  for (k,v) in possible_assignment.items()}
    # keys to remove:
    keys_to_remove_in_assignment = []
    print('possible_assignment', possible_assignment)
    for k, v in possible_assignment.items():
        if v in cells_and_their_neighbors_cur.keys():
            # del possible_assignment[k]
            keys_to_remove_in_assignment.append(k)

    for k in keys_to_remove_in_assignment:
        del possible_assignment[k]

    print('filtered_assignments',
          possible_assignment)  # --> good really kept the right one, but keep in mind that the others are actually missegmented cells in current --> those are very useful and I have positional info there --> maybe keep those

    for cell, val in possible_assignment.items():
        # just change the fate of the key and replace it in itself and in neighboring cells!!!
        # much less operations than in the other --> TODO
        neighbs_cell = cells_and_their_neighbors_cur[cell]
        for neigb in neighbs_cell:
            neighbs_of_neighb = cells_and_their_neighbors_cur[neigb]
            neighbs_of_neighb = [id if id != cell else val for id in neighbs_of_neighb]
            cells_and_their_neighbors_cur[neigb] = neighbs_of_neighb
            # print('cell in neighbs_of_neighb', cell in neighbs_of_neighb)  # FAlse --> ok indeed
            # print('val in neighbs_of_neighb',val in neighbs_of_neighb)  # True --> all is ok in fact

        # create a new cell then remove it
        cells_and_their_neighbors_cur[val] = cells_and_their_neighbors_cur[cell]
        del cells_and_their_neighbors_cur[cell]
        new_score = compute_neighbor_score(cells_and_their_neighbors_cur=cells_and_their_neighbors_cur,
                                           cells_and_their_neighbors_t_other=cells_and_their_neighbors_t_other)
        print('new_score, initial_score', new_score, initial_score)

        if new_score <= initial_score:
            # do not change if score is lower or equal --> TODO
            #  then restore change
            # cells_and_their_neighbors_cur[cell], cells_and_their_neighbors_cur[val] = cells_and_their_neighbors_cur[val], cells_and_their_neighbors_cur[cell]
            # reswap and restore
            # cells_and_their_neighbors_cur[cell]=bckup_neighbs_cell
            # cells_and_their_neighbors_cur[val]=bckup_neighbs_val
            cells_and_their_neighbors_cur = copy.deepcopy(cp)
            # else validate the list and continue
            # print('rejected assignment ', cell, val, r_g_b_from_rgb(cell), r_g_b_from_rgb(val), new_score,                  initial_score)


        else:
            # validate the change and update the score
            initial_score = new_score
            final_validated_changes[cell] = val
            cp = copy.deepcopy(cells_and_their_neighbors_cur)

            # do I have an inversion ??? --> maybe yes ????
            # prevent multi assignments ??? or update the image in fact --> that is necessary

            # print('accepted assignment', cell, val, r_g_b_from_rgb(cell), r_g_b_from_rgb(val), new_score, initial_score)
            # print(cell in cells_and_their_neighbors_cur.keys()) # FAlse --> ok indeed
            # print(val in cells_and_their_neighbors_cur.keys()) # True --> all is ok in fact

            # update correspondance between cell and its mask
            print('before mapping0', cell, val, tmp_map_tracks_n_label_t_cur[cell])
            tmp_map_tracks_n_label_t_cur[val] = tmp_map_tracks_n_label_t_cur[cell]
            del tmp_map_tracks_n_label_t_cur[cell]
            print('after mapping0', cell, val, tmp_map_tracks_n_label_t_cur[val])

    cp = copy.deepcopy(cells_and_their_neighbors_cur)
    # TODO by construction it does it both ways and I don't want that because it's a waste of time --> need fix that and change that --> if already validated prevent reswap
    # en fait c'est pas des swaps c'est des connexions de tracks
    for key, val in possible_swapping_pairs.items():
        # I need get the score after swap vs before for the cell and all of its neighbors recusrsively and only to that if all improves !!!
        for cell, neighbs in cells_and_their_neighbors_cur.items():
            if cell == key:
                # do the swap
                if val in cells_and_their_neighbors_cur.keys():

                    if val in final_validated_changes.keys():
                        if final_validated_changes[val] == key:
                            # prevent reswapping an already swapped stuff --> move on
                            print('already swapped --> skipping', cell, val)
                            continue

                    # in fact I need also store and restore all neighbs --> need do complex copies of the stuff
                    # bckup_neighbs_cell = list(cells_and_their_neighbors_cur[cell])
                    # should I do that before or after by the way --> before is better
                    neighbs_cell = cells_and_their_neighbors_cur[cell]
                    for neigb in neighbs_cell:
                        neighbs_of_neighb = cells_and_their_neighbors_cur[neigb]
                        if cell == 10287288:
                            print('neighbval begin', neighbs_of_neighb)
                        neighbs_of_neighb = [id if id != cell else val for id in neighbs_of_neighb]
                        cells_and_their_neighbors_cur[neigb] = neighbs_of_neighb
                        if cell == 10287288:
                            print('neighbval end', neighbs_of_neighb)

                    # print("change", bckup_neighbs_cell, neighbs_cell, cell, val) # no change here !!! --> bug

                    # c pas ça --> pr ses neighbs --> change all values
                    # neighbs_cell = [id if id!=cell else val for id in neighbs_cell]

                    # bckup_neighbs_val = list(cells_and_their_neighbors_cur[val])
                    neighbs_val = cells_and_their_neighbors_cur[val]
                    # if cell == 10287288:
                    #     print('neighbs_val before', neighbs_val)
                    for neigb in neighbs_val:
                        neighbs_of_neighb = cells_and_their_neighbors_cur[neigb]
                        if cell == 10287288:
                            print('neighbval begin', neighbs_of_neighb)
                        neighbs_of_neighb = [id if id != val else cell for id in neighbs_of_neighb]
                        cells_and_their_neighbors_cur[neigb] = neighbs_of_neighb

                        if cell == 10287288:
                            print('neighbval end', neighbs_of_neighb)

                    # if cell == 10287288:
                    #     print('neighbs_val after', neighbs_val)
                    # print("change2", bckup_neighbs_val, neighbs_val, cell, val)

                    # neighbs_val = [id if id != val else cell for id in neighbs_val]
                    # need replace val by cell in one and vice versa

                    cells_and_their_neighbors_cur[cell], cells_and_their_neighbors_cur[val] = \
                        cells_and_their_neighbors_cur[val], cells_and_their_neighbors_cur[cell]

                    # print(type(cells_and_their_neighbors_cur), type(cells_and_their_neighbors_t_other)) # le second est une liste mais ne devrait pas !!!! --> un bug qq part

                    # here I could gain a lot of time by just computing the delta score change for the changed values only !!! --> in fact very good idea and would save a lot of time

                    # if cell == 10287288:
                    #     try:
                    #         print('cells for scoring', cells_and_their_neighbors_cur[10287288],
                    #               cells_and_their_neighbors_t_other[10287288], cells_and_their_neighbors_cur[4367012],
                    #               cells_and_their_neighbors_t_other[4367012])
                    #     except:
                    #         pass

                    new_score = compute_neighbor_score(cells_and_their_neighbors_cur=cells_and_their_neighbors_cur,
                                                       cells_and_their_neighbors_t_other=cells_and_their_neighbors_t_other)
                    print('new_score, initial_score', new_score, initial_score)

                    if new_score <= initial_score:
                        # do not change if score is lower or equal --> TODO
                        #  then restore change
                        # cells_and_their_neighbors_cur[cell], cells_and_their_neighbors_cur[val] = cells_and_their_neighbors_cur[val], cells_and_their_neighbors_cur[cell]
                        # reswap and restore
                        # cells_and_their_neighbors_cur[cell]=bckup_neighbs_cell
                        # cells_and_their_neighbors_cur[val]=bckup_neighbs_val
                        cells_and_their_neighbors_cur = copy.deepcopy(cp)
                        # else validate the list and continue
                        # print('rejected changes ', cell, val, r_g_b_from_rgb(cell), r_g_b_from_rgb(val), new_score,                              initial_score)

                    else:
                        # validate the change and update the score
                        initial_score = new_score
                        final_validated_changes[cell] = val
                        cp = copy.deepcopy(cells_and_their_neighbors_cur)
                        # print('accepted change', cell, val, r_g_b_from_rgb(cell), r_g_b_from_rgb(val), new_score,                              initial_score)

                        # apply changes to the dict and return it too

                        print('before mapping', cell, val, tmp_map_tracks_n_label_t_cur[cell],                              tmp_map_tracks_n_label_t_cur[val])
                        tmp_map_tracks_n_label_t_cur[cell], tmp_map_tracks_n_label_t_cur[val] = \
                            tmp_map_tracks_n_label_t_cur[val], tmp_map_tracks_n_label_t_cur[cell]
                        print('after mapping', cell, val, tmp_map_tracks_n_label_t_cur[cell],                              tmp_map_tracks_n_label_t_cur[val])

                    # TODO also need update their neighbors

                    # if global_score is improved keep change otherwise reset it
                    # just do paiwise score --> easiest and if better --> keep

    print('final_validated_changes', final_validated_changes)



    # then recompute the score and if better --> change --> need compute the score for all cells implicated in

    # ne pas utiliser un dict mais plutot le truc splitte et ne pas causer de
    #
    # d['A'], d['B'] = d['B'], d['A'] --> this is how to swapp values
    # to swap I should compare the dict to itself --> in that case this is pure swap --> but still need score with respect to the prev image !!!
    # sinon c'est assignment --> mais peut dupliquer id --> need swap
    # see how to do ???

    # in the end apply this maybe one by one

    # convert dict back to an array

    print('tmp_map_tracks_n_label_t_cur2', len(tmp_map_tracks_n_label_t_cur))

    tmp_map_tracks_n_label_t_cur = np.array(list(tmp_map_tracks_n_label_t_cur.items()))

    print('tmp_map_tracks_n_label_t_curshp', tmp_map_tracks_n_label_t_cur.shape)


    # nb could also return score to see if worth continuing

    return cells_and_their_neighbors_cur, tmp_map_tracks_n_label_t_cur, initial_score  # need reconvert it to an array


# this is a local and thereby faster version of the swap stuff
# local and makes sure always the best cell is checked --> maybe smarter...
# vachement de swap mais des bonnes idees faudrait faire des tests pr comprendre où ça bugge
def optimize_score_local(cells_and_their_neighbors_cur, cells_and_their_neighbors_t_other, current_cells_n_score,
                   map_tracks_n_label_t_cur, threshold=0.5):
    # if shift is validated I need change the mappings --> either swap cells or swap their values --> the pb is that this will change the relationship to the rps --> need be smart but ok
    # convert map_tracks_n_label_t_cur to a dict then convert it back to array after all changes are applied

    tmp_map_tracks_n_label_t_cur = {k: v for k, v in map_tracks_n_label_t_cur}

    print('tmp_map_tracks_n_label_t_cur', tmp_map_tracks_n_label_t_cur)  # --> ok

    # sort current_cells_n_score values and keys from min to max until a threshold try to optimize score
    # find if any other pair improves score !!! and if so do the swap and recompute the score
    # probably need update all in the lists too --> TODO
    # or do not do recursively --> check that later --> first try find and see if score improves
    possible_swapping_pairs = {}
    possible_assignment = {}

    # nb maybe compute the score here --> no need to do it outside then --> simpler
    scores_sorted_ascending = dict(sorted(current_cells_n_score.items(), key=operator.itemgetter(1)))
    low_scores_to_check = [cell for cell in scores_sorted_ascending.keys() if
                           scores_sorted_ascending[cell] <= threshold]

    print('scores_sorted_descending', scores_sorted_ascending)
    # print('scores_sorted_descending', [r_g_b_from_rgb(cell) for cell in scores_sorted_ascending.keys()])
    # if r_g_b_to_rgb(252, 78, 252) in scores_sorted_ascending.keys():
    #     print('score ', scores_sorted_ascending[r_g_b_to_rgb(252, 78, 252)], r_g_b_to_rgb(252, 78, 252))
    # else:
    #     print('score not in')
    print('low_scores_to_check', low_scores_to_check)
    # print('in stuff to chekc', r_g_b_to_rgb(156, 248, 184) in low_scores_to_check,          r_g_b_to_rgb(156, 248, 184))  # false --> this cell has a score of 1.0 --> how is it possible ???
    # print('in stuff to chekc', r_g_b_to_rgb(66, 162, 164) in low_scores_to_check,          r_g_b_to_rgb(66, 162, 164))  # true --> score 0.6 --> ok
    # print('in stuff to chekc2', r_g_b_to_rgb(137, 134, 15) in low_scores_to_check,          r_g_b_to_rgb(137, 134, 15))  # true --> score 0.6 --> ok

    # try:
    #     # print(r_g_b_to_rgb(156, 248, 184) in cells_and_their_neighbors_cur,              r_g_b_to_rgb(156, 248, 184) in cells_and_their_neighbors_t_other)
    #     print('manually calculated score', pairwise_score(cells_and_their_neighbors_cur[r_g_b_to_rgb(156, 248, 184)],                                                          cells_and_their_neighbors_t_other[r_g_b_to_rgb(156, 248, 184)]),
    #           cells_and_their_neighbors_cur[r_g_b_to_rgb(156, 248, 184)],
    #           cells_and_their_neighbors_t_other[r_g_b_to_rgb(156, 248, 184)])
    #     print('manually calculated score2', pairwise_score(cells_and_their_neighbors_cur[r_g_b_to_rgb(66, 162, 164)],
    #                                                        cells_and_their_neighbors_t_other[
    #                                                            r_g_b_to_rgb(156, 248, 184)]),
    #           cells_and_their_neighbors_cur[r_g_b_to_rgb(66, 162, 164)],
    #           cells_and_their_neighbors_t_other[r_g_b_to_rgb(156, 248, 184)])
    #     print('manually calculated score3', pairwise_score(cells_and_their_neighbors_cur[r_g_b_to_rgb(156, 248, 184)],
    #                                                        cells_and_their_neighbors_t_other[
    #                                                            r_g_b_to_rgb(66, 162, 164)]),
    #           cells_and_their_neighbors_cur[r_g_b_to_rgb(156, 248, 184)],
    #           cells_and_their_neighbors_t_other[r_g_b_to_rgb(66, 162, 164)])
    # except:
    #     print('missing cell')
    #     pass

    # low_scores_to_check[980445, 4479535, 9829855, 10240303, 10369245, 10652374, 11560212, 16222777, 16404369] --> a l'air de marcher
    # loop those over the entire list

    # possible swap 16703757 980445
    # possible swap 16703757 4479535
    # possible swap 16703757 9829855
    # possible swap None 10240303
    # possible swap None 10369245
    # possible swap 16703757 10652374
    # possible swap None 11560212
    # possible swap 16703757 16222777
    # possible swap 16703757 16404369

    # shall I store all possible swaps and then apply them in the end
    # to improve the glocal score should improve

    # it is in the stuff to check but never makes it to the end --> WHY

    # possible assignment peut aussi detecter les paires en division--> très bonne idée

    #  TODO also check all possible swaps between pairs of cells with low score non 0 that would optimize global score
    # pb is score if they share neighbs --> individual score not better because need take into account all
    # maybe do brute force scoring for low cells once just to see

    # brute force swap --> create all swapping pairs non zero and check them

    for cell in low_scores_to_check:
        current_score = current_cells_n_score[cell]
        print(current_score)
        # TODO en fait c'est impossible car le score est tjrs inferieur à 1 sauf si l'utilisateur veut vraiment mettre 1 --> remove that or keep it to force analyze the whole image ??? --> I guess remove it
        if current_score >= 1.0:
            continue
        final_cell_to_swap = None
        # ssqdqsdqdqs
        best_score = 0
        # shall I really try all possible swap pairs ??? rather than try best or should I at least try one --> maybe yes --> the one with the best score
        possible_swaps, possible_assignments = find_neighbors_to_check(cell, cells_and_their_neighbors_cur, cells_and_their_neighbors_t_other)
        # should I pass all those cells directly without recomputing the score --> probably yes in fact
        if cell in cells_and_their_neighbors_t_other.keys():
            # could even add its own neighbs because if a swap helps then really do it!!!
            neigbsb = cells_and_their_neighbors_t_other[cell]

            # PB IS THAT I SHOULD ASSUME SCORE WITH ASSUMING BETTER SWAP
            # look in neighborhood if there is a better match
            # nb I could add its own neighbors by the way --> no need to only take those of the neighbor in prev --> need to check if it is a swap or an
            # brute force too --> does check all the cells in the image
            # for cell_to_swap_with, neigbs in cells_and_their_neighbors_cur.items():
            for cell_to_swap_with in possible_swaps:
                if cell_to_swap_with == cell:
                    continue
                neigbs = cells_and_their_neighbors_cur[cell_to_swap_with]

                # (len(list(set(neigbs) & set(neigbsb))) * 2) / (len(neigbsb) + len(neigbs))
                score = pairwise_score(neigbs, neigbsb)

                if cell_to_swap_with == 4367012 and cell == 10287288:
                    print('in da sgit', score, neigbs, neigbsb, current_score)

                if score >= best_score :
                    best_score = score
                    final_cell_to_swap = cell_to_swap_with
            if final_cell_to_swap is not None and best_score>0.:
                # print('possible swap', cell, final_cell_to_swap, current_score, r_g_b_from_rgb(cell),
                #       r_g_b_from_rgb(final_cell_to_swap))  # indeed a perfect swap
                possible_swapping_pairs[cell] = final_cell_to_swap
        else:
            # same code in fact -->
            # la cellule n'est pas dans other --> verif si le swap est  dans celui là sinon assigner le fate sinon creer une nouvelle id unique qui n'est ni dans l'un ni dans l'autre
            # possible_assignment
            # find any cell id that would share the max of the neighbors --> loop over all or loop over all that have low score and look for better

            # probably only need loop over low scores --> most of the cells will not have that then!!!
            # IN FCAT ONLY NEED BE DONE FOR NEW CELLS --> HUGE GAIN OF TIME!!!!
            # for cellb, neigbsb in cells_and_their_neighbors_t_other.items():
            # neigbsb = cells_and_their_neighbors_t_other[cell]
            neigbsb = cells_and_their_neighbors_cur[cell]

            # for cell_to_swap_with, neigbs in cells_and_their_neighbors_t_other.items():
            for cell_to_swap_with in possible_assignments:
                if cell_to_swap_with == cell:
                    continue
                neigbs = cells_and_their_neighbors_t_other[cell_to_swap_with]
                score = pairwise_score(neigbs,
                                       neigbsb)  # (len(list(set(neigbs) & set(neigbsb))) * 2) / (len(neigbsb) + len(neigbs))
                if score > best_score:
                    best_score = score
                    final_cell_to_swap = cell_to_swap_with
            if final_cell_to_swap is not None and best_score>0:
                # print('possible assignment', cell, final_cell_to_swap, current_score, r_g_b_from_rgb(cell),
                #       r_g_b_from_rgb(final_cell_to_swap), current_score)  # indeed a perfect swap
                possible_assignment[cell] = final_cell_to_swap
            # pass

    # check if any of the lost cells in current would give a better score than a cell in the current because if that is the case --> offer assignment
    missing_cells_from_prev = set(cells_and_their_neighbors_t_other.keys()) - set(cells_and_their_neighbors_cur.keys())

    print('in stuff to chekc2b', r_g_b_to_rgb(137, 134, 15) in missing_cells_from_prev,
          r_g_b_to_rgb(137, 134, 15), r_g_b_to_rgb(44, 140, 93))  # true --> score 0.6 --> ok
    if True:
        for cell in missing_cells_from_prev:
            best_score = 0
            # comme c'est du brute force ça rallonge pas mal le processus, pr finalement très peu de cellules --> is that really worth it ???
            final_cell_to_swap = None
            neigbsb = cells_and_their_neighbors_t_other[cell]
            for cell_to_swap_with, neigbs in cells_and_their_neighbors_cur.items():
                current_score = current_cells_n_score[cell_to_swap_with]
                if current_score >= 1.:
                    continue

                score = pairwise_score(neigbs,
                                       neigbsb)  # (len(list(set(neigbs) & set(neigbsb))) * 2) / (len(neigbsb) + len(neigbs))

                if score >= current_score:
                    # current_score = score
                    if score >= best_score:
                        best_score = score
                        final_cell_to_swap = cell_to_swap_with
            if final_cell_to_swap is not None:
                # print('possible assignment2', cell, final_cell_to_swap, current_score, r_g_b_from_rgb(cell),
                #       r_g_b_from_rgb(final_cell_to_swap), current_score)  # indeed a perfect swap
                possible_assignment[final_cell_to_swap] = cell
    # if any of these cells gives a better score than the current offer a swap

    # this cell never enters here although there has to be an equal cell maybe
    # print('inside swaps and so on', r_g_b_to_rgb(156, 248, 184) in possible_assignment.keys(),  r_g_b_to_rgb(156, 248, 184) in possible_assignment.values(),  r_g_b_to_rgb(156, 248, 184) in possible_swapping_pairs.keys(), r_g_b_to_rgb(156, 248, 184) in possible_swapping_pairs.values())

    # the two cells never enter in this !!!
    print('inside swaps and so on', r_g_b_to_rgb(156, 248, 184) in possible_assignment.keys(),
          r_g_b_to_rgb(156, 248, 184) in possible_assignment.values(),
          r_g_b_to_rgb(156, 248, 184) in possible_swapping_pairs.keys(),
          r_g_b_to_rgb(156, 248, 184) in possible_swapping_pairs.values())
    print('inside swaps and so on', r_g_b_to_rgb(66, 162, 164) in possible_assignment.keys(),
          r_g_b_to_rgb(66, 162, 164) in possible_assignment.values(),
          r_g_b_to_rgb(66, 162, 164) in possible_swapping_pairs.keys(),
          r_g_b_to_rgb(66, 162, 164) in possible_swapping_pairs.values())

    # maybe store al the possible swaps and apply them and try
    # low scores could even contain new cells
    # score TODO finalize that
    # print('total score after optimization', sum(cells_present_in_t_cur_but_absent_in_t_minus_1_score.values()))

    # get the dict and copy it

    import copy
    cp = copy.deepcopy(cells_and_their_neighbors_cur)
    initial_score = sum(current_cells_n_score.values())
    # new_score= initial_score

    print('initila score', initial_score)

    final_validated_changes = {}

    # I need treat the swapping and assignment differently!!!

    # in fact if the cell is present in the image --> real swapping and can be added otherwise it's an assignment --> do things differently
    # need further filter

    # FILTER AGAIN assignments
    # AN ASSIGNMENT SHOULD HAVE ONLY ONE CELL IN THE IMAGE

    # filtered_assignments = {k : v  for (k,v) in possible_assignment.items()}
    # keys to remove:
    keys_to_remove_in_assignment = []
    print('possible_assignment', possible_assignment)
    for k, v in possible_assignment.items():
        if v in cells_and_their_neighbors_cur.keys():
            # del possible_assignment[k]
            keys_to_remove_in_assignment.append(k)

    for k in keys_to_remove_in_assignment:
        del possible_assignment[k]

    print('filtered_assignments',
          possible_assignment)  # --> good really kept the right one, but keep in mind that the others are actually missegmented cells in current --> those are very useful and I have positional info there --> maybe keep those

    for cell, val in possible_assignment.items():
        # just change the fate of the key and replace it in itself and in neighboring cells!!!
        # much less operations than in the other --> TODO
        neighbs_cell = cells_and_their_neighbors_cur[cell]
        for neigb in neighbs_cell:
            neighbs_of_neighb = cells_and_their_neighbors_cur[neigb]
            neighbs_of_neighb = [id if id != cell else val for id in neighbs_of_neighb]
            cells_and_their_neighbors_cur[neigb] = neighbs_of_neighb
            # print('cell in neighbs_of_neighb', cell in neighbs_of_neighb)  # FAlse --> ok indeed
            # print('val in neighbs_of_neighb',val in neighbs_of_neighb)  # True --> all is ok in fact

        # create a new cell then remove it
        cells_and_their_neighbors_cur[val] = cells_and_their_neighbors_cur[cell]
        del cells_and_their_neighbors_cur[cell]
        new_score = compute_neighbor_score(cells_and_their_neighbors_cur=cells_and_their_neighbors_cur,
                                           cells_and_their_neighbors_t_other=cells_and_their_neighbors_t_other)
        print('new_score, initial_score', new_score, initial_score)

        if new_score < initial_score:
            # do not change if score is lower or equal --> TODO
            #  then restore change
            # cells_and_their_neighbors_cur[cell], cells_and_their_neighbors_cur[val] = cells_and_their_neighbors_cur[val], cells_and_their_neighbors_cur[cell]
            # reswap and restore
            # cells_and_their_neighbors_cur[cell]=bckup_neighbs_cell
            # cells_and_their_neighbors_cur[val]=bckup_neighbs_val
            cells_and_their_neighbors_cur = copy.deepcopy(cp)
            # else validate the list and continue
            # print('rejected assignment ', cell, val, r_g_b_from_rgb(cell), r_g_b_from_rgb(val), new_score,
            #       initial_score)


        else:
            # validate the change and update the score
            initial_score = new_score
            final_validated_changes[cell] = val
            cp = copy.deepcopy(cells_and_their_neighbors_cur)

            # do I have an inversion ??? --> maybe yes ????
            # prevent multi assignments ??? or update the image in fact --> that is necessary

            # print('accepted assignment', cell, val, r_g_b_from_rgb(cell), r_g_b_from_rgb(val), new_score, initial_score)
            # print(cell in cells_and_their_neighbors_cur.keys()) # FAlse --> ok indeed
            # print(val in cells_and_their_neighbors_cur.keys()) # True --> all is ok in fact

            # update correspondance between cell and its mask
            print('before mapping0', cell, val, tmp_map_tracks_n_label_t_cur[cell])
            tmp_map_tracks_n_label_t_cur[val] = tmp_map_tracks_n_label_t_cur[cell]
            del tmp_map_tracks_n_label_t_cur[cell]
            print('after mapping0', cell, val, tmp_map_tracks_n_label_t_cur[val])

    cp = copy.deepcopy(cells_and_their_neighbors_cur)
    # TODO by construction it does it both ways and I don't want that because it's a waste of time --> need fix that and change that --> if already validated prevent reswap
    # en fait c'est pas des swaps c'est des connexions de tracks
    for key, val in possible_swapping_pairs.items():
        # I need get the score after swap vs before for the cell and all of its neighbors recusrsively and only to that if all improves !!!
        for cell, neighbs in cells_and_their_neighbors_cur.items():
            if cell == key:
                # do the swap
                if val in cells_and_their_neighbors_cur.keys():

                    if val in final_validated_changes.keys():
                        if final_validated_changes[val] == key:
                            # prevent reswapping an already swapped stuff --> move on
                            print('already swapped --> skipping', cell, val)
                            continue

                    # in fact I need also store and restore all neighbs --> need do complex copies of the stuff
                    # bckup_neighbs_cell = list(cells_and_their_neighbors_cur[cell])
                    # should I do that before or after by the way --> before is better
                    neighbs_cell = cells_and_their_neighbors_cur[cell]
                    for neigb in neighbs_cell:
                        neighbs_of_neighb = cells_and_their_neighbors_cur[neigb]
                        if cell == 10287288:
                            print('neighbval begin', neighbs_of_neighb)
                        neighbs_of_neighb = [id if id != cell else val for id in neighbs_of_neighb]
                        cells_and_their_neighbors_cur[neigb] = neighbs_of_neighb
                        if cell == 10287288:
                            print('neighbval end', neighbs_of_neighb)

                    # print("change", bckup_neighbs_cell, neighbs_cell, cell, val) # no change here !!! --> bug

                    # c pas ça --> pr ses neighbs --> change all values
                    # neighbs_cell = [id if id!=cell else val for id in neighbs_cell]

                    # bckup_neighbs_val = list(cells_and_their_neighbors_cur[val])
                    neighbs_val = cells_and_their_neighbors_cur[val]
                    # if cell == 10287288:
                    #     print('neighbs_val before', neighbs_val)
                    for neigb in neighbs_val:
                        neighbs_of_neighb = cells_and_their_neighbors_cur[neigb]
                        if cell == 10287288:
                            print('neighbval begin', neighbs_of_neighb)
                        neighbs_of_neighb = [id if id != val else cell for id in neighbs_of_neighb]
                        cells_and_their_neighbors_cur[neigb] = neighbs_of_neighb

                        if cell == 10287288:
                            print('neighbval end', neighbs_of_neighb)

                    # if cell == 10287288:
                    #     print('neighbs_val after', neighbs_val)
                    # print("change2", bckup_neighbs_val, neighbs_val, cell, val)

                    # neighbs_val = [id if id != val else cell for id in neighbs_val]
                    # need replace val by cell in one and vice versa

                    cells_and_their_neighbors_cur[cell], cells_and_their_neighbors_cur[val] = \
                        cells_and_their_neighbors_cur[val], cells_and_their_neighbors_cur[cell]

                    # print(type(cells_and_their_neighbors_cur), type(cells_and_their_neighbors_t_other)) # le second est une liste mais ne devrait pas !!!! --> un bug qq part

                    # here I could gain a lot of time by just computing the delta score change for the changed values only !!! --> in fact very good idea and would save a lot of time

                    if cell == 10287288:
                        try:
                            print('cells for scoring', cells_and_their_neighbors_cur[10287288],
                                  cells_and_their_neighbors_t_other[10287288], cells_and_their_neighbors_cur[4367012],
                                  cells_and_their_neighbors_t_other[4367012])
                        except:
                            # no big deal debug
                            pass

                    new_score = compute_neighbor_score(cells_and_their_neighbors_cur=cells_and_their_neighbors_cur,
                                                       cells_and_their_neighbors_t_other=cells_and_their_neighbors_t_other)
                    print('new_score, initial_score', new_score, initial_score)

                    if new_score < initial_score:
                        # do not change if score is lower or equal --> TODO
                        #  then restore change
                        # cells_and_their_neighbors_cur[cell], cells_and_their_neighbors_cur[val] = cells_and_their_neighbors_cur[val], cells_and_their_neighbors_cur[cell]
                        # reswap and restore
                        # cells_and_their_neighbors_cur[cell]=bckup_neighbs_cell
                        # cells_and_their_neighbors_cur[val]=bckup_neighbs_val
                        cells_and_their_neighbors_cur = copy.deepcopy(cp)
                        # else validate the list and continue
                        # print('rejected changes ', cell, val, r_g_b_from_rgb(cell), r_g_b_from_rgb(val), new_score,
                        #       initial_score)

                    else:
                        # validate the change and update the score
                        initial_score = new_score
                        final_validated_changes[cell] = val
                        cp = copy.deepcopy(cells_and_their_neighbors_cur)
                        # print('accepted change', cell, val, r_g_b_from_rgb(cell), r_g_b_from_rgb(val), new_score,
                        #       initial_score)

                        # apply changes to the dict and return it too

                        print('before mapping', cell, val, tmp_map_tracks_n_label_t_cur[cell],
                              tmp_map_tracks_n_label_t_cur[val])
                        tmp_map_tracks_n_label_t_cur[cell], tmp_map_tracks_n_label_t_cur[val] = \
                            tmp_map_tracks_n_label_t_cur[val], tmp_map_tracks_n_label_t_cur[cell]
                        print('after mapping', cell, val, tmp_map_tracks_n_label_t_cur[cell],
                              tmp_map_tracks_n_label_t_cur[val])

                    # TODO also need update their neighbors

                    # if global_score is improved keep change otherwise reset it
                    # just do paiwise score --> easiest and if better --> keep

    print('final_validated_changes', final_validated_changes)
    # then recompute the score and if better --> change --> need compute the score for all cells implicated in

    # ne pas utiliser un dict mais plutot le truc splitte et ne pas causer de
    #
    # d['A'], d['B'] = d['B'], d['A'] --> this is how to swapp values
    # to swap I should compare the dict to itself --> in that case this is pure swap --> but still need score with respect to the prev image !!!
    # sinon c'est assignment --> mais peut dupliquer id --> need swap
    # see how to do ???

    # in the end apply this maybe one by one

    # convert dict back to an array
    tmp_map_tracks_n_label_t_cur = np.array(list(tmp_map_tracks_n_label_t_cur.items()))

    # nb could also return score to see if worth continuing

    return cells_and_their_neighbors_cur, tmp_map_tracks_n_label_t_cur, initial_score  # need reconvert it to an array


def track_cells(files):
    for iii in range(len(files)):
        # file_t0 = files[iii-1]
        # file_t1 = files[iii]
        #
        # print(file_t0, file_t1)

        # TODO instead of getting 3 files --> just make it able to get n files with specific rules --> TODO
        files_to_read = get_n_files_from_list(files, iii, -1, 1)
        # print(files_to_read)

        # TODO handle first ad last

        if files_to_read[0] is None or files_to_read[len(files_to_read) - 1] is None:
            # first or last image missing --> skipping for now but generate code at some point
            print('missing files', files_to_read)

            # in fact here I could make use of the data of two consecutive images
            # or should I try directly with 3 images ????

            # think about it
            continue

        # for file_to_read in files_to_read:
        extension = '.tif'  # '.png' # '.tif' # '.png' # '.tif'
        TA_path_minus_1, track_t_minus_1 = smart_name_parser(files_to_read[0],
                                                             ordered_output=['TA', 'tracked_cells_resized' + extension])
        TA_path_cur, track_t_cur = smart_name_parser(files_to_read[1],
                                                     ordered_output=['TA', 'tracked_cells_resized' + extension])
        TA_path_plus_1, track_t_plus_1 = smart_name_parser(files_to_read[2],
                                                           ordered_output=['TA', 'tracked_cells_resized' + extension])

        # print(tracked_cells_resized)
        track_t_minus_1 = RGB_to_int24(Img(track_t_minus_1))
        track_t_cur = RGB_to_int24(Img(track_t_cur))
        track_t_plus_1 = RGB_to_int24(Img(track_t_plus_1))

        # find_vertices(img)

        # could compare this to the other
        # vertices_t_minus_1 = find_vertices(track_t_minus_1, return_vertices=True, return_bonds=False)
        # vertices_t_cur = find_vertices(track_t_cur, return_vertices=True, return_bonds=False)
        # vertices_t_plus_1 = find_vertices(track_t_plus_1, return_vertices=True, return_bonds=False)
        vertices_t_minus_1 = np.stack(np.where(track_t_minus_1 == 0xFFFFFF), axis=1) #find_vertices(track_t_minus_1, return_vertices=True, return_bonds=False)
        vertices_t_cur = np.stack(np.where(track_t_cur == 0xFFFFFF), axis=1) # find_vertices(track_t_cur, return_vertices=True, return_bonds=False)
        vertices_t_plus_1 = np.stack(np.where(track_t_plus_1 == 0xFFFFFF), axis=1) # find_vertices(track_t_plus_1, return_vertices=True, return_bonds=False)
        # plt.imshow(vertices)
        # plt.show()

        # print(get_vx_neighbors(vertices, RGB24img=img)) # is that slow??? --

        # get_vx_neighbors(vertices_t_minus_1,                     RGB24img=track_t_minus_1)  # ça marche et pas trop long par contre ça me donne juste la correspondance one to one avec le vx array ce qui est peut etre ce que je veux d'ailleurs mais faut pas changer l'ordre

        # ce truc est slow --> can I speed up using vertices --> most likely yes
        # print(np.unique(np.asarray(associate_cell_to_its_neighbors(img)), axis=0)) # really get cell neighbors # ça a l'air de marcher # maybe I can speed up by using just the vertices or not ???? --> think about it
        # print(associate_cell_to_its_neighbors(img)) # really get cell neighbors # ça a l'air de marcher # maybe I can speed up by using just the vertices or not ???? --> think about it
        # aussi voir comment associer les vertices à une seule cellule --> car très utile
        # print()  # a bit faster indeed but not outstanding

        # get cells and their neigbs for -1 and 0 and do a scoring --> will highlight potential swapping and or division and or other things in a way --> REALLY TRY THAT
        cells_and_their_neighbors_minus_1 = np.unique(
            np.asarray(associate_cell_to_its_neighbors2(vertices_t_minus_1, track_t_minus_1)), axis=0)
        cells_and_their_neighbors_cur = np.unique(
            np.asarray(associate_cell_to_its_neighbors2(vertices_t_cur, track_t_cur)), axis=0)
        cells_and_their_neighbors_plus_1 = np.unique(
            np.asarray(associate_cell_to_its_neighbors2(vertices_t_plus_1, track_t_plus_1)), axis=0)

        cells_and_their_neighbors_minus_1 = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_minus_1)
        cells_and_their_neighbors_cur = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_cur)
        cells_and_their_neighbors_plus_1 = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_plus_1)

        print('cells_and_their_neighbors_minus_1', cells_and_their_neighbors_minus_1)
        print('cells_and_their_neighbors_cur', cells_and_their_neighbors_cur)
        print('cells_and_their_neighbors_plus_1', cells_and_their_neighbors_plus_1)

        # for all the common cells --> compute a score
        # nb of matching neighbs / total nb of neighbs in both

        # create and color

        cells_present_in_t_cur_but_absent_in_t_plus_1 = []
        cells_present_in_t_cur_but_absent_in_t_plus_1_score = {}
        matching_neighorhood_score_t_cur_with_t_plus_1 = np.zeros_like(track_t_cur, dtype=float)

        compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_plus_1,
                               cells_present_in_t_cur_but_absent_in_t_plus_1,
                               cells_present_in_t_cur_but_absent_in_t_plus_1_score,
                               matching_neighorhood_score_t_cur_with_t_plus_1, track_t_cur)

        # for cell, neigbs in cells_and_their_neighbors_cur.items():
        #     if cell in cells_and_their_neighbors_plus_1:
        #         # compute score and color cell
        #         neigbsb = cells_and_their_neighbors_plus_1[cell]
        #         score = (len(list(set(neigbs) & set(neigbsb))) * 2) / (len(neigbsb) + len(neigbs))
        #         cells_present_in_t_cur_but_absent_in_t_plus_1_score[cell]=score
        #         matching_neighorhood_score_t_cur_with_t_plus_1[track_t_cur == cell] = score
        #     else:
        #         cells_present_in_t_cur_but_absent_in_t_plus_1.append(cell)
        #         cells_present_in_t_cur_but_absent_in_t_plus_1_score[cell] = 0

        cells_present_in_t_cur_but_absent_in_t_minus_1 = []
        cells_present_in_t_cur_but_absent_in_t_minus_1_score = {}
        matching_neighorhood_score_t_cur_with_t_minus_1 = np.zeros_like(track_t_cur, dtype=float)

        compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                               cells_present_in_t_cur_but_absent_in_t_minus_1,
                               cells_present_in_t_cur_but_absent_in_t_minus_1_score,
                               matching_neighorhood_score_t_cur_with_t_minus_1, track_t_cur)
        # for cell, neigbs in cells_and_their_neighbors_cur.items():
        #     if cell in cells_and_their_neighbors_minus_1:
        #         # compute score and color cell
        #         neigbsb=cells_and_their_neighbors_minus_1[cell]
        #         score = (len(list(set(neigbs) & set(neigbsb)))*2)/(len(neigbsb)+len(neigbs))
        #         cells_present_in_t_cur_but_absent_in_t_minus_1_score[cell]=score
        #         matching_neighorhood_score_t_cur_with_t_minus_1[track_t_cur == cell]=score
        #     else:
        #         cells_present_in_t_cur_but_absent_in_t_minus_1.append(cell)
        #         cells_present_in_t_cur_but_absent_in_t_minus_1_score[cell] = 0

        # offer solutions based on scores --> TODO

        # do I have a bug here ???

        print(type(cells_and_their_neighbors_cur), type(
            cells_and_their_neighbors_minus_1))  # le second est une liste mais ne devrait pas !!!! --> un bug qq part
        # test after optimization

        # ça rend le code un peu plus long mais vraiment pas deconnant --> doit etre assez facile à faire
        lab_t_minus_1 = label(track_t_minus_1, connectivity=1, background=0xFFFFFF)
        rps_t_minus_1 = regionprops(lab_t_minus_1)
        lab_t_cur = label(track_t_cur, connectivity=1, background=0xFFFFFF)
        rps_t_cur = regionprops(lab_t_cur)
        lab_t_plus_1 = label(track_t_plus_1, connectivity=1, background=0xFFFFFF)
        rps_t_plus_1 = regionprops(lab_t_plus_1)

        # can I read all the neighbors in a smart way now by first index of one

        # can compare neigbors for a cell in two instances --> give it a try
        # all should be easy I guess !!!

        # to get cells in one I can get unique of the first col and compare it to the other --> in fact not that hard I think
        # pb if a cell has no vertex it will be ignored
        # --> purely isolated cell

        # in fact I can simply do unique otherwise directly on the image

        cells_in_t_plus_1 = np.unique(track_t_plus_1)
        cells_in_t_cur, first_pixels_t_cur = get_cells_in_image_n_fisrt_pixel(track_t_cur)  # np.unique(track_t_cur)
        cells_in_t_minus_1 = np.unique(track_t_minus_1)

        map_tracks_n_label_t_cur = map_track_id_to_label(first_pixels_t_cur, track_t_cur, lab_t_cur)

        print('first_pixels_t_cur', first_pixels_t_cur)  # this is the ravel index --> how can I convert it back ???
        print('cells_in_t_cur', cells_in_t_cur)
        print('map_tracks_n_label_t_cur', map_tracks_n_label_t_cur)

        # I'm ready to use this for the tracking of cells
        # just see how fast and efficient this is
        # do two versions, one with just showing errors and letting the user fix them  and one with other
        if FIX_SWAPS:
            # DETECT SWAPS AND FIX THEM AND COMPUTE SCORE --> TODO also fix cell correspondance then --> really necessary!!!
            # we update both the cells and their local correspondance --> I think I have it --> just need few more crosses to identify more errors
            # can I use that as a tracking algo on top of the other with a minimization of the stuff and just do the coloring in the end --> TODO

            cells_and_their_neighbors_cur, map_tracks_n_label_t_cur, last_score_reached = optimize_score(
                cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                cells_present_in_t_cur_but_absent_in_t_minus_1_score, map_tracks_n_label_t_cur)
            # print('map_tracks_n_label_t_cur2',map_tracks_n_label_t_cur2)
            # print('map_tracks_n_label_t_cur',map_tracks_n_label_t_cur)

            # depending on the corrections I apply I could also identify cells that have changed between -1 and 1 and that are missassigned cells in 3 and
            # could in fact try tracking with three images to be more efficient in fact directly because post correction will be time consuming in fact especially for swapped cells
            # but still i need that
            # do a version of the tracking using pyramidal reg that relies on that!!! --> TODO

            # possible assignment 10240303 6824721 0.6666666666666666 (156, 65, 47) (104, 35, 17) 0.6666666666666666  --> a dividing cell --> really a good peak
            # 0
            # possible assignment 10369245 10443693 0.5714285714285714 (158, 56, 221) (159, 91, 173) 0.5714285714285714 --> une misegmented cell en bas au milieu) et une cellule adjacente qui elle est bien placee mais je suppose que faire ce changment fera baisser le core et que donc ce changement sera ignoré...
            # 0
            # possible assignment 11560212 8271832 0.5714285714285714 (176, 101, 20) (126, 55, 216) 0.5714285714285714 --> une autre misegmented cell à gauche un peu apres le centre de l'image!!! et une cellule bien trackee à coté --> pareil faire le score et voir
            # 0
            # possible assignment 12248378 7043456 1.0 (186, 229, 58) (107, 121, 128) 1.0 --> mis tracked cell

            # cells_in_t_cur = cells_and_their_neighbors_cur.keys()

            # TO MANUALLY FIND A CELL
            # for jjj, missing in enumerate(cells_in_t_cur):
            #     if missing == 0xFFFFFF:
            #         continue
            #
            #     # if missing != r_g_b_to_rgb( 4,77,109):# for sure it's a swapped cell
            #     #     continue
            #
            #     # the cell 9829855 is not shown properly --> it has no score --> why ?? maybe because no vx ???
            #     if missing == 10369245:# 11560212
            #         label_id = map_tracks_n_label_t_cur[jjj][1]
            #         rps = rps_t_cur[label_id - 1]
            #         bbox = rps.bbox
            #         # bbox = [bbox[0] - 50, bbox[1] - 50, bbox[2] + 50, bbox[3] + 50]
            #         # Bounding box ``(min_row, min_col, max_row, max_col)
            #         # do a crop of the region of interest
            #
            #         print('bbox pos of cell to assign' ,bbox)

            # plot change

            # print('tests of alm', 12248378 in cells_and_their_neighbors_cur.keys(), 12248378 in cells_and_their_neighbors_minus_1.keys(), 7043456 in cells_and_their_neighbors_cur.keys(), 7043456 in cells_and_their_neighbors_minus_1.keys())
            # False False True True --> ok in fact

            compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                                   cells_present_in_t_cur_but_absent_in_t_minus_1,
                                   cells_present_in_t_cur_but_absent_in_t_minus_1_score,
                                   matching_neighorhood_score_t_cur_with_t_minus_1, track_t_cur)

            # plt.imshow(matching_neighorhood_score_t_cur_with_t_minus_1)
            # plt.show()

        # if I really need swap then I also need

        # for all
        print('cells_present_in_t_cur_but_absent_in_t_minus_1', cells_present_in_t_cur_but_absent_in_t_minus_1)
        print('cells_present_in_t_cur_but_absent_in_t_plus_1', cells_present_in_t_cur_but_absent_in_t_plus_1)

        print('total score before optimization',
              sum(cells_present_in_t_cur_but_absent_in_t_minus_1_score.values()))  # 305.82 initially then
        # for all cells with low score if they are try a swapping but only if present before

        # NB I HAVE FOUND A SWAPPED CELL --> 37,51, 138 located at 437, 487 in 0
        # I need map one to one the label id and the track id!!!
        # anyways that is very useful

        # need map track to the label image at least for the current but also for the two others and need be able to get the coords from that !!!!

        # TODO see if I can improve score between -1 and 0 by swapping
        # see how fast/slow this could be --> just restrict myself to cells with low

        # TODO find swaps that maximize score

        # compute sum score and as long as it improves --> continue

        # remove purewhite from this array
        for jjj, missing in enumerate(cells_in_t_cur):
            if missing == 0xFFFFFF:
                continue

            # if missing != r_g_b_to_rgb( 4,77,109):# for sure it's a swapped cell
            #     continue

            # the cell 9829855 is not shown properly --> it has no score --> why ?? maybe because no vx ???
            if False:
                if missing == 9829855:
                    # indeed this is an isolated cell that has one or no vertices --> really need handle that at some point but ok for now
                    label_id = map_tracks_n_label_t_cur[jjj][1]
                    rps = rps_t_cur[label_id - 1]
                    bbox = rps.bbox
                    # bbox = [bbox[0] - 50, bbox[1] - 50, bbox[2] + 50, bbox[3] + 50]
                    # Bounding box ``(min_row, min_col, max_row, max_col)
                    # do a crop of the region of interest

                    print(bbox)
                    try:

                        plot_n_images_in_line(int24_to_RGB(track_t_minus_1[bbox[0]:bbox[2], bbox[1]:bbox[3]]),
                                              int24_to_RGB(track_t_cur[bbox[0]:bbox[2], bbox[1]:bbox[3]]),
                                              int24_to_RGB(track_t_plus_1[bbox[0]:bbox[2], bbox[1]:bbox[3]]),
                                              matching_neighorhood_score_t_cur_with_t_minus_1[bbox[0]:bbox[2],
                                              bbox[1]:bbox[3]],
                                              matching_neighorhood_score_t_cur_with_t_plus_1[bbox[0]:bbox[2],
                                              bbox[1]:bbox[3]],
                                              title='bug' + str(r_g_b_from_rgb(missing)))
                    except:
                        traceback.print_exc()
                        print('diplay error --> ifnore for naow')

            try:
                decision = take_decision(missing, cells_and_their_neighbors_minus_1, cells_and_their_neighbors_cur,
                                         cells_and_their_neighbors_plus_1,
                                         cells_present_in_t_cur_but_absent_in_t_minus_1_score,
                                         cells_present_in_t_cur_but_absent_in_t_plus_1_score,
                                         assume_previous_image_is_GT=True)
            except:
                traceback.print_exc()
                print('pb with decision --> contuinuing but need a fix some day !!!')
                continue

            print(missing, ' --> ', decision)  # decide whether to keep or discard a cell

            # division
            # death
            # ignore
            # None

            # see unhandled cases

            # cell 4,77,109 is clearly swapped --> can I detect it and fix it in a smart way ?????

            # if decision is None: # --> a lot of these are artifacts --> see how to fix??? or ignore or have mire stringent rules
            # if decision is not None and death in decision:
            # there is a bug somewhere !!!
            # if decision is not None and not ignore in decision:
            # if decision is not None and (ignore in decision and (cells_present_in_t_cur_but_absent_in_t_minus_1_score[missing] != 1 or cells_present_in_t_cur_but_absent_in_t_plus_1_score[missing] != 1)): # cells ignore due to sensitivity

            # nb some of the local swap are above 0.5 (maybe it's due to my test sample because I swapped big cells which is unlikely to happen)
            #
            # shall I start by trying to maximize score for all the cells below or equal 0.5 then do the rest of the code
            # most likely swapped cells would have to

            # if score of one is 0 --> probably a far away swap in the track in the current frame --> need fix it
            # if first score is 0 --> probably a far away swap in the track in the current frame --> need fix it in current
            if decision is None and ((cells_present_in_t_cur_but_absent_in_t_minus_1_score[missing] < 0.5 or
                                      cells_present_in_t_cur_but_absent_in_t_plus_1_score[missing] < 0.5)):
                # label_id = np.argwhere(map_tracks_n_label_t_cur[0]==missing)

                print('decision scores', cells_present_in_t_cur_but_absent_in_t_minus_1_score[missing],
                      cells_present_in_t_cur_but_absent_in_t_plus_1_score[missing])

                label_id = map_tracks_n_label_t_cur[jjj][1]

                # for val in map_tracks_n_label_t_cur[0]:
                #     if val == missing:
                #         print('found')
                #         break
                # en effet ça n'y est pas mais pkoi

                # why is cell not found ??
                # this is not what it's supposed to be
                # try:

                print(missing, '--> ', r_g_b_from_rgb(missing))
                print(map_tracks_n_label_t_cur.shape)
                # why empty ???? --> should not be so!!!
                print(label_id)
                # print('track_t_cur==missing',track_t_cur==missing,'track_t_cur==missing')
                # check that I have no bug otherwise it will be complex
                rps = rps_t_cur[label_id - 1]
                bbox = rps.bbox
                bbox = [bbox[0] - 50, bbox[1] - 50, bbox[2] + 50, bbox[3] + 50]
                # Bounding box ``(min_row, min_col, max_row, max_col)
                # do a crop of the region of interest

                print(bbox)
                try:

                    plot_n_images_in_line(int24_to_RGB(track_t_minus_1[bbox[0]:bbox[2], bbox[1]:bbox[3]]),
                                          int24_to_RGB(track_t_cur[bbox[0]:bbox[2], bbox[1]:bbox[3]]),
                                          int24_to_RGB(track_t_plus_1[bbox[0]:bbox[2], bbox[1]:bbox[3]]),
                                          matching_neighorhood_score_t_cur_with_t_minus_1[bbox[0]:bbox[2],
                                          bbox[1]:bbox[3]],
                                          matching_neighorhood_score_t_cur_with_t_plus_1[bbox[0]:bbox[2],
                                          bbox[1]:bbox[3]], title=str(decision) + ' ' + str(r_g_b_from_rgb(missing)))
                except:
                    traceback.print_exc()
                    print('diplay error --> ifnore for naow')
                # except:
                #     pass
                # TODO get the region of the cell and crop the image so that I can see it
                # I could crop the image

        # would be great to see the cells just to see if things are indeed ok and or fix the stuff otherwise

        # show and center on cell

        # start from cells with lowest score and then progressively go up
        # by default new cells will have the lowest score they are either overseg in current or non segmented cells in previous, if we assume that the first image is good, perfectly corrected then the cell is definitely an overseg in cur
        # TODO do a brain that takes decision
        # the brain could also generate various options

        # cells_and_their_neighbors_minus_1 = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_minus_1)
        # cells_and_their_neighbors_cur = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_cur)
        # cells_and_their_neighbors_plus_1

        # TODO I could try to reidentify swapped cells --> by default they should be close
        # TO MAP THE CELL I COULD USE THE TRANSLATION BETWEEN THE TWO IMAGES BASED ON ITS NEIGHBOR CELLS TO IDENTIFY IT BACK --> VERY SMART IDEA AND SO EASY TO IMPLEMENT
        # --> TODO MAYBE CHECK AND DO THIS CODE
        # AS LONG AS I IMPROVE SCORE I CAN CONTINUE SWAPPING
        # I NEED CHANGE THE ID IN THE CURRENT CELL AND ALL ITS NEIGHBORS
        # WHEN I THINK I HAVE IDENTIFIED A SWAPPING I NEED CHECK JUST IN THE NEIGHBORHOOD --> ONE OR TWO ROWS AROUND THE LOST CELL --> VERY GOOD IDEA IN FACT

        # need two mappings --> one on orig and one on tracking --> fairly good idea in fact!!!
        # je pense que le neighborhood est la solution

        # ça marche pas mal --> ça attire l'attention  sur les cellules qui changent de voisin --> marche plutot bien en fait !!!
        # low score can be due to a division or to a new cell
        # if cell has low score and is adjacent to a new cell then most likely a division
        # else this can highlight T1s indeirectly !!!

        # maybe if this is combined to lost cells and smart algo I can reach something really magic
        # TODO --> do compare all --> TODO --> REALLY FAST !!!
        # this is really good because it can also identify swapping whereas the other algos cannot --> try combine this to other to differentiate swapping from other features and maybe use this to identify death and divisions
        # TODO finish that soon

        # when match is poor --> try find best match

        # compute a score for every cell --> shall I reconvert it to a dict to ease the comparison ????
        # or create

        plot_n_images_in_line(int24_to_RGB(track_t_minus_1), int24_to_RGB(track_t_cur), int24_to_RGB(track_t_plus_1),
                              matching_neighorhood_score_t_cur_with_t_minus_1,
                              matching_neighorhood_score_t_cur_with_t_plus_1)

        # in some case it can be useful to check even more frames
        # for example

        # this can also be done using label and regionprops --> see what is the best choice for this
        # in a way I could get properties only of the interesting cells
        # check the intersections I need

        # overseg is smthg that is present on the middle frame and not on the one before and after
        # an underseg is something that is absent from middle image and present in the two others # pb is that since tracks don't match I will not necessarily detect it but I may be able to detect it based on neighborhood but the algo need be super smart --> see if I can find a way to do that !!!

        # underseg could be smthg present in the two fisrt and not in the last --> but some apoptotic cells would be misidentified --> also these errors most likely require human intervention
        # could do a code that follows the error and lets the user decide what TODO
        # check also swappings and see if I can fix them!!!!

        # I could pass underseg to the next
        # can I identify a cell based on its neighbors --> most likely yes and if so I should be able to fix most errors --> provided there are little errors

        # print missing between consecutive frames
        missing_between_t_cur_and_t_minus_1 = np.setdiff1d(track_t_cur, track_t_minus_1)  # potentially underseg
        missing_between_t_cur_and_t_plus_1 = np.setdiff1d(track_t_cur, track_t_plus_1)  # potentially overeseg

        # these are true overseg
        overseg_t_cur = np.intersect1d(missing_between_t_cur_and_t_minus_1,
                                       missing_between_t_cur_and_t_plus_1)

        # can I also get infos from partial intersections between two consecutive frames --> most likely yes
        # for example if a cell is newly appearing in last and absent in the middle and if its neigbors match that in the first then there is likely to be an underseg that need be fixed !!!
        # check and reimplement my swapping algo to see how I can deal with that

        # do checks only if things are missing or if some pbs are encountered --> if so --> then fix them!!!

        # NB IN FACT AN OVERSEG IN CUR CAN BE DUE TO AN UNDERSEG IN PREVIOUS --> KEEP IN MIND AND TRY TO DESIGN SOME SUPRA INTELLIGENT CODE TO FIX THAT!!!

        # ça a l'air de marcher mais faut vraiment que le tracking soit perfect --> see how
        # maybe I can do an hybrid correction where the soft detects the error but the user is asked to fix it or to chose between automated solutions --> TODO !!!
        print('overseg_t_cur', overseg_t_cur)
        # qd meme ça marche pas mal!!!!

        # TODO try show the overseg cells or save them so that I can check by myself
        # do a plotter that plots 3 images + the ovgerseg cells

        # in fact I may have smarter rules --> for example if something is present in two consecutive images it is likely to be present in the previous one too!!!
        # see the best way TODO that
        # TODO also identify swappings

        overseg_mask = np.zeros_like(track_t_cur)
        overseg_mask[track_t_cur == 0xFFFFFF] = 64

        for missing in overseg_t_cur:
            overseg_mask[track_t_cur == missing] = 255

        for missing in missing_between_t_cur_and_t_minus_1:
            overseg_mask[track_t_cur == missing] = 128

        for missing in missing_between_t_cur_and_t_minus_1:
            overseg_mask[
                track_t_cur == missing] = 192  # practically these are all the same cells as in overseg_t_cur --> DO I HAVE A BUG ????

        # ça parait realiste mais checker usr un vrai example avec 3 images

        # plt.imshow(overseg_mask)
        # plt.show()
        plot_n_images_in_line(int24_to_RGB(track_t_minus_1), int24_to_RGB(track_t_cur), int24_to_RGB(track_t_plus_1),
                              overseg_mask)

        underseg_mask = np.zeros_like(track_t_cur)

        # print missing between consecutive frames

        # nb also don't forget that the intersections can be done both ways and that some info is stored in there too !!! --> TODO do fix all!!!
        lost_cells_between_t_minus_1_and_current = np.setdiff1d(track_t_minus_1, track_t_cur)  # potentially underseg
        lost_cells_between_t_plus_1_and_current = np.setdiff1d(track_t_plus_1, track_t_cur)  # potentially overeseg

        for missing in lost_cells_between_t_minus_1_and_current:
            underseg_mask[track_t_minus_1 == missing] = 255

        for missing in lost_cells_between_t_plus_1_and_current:
            underseg_mask[track_t_plus_1 == missing] = 128

        underseg_mask[track_t_cur == 0xFFFFFF] = 64

        # there is indeed important infos in that too
        # if cells do really match well --> then I could check them for sure

        # shall I try a resegmentation !!!

        # TODO maybe take a simpler example than in the ovipo because it's too complex

        # ça parait realiste mais checker usr un vrai example avec 3 images

        # plt.imshow(overseg_mask)
        # plt.show()
        plot_n_images_in_line(int24_to_RGB(track_t_minus_1), int24_to_RGB(track_t_cur), int24_to_RGB(track_t_plus_1),
                              underseg_mask)

        # these are true overseg
        # overseg_t_cur = np.intersect1d(missing_between_t_cur_and_t_minus_1,
        #                                missing_between_t_cur_and_t_plus_1)

        #  should I care about the common cells --> yes maybe if they do not share the same neighbors --> in that case this most likely means they are swapped -->

        # then I could fix them
        # in fact I could detect neighborhood only for the common cells to see if ok
        #

        #
        #
        # print(overseg_t_cur)
        #
        # common_cells = np.intersect1d(cells_in_current, cells_in_prev)
        # print(common_cells)
        # # print("Unique values in array1 that are not in array2:")
        # lost_cells =np.setdiff1d(cells_in_current, cells_in_prev)
        # print(lost_cells)

        # that is ok all of this is super fast --> now try to do it for real and for the magic of it

        # maybe also store seg/labels  so that I don't need redo them
        # for all diverging cells I could check them and do all sorts to stuff with them

        # can I correct for swapping too

        # not bad --> I could then see if the cell comes back or vanishes
        # I could maybe compare neighbours TODO magics

        # TODO --> see what I can do with that !!!

        # lost cells and new cells

        # pas mal en fait

        # avec ça je dois avoir à peu pres tout ce que je veux

        # maybe get bonds coords too using coords rather than images --> in a way that is a good idea

        # TODO maybe also associate bonds to cells --> can eb useful and not that hard to do --> then I have evrything to do my desired code

        #

        # not bad in fact generate all the things I need and just get the desired output

        # really not bad !!!

        # ou bien associer les neighbors à une cellule

        # loop pour chaque cellule

        # nb one can call a numba function this way --> very easy in fact
        # sum_array_numba = jit()(sum_array)

        # plt.imshow(vertices)
        # plt.show()
        # bonds = find_bonds(img) # chaque passage double le temps --> a faire que si necessaire donc
        # plt.imshow(bonds)
        # plt.show()

        # we do have our three files --> generate something useful out of them
        #
        # for file_to_read in files_to_read:
        #     try:
        #         tracked_cells_resized, TA_path = smart_name_parser(file_to_read,        ordered_output=['TA', 'tracked_cells_resized.png'])
        #     except:
        #         tracked_cells_resized = TA_path = None
        #     print(tracked_cells_resized, TA_path)

        print('files are useful', files_to_read)

# TODO very dirty --> clean it but very powerful!!!
def createTAmask(orig, mask):
    # if len(orig.shape)!=3:
        # print(orig)
    red =orig[...,0]
    red[mask[...,0]!=0]=255
    orig[...,0]=red
    return orig

# TODO
def help_user_correct_errors(files, channel=None, progress_callback=None):
    for iii in range(len(files)):

        try:
            if early_stop.stop:
                return
            if progress_callback is not None:
                progress_callback.emit(int((iii / len(files)) * 100))
            else:
                print(str((iii / len(files)) * 100) + '%')
        except:
            pass

        # file_t0 = files[iii-1]
        # file_t1 = files[iii]
        #
        # print(file_t0, file_t1)

        # TODO instead of getting 3 files --> just make it able to get n files with specific rules --> TODO
        files_to_read = get_n_files_from_list(files, iii, -1, 1)
        # print(files_to_read)

        # TODO handle first ad last

        if files_to_read[0] is None or files_to_read[len(files_to_read) - 1] is None:
            # first or last image missing --> skipping for now but generate code at some point
            print('missing files', files_to_read)

            # in fact here I could make use of the data of two consecutive images
            # or should I try directly with 3 images ????

            # think about it
            continue

        # for file_to_read in files_to_read:
        extension = '.tif'  # '.png' # '.tif' # '.png' # '.tif'
        TA_path_minus_1, track_t_minus_1 = smart_name_parser(files_to_read[0],
                                                             ordered_output=['TA', 'tracked_cells_resized' + extension])
        TA_path_cur, track_t_cur = smart_name_parser(files_to_read[1],
                                                     ordered_output=['TA', 'tracked_cells_resized' + extension])
        TA_path_plus_1, track_t_plus_1 = smart_name_parser(files_to_read[2],
                                                           ordered_output=['TA', 'tracked_cells_resized' + extension])

        # print(tracked_cells_resized)
        track_t_minus_1 = RGB_to_int24(Img(track_t_minus_1))
        track_t_cur = RGB_to_int24(Img(track_t_cur))
        track_t_plus_1 = RGB_to_int24(Img(track_t_plus_1))


        # show masks too
        _, handCorrection_t_minus_1 = smart_name_parser(files_to_read[0],
                                                             ordered_output=['TA', 'handCorrection.tif'])
        _, handCorrection_t_cur = smart_name_parser(files_to_read[1],
                                                     ordered_output=['TA', 'handCorrection.tif'])
        _, handCorrection_t_plus_1 = smart_name_parser(files_to_read[2],
                                                           ordered_output=['TA', 'handCorrection.tif'])

        TA_path_handCorrection_t_minus_1 =handCorrection_t_minus_1
        TA_path_handCorrection_t_cur=handCorrection_t_cur
        TA_path_handCorrection_t_plus_1=handCorrection_t_plus_1

        # TODO really do the mask à la TA
        handCorrection_t_minus_1 = Img(handCorrection_t_minus_1)
        handCorrection_t_cur = Img(handCorrection_t_cur)
        handCorrection_t_plus_1 = Img(handCorrection_t_plus_1)
        if len(handCorrection_t_minus_1.shape)==3:
            handCorrection_t_minus_1=handCorrection_t_minus_1[...,0]
        if len(handCorrection_t_cur.shape)==3:
            handCorrection_t_cur=handCorrection_t_cur[...,0]
        if len(handCorrection_t_plus_1.shape)==3:
            handCorrection_t_plus_1=handCorrection_t_plus_1[...,0]

        original_t_minus_1 = Img(files_to_read[0])
        original_t_cur = Img(files_to_read[1])
        original_t_plus_1 = Img(files_to_read[2])

        if channel is not None:
            # we reensure image has channel otherwise skip
            if len(original_t_minus_1.shape) > 2:
                original_t_minus_1=original_t_minus_1[...,channel]
            if len(original_t_cur.shape) > 2:
                original_t_cur=original_t_cur[...,channel]
            if len(original_t_plus_1.shape) > 2:
                original_t_plus_1=original_t_plus_1[...,channel]


        # find_vertices(img)

        # could compare this to the other
        # vertices_t_minus_1 = find_vertices(track_t_minus_1, return_vertices=True, return_bonds=False)
        # vertices_t_cur = find_vertices(track_t_cur, return_vertices=True, return_bonds=False)
        # vertices_t_plus_1 = find_vertices(track_t_plus_1, return_vertices=True, return_bonds=False)
        vertices_t_minus_1 = np.stack(np.where(track_t_minus_1 == 0xFFFFFF),axis=1)  # find_vertices(track_t_minus_1, return_vertices=True, return_bonds=False)
        vertices_t_cur = np.stack(np.where(track_t_cur == 0xFFFFFF),axis=1)  # find_vertices(track_t_cur, return_vertices=True, return_bonds=False)
        vertices_t_plus_1 = np.stack(np.where(track_t_plus_1 == 0xFFFFFF),axis=1)  # find_vertices(track_t_plus_1, return_vertices=True, return_bonds=False)

        # plt.imshow(vertices)
        # plt.show()

        # print(get_vx_neighbors(vertices, RGB24img=img)) # is that slow??? --

        # get_vx_neighbors(vertices_t_minus_1,                     RGB24img=track_t_minus_1)  # ça marche et pas trop long par contre ça me donne juste la correspondance one to one avec le vx array ce qui est peut etre ce que je veux d'ailleurs mais faut pas changer l'ordre

        # ce truc est slow --> can I speed up using vertices --> most likely yes
        # print(np.unique(np.asarray(associate_cell_to_its_neighbors(img)), axis=0)) # really get cell neighbors # ça a l'air de marcher # maybe I can speed up by using just the vertices or not ???? --> think about it
        # print(associate_cell_to_its_neighbors(img)) # really get cell neighbors # ça a l'air de marcher # maybe I can speed up by using just the vertices or not ???? --> think about it
        # aussi voir comment associer les vertices à une seule cellule --> car très utile
        # print()  # a bit faster indeed but not outstanding

        # get cells and their neigbs for -1 and 0 and do a scoring --> will highlight potential swapping and or division and or other things in a way --> REALLY TRY THAT
        cells_and_their_neighbors_minus_1 = np.unique(
            np.asarray(associate_cell_to_its_neighbors2(vertices_t_minus_1, track_t_minus_1)), axis=0)
        cells_and_their_neighbors_cur = np.unique(
            np.asarray(associate_cell_to_its_neighbors2(vertices_t_cur, track_t_cur)), axis=0)
        cells_and_their_neighbors_plus_1 = np.unique(
            np.asarray(associate_cell_to_its_neighbors2(vertices_t_plus_1, track_t_plus_1)), axis=0)

        cells_and_their_neighbors_minus_1 = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_minus_1)
        cells_and_their_neighbors_cur = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_cur)
        cells_and_their_neighbors_plus_1 = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_plus_1)

        # print('cells_and_their_neighbors_minus_1', cells_and_their_neighbors_minus_1)
        # print('cells_and_their_neighbors_cur', cells_and_their_neighbors_cur)
        # print('cells_and_their_neighbors_plus_1', cells_and_their_neighbors_plus_1)

        # for all the common cells --> compute a score
        # nb of matching neighbs / total nb of neighbs in both

        # create and color

        cells_present_in_t_cur_but_absent_in_t_plus_1 = []
        cells_present_in_t_cur_but_absent_in_t_plus_1_score = {}
        matching_neighorhood_score_t_cur_with_t_plus_1 = np.zeros_like(track_t_cur, dtype=float)

        compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_plus_1,
                               cells_present_in_t_cur_but_absent_in_t_plus_1,
                               cells_present_in_t_cur_but_absent_in_t_plus_1_score,
                               matching_neighorhood_score_t_cur_with_t_plus_1, track_t_cur)

        # for cell, neigbs in cells_and_their_neighbors_cur.items():
        #     if cell in cells_and_their_neighbors_plus_1:
        #         # compute score and color cell
        #         neigbsb = cells_and_their_neighbors_plus_1[cell]
        #         score = (len(list(set(neigbs) & set(neigbsb))) * 2) / (len(neigbsb) + len(neigbs))
        #         cells_present_in_t_cur_but_absent_in_t_plus_1_score[cell]=score
        #         matching_neighorhood_score_t_cur_with_t_plus_1[track_t_cur == cell] = score
        #     else:
        #         cells_present_in_t_cur_but_absent_in_t_plus_1.append(cell)
        #         cells_present_in_t_cur_but_absent_in_t_plus_1_score[cell] = 0

        cells_present_in_t_cur_but_absent_in_t_minus_1 = []
        cells_present_in_t_cur_but_absent_in_t_minus_1_score = {}
        matching_neighorhood_score_t_cur_with_t_minus_1 = np.zeros_like(track_t_cur, dtype=float)

        compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                               cells_present_in_t_cur_but_absent_in_t_minus_1,
                               cells_present_in_t_cur_but_absent_in_t_minus_1_score,
                               matching_neighorhood_score_t_cur_with_t_minus_1, track_t_cur)
        # for cell, neigbs in cells_and_their_neighbors_cur.items():
        #     if cell in cells_and_their_neighbors_minus_1:
        #         # compute score and color cell
        #         neigbsb=cells_and_their_neighbors_minus_1[cell]
        #         score = (len(list(set(neigbs) & set(neigbsb)))*2)/(len(neigbsb)+len(neigbs))
        #         cells_present_in_t_cur_but_absent_in_t_minus_1_score[cell]=score
        #         matching_neighorhood_score_t_cur_with_t_minus_1[track_t_cur == cell]=score
        #     else:
        #         cells_present_in_t_cur_but_absent_in_t_minus_1.append(cell)
        #         cells_present_in_t_cur_but_absent_in_t_minus_1_score[cell] = 0

        # offer solutions based on scores --> TODO

        # do I have a bug here ???

        # print(type(cells_and_their_neighbors_cur), type(cells_and_their_neighbors_minus_1))  # le second est une liste mais ne devrait pas !!!! --> un bug qq part
        # test after optimization

        # ça rend le code un peu plus long mais vraiment pas deconnant --> doit etre assez facile à faire
        # lab_t_minus_1 = label(track_t_minus_1, connectivity=1, background=0xFFFFFF)
        # rps_t_minus_1 = regionprops(lab_t_minus_1)
        # lab_t_cur = label(track_t_cur, connectivity=1, background=0xFFFFFF)
        # rps_t_cur = regionprops(lab_t_cur)
        # lab_t_plus_1 = label(track_t_plus_1, connectivity=1, background=0xFFFFFF)
        # rps_t_plus_1 = regionprops(lab_t_plus_1)

        # can I read all the neighbors in a smart way now by first index of one

        # can compare neigbors for a cell in two instances --> give it a try
        # all should be easy I guess !!!

        # to get cells in one I can get unique of the first col and compare it to the other --> in fact not that hard I think
        # pb if a cell has no vertex it will be ignored
        # --> purely isolated cell

        # in fact I can simply do unique otherwise directly on the image

        # cells_in_t_plus_1 = np.unique(track_t_plus_1)
        # cells_in_t_cur, first_pixels_t_cur = get_cells_in_image_n_fisrt_pixel(track_t_cur)  # np.unique(track_t_cur)
        # cells_in_t_minus_1 = np.unique(track_t_minus_1)
        #
        # map_tracks_n_label_t_cur = map_track_id_to_label(first_pixels_t_cur, track_t_cur, lab_t_cur)
        #
        # print('first_pixels_t_cur', first_pixels_t_cur)  # this is the ravel index --> how can I convert it back ???
        # print('cells_in_t_cur', cells_in_t_cur)
        # print('map_tracks_n_label_t_cur', map_tracks_n_label_t_cur)
        #
        # # I'm ready to use this for the tracking of cells
        # # just see how fast and efficient this is
        # # do two versions, one with just showing errors and letting the user fix them  and one with other
        # if FIX_SWAPS:
        #     # DETECT SWAPS AND FIX THEM AND COMPUTE SCORE --> TODO also fix cell correspondance then --> really necessary!!!
        #     # we update both the cells and their local correspondance --> I think I have it --> just need few more crosses to identify more errors
        #     # can I use that as a tracking algo on top of the other with a minimization of the stuff and just do the coloring in the end --> TODO
        #
        #     cells_and_their_neighbors_cur, map_tracks_n_label_t_cur, last_score_reached = optimize_score(
        #         cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
        #         cells_present_in_t_cur_but_absent_in_t_minus_1_score, map_tracks_n_label_t_cur)
        #     # print('map_tracks_n_label_t_cur2',map_tracks_n_label_t_cur2)
        #     # print('map_tracks_n_label_t_cur',map_tracks_n_label_t_cur)
        #
        #     # depending on the corrections I apply I could also identify cells that have changed between -1 and 1 and that are missassigned cells in 3 and
        #     # could in fact try tracking with three images to be more efficient in fact directly because post correction will be time consuming in fact especially for swapped cells
        #     # but still i need that
        #     # do a version of the tracking using pyramidal reg that relies on that!!! --> TODO
        #
        #     # possible assignment 10240303 6824721 0.6666666666666666 (156, 65, 47) (104, 35, 17) 0.6666666666666666  --> a dividing cell --> really a good peak
        #     # 0
        #     # possible assignment 10369245 10443693 0.5714285714285714 (158, 56, 221) (159, 91, 173) 0.5714285714285714 --> une misegmented cell en bas au milieu) et une cellule adjacente qui elle est bien placee mais je suppose que faire ce changment fera baisser le core et que donc ce changement sera ignoré...
        #     # 0
        #     # possible assignment 11560212 8271832 0.5714285714285714 (176, 101, 20) (126, 55, 216) 0.5714285714285714 --> une autre misegmented cell à gauche un peu apres le centre de l'image!!! et une cellule bien trackee à coté --> pareil faire le score et voir
        #     # 0
        #     # possible assignment 12248378 7043456 1.0 (186, 229, 58) (107, 121, 128) 1.0 --> mis tracked cell
        #
        #     # cells_in_t_cur = cells_and_their_neighbors_cur.keys()
        #
        #     # TO MANUALLY FIND A CELL
        #     # for jjj, missing in enumerate(cells_in_t_cur):
        #     #     if missing == 0xFFFFFF:
        #     #         continue
        #     #
        #     #     # if missing != r_g_b_to_rgb( 4,77,109):# for sure it's a swapped cell
        #     #     #     continue
        #     #
        #     #     # the cell 9829855 is not shown properly --> it has no score --> why ?? maybe because no vx ???
        #     #     if missing == 10369245:# 11560212
        #     #         label_id = map_tracks_n_label_t_cur[jjj][1]
        #     #         rps = rps_t_cur[label_id - 1]
        #     #         bbox = rps.bbox
        #     #         # bbox = [bbox[0] - 50, bbox[1] - 50, bbox[2] + 50, bbox[3] + 50]
        #     #         # Bounding box ``(min_row, min_col, max_row, max_col)
        #     #         # do a crop of the region of interest
        #     #
        #     #         print('bbox pos of cell to assign' ,bbox)
        #
        #     # plot change
        #
        #     # print('tests of alm', 12248378 in cells_and_their_neighbors_cur.keys(), 12248378 in cells_and_their_neighbors_minus_1.keys(), 7043456 in cells_and_their_neighbors_cur.keys(), 7043456 in cells_and_their_neighbors_minus_1.keys())
        #     # False False True True --> ok in fact
        #
        #     compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
        #                            cells_present_in_t_cur_but_absent_in_t_minus_1,
        #                            cells_present_in_t_cur_but_absent_in_t_minus_1_score,
        #                            matching_neighorhood_score_t_cur_with_t_minus_1, track_t_cur)
        #
        #     # plt.imshow(matching_neighorhood_score_t_cur_with_t_minus_1)
        #     # plt.show()
        #
        # # if I really need swap then I also need

        # for all
        # print('cells_present_in_t_cur_but_absent_in_t_minus_1', cells_present_in_t_cur_but_absent_in_t_minus_1)
        # print('cells_present_in_t_cur_but_absent_in_t_plus_1', cells_present_in_t_cur_but_absent_in_t_plus_1)
        #
        # print('total score before optimization',
        #       sum(cells_present_in_t_cur_but_absent_in_t_minus_1_score.values()))  # 305.82 initially then
        # for all cells with low score if they are try a swapping but only if present before

        # NB I HAVE FOUND A SWAPPED CELL --> 37,51, 138 located at 437, 487 in 0
        # I need map one to one the label id and the track id!!!
        # anyways that is very useful

        # need map track to the label image at least for the current but also for the two others and need be able to get the coords from that !!!!

        # TODO see if I can improve score between -1 and 0 by swapping
        # see how fast/slow this could be --> just restrict myself to cells with low

        # TODO find swaps that maximize score

        # compute sum score and as long as it improves --> continue

        # remove purewhite from this array
        # for jjj, missing in enumerate(cells_in_t_cur):
        #     if missing == 0xFFFFFF:
        #         continue
        #
        #     # if missing != r_g_b_to_rgb( 4,77,109):# for sure it's a swapped cell
        #     #     continue
        #
        #     # the cell 9829855 is not shown properly --> it has no score --> why ?? maybe because no vx ???
        #     if False:
        #         if missing == 9829855:
        #             # indeed this is an isolated cell that has one or no vertices --> really need handle that at some point but ok for now
        #             label_id = map_tracks_n_label_t_cur[jjj][1]
        #             rps = rps_t_cur[label_id - 1]
        #             bbox = rps.bbox
        #             # bbox = [bbox[0] - 50, bbox[1] - 50, bbox[2] + 50, bbox[3] + 50]
        #             # Bounding box ``(min_row, min_col, max_row, max_col)
        #             # do a crop of the region of interest
        #
        #             print(bbox)
        #             try:
        #
        #                 plot_n_images_in_line(int24_to_RGB(track_t_minus_1[bbox[0]:bbox[2], bbox[1]:bbox[3]]),
        #                                       int24_to_RGB(track_t_cur[bbox[0]:bbox[2], bbox[1]:bbox[3]]),
        #                                       int24_to_RGB(track_t_plus_1[bbox[0]:bbox[2], bbox[1]:bbox[3]]),
        #                                       matching_neighorhood_score_t_cur_with_t_minus_1[bbox[0]:bbox[2],
        #                                       bbox[1]:bbox[3]],
        #                                       matching_neighorhood_score_t_cur_with_t_plus_1[bbox[0]:bbox[2],
        #                                       bbox[1]:bbox[3]],
        #                                       title='bug' + str(r_g_b_from_rgb(missing)))
        #             except:
        #                 traceback.print_exc()
        #                 print('diplay error --> ifnore for naow')
        #
        #     try:
        #         decision = take_decision(missing, cells_and_their_neighbors_minus_1, cells_and_their_neighbors_cur,
        #                                  cells_and_their_neighbors_plus_1,
        #                                  cells_present_in_t_cur_but_absent_in_t_minus_1_score,
        #                                  cells_present_in_t_cur_but_absent_in_t_plus_1_score,
        #                                  assume_previous_image_is_GT=True)
        #     except:
        #         traceback.print_exc()
        #         print('pb with decision --> contuinuing but need a fix some day !!!')
        #         continue
        #
        #     print(missing, ' --> ', decision)  # decide whether to keep or discard a cell
        #
        #     # division
        #     # death
        #     # ignore
        #     # None
        #
        #     # see unhandled cases
        #
        #     # cell 4,77,109 is clearly swapped --> can I detect it and fix it in a smart way ?????
        #
        #     # if decision is None: # --> a lot of these are artifacts --> see how to fix??? or ignore or have mire stringent rules
        #     # if decision is not None and death in decision:
        #     # there is a bug somewhere !!!
        #     # if decision is not None and not ignore in decision:
        #     # if decision is not None and (ignore in decision and (cells_present_in_t_cur_but_absent_in_t_minus_1_score[missing] != 1 or cells_present_in_t_cur_but_absent_in_t_plus_1_score[missing] != 1)): # cells ignore due to sensitivity
        #
        #     # nb some of the local swap are above 0.5 (maybe it's due to my test sample because I swapped big cells which is unlikely to happen)
        #     #
        #     # shall I start by trying to maximize score for all the cells below or equal 0.5 then do the rest of the code
        #     # most likely swapped cells would have to
        #
        #     # if score of one is 0 --> probably a far away swap in the track in the current frame --> need fix it
        #     # if first score is 0 --> probably a far away swap in the track in the current frame --> need fix it in current
        #     if decision is None and ((cells_present_in_t_cur_but_absent_in_t_minus_1_score[missing] < 0.5 or
        #                               cells_present_in_t_cur_but_absent_in_t_plus_1_score[missing] < 0.5)):
        #         # label_id = np.argwhere(map_tracks_n_label_t_cur[0]==missing)
        #
        #         print('decision scores', cells_present_in_t_cur_but_absent_in_t_minus_1_score[missing],
        #               cells_present_in_t_cur_but_absent_in_t_plus_1_score[missing])
        #
        #         label_id = map_tracks_n_label_t_cur[jjj][1]
        #
        #         # for val in map_tracks_n_label_t_cur[0]:
        #         #     if val == missing:
        #         #         print('found')
        #         #         break
        #         # en effet ça n'y est pas mais pkoi
        #
        #         # why is cell not found ??
        #         # this is not what it's supposed to be
        #         # try:
        #
        #         print(missing, '--> ', r_g_b_from_rgb(missing))
        #         print(map_tracks_n_label_t_cur.shape)
        #         # why empty ???? --> should not be so!!!
        #         print(label_id)
        #         # print('track_t_cur==missing',track_t_cur==missing,'track_t_cur==missing')
        #         # check that I have no bug otherwise it will be complex
        #         rps = rps_t_cur[label_id - 1]
        #         bbox = rps.bbox
        #         bbox = [bbox[0] - 50, bbox[1] - 50, bbox[2] + 50, bbox[3] + 50]
        #         # Bounding box ``(min_row, min_col, max_row, max_col)
        #         # do a crop of the region of interest
        #
        #         print(bbox)
        #         try:
        #
        #             plot_n_images_in_line(int24_to_RGB(track_t_minus_1[bbox[0]:bbox[2], bbox[1]:bbox[3]]),
        #                                   int24_to_RGB(track_t_cur[bbox[0]:bbox[2], bbox[1]:bbox[3]]),
        #                                   int24_to_RGB(track_t_plus_1[bbox[0]:bbox[2], bbox[1]:bbox[3]]),
        #                                   matching_neighorhood_score_t_cur_with_t_minus_1[bbox[0]:bbox[2],
        #                                   bbox[1]:bbox[3]],
        #                                   matching_neighorhood_score_t_cur_with_t_plus_1[bbox[0]:bbox[2],
        #                                   bbox[1]:bbox[3]], title=str(decision) + ' ' + str(r_g_b_from_rgb(missing)))
        #         except:
        #             traceback.print_exc()
        #             print('diplay error --> ifnore for naow')
        #         # except:
        #         #     pass
        #         # TODO get the region of the cell and crop the image so that I can see it
        #         # I could crop the image

        # would be great to see the cells just to see if things are indeed ok and or fix the stuff otherwise

        # show and center on cell

        # start from cells with lowest score and then progressively go up
        # by default new cells will have the lowest score they are either overseg in current or non segmented cells in previous, if we assume that the first image is good, perfectly corrected then the cell is definitely an overseg in cur
        # TODO do a brain that takes decision
        # the brain could also generate various options

        # cells_and_their_neighbors_minus_1 = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_minus_1)
        # cells_and_their_neighbors_cur = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_cur)
        # cells_and_their_neighbors_plus_1

        # TODO I could try to reidentify swapped cells --> by default they should be close
        # TO MAP THE CELL I COULD USE THE TRANSLATION BETWEEN THE TWO IMAGES BASED ON ITS NEIGHBOR CELLS TO IDENTIFY IT BACK --> VERY SMART IDEA AND SO EASY TO IMPLEMENT
        # --> TODO MAYBE CHECK AND DO THIS CODE
        # AS LONG AS I IMPROVE SCORE I CAN CONTINUE SWAPPING
        # I NEED CHANGE THE ID IN THE CURRENT CELL AND ALL ITS NEIGHBORS
        # WHEN I THINK I HAVE IDENTIFIED A SWAPPING I NEED CHECK JUST IN THE NEIGHBORHOOD --> ONE OR TWO ROWS AROUND THE LOST CELL --> VERY GOOD IDEA IN FACT

        # need two mappings --> one on orig and one on tracking --> fairly good idea in fact!!!
        # je pense que le neighborhood est la solution

        # ça marche pas mal --> ça attire l'attention  sur les cellules qui changent de voisin --> marche plutot bien en fait !!!
        # low score can be due to a division or to a new cell
        # if cell has low score and is adjacent to a new cell then most likely a division
        # else this can highlight T1s indeirectly !!!

        # maybe if this is combined to lost cells and smart algo I can reach something really magic
        # TODO --> do compare all --> TODO --> REALLY FAST !!!
        # this is really good because it can also identify swapping whereas the other algos cannot --> try combine this to other to differentiate swapping from other features and maybe use this to identify death and divisions
        # TODO finish that soon

        # when match is poor --> try find best match

        # compute a score for every cell --> shall I reconvert it to a dict to ease the comparison ????
        # or create

        # plot_n_images_in_line(int24_to_RGB(track_t_minus_1), int24_to_RGB(track_t_cur), int24_to_RGB(track_t_plus_1),
        #                       matching_neighorhood_score_t_cur_with_t_minus_1,
        #                       matching_neighorhood_score_t_cur_with_t_plus_1)

        # replace this by my multiviewer


        # best is to show this as a dialog and get ok or cancel !!
        # multi_display = ImgDisplayWindow(nb_rows=3, nb_cols=3)
        # multi_display.set_images(int24_to_RGB(track_t_minus_1), int24_to_RGB(track_t_cur), int24_to_RGB(track_t_plus_1),
        #                       matching_neighorhood_score_t_cur_with_t_minus_1,
        #                       matching_neighorhood_score_t_cur_with_t_plus_1)
        # multi_display.show()

        # ok but would be great to have this full screen and fullscreen size


        # TODO also show a mask overlay à la TA ??? need a channel sepcified on orig
        # see how I do that normally ???



        # print(TA_path_handCorrection_t_minus_1)

        # plt.imshow(apply_lut(matching_neighorhood_score_t_cur_with_t_minus_1,PaletteCreator().create3(PaletteCreator.DNA),convert_to_RGB=True))
        # plt.show()
        # do the mask overlap à la TA
        augment, ok = ImgDisplayWindow.display(draw_mode='pen', nb_rows=3, nb_cols=3, images=[
            # handCorrection_t_minus_1,
            #                                                                           handCorrection_t_cur,
            #                                                                           handCorre««««ction_t_plus_1,
            # createTAmask(original_t_minus_1, handCorrection_t_minus_1),
            # createTAmask(original_t_cur,handCorrection_t_cur),«
            # createTAmask( original_t_plus_1,handCorrection_t_plus_1),
            [original_t_minus_1, handCorrection_t_minus_1],
            [original_t_cur,handCorrection_t_cur],
            [original_t_plus_1,handCorrection_t_plus_1],
            int24_to_RGB(track_t_minus_1), int24_to_RGB(track_t_cur), int24_to_RGB(track_t_plus_1),
                              # apply_lut(matching_neighorhood_score_t_cur_with_t_minus_1,PaletteCreator().create3(PaletteCreator.DNA),convert_to_RGB=True),
                              apply_lut(matching_neighorhood_score_t_cur_with_t_minus_1,matplotlib_to_TA(),convert_to_RGB=True),
                              None,
                              # apply_lut(matching_neighorhood_score_t_cur_with_t_plus_1,PaletteCreator().create3(PaletteCreator.DNA),convert_to_RGB=True)], synced=True)
                              apply_lut(matching_neighorhood_score_t_cur_with_t_plus_1,matplotlib_to_TA(),convert_to_RGB=True)],
                                               labels=['image t-1','current image','image t+1',
                                                       'track t-1', 'current track', 'track t+1',
                                                       # 'matching_neighorhood_score_t_cur_with_t_minus_1', None, 'matching_neighorhood_score_t_cur_with_t_plus_1'],
                                                       'score t-1/current', None, 'score t+1/current'],
                                               # score current image _t_cur_with_t_minus_1
                                               synced=True, lst=files, cur_frame_idx=iii)

        # for the first three painters save their mask to the mask of TA --> TODO
        # PaletteCreator().create3(PaletteCreator.DNA)

        # print('ok0', ok)
        if not ok:
            return

        # this is to autosave the edited masks maybe some day also support track connection or swapping --> TODO
        mask_save = [TA_path_handCorrection_t_minus_1, TA_path_handCorrection_t_cur, TA_path_handCorrection_t_plus_1]
        for i in range(3):
            mask = augment[i].get_mask()
            if mask is not None:
                Img(mask, dimensions='hw').save(mask_save[i])


        # it


        # when the user is done --> just save the modified masks
        # need get painters out or all masks out
        # and save them


        # in some case it can be useful to check even more frames
        # for example

        # this can also be done using label and regionprops --> see what is the best choice for this
        # in a way I could get properties only of the interesting cells
        # check the intersections I need

        # overseg is smthg that is present on the middle frame and not on the one before and after
        # an underseg is something that is absent from middle image and present in the two others # pb is that since tracks don't match I will not necessarily detect it but I may be able to detect it based on neighborhood but the algo need be super smart --> see if I can find a way to do that !!!

        # underseg could be smthg present in the two fisrt and not in the last --> but some apoptotic cells would be misidentified --> also these errors most likely require human intervention
        # could do a code that follows the error and lets the user decide what TODO
        # check also swappings and see if I can fix them!!!!

        # I could pass underseg to the next
        # can I identify a cell based on its neighbors --> most likely yes and if so I should be able to fix most errors --> provided there are little errors

        # print missing between consecutive frames
        # missing_between_t_cur_and_t_minus_1 = np.setdiff1d(track_t_cur, track_t_minus_1)  # potentially underseg
        # missing_between_t_cur_and_t_plus_1 = np.setdiff1d(track_t_cur, track_t_plus_1)  # potentially overeseg
        #
        # # these are true overseg
        # overseg_t_cur = np.intersect1d(missing_between_t_cur_and_t_minus_1,
        #                                missing_between_t_cur_and_t_plus_1)

        # can I also get infos from partial intersections between two consecutive frames --> most likely yes
        # for example if a cell is newly appearing in last and absent in the middle and if its neigbors match that in the first then there is likely to be an underseg that need be fixed !!!
        # check and reimplement my swapping algo to see how I can deal with that

        # do checks only if things are missing or if some pbs are encountered --> if so --> then fix them!!!

        # NB IN FACT AN OVERSEG IN CUR CAN BE DUE TO AN UNDERSEG IN PREVIOUS --> KEEP IN MIND AND TRY TO DESIGN SOME SUPRA INTELLIGENT CODE TO FIX THAT!!!

        # ça a l'air de marcher mais faut vraiment que le tracking soit perfect --> see how
        # maybe I can do an hybrid correction where the soft detects the error but the user is asked to fix it or to chose between automated solutions --> TODO !!!
        # print('overseg_t_cur', overseg_t_cur)
        # # qd meme ça marche pas mal!!!!
        #
        # # TODO try show the overseg cells or save them so that I can check by myself
        # # do a plotter that plots 3 images + the ovgerseg cells
        #
        # # in fact I may have smarter rules --> for example if something is present in two consecutive images it is likely to be present in the previous one too!!!
        # # see the best way TODO that
        # # TODO also identify swappings
        #
        # overseg_mask = np.zeros_like(track_t_cur)
        # overseg_mask[track_t_cur == 0xFFFFFF] = 64
        #
        # for missing in overseg_t_cur:
        #     overseg_mask[track_t_cur == missing] = 255
        #
        # for missing in missing_between_t_cur_and_t_minus_1:
        #     overseg_mask[track_t_cur == missing] = 128
        #
        # for missing in missing_between_t_cur_and_t_minus_1:
        #     overseg_mask[
        #         track_t_cur == missing] = 192  # practically these are all the same cells as in overseg_t_cur --> DO I HAVE A BUG ????
        #
        # # ça parait realiste mais checker usr un vrai example avec 3 images
        #
        # # plt.imshow(overseg_mask)
        # # plt.show()
        # plot_n_images_in_line(int24_to_RGB(track_t_minus_1), int24_to_RGB(track_t_cur), int24_to_RGB(track_t_plus_1),
        #                       overseg_mask)
        #
        # underseg_mask = np.zeros_like(track_t_cur)
        #
        # # print missing between consecutive frames
        #
        # # nb also don't forget that the intersections can be done both ways and that some info is stored in there too !!! --> TODO do fix all!!!
        # lost_cells_between_t_minus_1_and_current = np.setdiff1d(track_t_minus_1, track_t_cur)  # potentially underseg
        # lost_cells_between_t_plus_1_and_current = np.setdiff1d(track_t_plus_1, track_t_cur)  # potentially overeseg
        #
        # for missing in lost_cells_between_t_minus_1_and_current:
        #     underseg_mask[track_t_minus_1 == missing] = 255
        #
        # for missing in lost_cells_between_t_plus_1_and_current:
        #     underseg_mask[track_t_plus_1 == missing] = 128
        #
        # underseg_mask[track_t_cur == 0xFFFFFF] = 64
        #
        # # there is indeed important infos in that too
        # # if cells do really match well --> then I could check them for sure
        #
        # # shall I try a resegmentation !!!
        #
        # # TODO maybe take a simpler example than in the ovipo because it's too complex
        #
        # # ça parait realiste mais checker usr un vrai example avec 3 images
        #
        # # plt.imshow(overseg_mask)
        # # plt.show()
        # plot_n_images_in_line(int24_to_RGB(track_t_minus_1), int24_to_RGB(track_t_cur), int24_to_RGB(track_t_plus_1),
        #                       underseg_mask)
        #
        # # these are true overseg
        # # overseg_t_cur = np.intersect1d(missing_between_t_cur_and_t_minus_1,
        # #                                missing_between_t_cur_and_t_plus_1)
        #
        # #  should I care about the common cells --> yes maybe if they do not share the same neighbors --> in that case this most likely means they are swapped -->
        #
        # # then I could fix them
        # # in fact I could detect neighborhood only for the common cells to see if ok
        # #
        #
        # #
        # #
        # # print(overseg_t_cur)
        # #
        # # common_cells = np.intersect1d(cells_in_current, cells_in_prev)
        # # print(common_cells)
        # # # print("Unique values in array1 that are not in array2:")
        # # lost_cells =np.setdiff1d(cells_in_current, cells_in_prev)
        # # print(lost_cells)
        #
        # # that is ok all of this is super fast --> now try to do it for real and for the magic of it
        #
        # # maybe also store seg/labels  so that I don't need redo them
        # # for all diverging cells I could check them and do all sorts to stuff with them
        #
        # # can I correct for swapping too
        #
        # # not bad --> I could then see if the cell comes back or vanishes
        # # I could maybe compare neighbours TODO magics
        #
        # # TODO --> see what I can do with that !!!
        #
        # # lost cells and new cells
        #
        # # pas mal en fait
        #
        # # avec ça je dois avoir à peu pres tout ce que je veux
        #
        # # maybe get bonds coords too using coords rather than images --> in a way that is a good idea
        #
        # # TODO maybe also associate bonds to cells --> can eb useful and not that hard to do --> then I have evrything to do my desired code
        #
        # #
        #
        # # not bad in fact generate all the things I need and just get the desired output
        #
        # # really not bad !!!
        #
        # # ou bien associer les neighbors à une cellule
        #
        # # loop pour chaque cellule
        #
        # # nb one can call a numba function this way --> very easy in fact
        # # sum_array_numba = jit()(sum_array)
        #
        # # plt.imshow(vertices)
        # # plt.show()
        # # bonds = find_bonds(img) # chaque passage double le temps --> a faire que si necessaire donc
        # # plt.imshow(bonds)
        # # plt.show()
        #
        # # we do have our three files --> generate something useful out of them
        # #
        # # for file_to_read in files_to_read:
        # #     try:
        # #         tracked_cells_resized, TA_path = smart_name_parser(file_to_read,        ordered_output=['TA', 'tracked_cells_resized.png'])
        # #     except:
        # #         tracked_cells_resized = TA_path = None
        # #     print(tracked_cells_resized, TA_path)

        # print('files are useful', files_to_read)

# hack my viewer code to synchronize everything
# get the code back from the stuff of Amrutha and even further finish it --> would also be a nice free floating window!!!
# now try the synchronous multiview of images --> should not be that hard and do the multi view
# then compute the changes and correction using multiview stuff --> write a set of rules and see if that works and if things are missed then offer corrections --> TODO
# TODO --> do all
if __name__ == '__main__':
    start = timer()

    if True:
        # img_t_minus1 = RGB_to_int24(
        #     Img('/E/Sample_images/tracking_test/test_uncropped/200319.lif_t021/tracked_cells_resized.tif'))
        # img_t_cur = RGB_to_int24(
        #     Img('/E/Sample_images/tracking_test/test_uncropped/200319.lif_t022/tracked_cells_resized.tif'))
        img_t_minus1 = RGB_to_int24(
            Img('/E/Sample_images/sample_images_PA/mini10_fake_swaps/focused_Series012/tracked_cells_resized.tif'))
        img_t_cur = RGB_to_int24(
            Img('/E/Sample_images/sample_images_PA/mini10_fake_swaps/focused_Series014/tracked_cells_resized.tif'))
        divs = detect_divisions(img_t_cur, img_t_minus1)
        print(divs)

        # very cool set of tools by the way!!!
        border_cells = get_border_cells(img_t_cur)
        border_cells_plus_one = get_border_cells_plus_one(get_cells_and_their_neighbors_from_image(img_t_cur), border_cells, remove_border_cells=True)# pb is that it includes border cells --> is now an option

        border_cells = plot_anything_a_la_TA(img_t_cur, border_cells, color=0xFF0000)
        border_cells_plus_one = plot_anything_a_la_TA(img_t_cur, border_cells_plus_one, color=0xFFFF00)

        divisions = plot_dividing_cells_a_la_TA(img_t_cur, divs, plot_cell_outline=False) # less stringent parameters to really see all the divisions --> exclude_cells_with_size_difference_superior=12, remove_divisions_involving_a_border_cell=False)

        plt.imshow(divisions)
        plt.show()

        plot_n_images_in_line(int24_to_RGB(img_t_minus1),int24_to_RGB(img_t_cur), divisions, int24_to_RGB(border_cells), int24_to_RGB(border_cells_plus_one))
        # plt.imshow(divisions)
        # plt.show()

        import sys

        sys.exit(0)

    if False:
        # NB THERE IS SOMETHING VERY INTERESTING IN THIS OVERLAP METHOD AND IN ADDITION IT IS VERY FAST --> REALLY CAN BE USEFUL MAYBE ALSO TO RESTRICT TO SOMETHING REALISTIC THE CELLS

        # what if I do an and between the two images --> I can see if its color is fitting --> if not could try to find a better match --> maybe a good idea

        # and_img = np.logical_and(RGB_to_int24(Img('/E/Sample_images/tracking_test/test_uncropped/200319.lif_t021/tracked_cells_resized.tif')), RGB_to_int24(Img('/E/Sample_images/tracking_test/test_uncropped/200319.lif_t022/tracked_cells_resized.tif')))
        # and_img = np.logical_and(RGB_to_int24(Img('/E/Sample_images/tracking_test/test_uncropped/200319.lif_t021/tracked_cells_resized.tif')), RGB_to_int24(Img('/E/Sample_images/tracking_test/test_uncropped/200319.lif_t022/tracked_cells_resized.tif')))
        img_t_minus1 = RGB_to_int24(
            Img('/E/Sample_images/tracking_test/test_uncropped/200319.lif_t021/tracked_cells_resized.tif'))
        img_t_cur = RGB_to_int24(
            Img('/E/Sample_images/tracking_test/test_uncropped/200319.lif_t022/tracked_cells_resized.tif'))

        # find_best_matches_both_ways(img_t_cur, img_t_another)

        and_img = np.copy(img_t_cur)
        and_img[img_t_cur != img_t_minus1] = 0
        # and_img[img_t_cur==img_t_minus1]=img_t_minus1[img_t_cur==img_t_minus1]
        plt.imshow(int24_to_RGB(and_img))
        plt.show()

        # matching

        # maybe I could use this trick to get the translation
        xor_img = np.zeros_like(img_t_cur)  # np.copy(img_t_cur)
        # xor_img[img_t_cur==img_t_minus1]=0
        xor_img[img_t_cur != img_t_minus1] = img_t_minus1[img_t_cur != img_t_minus1]

        # img_t_cur[xor_img!=0]=xor_img[xor_img!=0]

        plt.imshow(int24_to_RGB(xor_img))
        plt.show()

        # maybe use that to find the most frequent value !!!!
        # counts = np.bincount(a)
        # print(np.argmax(counts))

        # print(xor_img==0)

        # merge = np.copy(xor_img)
        # # merge[and_img+xor_img)!=0]=and_img[(and_img+xor_img)!=0]
        # # merge[xor_img!=0]=xor_img[xor_img!=0]
        # # merge[xor_img!=0]=xor_img[(and_img+xor_img)!=0]
        # merge[xor_img==0]=and_img[xor_img==0]
        # plt.imshow(int24_to_RGB(merge))
        # plt.show()
        # if I combine both images I should get the most likely id for every cell

        Img(int24_to_RGB(xor_img), dimensions='hwc').save('/E/Sample_images/Consensus_learning/gray/xor_img.tif',
                                                          mode='raw')

        Img(int24_to_RGB(and_img), dimensions='hwc').save('/E/Sample_images/Consensus_learning/gray/and_img.tif',
                                                          mode='raw')
        Img(int24_to_RGB(img_t_cur), dimensions='hwc').save('/E/Sample_images/Consensus_learning/gray/img_t_cur.tif',
                                                            mode='raw')
        Img(int24_to_RGB(img_t_minus1), dimensions='hwc').save(
            '/E/Sample_images/Consensus_learning/gray/img_t_minus1.tif', mode='raw')

        xor_img = xor_img & and_img

        print(xor_img.shape)
        # xor_img[xor_img==0]=and_img[xor_img==0]
        Img(int24_to_RGB(xor_img), dimensions='hwc').save('/E/Sample_images/Consensus_learning/gray/comb.tif',
                                                          mode='raw')
        # la somme redonne l'image t-1 --> why not t0

        # if a cell is almost full black
        # I can do xor to find non matching color
        # that is an easy and fast trick to find possible match for a cell or when cells do not match -->

        print('total time',
              timer() - start)

        # --> seven secs
        import sys

        sys.exit(0)

    if False:

        # --> ok c'est ce truc que je dois hacker

        # files = loadlist('/E/Sample_images/sample_images_denoise_manue/210312_armGFP_line2_suz_39h30APF/predict/predict_model_nb_0/list.lst')
        # files = loadlist('/E/Sample_images/sample_images_PA/mini10/liste.lst')
        files = loadlist(
            '/E/Sample_images/sample_images_PA/mini10_fake_swaps/liste.lst')  # fake manually added swaped cells
        # files = loadlist('/E/Sample_images/tracking_test/old') # very good for testing cause a lot of errors and a few moderate swap # maybe too many errors to start with
        print(files)

        # TODO try restrict in clone only

        # /usr/local/bin/python3.7 /home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/tracking/tracking_error_detector_and_fixer.py
        # ['/E/Sample_images/sample_images_PA/mini/focused_Series012.png', '/E/Sample_images/sample_images_PA/mini/focused_Series014.png', '/E/Sample_images/sample_images_PA/mini/focused_Series015.png', '/E/Sample_images/sample_images_PA/mini/focused_Series016.png', '/E/Sample_images/sample_images_PA/mini/focused_Series018.png', '/E/Sample_images/sample_images_PA/mini/focused_Series019.png']
        # [('/E/Sample_images/sample_images_PA/mini/focused_Series014.png', '/E/Sample_images/sample_images_PA/mini/focused_Series014'), ('/E/Sample_images/sample_images_PA/mini/focused_Series014.png', '/E/Sample_images/sample_images_PA/mini/focused_Series014')]
        #
        # Process finished with exit code 0

        track_cells(files)

    if True:

        # ça marche super --> appliquer ça à l'ovipo !!! --> TODO
        import sys
        app = QApplication(sys.argv)
        # --> ok c'est ce truc que je dois hacker

        # files = loadlist('/E/Sample_images/sample_images_denoise_manue/210312_armGFP_line2_suz_39h30APF/predict/predict_model_nb_0/list.lst')
        # files = loadlist('/E/Sample_images/sample_images_PA/mini10/liste.lst')
        files = loadlist(
            '/E/Sample_images/sample_images_PA/mini10_fake_swaps/liste.lst')  # fake manually added swaped cells
        # files = loadlist('/E/Sample_images/tracking_test/old') # very good for testing cause a lot of errors and a few moderate swap # maybe too many errors to start with
        print(files)

        # TODO try restrict in clone only

        # /usr/local/bin/python3.7 /home/aigouy/mon_prog/Python/epyseg_pkg/personal/pyTA/tracking/tracking_error_detector_and_fixer.py
        # ['/E/Sample_images/sample_images_PA/mini/focused_Series012.png', '/E/Sample_images/sample_images_PA/mini/focused_Series014.png', '/E/Sample_images/sample_images_PA/mini/focused_Series015.png', '/E/Sample_images/sample_images_PA/mini/focused_Series016.png', '/E/Sample_images/sample_images_PA/mini/focused_Series018.png', '/E/Sample_images/sample_images_PA/mini/focused_Series019.png']
        # [('/E/Sample_images/sample_images_PA/mini/focused_Series014.png', '/E/Sample_images/sample_images_PA/mini/focused_Series014'), ('/E/Sample_images/sample_images_PA/mini/focused_Series014.png', '/E/Sample_images/sample_images_PA/mini/focused_Series014')]
        #
        # Process finished with exit code 0

        # TODO need handle tracking and need break buttons to stop the loop
        help_user_correct_errors(files)
        print('total time',
              timer() - start)  # --> total time 4.510682744003134 in numba vs 740 secs (forever) otherwise !!!  --> too bad I didn't find this earlier --> that would have increased the speed of my code so much

        # sys.exit(app.exec_())
        sys.exit(0)



    print('total time',
          timer() - start)  # --> total time 4.510682744003134 in numba vs 740 secs (forever) otherwise !!!  --> too bad I didn't find this earlier --> that would have increased the speed of my code so much
