from epyseg.img import int24_to_RGB, RGB_to_int24
from timeit import default_timer as timer
import numpy as np
from epyseg.img import Img
# really very good --> now make it recursive then done --> extract the cell from there and continue
from natsort import natsorted
import os

# parameters
from epyseg.utils.loadlist import loadlist
from epyseg.ta.tracking.rapid_detection_of_vertices import associate_vertices_to_cells, detect_vertices_and_bonds

__DEBUG__ = False



def _fix_swapping_internal(tracked_cells_t0, tracked_cells_t1, vertices_t0=None, vertices_t1=None, MIN_CUT_OFF_SWAPPING=None):
    """
    Fixes swapping of cells between two time steps.

    Args:
        tracked_cells_t0 (numpy.ndarray): Array of tracked cells at time t0.
        tracked_cells_t1 (numpy.ndarray): Array of tracked cells at time t1.
        vertices_t0 (numpy.ndarray, optional): Array of vertices at time t0. Defaults to None.
        vertices_t1 (numpy.ndarray, optional): Array of vertices at time t1. Defaults to None.
        MIN_CUT_OFF_SWAPPING (float, optional): Minimum cutoff score for swapping. Defaults to None.

    Returns:
        numpy.ndarray: Updated array of tracked cells at time t1.

    """
    if __DEBUG__:
        cell_to_check = 0x29d785  # 0x976494 #0x29d785  # 0x746502 #0x96c9ef #0x542358  # 0xf36619 #0x496bbd #0x1bf38f
        # map           9921684   2742149   7628034  9882095   5514072   15951385   4811709
        print('cell to check', cell_to_check, format(cell_to_check, 'x'))


    # inner_time = timer()
    if vertices_t0 is None:
        vertices_t0 = detect_vertices_and_bonds(tracked_cells_t0)
        vertices_t0 = np.where(vertices_t0 == 255)
    if vertices_t1 is None:
        vertices_t1 = detect_vertices_and_bonds(tracked_cells_t1)
        vertices_t1 = np.where(vertices_t1 == 255)
    # print('current time0', timer() - inner_time)

    # inner_time = timer()
    t0_cells_n_vertices = associate_vertices_to_cells(tracked_cells_t0, vertices_t0,
                                                      forbidden_colors=(0, 0xFFFFFF),
                                                      output_cells_and_their_neighbors=True)
    t1_cells_n_vertices = associate_vertices_to_cells(tracked_cells_t1, vertices_t1,
                                                      forbidden_colors=(0, 0xFFFFFF),
                                                      output_cells_and_their_neighbors=True)


    if __DEBUG__:
        print('t0_cells_n_vertices', t0_cells_n_vertices)
        print('t1_cells_n_vertices', t1_cells_n_vertices)

    # inner_time = timer()
    best_pairs = {}  # dict that contains pairs -->  a good idea
    best_pairs_score = {}

    for key, value in t0_cells_n_vertices.items():
        max_match = 0.
        best_match = 0
        neighs_at_t0 = t0_cells_n_vertices[key]
        best_pair_size = 0

        for key2, value2 in t1_cells_n_vertices.items():
            neighs_at_t1 = t1_cells_n_vertices[key2]
            common_neighs = set(neighs_at_t1).intersection(neighs_at_t0)

            # if two are equal --> try to take the best --> find a way to do it smartly
            if len(common_neighs) > 0 and len(
                    common_neighs) >= max_match:  # if found twice --> find a way to select which cells to take
                # max_match = len(common_neighs) / ((len(neighs_at_t1)+len(neighs_at_t0))/2)

                conti = True
                if len(common_neighs) == max_match:
                    if len(neighs_at_t1) == best_pair_size:
                        if key == key2:
                            # do favor identity as much as possible
                            conti = True
                        # find a trick or random
                        elif key2 in neighs_at_t0:  # not sure this is very smart but give it a try NB recursion may fix bugs
                            conti = False
                    elif len(neighs_at_t1) > best_pair_size:
                        conti = False
                    # si les 2 font la meme taille --> pb à nouveau --> y a t'il une astuce...

                if conti:
                    max_match = len(common_neighs)  # / ((len(neighs_at_t1)+len(neighs_at_t0))/2)
                    best_match = key2
                    best_pair_size = len(neighs_at_t1)
                    if __DEBUG__:
                        if key == cell_to_check:
                            hex_list_0 = [format(x, 'x') for x in neighs_at_t0]
                            hex_list_1 = [format(x, 'x') for x in neighs_at_t1]
                            print('pairs for a test', neighs_at_t0, neighs_at_t1, len(common_neighs), max_match,
                                  format(key, 'x'), format(key2, 'x'), key, key2, hex_list_0,
                                  hex_list_1)  # that works --> so indeed the best match is found but then cells are lost and swapped --> why is that --> I still miss a ctrl somewhere --> or a secondary swap that is less good somehow --> check that or a coding error!!
                else:
                    if __DEBUG__:
                        if key == cell_to_check:
                            hex_list_0 = [format(x, 'x') for x in neighs_at_t0]
                            hex_list_1 = [format(x, 'x') for x in neighs_at_t1]
                            print('discarded pairs for a test', neighs_at_t0, neighs_at_t1, len(common_neighs),
                                  max_match,
                                  format(key, 'x'), format(key2, 'x'), key, key2, hex_list_0,
                                  hex_list_1)
        if __DEBUG__:
            if key == cell_to_check:
                print('final match', max_match, cell_to_check, best_match, format(cell_to_check, 'x'),
                      format(best_match, 'x'))
        # or best --> compare to its current match and if better then take it otherwise ignore  --> maybe

        # in fact I take all here but is that really worth ??? --> maybe take one but only if the cell has just one neighbor ???? --> shall I start with 2 ??? that would dramatically reduce the nb of cells I guess --> think about it
        if max_match >= 0.5:  # take all cells that have at least half of their neighbs --> 40--> maybe ok

            # compare neighbors of current match versus those in the previous
            best_pairs[key] = best_match
            best_pairs_score[key] = max_match  # will be used later to avoid issues

    # remove cells that did not change much
    if MIN_CUT_OFF_SWAPPING is not None and MIN_CUT_OFF_SWAPPING>0:
        pairs_to_ignore = [k for k, v in best_pairs_score.items() if v <= MIN_CUT_OFF_SWAPPING]
        for ids in pairs_to_ignore:
            del best_pairs[ids]

    # finding duplicate values
    # from dictionary using set
    rev_dict = {}
    for key, value in best_pairs.items():
        rev_dict.setdefault(value, set()).add(key)

    dupes_matches_to_be_removed = []
    for key, value in rev_dict.items():

        if __DEBUG__:
            if key == cell_to_check:
                print('MEGA THERE######')

        if len(value) > 1:
            best_match = 0
            max_match = 0
            # keep only best and remove all others from the table
            for iii, val in enumerate(value):
                if __DEBUG__:
                    if key == cell_to_check:
                        print(iii, val, best_pairs_score[val])

                if best_pairs_score[val] == max_match:
                    if val == key:
                        max_match = best_pairs_score[val]
                        best_match = val
                elif best_pairs_score[val] > max_match:
                    max_match = best_pairs_score[val]
                    best_match = val

            if __DEBUG__:
                if key == cell_to_check:
                    print(value, val, max_match)

            value.remove(best_match)
            # I do have a bug here but I don't get it in fact but ok for now
            dupes_matches_to_be_removed.extend(
                value)  # not a good idea --> just need to remove it from the value not from everywhere... is that correct ???? remove it only if present again in duplicates in final stuff after cleaning

    if __DEBUG__:
        print('dupes_matches_to_be_removed', dupes_matches_to_be_removed,
              [format(x, 'x') for x in dupes_matches_to_be_removed])

    for key in dupes_matches_to_be_removed:
        if key in best_pairs:
            del best_pairs[key]

    def swap(pairs_of_cells_to_swap,
             tracked_cells_t1):  # this is super slow and would benefit from recoding it in C --> this takes most of the time of the function
        # inner_time = timer()
        clone_tracks = np.array(tracked_cells_t1, copy=True)
        # print('current time', timer() - inner_time)

        for cell1, cell2 in pairs_of_cells_to_swap.items():
            if cell1 == cell2: # remove identity cause it slows down a lot the process
                continue

            # if cell2 not in pairs_of_cells_to_swap.keys(): # very slow --> find a way to make that faster
            clone_tracks[tracked_cells_t1 == cell1] = cell2
            clone_tracks[tracked_cells_t1 == cell2] = cell1

            if __DEBUG__:
                print('converting', cell2, '--> ', cell1, format(cell2, 'x'), '-->', format(cell1, 'x'))

            # find a way todo things and to not duplicate stuff

            # pb --> this creates duplications but otherwise it does swap correct cells --> find a way to prevent this --> I need to order swaps and other stuff
        # print('current time2', timer() - inner_time)
        return clone_tracks

    # inner_time = timer()
    tracked_cells_t1 = swap(best_pairs,
                            tracked_cells_t1)  # c'est ce truc qui est super lent --> plus de la moitié du temps --> try improve that

    if __DEBUG__:
        print('best_pairs', len(best_pairs), best_pairs)  # maybe best ratio versus its own size
    # TODO maybe save swapped cells --> only show those and blacken all the rest

    return tracked_cells_t1


if __name__ == '__main__':

    swap_fix_recursions = 3

    # pb is because after translation the cell is also present in both --> skipped by my stuff --> need detect those cells and chose to get only one in fact
    # pb is because after translation the cell is also present in both --> skipped by my stuff --> need detect those cells and chose to get only one in fact
    # or first do the

    start_all = timer()

    # path = '/E/Sample_images/tracking_test/leg_uncropped'  # quick test of the tracking algo
    # path = '/E/Sample_images/tracking_test/leg_cropped'  # quick test of the tracking algo
    path = '/E/Sample_images/tracking_test/test_uncropped'  # quick test of the tracking algo
    # path = '/E/Sample_images/tracking_test/test_cropped'  # quick test of the tracking algo

    if not path.lower().endswith('.txt') and not path.lower().endswith('.lst'):
        images_to_analyze = os.listdir(path)
        images_to_analyze = [os.path.join(path, f) for f in images_to_analyze if
                             os.path.isfile(os.path.join(path, f))]  # list only files and only if they end by tif
        images_to_analyze = natsorted(images_to_analyze)
    else:
        images_to_analyze = loadlist(path)

    # get current and next in fact and map next

    # process first image directly then move on to next
    # images_to_analyze = []

    # for image 0 --> just randomly fill it with colors --> make sure no dupe and that's it

    # TODO --> do the first

    print(images_to_analyze)

    for l in range(len(images_to_analyze) - 1):

        start_loop = timer()

        file_path_0 = images_to_analyze[l]
        file_path_1 = images_to_analyze[l + 1]

        print('files', file_path_1, file_path_0)

        if not file_path_1.endswith('.tif') or not file_path_0.endswith('.tif'):
            continue

        # print('files', file_path_1, file_path_0)

        filename0_without_ext = os.path.splitext(file_path_0)[0]
        filename1_without_ext = os.path.splitext(file_path_1)[0]

        tracked_cells_t0 = Img(os.path.join(filename0_without_ext, 'tracked_cells_resized.tif'))
        tracked_cells_t1 = Img(os.path.join(filename1_without_ext, 'tracked_cells_resized.tif'))

    vertices_t0 = detect_vertices_and_bonds(RGB_to_int24(tracked_cells_t0))
    vertices_t0 = np.where(vertices_t0==255)
    vertices_t1 = detect_vertices_and_bonds(RGB_to_int24(tracked_cells_t1))
    vertices_t1 = np.where(vertices_t1 == 255)

    # time is always increasing but why is that is some set not reset
    for fix in range(swap_fix_recursions):
        print('swapping fix recursion #', fix)
        begin_loop = timer()
        tracked_cells_t1 = _fix_swapping_internal(tracked_cells_t0, tracked_cells_t1, vertices_t0=vertices_t0, vertices_t1=vertices_t1)
        # tracked_cells_t1 = _fix_swapping_internal(tracked_cells_t0, tracked_cells_t1)
        print('current time', timer() - begin_loop)
        # Img(tracked_cells_t1).save(os.path.join(filename0_without_ext, 'track_after_swap_correction'+ str(fix)+ '.tif'), mode='raw')

    # example of swapped cells
    print('total time', timer() - start_all)  # --> 47 secs --> un peu lent qd meme

    Img(int24_to_RGB(tracked_cells_t1)).save(os.path.join(filename0_without_ext, 'track_after_swap_correction.tif'), mode='raw')

    # puis je la faire dynamiquement pr eviter les swaps en live
    # --> void aussi si rentable ou pas

    # TODO -->
    # this now seems ok and could be done recursively...
    # could also use neighbor cells to reassign most likely cell --> cell that matches most of the surrounding IDs

    # pas mal --> see how many recursions might be needed to get the stuff to work --> might be time consuming to do plenty of recursion... --> maybe just update the map so that not everything needs be calculated again

    # if a cell is unsure try reassign it by its neighborood...

    # TODO --> reidentify cells by neighborhood --> simply use the vertices to detect the neighbors --> split the code
