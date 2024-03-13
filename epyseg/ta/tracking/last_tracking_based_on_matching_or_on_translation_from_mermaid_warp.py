import os.path
import traceback

import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import regionprops
import numpy as np
from epyseg.img import Img, RGB_to_int24, int24_to_RGB
from timeit import default_timer as timer

# if the stuff has been done by the module then
from epyseg.ta.colors.colorgen import get_forbidden_colors_int24
from epyseg.ta.tracking.registration import pre_register_images, apply_translation
from epyseg.ta.tracking.tools import assign_random_ID_to_missing_cells, get_TA_file, first_image_tracking
from epyseg.ta.tracking.tracking_error_detector_and_fixer import find_vertices, associate_cell_to_its_neighbors2, \
    associate_cells_to_neighbors_ID_in_dict, compute_neighbor_score, optimize_score, get_cells_in_image_n_fisrt_pixel, \
    map_track_id_to_label, apply_color_to_labels

# TODO also finalize this tracking code and offer it as a registration
# also offer it as a check for segmentation --> try run the wshed locally with that

# TODO --> if two or more cells match then maybe try to find the best fit
# --> a good idea --> maybe could use area or cell elongation as the main criterion


# def sort_rps_by_area(rps):
# --> TODO add the possibility to warp
# also do a code to reconnect tracks that I may also use to MT the tracking --> I may need to know how to handle shared colored cell list or do two by two and do a code that can handle the junction of al the two by two pairs --> TODO
# indeed try to do that
# hack the other tracking code so that it does not need the cells
# TODO add checks...
# I still need to have and store all the cells to avoid duplicating IDs --> also need check that this is done properly ...

# le code doit surement etre ok --> à finaliser... et à tester
# en fait le code peut aussi etre utilisé pour de la segmentation  --> à essayer...
# --> warper une segmentation existante et l'appliquer au suivant et si une autre seg est dispo alors voir si on peut prendre le meilleur des deux mondes...

# TODO --> also finalize the code so that it can be used on a regular image to track
# if first image --> need generate the iamge then I need to be sure whether the stuff is working or not
# I need to have the already assigned colors passed in too
from epyseg.tools.early_stopper_class import early_stop
from epyseg.utils.loadlist import loadlist

def match_by_max_overlap(name_t1, name_t0, channel_of_interest=None, assigned_IDs=[], recursive_assignment=True, warp_using_mermaid_if_map_is_available=True, pre_register=True):
    """
    Match cells between two timepoints based on maximizing overlap.

    Args:
        name_t1 (str): Filename of the first timepoint.
        name_t0 (str): Filename of the second timepoint.
        channel_of_interest (str): Name of the channel of interest.
        assigned_IDs (list): List of already assigned cell IDs.
        recursive_assignment (bool): Flag indicating whether to perform recursive assignment.
        warp_using_mermaid_if_map_is_available (bool): Flag indicating whether to warp using Mermaid if a map is available.
        pre_register (bool): Flag indicating whether to perform pre-registration.

    Returns:
        None
    """

    # Get the filename of the first timepoint without the extension
    filename1_without_ext = os.path.splitext(name_t1)[0]

    # Load the hand-corrected mask for the first timepoint
    mask_t1 = Img(os.path.join(filename1_without_ext, 'handCorrection.tif'))

    # Load the warped and resized cell tracks for the second timepoint
    int_24_warped_track_t0 = RGB_to_int24(Img(os.path.join(os.path.splitext(name_t0)[0], 'tracked_cells_resized.tif')))

    if warp_using_mermaid_if_map_is_available:
        try:
            from personal.mermaid.deep_warping_uing_mermaid_minimal import warp_image_directly_using_phi

            filename0_without_ext = os.path.splitext(name_t0)[0]

            if os.path.exists(os.path.join(filename0_without_ext, 'mermaid_map.tif')):
                # Warp the cell tracks using Mermaid if the map is available
                warp_map = Img(os.path.join(filename0_without_ext, 'mermaid_map.tif'))
                int_24_warped_track_t0 = warp_image_directly_using_phi(int_24_warped_track_t0, warp_map, pre_registration=name_t0 if pre_register else None).astype(np.uint32)
            else:
                if pre_register:
                    # Perform pre-registration if specified
                    trans_dim_0, trans_dim_1 = _pre_reg(name_t0, name_t1, channel_of_interest)
                    int_24_warped_track_t0 = apply_translation(int_24_warped_track_t0, -trans_dim_0, -trans_dim_1)
        except:
            print("Mermaid registration failed... continuing without")
            if pre_register:
                # Perform pre-registration if specified
                trans_dim_0, trans_dim_1 = _pre_reg(name_t0, name_t1, channel_of_interest)
                int_24_warped_track_t0 = apply_translation(int_24_warped_track_t0, -trans_dim_0, -trans_dim_1)
    else:
        if pre_register:
            # Perform pre-registration if specified
            trans_dim_0, trans_dim_1 = _pre_reg(name_t0, name_t1, channel_of_interest)
            int_24_warped_track_t0 = apply_translation(int_24_warped_track_t0, -trans_dim_0, -trans_dim_1)

    # Convert the mask to label image
    if len(mask_t1.shape) == 3:
        mask_t1 = mask_t1[..., 0]

    label_t1 = measure.label(mask_t1, connectivity=1, background=255)

    rps_label_t1 = regionprops(label_t1)

    track_t1 = np.zeros_like(mask_t1, dtype=np.uint64)

    matched_IDs = []

    # Iterate over the labeled regions in the first timepoint, sorted by area in descending order
    for rps in sorted(rps_label_t1, key=lambda r: r.area, reverse=True):
        if track_t1[rps.coords[0][0], rps.coords[0][1]] != 0:
            # Skip the region if it has already been assigned
            continue

        pixels = int_24_warped_track_t0[rps.coords[:, 0], rps.coords[:, 1]]
        unique, counts = np.unique(pixels, return_counts=True)

        color_to_assign = unique[np.argmax(counts)]

        if color_to_assign in matched_IDs:
            # Skip the region if the assigned color has already been matched
            continue

        matched_IDs.append(color_to_assign)

        track_t1[rps.coords[:, 0], rps.coords[:, 1]] = color_to_assign

    track_t1[label_t1 == 0] = 0xFFFFFF

    if recursive_assignment:
        # Perform recursive assignment to further identify cells
        run_swapping_correction_recursively_to_further_identify_cells(track_t1, int_24_warped_track_t0, label_t1, rps_label_t1, assigned_IDs, filename1_without_ext, MAX_ITER=15)
    else:
        # I need at least to make sure that all the black cells have been assigned an ID
        track_t1 = assign_random_ID_to_missing_cells(track_t1, label_t1, regprps=rps_label_t1,
                                                                assigned_ids=assigned_IDs)

        # Save the tracked cells for the first timepoint
        Img(int24_to_RGB(track_t1)).save(get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'), mode='raw')

def _pre_reg(name_t0, name_t1, channel_of_interest):
    """
    Perform pre-registration between two images.

    Args:
        name_t0 (str): Filename of the first image.
        name_t1 (str): Filename of the second image.
        channel_of_interest (str): Name of the channel of interest.

    Returns:
        tuple: Tuple containing the translation dimensions (trans_dim_0, trans_dim_1).
    """

    I0 = Img(name_t0)
    if channel_of_interest is not None and len(I0.shape) > 2:
        I0 = I0[..., channel_of_interest]
    I1 = Img(name_t1)
    if channel_of_interest is not None and len(I1.shape) > 2:
        I1 = I1[..., channel_of_interest]
    trans_dim_0, trans_dim_1 = pre_register_images(orig_t0=I0, orig_t1=I1)
    return trans_dim_0, trans_dim_1


def get_matched_ids(rps_t1_mask, tracks):
    """
    Get the matched cell IDs from the labeled regions.

    Args:
        rps_t1_mask (list): List of labeled regions from the first timepoint mask.
        tracks (ndarray): Array containing the cell tracks.

    Returns:
        list: List of matched cell IDs.
    """

    matched_cells_ids = []

    for iii, region in enumerate(rps_t1_mask):
        color = tracks[region.coords[0][0], region.coords[0][1]]
        if color == 0:
            # missing_cells_t1.append(iii)
            pass
        else:
            matched_cells_ids.append(color)

    return matched_cells_ids


def run_swapping_correction_recursively_to_further_identify_cells(tracks, tracked_cells_t0, labels_t1, rps_t1_mask,
                                                                  assigned_ids, filename1_without_ext, MAX_ITER=15):
    start_all = timer()
    # print('intermediate time before swap correction0', timer() - start_loop)

    # THIS IS STARTING FROM HERE THAT I SHOULD USE THE NEW SCORING AND SWAPPING ALGO --> TODO
    # Img(int24_to_RGB(tracks)).save(get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'),mode='raw')

    ##################################################################################NEO CODE

    # en fait c'est très bon j'ai juste à peaufiner pr sauver du temps de processing et aussi ajouter le score --> avec un nb max de recursion
    # prevent analysing too many times the images

    # get score and if not imrpoving anymore --> early stop and if improves get a max nb of recursion and really speed up the whole process
    # see how to better handle missing cells without reprocessing everything !!!



    # just for mapping
    track_t_cur = tracks
    track_t_minus_1 = tracked_cells_t0

    # recursion is really needed but I don't need do all that for the recursion
    # assign new cell

    # could compare this to the other
    # vertices_t_minus_1 = find_vertices(track_t_minus_1, return_vertices=True, return_bonds=False)
    vertices_t_minus_1 = np.stack(np.where(track_t_minus_1 == 0xFFFFFF), axis=1)
    # vertices_t_cur = find_vertices(track_t_cur, return_vertices=True, return_bonds=False)
    vertices_t_cur = np.stack(np.where(track_t_cur == 0xFFFFFF), axis=1)

    # print('vertices_t_minus_1',vertices_t_minus_1)
    # print('vertices_t_cur',vertices_t_cur)

    # print('sum all', np.sum(vertices_t_minus_1-vertices_t_cur))
    # somehow the two images are the same --> did I do a mistake somewhere ???

    # print(vertices_t_minus_1.shape)

    # plt.imshow(vertices_t_minus_1)
    # plt.show()

    # maybe also if too many rejected changes --> ignore
    # DO I NEED SO MUCH PYRAMIDAL REG WITH SUCH A POWERFUL ALGO ??? --> probably not ??? --> check that and really compare to the other method and clean code heavily !!!

    cells_and_their_neighbors_minus_1 = np.unique(
        np.asarray(associate_cell_to_its_neighbors2(vertices_t_minus_1, track_t_minus_1)), axis=0)
    cells_and_their_neighbors_minus_1 = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_minus_1)


    # cells_in_t_minus_1 = np.unique(track_t_minus_1)
    # print('neighbs found', cells_and_their_neighbors_minus_1[10287288])


    # il y a des bugs dans ce code --> maybe recode all
    # some of the cells are not

    initial_score = -1

    # it's better but there are still some swapping --> need be fixed
    for lll in range(MAX_ITER):
        start_loop = timer()

        # print(get_vx_neighbors(vertices, RGB24img=img)) # is that slow??? --

        # get_vx_neighbors(vertices_t_minus_1,                     RGB24img=track_t_minus_1)  # ça marche et pas trop long par contre ça me donne juste la correspondance one to one avec le vx array ce qui est peut etre ce que je veux d'ailleurs mais faut pas changer l'ordre

        # ce truc est slow --> can I speed up using vertices --> most likely yes
        # print(np.unique(np.asarray(associate_cell_to_its_neighbors(img)), axis=0)) # really get cell neighbors # ça a l'air de marcher # maybe I can speed up by using just the vertices or not ???? --> think about it
        # print(associate_cell_to_its_neighbors(img)) # really get cell neighbors # ça a l'air de marcher # maybe I can speed up by using just the vertices or not ???? --> think about it
        # aussi voir comment associer les vertices à une seule cellule --> car très utile
        # print()  # a bit faster indeed but not outstanding

        # get cells and their neigbs for -1 and 0 and do a scoring --> will highlight potential swapping and or division and or other things in a way --> REALLY TRY THAT

        cells_and_their_neighbors_cur = np.unique(
            np.asarray(associate_cell_to_its_neighbors2(vertices_t_cur, track_t_cur)), axis=0)
        cells_and_their_neighbors_cur = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_cur)


        # because this is the label of the cell and not its id ???
        # print('difference', set(cells_and_their_neighbors_cur.keys())-set(cells_and_their_neighbors_minus_1.keys())) # this difference is 0 --> but why
        # print('difference2', set(cells_and_their_neighbors_minus_1.keys())-set(cells_and_their_neighbors_cur.keys())) # this difference is 0 --> but why # both differences are 0 --> does not make any sense then
        # print('1',len(cells_and_their_neighbors_cur.keys())) # both exactly have the same size --> have I done a mistake ???
        # print('2',len(cells_and_their_neighbors_minus_1.keys()))





        # FOR DEBUG
        # print('cells_and_their_neighbors_minus_1', cells_and_their_neighbors_minus_1)
        # print('cells_and_their_neighbors_cur', cells_and_their_neighbors_cur)

        # print('neighbs found2', cells_and_their_neighbors_minus_1[10287288])
        # print('neighbs found', cells_and_their_neighbors_cur[10287288])
        # print('cells_and_their_neighbors_plus_1', cells_and_their_neighbors_plus_1)

        # for all the common cells --> compute a score
        # nb of matching neighbs / total nb of neighbs in both

        # create and color

        # cells_present_in_t_cur_but_absent_in_t_plus_1 = []
        # cells_present_in_t_cur_but_absent_in_t_plus_1_score={}
        # matching_neighorhood_score_t_cur_with_t_plus_1 = np.zeros_like(track_t_cur, dtype=float)
        #
        # compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_plus_1,cells_present_in_t_cur_but_absent_in_t_plus_1, cells_present_in_t_cur_but_absent_in_t_plus_1_score, matching_neighorhood_score_t_cur_with_t_plus_1)

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

        # matching_neighorhood_score_t_cur_with_t_minus_1 = np.zeros_like(track_t_cur, dtype=float)

        score = compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                                       cells_present_in_t_cur_but_absent_in_t_minus_1,
                                       cells_present_in_t_cur_but_absent_in_t_minus_1_score)#,score_plotting_image=matching_neighorhood_score_t_cur_with_t_minus_1)  # , cells_present_in_t_cur_but_absent_in_t_minus_1, cells_present_in_t_cur_but_absent_in_t_minus_1_score, matching_neighorhood_score_t_cur_with_t_minus_1)


        print('cells_present_in_t_cur_but_absent_in_t_minus_1', cells_present_in_t_cur_but_absent_in_t_minus_1) # --> big bug because this is empty and it does not make sense
        # plt.imshow(matching_neighorhood_score_t_cur_with_t_minus_1)
        # plt.show()

        # print('defined score', cells_present_in_t_cur_but_absent_in_t_minus_1_score[10287288])

        if initial_score < 0:
            initial_score = score
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

        # can probably store that

        # ça rend le code un peu plus long mais vraiment pas deconnant --> doit etre assez facile à faire
        # lab_t_minus_1 = label(track_t_minus_1, connectivity=1, background=0xFFFFFF)
        # rps_t_minus_1 = regionprops(lab_t_minus_1)
        lab_t_cur = measure.label(track_t_cur, connectivity=1, background=0xFFFFFF)
        # rps_t_cur = regionprops(lab_t_cur)

        # can I read all the neighbors in a smart way now by first index of one

        # can compare neigbors for a cell in two instances --> give it a try
        # all should be easy I guess !!!

        # to get cells in one I can get unique of the first col and compare it to the other --> in fact not that hard I think
        # pb if a cell has no vertex it will be ignored
        # --> purely isolated cell

        # in fact I can simply do unique otherwise directly on the image

        cells_in_t_cur, first_pixels_t_cur = get_cells_in_image_n_fisrt_pixel(track_t_cur)  # np.unique(track_t_cur)
        map_tracks_n_label_t_cur = map_track_id_to_label(first_pixels_t_cur, track_t_cur, lab_t_cur)

        # print('first_pixels_t_cur', first_pixels_t_cur)  # this is the ravel index --> how can I convert it back ???
        # print('cells_in_t_cur', cells_in_t_cur)
        # print('map_tracks_n_label_t_cur', map_tracks_n_label_t_cur)

        # I'm ready to use this for the tracking of cells
        # just see how fast and efficient this is

        # l'ancien code est bien meilleur en fait --> voir pkoi parce que le code là est plus beau et plus logique --> essayer de comprendre pkoi moins bon en fait est -ce trop stringent au niveau des scores ????
        # essayer de comprendre le pb!!!!

        FIX_SWAPS = True
        # threshold = 1.0 # 0.5 # 1.0
        threshold = 0.8  # 0.76 # 0.5 # 1.0 # really required to be above 1 to cope with concurrent drifts of several cells


        # even though there are errors it really doe not find them
        if FIX_SWAPS:
            # DETECT SWAPS AND FIX THEM AND COMPUTE SCORE --> TODO also fix cell correspondance then --> really necessary!!!
            # we update both the cells and their local correspondance --> I think I have it --> just need few more crosses to identify more errors
            # can I use that as a tracking algo on top of the other with a minimization of the stuff and just do the coloring in the end --> TODO

            cells_and_their_neighbors_cur, map_tracks_n_label_t_cur, last_score_reached = optimize_score(
                cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                cells_present_in_t_cur_but_absent_in_t_minus_1_score, map_tracks_n_label_t_cur, threshold=threshold)
            # print('map_tracks_n_label_t_cur2',map_tracks_n_label_t_cur2)
            # print('map_tracks_n_label_t_cur',map_tracks_n_label_t_cur)



            # style if improvment < 5 early stop
            # score improved from 617.7013813251964 to 692.6711356897581
            # score improved from 692.6711356897581 to 708.2236112791516
            # score improved from 708.2236112791516 to 711.5193129630884
            # score improved from 711.5193129630884 to 712.606217724993
            # score improved from 712.606217724993 to 713.5926725744915

            # max reached after 10 iter on the same just for fun it stopped at 6 and I couldn't detect any improvment --> ok in fact!!!
            # use a detector of divisions and of T1s
            # can also be used to detect intercalation maybe ???

            # could also have a minimum improvment and break if not really improving anymore
            # 1388
            # do i have a bug because no possible alignment
            if last_score_reached == initial_score:
                print(last_score_reached, initial_score)
                print('early stop', lll)
                break

            if last_score_reached > initial_score:
                print('score improved from', initial_score, 'to', last_score_reached)
                initial_score = last_score_reached
            else:
                print('score did not improve', initial_score)

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

            # check why some of the cells are swapped --> especially because it cannot imrpove the code --> see why not detected and really need change this !!!
            #
            compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
                                   cells_present_in_t_cur_but_absent_in_t_minus_1,
                                   cells_present_in_t_cur_but_absent_in_t_minus_1_score)

            # FOR DEBUG KEEP
            # matching_neighorhood_score_t_cur_with_t_minus_1 = np.zeros_like(track_t_cur, dtype=float)
            # compute_neighbor_score(cells_and_their_neighbors_cur, cells_and_their_neighbors_minus_1,
            #                        cells_present_in_t_cur_but_absent_in_t_minus_1,
            #                        cells_present_in_t_cur_but_absent_in_t_minus_1_score,
            #                        matching_neighorhood_score_t_cur_with_t_minus_1, track_t_cur)
            #
            #
            # plt.imshow(matching_neighorhood_score_t_cur_with_t_minus_1)
            # plt.show()

            # if I really need swap then I also need

            # ça marche mais du coup je dois lui faire faire la coloration maintenant

            corrected_track = apply_color_to_labels(lab_t_cur, map_tracks_n_label_t_cur)
            # plt.imshow(int24_to_RGB(corrected_track))
            # plt.show()

            corrected_track = assign_random_ID_to_missing_cells(corrected_track, labels_t1, regprps=rps_t1_mask,
                                                                assigned_ids=assigned_ids)
            # update mask!!!
            track_t_cur = corrected_track

        ##################################################################################END NEO CODE

        # # now we really do the swapping the swapping correction --> I could reuse this code in the other
        # # MEGA TODO just to speed up things here just restrict the swapping to the problematic cells and new cells could also exclude cells that match full dark of main image --> indeed could gain some time when big translations
        # vertices_t0 = detect_vertices_and_bonds(labels_t0)
        # vertices_t0 = np.where(vertices_t0 == 255)
        # vertices_t1 = detect_vertices_and_bonds(labels_t1)
        # vertices_t1 = np.where(vertices_t1 == 255)
        #
        # # ça c'est facile mais faut juste looper sur les cellules problematiques --> ajouter une liste de cellules à checker et les supprimer si ok ??? peut etre --> à faire
        # print('intermediate time before swap correction1', timer() - start_loop)
        # for fix in range(swap_fix_recursions):
        #     if __DEBUG__:
        #         print('swapping fix recursion #', fix)
        #     # print(tracked_cells_t1.shape)
        #     tracked_cells_t1 = _fix_swapping_internal(tracked_cells_t0, tracked_cells_t1, vertices_t0=vertices_t0,
        #                                               vertices_t1=vertices_t1, MIN_CUT_OFF_SWAPPING=MIN_CUT_OFF_SWAPPING)
        #     # print('-->', tracked_cells_t1.shape)
        #     # Img(tracked_cells_t1).save(get_TA_file(filename0_without_ext, 'track_after_swap_correction'+ str(fix)+ '.tif'), mode='raw')

        # example of swapped cells
        # print('total time', timer() - start_all)  # --> 47 secs --> un peu lent qd meme

        # Img(tracked_cells_t1).save(get_TA_file(filename0_without_ext, 'track_after_swap_correction.tif'), mode='raw')
        print('end loop', timer() - start_loop)

    # make sure to fix missing not found cells
    track_t_cur = assign_random_ID_to_missing_cells(track_t_cur, labels_t1, regprps=rps_t1_mask,
                                                 assigned_ids=assigned_ids)

    print(get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'))

    # NB it still contains small swapping errors I don't get --> why
    # swap first then assign and assign if not swapped --> check cell

    # celui qui m'interesse c'est le 2
    # Img(int24_to_RGB(corrected_track)).save(get_TA_file(filename1_without_ext, 'tracked_cells_resized_v2.tif'),mode='raw')
    Img(int24_to_RGB(track_t_cur)).save(get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'),mode='raw')

    # with a recursion it's very good actually and really is optimized

    # print(r_g_b_to_rgb(137,134,15))

    print('total time all recusrsions', timer() - start_all)
    # return track_t_cur

def track_first_frame(original_t0, assigned_ids, seed):
    filename0_without_ext = os.path.splitext(original_t0)[0]
    mask_t0 = Img(get_TA_file(filename0_without_ext, 'handCorrection.tif'))

    if mask_t0.has_c():
        mask_t0 = mask_t0[..., 0]
    labels_t0 = measure.label(mask_t0, connectivity=1, background=255)  # FOUR_CONNECTED
    tracked_cells_t0 = first_image_tracking(mask_t0, labels_t0, assigned_ids=assigned_ids, seed=seed)
    Img(int24_to_RGB(tracked_cells_t0), dimensions='hwc').save(
        get_TA_file(filename0_without_ext, 'tracked_cells_resized.tif'), mode='raw')  # should work --> DO the job

def match_by_max_overlap_lst(lst, channel_of_interest=None, recursive_assignment=True,
                             warp_using_mermaid_if_map_is_available=True, pre_register=True,
                             progress_callback=None):
    """
    Performs tracking and matching on a list of images.

    Args:
        lst (list): The list of images.
        channel_of_interest (None, optional): The channel of interest. Defaults to None.
        recursive_assignment (bool, optional): Whether to use recursive assignment. Defaults to True.
        warp_using_mermaid_if_map_is_available (bool, optional): Whether to warp using Mermaid if a map is available.
            Defaults to True.
        pre_register (bool, optional): Whether to perform pre-registration. Defaults to True.
        progress_callback (None, optional): A callback function for progress updates. Defaults to None.

    Returns:
        None

    Notes:
        - If the list contains only one image, it performs tracking on that image and returns.
        - Otherwise, it tracks and matches each pair of consecutive images in the list.

    # Examples:
    #     >>> images = [image1, image2, image3]
    #     >>> match_by_max_overlap_lst(images, channel_of_interest=1, recursive_assignment=False)
    """

    start_all = timer()
    seed = 1  # always start tracking with the same seed to have roughly the same color
    assigned_ids = get_forbidden_colors_int24()

    try:
        zipped_list = zip(lst, lst[1:])
    except:
        # Assume list contains only one image --> return just the first image tracking
        track_first_frame(lst[0], assigned_ids, seed)
        return

    for iii, (original_t0, original_t1) in enumerate(zipped_list):
        try:
            if early_stop.stop:
                return
            if progress_callback is not None:
                progress_callback.emit(int((iii * 100) / len(zipped_list)))
            else:
                print(str((iii * 100) / len(zipped_list)) + '%')
        except:
            pass

        if iii == 0:
            # Need to create the track for the first image then recursively go on
            track_first_frame(original_t0, assigned_ids, seed)

        match_by_max_overlap(original_t1, original_t0, channel_of_interest=channel_of_interest,
                             recursive_assignment=recursive_assignment,
                             warp_using_mermaid_if_map_is_available=warp_using_mermaid_if_map_is_available,
                             pre_register=pre_register)

    print('total time', timer() - start_all)


if __name__ == '__main__':
    import sys

    if True:
        # lst = loadlist('/E/Sample_images/sample_images_PA/mini_empty/list.lst')
        lst = loadlist('/E/Sample_images/sample_images_PA/mini (copie)/*.png')

        print(lst)
        # match_by_max_overlap_lst(lst, recursive_assignment=False, warp_using_mermaid_if_map_is_available=True) # --> 2.76 secs without recursion
        # match_by_max_overlap_lst(lst, recursive_assignment=True, warp_using_mermaid_if_map_is_available=True) # --> 21 secs with recursion vs 2.76 secs without recursion... --> *10

        # shall I assign or not lost cells
        # I should probably ask for prereg or not --> check
        # or do pre reg by default because fast and useful --> I would need the channel
        # match_by_max_overlap_lst(lst, channel_of_interest=1, recursive_assignment=True, warp_using_mermaid_if_map_is_available=False, pre_register=True) # good and does not load mermaid files yet
        match_by_max_overlap_lst(lst, channel_of_interest=1, recursive_assignment=False, warp_using_mermaid_if_map_is_available=False, pre_register=True) # good and does not load mermaid files yet
        sys.exit(0)

    # if True:
    # ça marche meme si la key est dupliquée...
    #     student_tuples = [
    #         ('john', 'A', 15),
    #         ('jane', 'B', 12),
    #         ('dave', 'B', 10),
    #         ('dan', 'C', 10),
    #         ]
    #     print(sorted(student_tuples, key=lambda student: student[2]))  # sort by age
    #
    #     sys.exit(0)

    from timeit import default_timer as timer

    start = timer()
    match_by_max_overlap(
        '/E/Sample_images/sample_images_PA/trash_test_mem/trash_registration_network/1/210930_EcadKI_mel_40-54hAPF_ON.lif - Series011_t001.tif',
        '/E/Sample_images/sample_images_PA/trash_test_mem/trash_registration_network/1/210930_EcadKI_mel_40-54hAPF_ON.lif - Series011_t000_fake.tif')
    # match_by_max_overlap(Img('/E/Sample_images/sample_images_PA/trash_test_mem/trash_registration_network/1/210930_EcadKI_mel_40-54hAPF_ON.lif - Series011_t001/handCorrection.tif'), RGB_to_int24(Img('/E/Sample_images/sample_images_PA/trash_test_mem/trash_registration_network/1/warped_mask.tif')))
    # match_by_amx_overlap(Img('/E/Sample_images/sample_images_PA/trash_test_mem/trash_registration_network/1/210930_EcadKI_mel_40-54hAPF_ON.lif - Series011_t001/handCorrection.tif'),RGB_to_int24(Img('/E/Sample_images/sample_images_PA/trash_test_mem/trash_registration_network/1/210930_EcadKI_mel_40-54hAPF_ON.lif - Series011_t000/tracked_cells_resized.tif'))) # ça marche vraiment mieux en utilisant la pre registration en fait!!!

    # need pre reg it




    print('stop', timer() - start)

    # faire une pipeline qui peut etre va generer un shifted track or un shifted --> ideally I should just put this code within mermaid --> it would be a new tracking --> that would read the cell label file and

    # finalize this
