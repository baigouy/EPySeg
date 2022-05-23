# maybe detect those big shifts and see whether I could find a better solution to handle them ???
# marche mais qd meme big big à l'interface entre les duex tissus

# pb --> par moment il y a des shifts enormes de tonnes de cellules --> peut etre faut Il un truc plus mieux

# ce code est vraiment excellent maintenant --> a good starting point then get dividing cells --> mais un peu lent mais pourrait accélerer en limitant la recherche aux cellules les plus proches (anyway long swappings do not exist)
# then find under and oversegmented cells and I'll be done...
# could maybe try to speed up by computing overlap and calculating local translation of matching pairs


# TODO do a division code --> detect new cell in an image compared to the previous and see if cell is present in next maybe then the sum of neighbors should be the closest to the parent of the cell --> find parent
# à tester --> devrait pas être difficile


# just two cells flipped --> is there any fix
# do code to dissect stuff


# nb le  regular multiswap est meilleur --> mais pkoi ??--> moins de swapping
# really excell

# that seems to work miraculously well -->just try a comparison to the other method but at least it's much smarter
# 10 secs per image

# I can easily add
# this is really very good but can it be improved and made faster by reducing the cells to be checked and applying the dual assignment
# qd même 7h de processing c long, but on the other hand this is not human time so we don't CARE
# it seems to work and even looks faster now

from epyseg.img import RGB_to_int24, int24_to_RGB
from timeit import default_timer as timer
import traceback
import numpy as np
from skimage.measure import label
from epyseg.img import Img
from epyseg.ta.colors.colorgen import get_forbidden_colors_int24
from epyseg.ta.measurements.measurements3D.get_point_on_surface_if_centroid_is_bad import point_on_surface
from epyseg.ta.tracking.registration import get_pyramidal_registration
from epyseg.ta.tracking.tools import assign_random_ID_to_missing_cells, first_image_tracking, get_list_of_files, \
    get_input_files_and_output_folders, get_TA_file, smart_name_parser
from skimage.measure import regionprops
from skimage import measure

# parameters
__DEBUG__ = False

# gain correlates with depth!!!
# KEEP  for the parameter below if cells are big one can decrease it for some images it does make a big difference!!! --> TODO SET THIS AS A
from epyseg.ta.tracking.tracking_error_detector_and_fixer import find_vertices, associate_cell_to_its_neighbors2, \
    associate_cells_to_neighbors_ID_in_dict, compute_neighbor_score, optimize_score, get_cells_in_image_n_fisrt_pixel, \
    map_track_id_to_label, apply_color_to_labels


# PYRAMIDAL_DEPTH = 3  # 2 #3 # computing pyramidal translation can be time consuming and the gain vs time cost is not always worth it
# MAX_ITER = 15  # max nb of optimizatuion loops):
from epyseg.tools.early_stopper_class import early_stop
# from personal.pyTA.tsts.new_registration_code import criss_cross_reg


def track_cells_dynamic_tissue(path, channel=None, PYRAMIDAL_DEPTH=3, MAX_ITER=15, progress_callback=None):
    start_all = timer()

    # PYRAMIDAL_DEPTH = 3  # 2 #3 # computing pyramidal translation can be time consuming and the gain vs time cost is not always worth it

    # the swapping correction is so good that it does not require deep (hence slow pyramid) to work!!!

    # still some errors
    # why

    # PYRAMIDAL_SKIP=1 #0 # skip some division step
    # MIN_CUT_OFF_SWAPPING = None  # only swap/assign cells if they have a high score/high match
    # swap_fix_recursions = 3
    seed = 1  # always start tracking with same seed to have roughly the same color
    assigned_ids = get_forbidden_colors_int24()

    # TODO add forbidden colors to it
    images_to_analyze = get_list_of_files(path)

    # print(range(len(images_to_analyze) - 1))

    # if list is empty need enter at least once
    # print(range(len(images_to_analyze) - 1))

    # for cases that contain just one image --> this is a code duplication --> maybe try to get rid of it someday
    if len(images_to_analyze)==1:
        # file_path_0, file_path_1, filename0_without_ext, filename1_without_ext = get_input_files_and_output_folders(
        #     images_to_analyze, 0)
        filename0_without_ext = smart_name_parser(images_to_analyze[0], 'full_no_ext')
        mask_t0 = Img(get_TA_file(filename0_without_ext, 'handCorrection.tif'))
        if mask_t0.has_c():
            mask_t0 = mask_t0[..., 0]
        labels_t0 = measure.label(mask_t0, connectivity=1, background=255)  # FOUR_CONNECTED
        # if l == 0:
        tracked_cells_t0 = first_image_tracking(mask_t0, labels_t0, assigned_ids=assigned_ids, seed=seed)
        Img(int24_to_RGB(tracked_cells_t0), dimensions='hwc').save(
            get_TA_file(filename0_without_ext, 'tracked_cells_resized.tif'),
            mode='raw')  # should work --> DO the job
        return

    for l in range(len(images_to_analyze) - 1):
        try:
            if early_stop.stop == True:
                return
            if progress_callback is not None:
                progress_callback.emit((l * 100) / len(images_to_analyze))
            else:
                print(str((l * 100) / len(images_to_analyze)) + '%')
        except:
            pass

        start_loop = timer()

        file_path_0, file_path_1, filename0_without_ext, filename1_without_ext = get_input_files_and_output_folders(
            images_to_analyze, l)

        orig_t0 = Img(file_path_0)  # [455:455+122,544:544+127] #[384:512,512:640]
        orig_t1 = Img(file_path_1)  # [455:455+122,544:544+127] #[384:512,512:640]

        if channel is not None:
            # we reensure image has channel otherwise skip
            if len(orig_t0.shape) > 2:
                orig_t0 = orig_t0[..., channel]
            if len(orig_t1.shape) > 2:
                orig_t1 = orig_t1[..., channel]

        # gros bug qq part je pense
        # ça marche --> trop cool
        print('intermediate time before pyramidal registration', timer() - start_loop)

        # in fact this is really the pyramidal reg that is so slow now --> try speed it up

        # reducing depth makes a huge gain of time and the gain is limited --> maybe offer depth as a parameters
        translation_matrix = get_pyramidal_registration(orig_t0, orig_t1, depth=PYRAMIDAL_DEPTH,threshold_translation=20)  # , threshold_translation=20 , pyramidal_skip=PYRAMIDAL_SKIP
        # translation_matrix = criss_cross_reg(orig_t0, orig_t1) # , threshold_translation=20 , pyramidal_skip=PYRAMIDAL_SKIP # marche mieux mais pas encore ça --> comment comparer et trouver scores # can I exclude no specific objects from tracking --> so that it is working better
        print('intermediate time after pyramidal registration', timer() - start_loop)

        Img(translation_matrix).save(get_TA_file(filename0_without_ext, 'translation_matrix.tif'))

        mask_t0 = Img(get_TA_file(filename0_without_ext, 'handCorrection.tif'))

        if mask_t0.has_c():
            mask_t0 = mask_t0[..., 0]
        labels_t0 = measure.label(mask_t0, connectivity=1, background=255)  # FOUR_CONNECTED

        if __DEBUG__:
            print(mask_t0.shape, get_TA_file(filename0_without_ext, 'handCorrection.tif'))

        mask_t1 = Img(get_TA_file(filename1_without_ext, 'handCorrection.tif'))
        if mask_t1.has_c():
            mask_t1 = mask_t1[..., 0]
        if l == 0:
            tracked_cells_t0 = first_image_tracking(mask_t0, labels_t0, assigned_ids=assigned_ids, seed=seed)
            Img(int24_to_RGB(tracked_cells_t0), dimensions='hwc').save(
                get_TA_file(filename0_without_ext, 'tracked_cells_resized.tif'),
                mode='raw')  # should work --> DO the job
        else:
            tracked_cells_t0 = RGB_to_int24(Img(get_TA_file(filename0_without_ext, 'tracked_cells_resized.tif')))

        if __DEBUG__:
            print(mask_t0.shape)
            print(mask_t1.shape)

        height = mask_t0.shape[0]
        width = mask_t0.shape[1]

        centroids_t0 = []

        for iii, region in enumerate(regionprops(labels_t0)):
            # take regions with large enough areas
            # centroid = region.centroid
            centroid = point_on_surface(region, labels_t0)

            # print(labels_t0[int(centroid[0]), int(centroid[0])],iii+1)

            centroids_t0.append(centroid)

            # KEEP NB BE CAREFUL SOMEHOW THE CODE BELOW CREATES A LOT MORE SWAPPING ALTHOUGH IT SHOULD NOT --> NO CLUE WHY BUT BAD
            # TODO maybe if cell is big I should really take the center of area even if by luck the value falls in it can I identify big cells based on area ???
            # maybe it's really at the matching level that it becomes the most interesting this stuff
            # if labels_t0[int(centroid[0]), int(centroid[0])] == iii + 1:
            #     centroids_t0.append(centroid)
            # else:
            #     print('replacing centroid by center of area because of mismatch')
            #     # replace centroid by center of area to always be in the shape no matter how its shape looks like
            #     centroids_t0.append((region.coords[int(region.area / 2)][0], region.coords[int(region.area / 2)][
            #         1]))  # cree des erreurs --> take centroid except if mismatch

            # print(region.centroid)

        # points1 = get_ordered_list(points1,0,0)
        centroids_t0 = np.array(centroids_t0)  # parfait

        labels_t1 = measure.label(mask_t1, connectivity=1, background=255)  # FOUR_CONNECTED
        if __DEBUG__:
            print(labels_t1.shape)

        t0_cells_n_vertices = {}
        t1_cells_n_vertices = {}

        # NB if I wanna use that then I need 255 to be a forbidden color everywhere otherwise a pure blue cell will give an error --> see how I can do that otherwise really need invert but in a smart way --> think about it
        # could also get a mapping --> gain of time maybe... --> such as a mapping to the coords
        centroids_t1 = []
        for iii, region in enumerate(regionprops(labels_t1)):
            # take regions with large enough areas
            # centroid = region.centroid
            centroid = point_on_surface(region, labels_t1)
            centroids_t1.append(centroid)
            # if labels_t1[int(centroid[0]), int(centroid[0])] == iii + 1:
            #     centroids_t1.append(centroid)
            # else:
            #     print('replacing centroid by center of area because of mismatch')
            #     # replace centroid by center of area to always be in the shape no matter how its shape looks like
            #     centroids_t1.append((region.coords[int(region.area / 2)][0], region.coords[int(region.area / 2)][1]))

        centroids_t1 = np.array(centroids_t1)

        tile_width = 64  # 256 #128
        tile_height = 64  # 256 #128

        # nb sometimes further cropping cause the registration issue and sometimes it helps
        # if I could crop further I may be able to perform better --> see how I can do that though

        # otherwise could simply do a swap and reassign it...
        rps_t1_mask = regionprops(labels_t1)

        # TODO  --> do a table matching pair of id

        matched_cells = {}

        matched_cells_in_t1 = []
        # matched_cells_in_t1_correspondance = {}

        # TODO REMOVE just raw alignment to try to see if post process error
        # tracks_trash = np.zeros((*mask_t1.shape, 3), dtype=np.uint8)

        #

        tracks = np.zeros_like(mask_t1, dtype=np.uint32)
        for centroid_t0 in centroids_t0:

            t0 = translation_matrix[int(centroid_t0[0]), int(centroid_t0[1]), 0]
            t1 = translation_matrix[int(centroid_t0[0]), int(centroid_t0[1]), 1]

            try:
                # this is ok so far but could take the center of the area instead!!!
                cell_id_in_t0 = tracked_cells_t0[
                    int(centroid_t0[0]), int(centroid_t0[1])]

                translation_corrected_y = int(centroid_t0[0] + t0)
                translation_corrected_x = int(centroid_t0[1] + t1)
                if not (
                        translation_corrected_y < height and translation_corrected_y >= 0 and translation_corrected_x < width and translation_corrected_x >= 0):
                    # out of bounds --> continue
                    continue
                possible_matching_cell = labels_t1[translation_corrected_y, translation_corrected_x]

                if possible_matching_cell > 0:  # could check if was already matched

                    # in fact I could allow it and check doublons later
                    if possible_matching_cell in matched_cells_in_t1:
                        # best would be to keep the best of the two --> the one that preserves the best vector maybe
                        # en effet il y a des mismatch et des dupes mais pas une bonne idee de les enlever car ils sont pas bons
                        # ideally should keep the biggest cell here instead
                        # if __DEBUG__:
                        #     print('duplicated cell already matched', matched_cells_in_t1_correspondance[possible_matching_cell],
                        #           '-->', labels_t0[int(centroid_t0[0]), int(
                        #             centroid_t0[1])], 'color',cell_id_in_t0)  # 107 duplicates --> but not real dupes cause same cell I bet
                        continue

                    if __DEBUG__:
                        print('assigned id-->', cell_id_in_t0)

                    # bug of assigning a white cell as a label --> need fix but
                    if cell_id_in_t0 == 0xFFFFFF:
                        if __DEBUG__:
                            print(
                                'error assigning white --> continuing')  # TODO maybe fix this some day to make sure we try to reassign the cell afterwards
                        continue
                        # import sys
                        # sys.exit(0)

                    tracks[labels_t1 == possible_matching_cell] = cell_id_in_t0
                    matched_cells_in_t1.append(possible_matching_cell)

                    if __DEBUG__:
                        if labels_t0[int(centroid_t0[0]), int(centroid_t0[1])] in matched_cells:
                            print('duplicated cell', labels_t0[int(centroid_t0[0]), int(centroid_t0[1])],
                                  matched_cells[labels_t0[int(centroid_t0[0]), int(centroid_t0[1])]], '-->',
                                  possible_matching_cell)

                    # TODO check DO I REALLY NEED THAT ANYMORE ???
                    matched_cells[labels_t0[int(centroid_t0[0]), int(centroid_t0[1])]] = possible_matching_cell
                    # matched_cells_in_t1_correspondance[possible_matching_cell] = labels_t0[   int(centroid_t0[0]), int(centroid_t0[1])]
            except:
                # assume cell is out of bonds --> ignore ot could correct axis
                traceback.print_exc()

        if __DEBUG__:
            print('all_matched_cells', matched_cells)  # all is ok

        # very good in fact

        # missing_cells_t1 = []
        matched_cells_ids = []

        for iii, region in enumerate(rps_t1_mask):
            color = tracks[region.coords[0][0], region.coords[0][1]]
            if color == 0:
                # missing_cells_t1.append(iii)
                pass
            else:
                matched_cells_ids.append(color)

        if __DEBUG__:
            # print('missing_cells_t1', len(missing_cells_t1), '/', len(rps_t1_mask), missing_cells_t1)
            print('matched_cells_ids', len(matched_cells_ids), '/', len(rps_t1_mask), matched_cells_ids)

        labels_tracking_t0 = measure.label(tracked_cells_t0, connectivity=1, background=0xFFFFFF)

        # plt.imshow(tracked_cells_t0[...,0], cmap='gray') #
        # plt.show()

        rps_t0 = regionprops(labels_tracking_t0)

        unmatched_cells_in_t0 = []
        # all ok so why bug there
        for iii, region in enumerate(rps_t0):
            # there is a bug there and I don't get it
            color = tracked_cells_t0[region.coords[0][0], region.coords[0][1]]
            # print(color)  # why always the same color -->
            if color not in matched_cells_ids:  # why bug --> is it due to hash error
                unmatched_cells_in_t0.append(iii)

        if __DEBUG__:
            print('missing_cells_t0', len(unmatched_cells_in_t0), '/', len(rps_t0), unmatched_cells_in_t0)

        # I have already added them --> REDUNDANT AND I AM NOT EVEB SURE I USE THE MATCHING PAIRS --> REMOVE THEM no it's ok
        def associate_cell_id_to_idx(rps, image):
            cell_id_n_ips = {}
            for iii, region in enumerate(rps):
                cell_id = image[region.coords[0][0], region.coords[0][1]]
                cell_id_n_ips[cell_id] = iii
            return cell_id_n_ips

        # cell_id_n_ips_t0 = associate_cell_id_to_idx(rps_t0, tracked_cells_t0)
        # cell_id_n_ips_t1 = associate_cell_id_to_idx(rps_t1_mask, tracks)

        # check pixels by neighborhood
        # if several cells are missing together --> try to match them together --> best match maybe --> faut vraiment essayer
        # essayer de faire mieux

        # MEGA TODO TRY REIDENTIFY MISSING CELLS BASED ON NEIGHBORHOOD MAYBE

        # TODO add missing cells in the very end --> all cells in new image that haven't been assigned
        #
        # for iii, region in enumerate(rps_t1_mask):
        #     color_of_first_pixel_of_potential_match = tracks[region.coords[0][0], region.coords[0][1]]
        #
        #     if color_of_first_pixel_of_potential_match == 0:
        #             new_col = get_unique_random_color_int24(forbidden_colors=assigned_ids, assign_new_col_to_forbidden=True)
        #             tracks[labels_t1 == labels_t1[region.coords[0][0], region.coords[0][1]]] = new_col

        # NB does not seem to work as some cells clearly miss some data --> TODO
        tracks = assign_random_ID_to_missing_cells(tracks, labels_t1, regprps=rps_t1_mask, assigned_ids=assigned_ids)

        # TODO sort cells by the nb of neighbors missing take the ones with most neighbors to reidentify them soon

        # just for TA compatibility...
        tracks[labels_t1 == 0] = 0xFFFFFF

        # plt.imshow(int24_to_RGB(tracks))
        # plt.show()

        if __DEBUG__:
            print('out', get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'))
        # Img(tracks, dimensions='hwc').save(get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'), mode='raw')
        tracked_cells_t1 = tracks

        print('intermediate time before swap correction0', timer() - start_loop)

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
        # plt.imshow(vertices)
        # plt.show()

        # maybe also if too many rejected changes --> ignore
        # DO I NEED SO MUCH PYRAMIDAL REG WITH SUCH A POWERFUL ALGO ??? --> probably not ??? --> check that and really compare to the other method and clean code heavily !!!

        cells_and_their_neighbors_minus_1 = np.unique(
            np.asarray(associate_cell_to_its_neighbors2(vertices_t_minus_1, track_t_minus_1)), axis=0)
        cells_and_their_neighbors_minus_1 = associate_cells_to_neighbors_ID_in_dict(cells_and_their_neighbors_minus_1)
        cells_in_t_minus_1 = np.unique(track_t_minus_1)
        # print('neighbs found', cells_and_their_neighbors_minus_1[10287288])

        initial_score = -1

        # it's better but there are still some swapping --> need be fixed
        for lll in range(MAX_ITER):

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
                                           cells_present_in_t_cur_but_absent_in_t_minus_1_score)  # , cells_present_in_t_cur_but_absent_in_t_minus_1, cells_present_in_t_cur_but_absent_in_t_minus_1_score, matching_neighorhood_score_t_cur_with_t_minus_1)

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
            lab_t_cur = label(track_t_cur, connectivity=1, background=0xFFFFFF)
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
                if last_score_reached == initial_score:
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

        print(get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'))

        # NB it still contains small swapping errors I don't get --> why
        # swap first then assign and assign if not swapped --> check cell

        # celui qui m'interesse c'est le 2
        # Img(int24_to_RGB(corrected_track)).save(get_TA_file(filename1_without_ext, 'tracked_cells_resized_v2.tif'),mode='raw')
        Img(int24_to_RGB(corrected_track)).save(get_TA_file(filename1_without_ext, 'tracked_cells_resized.tif'),
                                                mode='raw')

        # with a recursion it's very good actually and really is optimized

    # print(r_g_b_to_rgb(137,134,15))

    print('total time', timer() - start_all)


if __name__ == '__main__':
    # TODO --> do that recursively in order to really track all --> need a loop over image

    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/200319/GT/predict/predict_model_nb_0'
    # path = '/E/Sample_images/sample_images_denoise_manue/210128_dpov1_female_48h10APF/predict/predict_model_nb_0'
    # path = '/E/Sample_images/sample_images_denoise_manue/210121_armGFP_suz_line2_47h30_APF/predict/before_crash/predict_model_nb_0'
    # path = '/E/Sample_images/sample_images_denoise_manue/210128_dpov1_female_48h10APF/registered_stack/splitted_stack'
    # path = '/E/Sample_images/sample_images_denoise_manue/201104_armGFP_different_lines_tila/predict/predict_model_nb_0/registered/splitted' # done slow
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/21-03/GT/predict/predict_model_nb_3'
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/25-1/GT/predict/predict_model_nb_3'
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/27-03/GT/predict/predict_model_nb_3'
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/05-04/GT/predict/predict_model_nb_3'
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/24-1/GT/predict/predict_model_nb_0'
    # path = '/E/Sample_images/sample_images_denoise_manue/210108_armGFP_48hAPF_female/predict/predict_model_nb_4'
    # path = '/E/Sample_images/sample_images_denoise_manue/210108_armGFP_48hAPF_female/predict/predict_model_nb_1'
    # path = '/E/Sample_images/sample_images_denoise_manue/200722_armGFP_suz_ON_47hAPF/predict/predict_model_nb_4/list.lst' # done slow
    # path = '/E/Sample_images/sample_images_denoise_manue/legs/100807_leg4_male/predict/predict_model_nb_0/list.lst' # done but very slow --> need improve and by quite a lot...
    # path = '/E/Sample_images/sample_images_denoise_manue/200709_armGFP_suz_46hAPF_ON/predict/predict_model_nb_0best/list.lst' # last test 190secs much better very slow using new code --> 500-600secs per loop --> now 215 secs per loop with the cleaned up algo --> much better
    # path = '/E/Sample_images/sample_images_denoise_manue/210324_ON_suz_22h45_armGFP_line2/predict/predict_model_nb_0/list.lst'
    # path = '/E/Sample_images/sample_images_denoise_manue/210312_armGFP_line2_suz_39h30APF/predict/predict_model_nb_0/list.lst'
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/05-04/GT/predict/predict_model_nb_3/list.lst'
    # path = '/E/Sample_images/sample_images_denoise_manue/210402_EcadKI_mel_female_26hAPF_pupae_from_old_tube/predict/predict_model_nb_0_not_best_but_very_good_still/list.lst' #TODO
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/210219/predict/predict_model_nb_0' #TODO
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/20-2/predict/predict_model_nb_0' #TODO
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/26-02/predict/predict_model_nb_0' #TODO
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/30-1/predict/predict_model_nb_0' #TODO
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/070219/predict/predict_model_nb_0' #TODO
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/05-2/predict/predict_model_nb_0' #TODO
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/27-02/predict/predict_model_nb_0' #TODO
    # path = '/E/Sample_images/sample_images_denoise_manue/210409_EcadKI_ovipo_female_around_40hAPF/predict/predict_model_nb_4' #TODO
    # path = '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/06-2/predict/predict_model_nb_0/list.lst' #TODO
    # path = '/E/Sample_images/sample_images_denoise_manue/legs/100611_female_leg/predict/predict_model_nb_2/list.lst' # done slow
    # path = '/E/Sample_images/sample_images_denoise_manue/legs/100706_leg4_female/predict/predict_model_nb_2' # latest measure 1103s --> even better --> ALMOST OK AND CAN INCREASE SPEED BY NB OF PROCS TOO --> 2473secs --> before and now is 1612 secs --> much better because results are identical and I just changhed two lines of code... --> again a bit slow --> try speed this up but really good in fact
    # path = '/E/Sample_images/sample_images_denoise_manue/legs/100608_leg3/predict/predict_model_nb_2best_maybe/list.lst' # --> 7491.503101537935secs --> a bit slow --> really try to improve that in the future but kinda ok for now
    # path = '/E/Sample_images/sample_images_denoise_manue/legs/legs_le_bivic/181023_ON_tila_SC_EcadKI_around16H_APF/predict/predict_model_nb_0/list.lst' # done slow
    # path = '/E/Sample_images/sample_images_denoise_manue/200709_armGFP_suz_46hAPF_ON/predict/predict_model_nb_0/list.lst' # done slow
    # path = '/E/Sample_images/segmentation_assistant/ovipo_uncropped'  # done test
    # path = '/E/Sample_images/sample_images_denoise_manue/201104_armGFP_different_lines_tila/predict/predict_model_nb_0/list.lst'  # done test
    # path = '/E/Sample_images/test_tracking_with_shift' # quick test of the algo
    # path = '/E/Sample_images/tracking_test'  # quick test of the tracking algo
    # path = '/E/Sample_images/tracking_test/test_uncropped'  # quick test of the tracking algo # local version is much worse --> not worth it --> do I have a bug ????
    path = '/E/Sample_images/tracking_test/test_quick_TA_tracking/list.lst'  #total time 296.99147817201447 -->  308.14404578204267 with criss cross reg --> mieux mais bcp de drift sur les bords --> pkoi ??? et des rigions de big shift car too coarse --> maybe need further split it like the other maybe
    # ideally I should detect and prevent the big difts
    # or need 6 and divide by
    # path = '/E/Sample_images/tracking_test/test_cropped'  # quick test of the tracking algo
    # path = '/E/Sample_images/tracking_test/leg_uncropped'  # quick test of the tracking algo # quite good --> very few swaps in fact # total time 152.38672435702756 sec nouveau vs total time 164.57967406092212 secs for the old one --> not really useful to restrict then best score global  =733.1310118292632, best score local =730.04 --> early stop 730.9823090293312
    # path = '/E/Sample_images/tracking_test/leg_cropped'  # quick test of the tracking algo

    track_cells_dynamic_tissue(path)
