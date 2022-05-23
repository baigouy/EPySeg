# zut le code est pas finalisé...

# USE a track object to store the click and its content...


# do a code to do smart track correction that avoids creating dupes
# checks for dupes, etc...
# can connect tracks and or swap tracks

# si les deux couleurs existent dans les deux images alors c'est du swap et faut swapper à partir de la dernière image sinon
# si chaque couleur existe dans une image et pas dans l'autre alors faire un connect track. Si le connect risque de dupliquer une cellule alors il faut faire un swap de la cellule puis connecter la track afin de ne pas dupliquer les cellules
# reflechir mais ça devrait etre bon
# existe t'il un moyen perenne de restaurer les tracks corrected ???? si oui lequel pr que le job ne soit pas perdu
# peut etre simplement faire un truc du genre update tracks et rajouter les cellules perdues de maniere aleatoire ? pas si simple en fait
# puis je aussi swapper plusieurs cellules en même temps ??? peut etre mais pas si simple
# comment faire un algo de swap intelligent


# need to store cell id or centroid or centre of area and the frame --> should be sufficient to do everything --> the most robust then detect the color of the other

# offer apply to all or just before or after the current image
# do the codes for that

# aussi faire un code pour reappliquer les tracks


from epyseg.img import RGB_to_int24, int24_to_RGB
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from epyseg.img import Img
from epyseg.utils.loadlist import loadlist
from epyseg.ta.tracking.tools import smart_name_parser
from epyseg.tools.logger import TA_logger  # logging

logger = TA_logger()

def swap(img, id1, id2):
    idcs1 = img == id1
    idcs2 = img == id2
    img[idcs1], img[idcs2] = id2, id1
    return img


def assign_id(img, old_id, new_id):
    img[img == old_id] = new_id
    return img

def swap_tracks(lst, frame_of_first_connection, id_t0, id_t1, __preview_only=False):
    correct_track(lst, frame_of_first_connection, id_t0, id_t1, correction_mode='swap', __preview_only=__preview_only)


def connect_tracks(lst, frame_of_first_connection, id_t0, id_t1,__preview_only=False):
    correct_track(lst, frame_of_first_connection, id_t0, id_t1, correction_mode='connect', __preview_only=__preview_only)


def correct_track(lst, frame_of_first_connection, id_t0, id_t1, correction_mode='connect', __preview_only=False, early_stop=True):
    if frame_of_first_connection >= len(lst):
        logger.error('error wrong connection frame nb')
        return

        # first thing to do is to check that we are not going to duplicate cells --> need check that both are not there together in the same frame otherwise need swap cells before --> try that
    for l in range(frame_of_first_connection, len(lst)):
        # file_path_0 = images_to_analyze[l]
        # filename0_without_ext = os.path.splitext(file_path_0)[0]


        tracked_cell_path = smart_name_parser(lst[l], ordered_output='tracked_cells_resized.tif') # os.path.join(filename0_without_ext, )
        tracked_cells_t0 = RGB_to_int24(Img(tracked_cell_path))

        cellpos_1 = id_t0 in tracked_cells_t0
        cellpos_2 = id_t1 in tracked_cells_t0

        if not cellpos_1 and not cellpos_2:
            logger.info('cells not found --> ignoring track correction')
            continue

        if cellpos_1 and cellpos_2 and correction_mode == 'connect':
            logger.info('IDs are both present in the same image, to avoid track ID duplication the desired change will be ignored, please ensure the selected cells are not wimply swapped in the image')
            # alternatively I could do a swap but not sure this is wise
            break

        if not cellpos_1 and cellpos_2 or not cellpos_2 and cellpos_1 and correction_mode == 'swap':
            logger.info('missing cell at frame '+str(l)+' swapping ignored')
            if early_stop:
                logger.info('quitting track edition')
                break
            else:
                continue

        if cellpos_1 and cellpos_2 and correction_mode == 'swap': # or (cellpos_1 and cellpos_2 and correction_mode == 'connect')
            logger.info('swapped cell ids: ' + str(id_t0) + ' and ' + str(id_t1))
            # or consider this an error and report it and ignore
            tracked_cells_t0 = swap(tracked_cells_t0, id_t0, id_t1)
            if __preview_only:
                plt.imshow(int24_to_RGB(tracked_cells_t0))
                plt.show()
            else:
                Img(int24_to_RGB(tracked_cells_t0), dimensions='hwc').save(tracked_cell_path, mode='raw')
        else:
            if cellpos_1:
                logger.info('changed id ' + str(id_t0) + ' to ' + str(id_t1))
                # tracked_cells_t0[tracked_cells_t0 == id_t0] = id_t1
                tracked_cells_t0 = assign_id(tracked_cells_t0, id_t0, id_t1)
            elif cellpos_2:
                logger.info('changed id ' + str(id_t1) + ' to ' + str(id_t0))
                # tracked_cells_t0[tracked_cells_t0 == id_t1] = id_t0
                tracked_cells_t0 = assign_id(tracked_cells_t0, id_t1, id_t0)
            if __preview_only:
                plt.imshow(int24_to_RGB(tracked_cells_t0))
                plt.show()
            else:
                Img(int24_to_RGB(tracked_cells_t0), dimensions='hwc').save(tracked_cell_path, mode='raw')

if __name__ == "__main__":
    start_all = timer()

    # path = '/E/Sample_images/tracking_test'
    path = '/E/Sample_images/sample_images_PA/trash_test_mem/mini10_fake_swaps/liste.lst'
    # path = '/E/Sample_images/test_tracking_with_shift' # quick test of the algo
    #
    # images_to_analyze = os.listdir(path)
    # images_to_analyze = [os.path.join(path, f) for f in images_to_analyze if
    #                      os.path.isfile(os.path.join(path, f))]  # list only files and only if they end by tif
    # images_to_analyze = natsorted(images_to_analyze)

    images_to_analyze = loadlist(path)

    # then do amrutha's code watershed  imagendarray (2-D, 3-D, …) of integers--> cool --> will also work for 3D --> then need segmentation in 3D
    #
    #
    # for l in range(len(images_to_analyze)):
    #     # in fact I don't even need to reopen the mask file because the track file should suffice ???
    #
    #     file_path_0 = images_to_analyze[l]
    #     if not file_path_0.endswith('.tif'):
    #         continue
    #
    #
    #     # print('files', file_path_1, file_path_0)
    #
    #     filename0_without_ext = os.path.splitext(file_path_0)[0]
    #
    #     tracked_cells_t0 = Img(os.path.join(filename0_without_ext,        'tracked_cells_resized.tif')) # 455:455 + 128, 435:435 + 128 #455:455 + 256, 435:435 + 290 #768:1024,0:256 #0:128, 0:128 #500:500+128, 500:500+128
    #
    #
    #
    #
    #     # TODO always get the mask instead of any other files in order to have the minimal nb of files available
    #     # mask_t0 = Img(os.path.join(filename0_without_ext, 'handCorrection.tif'))[
    #     #     ..., 0]  # 455:455 + 128, 435:435 + 128 #455:455 + 256, 435:435 + 290 #768:1024,0:256 #0:128, 0:128 #500:500+128, 500:500+128
    #
    #
    #     # labels_t0 = measure.label(mask_t0, connectivity=1,
    #     #                           background=255)  # FOUR_CONNECTED # use bg = 255 to avoid having to invert the image --> a gain of time I think and could be applied more generally
    #
    #     height = tracked_cells_t0.shape[0]
    #     width = tracked_cells_t0.shape[1]
    #
    #
    #     # could maybe store the ID of duplicate cell as a reseed --> can be a pb with extruding cells but should most often be ok though
    #     #
    #
    #     labels_tracking_t0 = measure.label(tracked_cells_t0[..., 0], connectivity=1, background=255)
    #     rps_t0 = regionprops(labels_tracking_t0)
    #
    #
    #     centroids_t0 = []
    #     for region in rps_t0:
    #         # take regions with large enough areas
    #         centroids_t0.append(region.centroid)

    # try create a dupe to see if it fails -> it will and I should really avoid that

    # swap_tracks(images_to_analyze, 1, 0x0ef5dd, 0xa28ad6, __preview_only=True)  # both cells exist


    swap_tracks(images_to_analyze, 0, 0x0ef5dd, 0xFAFAFF, __preview_only=True)  # one cells exist

    # ça marche mais reflechir que je ne puisse pas creer des pbs avec ça
    # connect_tracks(images_to_analyze, 0, 0xffdd5d, 0x3b2cdb, __preview_only=True)  # both cells exist
    # connect_tracks(images_to_analyze, 0, 0xffdd5d, 0x0ef5dd, __preview_only=True)  # create a duplicate cell # --> breaking

    # codes are the same --> just keep one and

    # tt a l'air de marcher --> maybe add a flag for recursive --> in fact not...

    print('total time', timer() - start_all)
