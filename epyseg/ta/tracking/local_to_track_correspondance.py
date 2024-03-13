from skimage.measure import regionprops, label
import numpy as np
from epyseg.ta.tracking.tools import smart_name_parser
from epyseg.img import Img, RGB_to_int24
from epyseg.tools.early_stopper_class import early_stop
from epyseg.utils.loadlist import loadlist
from epyseg.ta.database.sql import TAsql
import traceback
import os
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()

def save_local_id_and_track_correspondence(local_id_track_correspondence, db_file, type='cell'):
    """
    Saves the local ID and track correspondence in a database.

    Args:
        local_id_track_correspondence (dict): The dictionary containing the local ID and track correspondence.
        db_file (str): The path to the database file.
        type (str, optional): The type of correspondence. Defaults to 'cell'.

    """
    # Open the database and save the file in it
    db = TAsql(filename_or_connection=db_file)
    table_name = type + '_tracks'
    db.drop_table(table_name)

    if local_id_track_correspondence is None:
        db.close()
        return

    # Change the dict to another dict
    local_ids = list(local_id_track_correspondence.keys())
    track_ids = list(local_id_track_correspondence.values())
    data_with_headers = {'local_id': local_ids, 'track_id': track_ids}
    db.create_and_append_table(table_name, data_with_headers)
    db.close()


def get_local_id_n_track_correspondence_from_images(filename):
    """
    Retrieves the local ID and track correspondence from the images.

    Args:
        filename (str): The filename of the images.

    Returns:
        dict: The dictionary containing the local ID and track correspondence.

    """
    if filename is None:
        return None

    local_id_track_correspondence = {}

    try:
        tracked_image_path = smart_name_parser(filename, ordered_output='tracked_cells_resized.tif')

        cell_id_image = None

        handCorrection1, handCorrection2 = smart_name_parser(filename, ordered_output=['handCorrection.png', 'handCorrection.tif'])

        if os.path.isfile(handCorrection2):
            cell_id_image = Img(handCorrection2)
        elif os.path.isfile(handCorrection1):
            cell_id_image = Img(handCorrection1)

        if cell_id_image is None:
            logger.error('File not found ' + str(handCorrection2) + ' please segment the images first')
            return

        if len(cell_id_image.shape) >= 3:
            cell_id_image = cell_id_image[..., 0]
        cell_id_image = label(cell_id_image, connectivity=1, background=255)

        tracked_image = None
        if os.path.isfile(tracked_image_path):
            tracked_image = RGB_to_int24(Img(tracked_image_path))

        if tracked_image is None:
            logger.error('File not found ' + str(tracked_image_path) + ' please track cells first')
            return

        for region in regionprops(cell_id_image):
            color = tracked_image[region.coords[0][0], region.coords[0][1]]
            if color == 0:
                logger.warning("Tracks and cells don't match, correspondence will be meaningless, please update your files")
                return None
            local_id_track_correspondence[region.label] = color
    except:
        traceback.print_exc()
        logger.error('Something went wrong when converting local ID to tracks for file ' + str(filename))
        return None

    return local_id_track_correspondence

def add_localID_to_trackID_correspondance_in_DB(lst, progress_callback=None):
    """
    Adds the local ID to track ID correspondence to the database.

    Args:
        lst (list): The list of files.
        progress_callback (function, optional): The callback function for reporting progress. Defaults to None.

    """
    if lst is not None and lst:
        for lll, file in enumerate(lst):
            try:
                if early_stop.stop == True:
                    return
                if progress_callback is not None:
                    progress_callback.emit(int((lll / len(lst)) * 100))
                else:
                    print(str((lll / len(lst)) * 100) + '%')
            except:
                pass
            db_file = smart_name_parser(file, ordered_output='pyTA.db')
            local_to_global_correspondence = get_local_id_n_track_correspondence_from_images(file)
            save_local_id_and_track_correspondence(local_to_global_correspondence, db_file)


if __name__ == "__main__":

    if True:
        lst = loadlist('/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/list.lst')
        add_localID_to_trackID_correspondance_in_DB(lst)

    if False:
        # test_local_to_global_correspondece = {1: 0xFF0000, 2: 0x00FF00, 3: 0x0000FF}
        db_file_for_test = '/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012/pyTA.db'

        test_local_to_global_correspondece = get_local_id_n_track_correspondence_from_images('/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012.png')
        # print(test_local_to_global_correspondece)
        save_local_id_and_track_correspondence(test_local_to_global_correspondece, db_file_for_test)
        # try get the stuff

        # pass
