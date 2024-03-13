import traceback
import matplotlib.pyplot as plt

from epyseg.ta.measurements.measurements3D.get_point_on_surface_if_centroid_is_bad import point_on_surface
from epyseg.ta.utils.TA_tools import get_TA_clone_file_name
from epyseg.img import RGB_to_int24, Img
from epyseg.ta.tracking.tools import smart_name_parser
from skimage.measure import label, regionprops
import numpy as np
from epyseg.tools.early_stopper_class import early_stop
from epyseg.tools.logger import TA_logger  # logging

logger = TA_logger()

# deprecated --> use another code if possible
def convert_db_xy_coordiantes_to_local_cell_ID_for_clones(db_path, table_name, image_for_correspondance_label_or_RGB24, return_image=False):
    """
    Converts database coordinates to local cell IDs for clones.

    Args:
        db_path (str): The path to the database.
        table_name (str): The name of the table containing the coordinates.
        image_for_correspondance_label_or_RGB24 (ndarray): The image used for correspondence between labels and RGB24 values.
        return_image (bool, optional): A flag indicating whether to return an image or a set of cells.
                                       Defaults to False.

    Returns:
        list or ndarray: The output, either a list of cells or an image.

    """
    from epyseg.ta.database.sql import TAsql
    import numpy as np

    output = []  # Initialize the output variable
    db = None
    try:
        db = TAsql(db_path)  # Open the database connection

        # Get the coordinates from the database
        query_results = db.run_SQL_command_and_get_results('SELECT * FROM ' + table_name, return_header=False)

        if return_image:
            output = np.zeros_like(image_for_correspondance_label_or_RGB24)  # Create an empty image

        for coords in query_results:
            color = image_for_correspondance_label_or_RGB24[coords[0], coords[1]]  # Get the color value at the coordinates

            if return_image:
                output[image_for_correspondance_label_or_RGB24 == color] = color  # Assign the color to corresponding pixels in the image
            else:
                output.append(color)  # Append the color to the list of cells
    except:
        import traceback
        traceback.print_exc()
        logger.error('Something went wrong, selection could not be converted to 2D coordinates')
    finally:
        if db is not None:
            try:
                db.close()  # Close the database connection
            except:
                import traceback
                traceback.print_exc()

    return output

def get_colors_drawn_over(mask, image_to_analyze, forbidden_colors=[0, 0xFFFFFF]):
    """
    Retrieves the colors drawn over a mask in an image.

    Args:
        mask (ndarray): The mask indicating the drawn areas.
        image_to_analyze (ndarray): The image to analyze.
        forbidden_colors (list or int, optional): Colors to exclude from the result.
                                                  Defaults to [0, 0xFFFFFF].

    Returns:
        list: The list of colors drawn over the mask.

    """
    if mask is None:
        return None

    selected_colors = image_to_analyze[mask != 0]  # Get the colors corresponding to the non-zero regions of the mask

    if len(image_to_analyze.shape) == 3:
        selected_colors = RGB_to_int24(selected_colors)  # Convert RGB colors to int24 format

    selected_colors = set(selected_colors.ravel().tolist())  # Convert the colors to a set

    if forbidden_colors is not None:
        if not isinstance(forbidden_colors, list):
            forbidden_colors = [forbidden_colors]  # Convert single forbidden color to a list

        for color in forbidden_colors:
            if color in selected_colors:
                selected_colors.remove(color)  # Remove forbidden colors from the selected colors set

    selected_colors = list(selected_colors)  # Convert the set back to a list

    return selected_colors


def handCorrection_to_label(handCorrection):
    """
    Converts a hand correction image to a labeled image.

    Args:
        handCorrection (str or ndarray): The hand correction image as a filename or ndarray.

    Returns:
        ndarray: The labeled image.

    """
    if isinstance(handCorrection, str):
        handCorrection = Img(handCorrection)  # Load the hand correction image if it's a filename

    if len(handCorrection.shape) == 3:
        handCorrection = handCorrection[..., 0]  # Convert RGB image to grayscale

    handCorrection = label(handCorrection, connectivity=1, background=255)  # Label the image

    return handCorrection

def add_coords_to_db(db_path, table_name_to_store_coords_to, coords):
    """
    Adds coordinates to a database table.

    Args:
        db_path (str): The path to the database.
        table_name_to_store_coords_to (str): The name of the table to store the coordinates.
        coords (list): A list of coordinate tuples in the format (y, x).

    """
    from epyseg.ta.database.sql import TAsql

    coords_of_selection = {'pixel_within_cell_y': [], 'pixel_within_cell_x': []}

    db = None
    try:
        db = TAsql(db_path)  # Open the database connection

        # Extract y and x coordinates from the coordinate tuples and store them in separate lists
        for coord in coords:
            coords_of_selection['pixel_within_cell_y'].append(coord[0])
            coords_of_selection['pixel_within_cell_x'].append(coord[1])

        # Create a new table or append to an existing table in the database with the coordinate data
        db.create_and_append_table(table_name=table_name_to_store_coords_to, datas=coords_of_selection)
    except:
        traceback.print_exc()
        logger.error('Something went wrong while trying to write coordinates to the database')
    finally:
        if db is not None:
            try:
                db.close()  # Close the database connection
            except:
                traceback.print_exc()


def convert_selection_color_to_coords(img_to_analyze_RGB24_or_label, selected_cells, bg_color=None):
    """
    Converts selected cell colors to coordinates within the image.

    Args:
        img_to_analyze_RGB24_or_label (numpy.ndarray): The image to analyze, either in RGB24 format or label format.
        selected_cells (list): A list of selected cell colors.
        bg_color (int, optional): The background color. If not provided, it will be determined automatically.

    Returns:
        list: A list of coordinate tuples representing the selected cells.

    """
    # tmp = img_to_analyze
    coords_of_selection = []

    if img_to_analyze_RGB24_or_label is None:
        logger.error('No image provided. Nothing to do...')
        return coords_of_selection

    if len(img_to_analyze_RGB24_or_label.shape) == 3:
        img_to_analyze_RGB24_or_label = RGB_to_int24(img_to_analyze_RGB24_or_label)

    if bg_color is None:
        if 0xFFFFFF in img_to_analyze_RGB24_or_label:
            bg_color = 0xFFFFFF
        else:
            bg_color = 0

    cell_label = label(img_to_analyze_RGB24_or_label, connectivity=1, background=bg_color)

    # For all the selected cells, return the coordinates
    for region in regionprops(cell_label):
        color = img_to_analyze_RGB24_or_label[region.coords[0][0], region.coords[0][1]]
        if color in selected_cells:
            point_always_inside_the_cell = point_on_surface(region, cell_label)
            coords_of_selection.append(point_always_inside_the_cell)

    return coords_of_selection


def convert_coords_to_IDs(img_to_analyze_RGB24_or_label, selected_coords, forbidden_colors=[0xFFFFFF, 0], return_image=False, new_color_to_give_to_cells_if_return_image=None):
    """
    Converts selected coordinates to cell IDs in the image.

    Args:
        img_to_analyze_RGB24_or_label (numpy.ndarray): The image to analyze, either in RGB24 format or label format.
        selected_coords (list): A list of selected coordinates.
        forbidden_colors (list, optional): Colors to exclude from conversion. Defaults to [0xFFFFFF, 0].
        return_image (bool, optional): Whether to return an image with converted cell IDs. Defaults to False.
        new_color_to_give_to_cells_if_return_image (int, optional): The color to assign to cells in the returned image. Only applicable if return_image is True.

    Returns:
        list or numpy.ndarray: A list of cell IDs or an image with converted cell IDs, depending on the return_image parameter.

    """
    ids = []

    if img_to_analyze_RGB24_or_label is None:
        logger.error('Empty image. Nothing to do...')
        if return_image:
            return None
        return ids

    if len(img_to_analyze_RGB24_or_label.shape) == 3:
        img_to_analyze_RGB24_or_label = RGB_to_int24(img_to_analyze_RGB24_or_label)

    if return_image:
        ids = np.zeros_like(img_to_analyze_RGB24_or_label)

    for coords in selected_coords:
        color = img_to_analyze_RGB24_or_label[coords[0], coords[1]]

        if forbidden_colors is not None:
            if color in forbidden_colors:
                continue

        if not return_image:
            ids.append(color)
        else:
            # Can be used to plot clones
            if new_color_to_give_to_cells_if_return_image is not None:
                ids[img_to_analyze_RGB24_or_label == color] = new_color_to_give_to_cells_if_return_image
            else:
                ids[img_to_analyze_RGB24_or_label == color] = color

    return ids

def convert_selected_cells_to_local_db_coords(selected_cells, label_or_RGB24_image_corresponding_to_selected_cells,
                                              db_path, table_name_to_store_coords_to, bg_color_for_label=None):
    """
    Converts selected cells to local coordinates and stores them in a database.

    Args:
        selected_cells (list): List of selected cell IDs.
        label_or_RGB24_image_corresponding_to_selected_cells (numpy.ndarray): The label or RGB24 image corresponding to the selected cells.
        db_path (str): Path to the database.
        table_name_to_store_coords_to (str): Name of the table to store the coordinates.
        bg_color_for_label (int, optional): Background color for label image. Defaults to None.

    Returns:
        None

    """
    if not selected_cells:
        logger.error('Nothing selected. Nothing to do.')
        return

    if bg_color_for_label is None:
        # Auto mode to determine the label
        if 0xFFFFFF in label_or_RGB24_image_corresponding_to_selected_cells:
            bg_color_for_label = 0xFFFFFF
        else:
            bg_color_for_label = 0

    try:
        cell_label = label(label_or_RGB24_image_corresponding_to_selected_cells, connectivity=1,
                           background=bg_color_for_label)

        coords = []
        for region in regionprops(cell_label):
            color = label_or_RGB24_image_corresponding_to_selected_cells[region.coords[0][0], region.coords[0][1]]
            if color in selected_cells:
                # Get the coordinate always inside the cell and store it in the database
                point_awlays_inside_the_cell = point_on_surface(region, cell_label)
                coords.append(point_awlays_inside_the_cell)

        # Store the coordinates in the database
        add_coords_to_db(db_path, table_name_to_store_coords_to, coords)

    except:
        traceback.print_exc()
        logger.error('Something went wrong. Selection could not be converted to 2D coordinates')


if __name__ == '__main__':
    import sys

    if False:
        # test how to append selection to a clone database
        path_to_image = '/E/Sample_images/sample_images_PA/mini_empty/focused_Series012.png'
        selected_cells = [200, 201, 302]  # cells 1-3 have been selected
        handCorrection = Img(smart_name_parser(path_to_image, ordered_output='handCorrection.tif'))
        if len(handCorrection.shape) == 3:
            handCorrection = handCorrection[..., 0]
        cell_label = label(handCorrection, connectivity=1, background=255)
        # print(smart_name_parser(path_to_image, ordered_output='pyTA.db'))
        convert_selected_cells_to_local_db_coords(selected_cells, cell_label,
                                                  smart_name_parser(path_to_image, ordered_output='pyTA.db'),
                                                  'test_db_name', bg_color_for_label=0)  # smart
        # try relaod the coords as an image
        sys.exit(0)

    # ça marche vraiment bien en fait j'adore!!!
    if True:
        # load database from image and get the corresponding cell image or db depending on what is desired --> TODO
        path_to_image = '/E/Sample_images/sample_images_PA/mini_empty/focused_Series012.png'
        handCorrection = Img(smart_name_parser(path_to_image, ordered_output='handCorrection.tif'))
        if len(handCorrection.shape) == 3:
            handCorrection = handCorrection[..., 0]
        cell_label = label(handCorrection, connectivity=1, background=255)
        output = convert_db_xy_coordiantes_to_local_cell_ID_for_clones(smart_name_parser(path_to_image, ordered_output='pyTA.db'), 'test_db_name', cell_label, return_image=True) #return_image=Fals
        # ça marche vraiment super bien
        if isinstance(output, np.ndarray):
            plt.imshow(output)
            plt.show()
        else:
            print(output)
        # --> this does return cells 1 to 3
        sys.exit(0)
