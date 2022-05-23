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


# TODO --> I would really need an editor --> see how I can do that ???
# because that would really be super useful !!!

# this class will gather all the TA selection tools and conversion between the various formats

# if not return image then return
# converts a set of coordinates to a list of Ids or to an image output --> will be very useful to plot and store clones, dividing cells and dying cells...

# deprecated --> use another code if possible
def convert_db_xy_coordiantes_to_local_cell_ID_for_clones(db_path, table_name, image_for_correspondance_label_or_RGB24,
                                                          return_image=False):
    from epyseg.ta.database.sql import TAsql
    # read the coords from the table and make sure they can be used to generate a file
    # pass

    # open the db and get the coords and return a set of cells or an image depending on what the user wants to have -> maybe could also return a table or a temp table so that one can do a join on it --> very smart idea too!!
    output = []
    db = None
    try:

        db = TAsql(db_path)
        # now i need get the coords from the db and check the image for that

        # NB THIS TABLES ASSUMES THE QUERY RETURNS 2D COORDS --> ALWAYS AN X AND A Y
        #headers,
        query_results = db.run_SQL_command_and_get_results('SELECT * FROM ' + table_name, return_header=False)

        # formatted = np.asarray(query_results, dtype=object)
        # print(headers)
        # print(query_results)
        # print(formatted)

        # display = None

        if return_image:
            output = np.zeros_like(image_for_correspondance_label_or_RGB24)

        # TODO --> need a check to be sure the coords are really inside the cell !!! --> normally it must always be but that is a smart check anyway!!!
        for coords in query_results:
            # print(coords)
            # print(coords[0], coords[1])
            color = image_for_correspondance_label_or_RGB24[coords[0], coords[1]]
            # this gives me all the cells I need to get
            # then I could either get the corresponding label or the image corresponding to that
            # print(color)
            if return_image:
                output[image_for_correspondance_label_or_RGB24 == color] = color
            else:
                output.append(color)
    except:
        traceback.print_exc()
        logger.error('Something went wrong, selection could not be converted to 2D coordinates')
    finally:
        if db is not None:
            try:
                db.close()
            except:
                traceback.print_exc()
        return output




def get_colors_drawn_over(mask, image_to_analyze, forbidden_colors=[0,0xFFFFFF]):
    # print('I was called')
    if mask is None:
        return
    # print(mask)
    # need get all the colors of the raw image that correspond to user selection
    selected_colors = image_to_analyze[mask != 0]
    # print('selected_colors',selected_colors)

    # why is the user drawing null
    # plt.imshow(mask)
    # plt.show()

    if len(image_to_analyze.shape)==3:
        selected_colors = RGB_to_int24(selected_colors)
    selected_colors = set(selected_colors.ravel().tolist())


    # if 0xFFFFFF in selected_colors:
    #     selected_colors.remove(0xFFFFFF)
    if forbidden_colors is not None:
        if not isinstance(forbidden_colors, list):
            forbidden_colors = [forbidden_colors]
        for color in forbidden_colors:
            if color in selected_colors:
                selected_colors.remove(color)
    selected_colors = list(selected_colors)

    # print(selected_colors) # all is ok there -> so the error is afterwards
    return selected_colors

def handCorrection_to_label(handCorrection):
    if isinstance(handCorrection,str):
        # TODO shall I so more complex things such as try also create the handcorrection name if it does not exist ??? --> TODO but maybe later!!!
        handCorrection = Img(handCorrection)
    if len(handCorrection.shape) == 3:
        handCorrection= handCorrection[...,0]
    handCorrection = label(handCorrection, connectivity=1, background=255)
    return handCorrection

def add_coords_to_db(db_path, table_name_to_store_coords_to, coords):
    from epyseg.ta.database.sql import TAsql
    coords_of_selection = {'pixel_within_cell_y': [],
                           'pixel_within_cell_x': [],
                           }
    db = None
    try:
        db = TAsql(db_path)
        for coord in coords:
            coords_of_selection['pixel_within_cell_y'].append(coord[0])
            coords_of_selection['pixel_within_cell_x'].append(coord[1])
        db.create_and_append_table(table_name=table_name_to_store_coords_to, datas=coords_of_selection)
    except:
        traceback.print_exc()
        logger.error('something went wrong while trying to write coordinates to the DB')
    finally:
        if db is not None:
            try:
                db.close()
            except:
                traceback.print_exc()

# img_to_analyze
def convert_selection_color_to_coords(img_to_analyze_RGB24_or_label, selected_cells, bg_color=None):
    # tmp = img_to_analyze
    coords_of_selection = []
    if img_to_analyze_RGB24_or_label is None:
        logger.error('no image -> nothing to do...')
        return coords_of_selection

    if len(img_to_analyze_RGB24_or_label.shape) == 3:
        img_to_analyze_RGB24_or_label = RGB_to_int24(img_to_analyze_RGB24_or_label)


    if bg_color is None:
        if 0xFFFFFF in img_to_analyze_RGB24_or_label:
            bg_color = 0xFFFFFF
        else:
            bg_color = 0

    cell_label = label(img_to_analyze_RGB24_or_label, connectivity=1, background=bg_color)

    # for all the selected cells --> return the coords --> probably a quite good idea...
    for region in regionprops(cell_label):
        color = img_to_analyze_RGB24_or_label[region.coords[0][0], region.coords[0][1]]
        if color in selected_cells:
            point_awlays_inside_the_cell = point_on_surface(region, cell_label)
            coords_of_selection.append(point_awlays_inside_the_cell)

    return coords_of_selection

# def convert_coords_to_image(image_for_correspondance_label_or_RGB24, coords):
#     for coord in coords:
#             # print(coords)
#             # print(coords[0], coords[1])
#             color = image_for_correspondance_label_or_RGB24[coord[0], coord[1]]
#             # this gives me all the cells I need to get
#             # then I could either get the corresponding label or the image corresponding to that
#             # print(color)
#             if return_image:
#                 output[image_for_correspondance_label_or_RGB24 == color] = color
#             else:
#                 output.append(color)

# converts back some local coords to IDs
def convert_coords_to_IDs(img_to_analyze_RGB24_or_label, selected_coords, forbidden_colors = [0xFFFFFF,0], return_image=False, new_color_to_give_to_cells_if_return_image=None):
    ids = []
    if img_to_analyze_RGB24_or_label is None:
        logger.error('Empty image --> nothing to do...')
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
            # can be used to plot clones
            if new_color_to_give_to_cells_if_return_image is not None:
                ids[img_to_analyze_RGB24_or_label == color] = new_color_to_give_to_cells_if_return_image
            else:
                ids[img_to_analyze_RGB24_or_label==color]=color
    return ids

# also do the oposite --> convert cells to database coords
# maybe I need ask the color for the conversion --> that may make sense or assume balck or white by default --> think about it ???? -->
# otherwise I also need the db_path
# and the table name
# depreacated --> use another code if possible...
def convert_selected_cells_to_local_db_coords(selected_cells, label_or_RGB24_image_corresponding_to_selected_cells,
                                              db_path, table_name_to_store_coords_to, bg_color_for_label=None):
    if not selected_cells:
        logger.error('Nothing selected --> nothing to do')
        return

    if bg_color_for_label is None:
        # auto mode to determine the label --> may fail if the image is non conventional
        if 0xFFFFFF in label_or_RGB24_image_corresponding_to_selected_cells:
            bg_color_for_label = 0xFFFFFF
        else:
            bg_color_for_label = 0

    # db = None
    try:
        cell_label = label(label_or_RGB24_image_corresponding_to_selected_cells, connectivity=1,
                           background=bg_color_for_label)
        # rps =

        # check if region props contains a point
        # or get the color of the point and return the corresponding cell --> TODO
        # get the index of the

        # cells_2D['pixel_within_cell_y'].append(point_awlays_inside_the_cell[0])
        # cells_2D['pixel_within_cell_x'].append(point_awlays_inside_the_cell[1])

        # need create a db with the right keys --> TODO

        # coords_of_selection = {'pixel_within_cell_y': [],
        #                        'pixel_within_cell_x': [],
        #                        }

        # db = TAsql(db_path)
        # print('cells_2D'*20)
        coords = []
        for region in regionprops(cell_label):
            color = label_or_RGB24_image_corresponding_to_selected_cells[region.coords[0][0], region.coords[0][1]]
            if color in selected_cells:
                # get the coord always inside the cell and store it to a db --> TODO
                # can also be used for a reload --> very smart in fact!!!!
                point_awlays_inside_the_cell = point_on_surface(region, cell_label)
                coords.append(point_awlays_inside_the_cell)
                # print(point_awlays_inside_the_cell)
                # coords_of_selection['pixel_within_cell_y'].append(point_awlays_inside_the_cell[0])
                # coords_of_selection['pixel_within_cell_x'].append(point_awlays_inside_the_cell[1])

        # db.create_and_append_table(table_name=table_name_to_store_coords_to, datas=coords_of_selection)
        add_coords_to_db(db_path, table_name_to_store_coords_to, coords)

    except:
        traceback.print_exc()
        logger.error('Something went wrong, selection could not be converted to 2D coordinates')
    # finally:
    #     if db is not None:
    #         try:
    #             db.close()
    #         except:
    #             traceback.print_exc()

    # gets an image with some selected cells and convert it to a set of local coords that can then be shown and or updated
    # pass

# maybe store the ids of the cells here then!!!
# TODO --> see my other code and get it there
# faire differents selecteurs et les mettre à tel ou tel endroit --> TODO
# TODO hack ImgDisplayWindow.display to get the drawing of the user and the selection --> TODO

if __name__ == '__main__':
    import sys

    if False:
        # test how to append selection to a clone database
        path_to_image = '/E/Sample_images/sample_images_PA/trash_test_mem/mini_empty/focused_Series012.png'
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
        path_to_image = '/E/Sample_images/sample_images_PA/trash_test_mem/mini_empty/focused_Series012.png'
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
