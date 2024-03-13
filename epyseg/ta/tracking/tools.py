"""
TODO


"""

import numpy as np
from skimage.draw import polygon, polygon_perimeter
from skimage.measure import regionprops
from skimage import measure
from collections import Counter
from natsort import natsorted
from epyseg.utils.loadlist import loadlist
from epyseg.ta.colors.colorgen import get_unique_random_color_int24
import os

from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()


def get_TA_file(file_name_without_ext, TA_file):
    """
    Returns the path to the TA file by joining the file name without extension and the TA file.

    Args:
        file_name_without_ext (str): File name without extension.
        TA_file (str): TA file.

    Returns:
        str: Path to the TA file.

    """
    return os.path.join(file_name_without_ext, TA_file)


def get_n_consecutive_input_files_and_output_folders(images_to_analyze, start_idx, nb_of_images_to_get, incr=1):
    """
    Returns a list of consecutive input files and output folders.

    Args:
        images_to_analyze (list): List of images to analyze.
        start_idx (int): Starting index.
        nb_of_images_to_get (int): Number of consecutive images to get.
        incr (int, optional): Increment value. Defaults to 1.

    Returns:
        list: List of input files and output folders.

    """
    return get_n_input_files_and_output_folders(images_to_analyze, start_idx, 0, nb_of_images_to_get, incr=incr)


def get_n_input_files_and_output_folders(images_to_analyze, start_idx, start_range, end_range_inclusive, incr=1):
    """
    Returns a list of input files and output folders within a specified range.

    Args:
        images_to_analyze (list): List of images to analyze.
        start_idx (int): Starting index.
        start_range (int): Starting range.
        end_range_inclusive (int): Ending range (inclusive).
        incr (int, optional): Increment value. Defaults to 1.

    Returns:
        list: List of input files and output folders.

    """
    try:
        return images_to_analyze[start_idx + start_range: start_idx + end_range_inclusive: incr]
    except:
        output_data = []
        for i in range(start_range, end_range_inclusive + 1, incr):
            output_data.append(get_input_file_and_output_folder(images_to_analyze, start_idx + i))
        return output_data


def get_n_files_from_list(images_to_analyze, start_idx, start_range, end_range_inclusive, incr=1):
    """
    Returns a list of files from a given list within a specified range.

    Args:
        images_to_analyze (list): List of images to analyze.
        start_idx (int): Starting index.
        start_range (int): Starting range.
        end_range_inclusive (int): Ending range (inclusive).
        incr (int, optional): Increment value. Defaults to 1.

    Returns:
        list: List of files.

    Raises:
        Exception: Raises an exception if the list is empty.

    """
    if images_to_analyze is None or not images_to_analyze:
        raise Exception('Empty list, nothing to do...')
    try:
        if start_idx + start_range < 0 or start_idx + end_range_inclusive >= len(images_to_analyze):
            raise IndexError("List index out of range")
        return images_to_analyze[start_idx + start_range: start_idx + end_range_inclusive + 1: incr]
    except:
        output_data = []
        for i in range(start_range, end_range_inclusive + 1, incr):
            if start_idx + i < 0 or start_idx + i >= len(images_to_analyze):
                output_data.append(None)
                continue
            output_data.append(images_to_analyze[start_idx + i])
        return output_data


known_parsers = ['parent',
                'ext',
                'short',
                'full_no_ext',
                'TA',  # same as full no ext
                'short_no_ext',
                'full',
                'full_no_end_slash',
                'short_pop_all_exts',  # extreme version of short no ext where series of exts are removed, e.g. hiC_250522.pairs.txt.gz -−> hiC_250522
                ]


def replace_empty_string_by_none(input_string):
    """
    Replaces an empty string with None.

    Args:
        input_string (str): Input string.

    Returns:
        str: Input string with empty string replaced by None.

    """
    if not input_string:
        return None
    return input_string


def smart_name_appender(name, condition_boolean, text_to_append_if_condition_is_true=None,
                        text_to_append_if_condition_is_false=None):
    """
    Appends a string to a name depending on a condition.

    Args:
        name (str): Name to append the string.
        condition_boolean (bool): Condition.
        text_to_append_if_condition_is_true (str, optional): Text to append if the condition is True. Defaults to None.
        text_to_append_if_condition_is_false (str, optional): Text to append if the condition is False. Defaults to None.

    Returns:
        str: Modified name.

    """
    if condition_boolean:
        if text_to_append_if_condition_is_true is not None:
            name += text_to_append_if_condition_is_true
    else:
        if text_to_append_if_condition_is_false is not None:
            name += text_to_append_if_condition_is_false
    return name


def smart_name_parser(full_file_path, ordered_output=known_parsers, appenders=None, replace_empty_by_none=False):
    """
    Parses the full file path and returns the specified output.

    Args:
        full_file_path (str): Full file path.
        ordered_output (str or list, optional): Ordered list of output formats. Defaults to known_parsers.
        appenders (str or list, optional): String or list of strings to append to the output. Defaults to None.
        replace_empty_by_none (bool, optional): Replace empty strings by None. Defaults to False.

    Returns:
        str or list: Parsed output.

    """
    if full_file_path is None:
        logger.error('Please specify an input file name')
        return
    if ordered_output is None:
        logger.error('Parsing unspecified, nothing to do')
        return

    fixed_path = full_file_path.replace('\\\\', '/').replace('\\', '/')
    filename0_without_path = os.path.basename(fixed_path)
    filename0_without_ext, ext = os.path.splitext(filename0_without_path)
    parent_dir_of_filename0 = os.path.dirname(fixed_path)

    single_output = False
    if isinstance(ordered_output, str):
        single_output = True
        ordered_output = [ordered_output]

    sorted_output = []
    for parser in ordered_output:
        if parser not in known_parsers:
            sorted_output.append(os.path.join(parent_dir_of_filename0, os.path.join(filename0_without_ext, parser)))
            continue
        if parser == 'parent':
            sorted_output.append(parent_dir_of_filename0)
            continue
        if parser == 'short_no_ext':
            sorted_output.append(filename0_without_ext)
            continue
        if parser == 'short_pop_all_exts':
            extracted = filename0_without_ext
            while '.' in extracted:
                extracted, _ = os.path.splitext(extracted)
            sorted_output.append(extracted)
        if parser == 'full_no_ext' or parser == 'TA':
            sorted_output.append(os.path.join(parent_dir_of_filename0, filename0_without_ext))
            continue
        if parser == 'short':
            sorted_output.append(filename0_without_path)
            continue
        if parser == 'full':
            sorted_output.append(full_file_path)
            continue
        if parser == 'ext':
            sorted_output.append(ext)
            continue
        if parser == 'full_no_end_slash':
            out = fixed_path
            if out.endswith('/'):
                out = out[:-1]
            sorted_output.append(out)

    if appenders is not None:
        if not isinstance(appenders, list):
            appenders = [appenders]
        sorted_output = [os.path.join(txt, *appenders) for txt in sorted_output]

    if replace_empty_by_none:
        sorted_output = [replace_empty_string_by_none(input_string) for input_string in sorted_output]

    if single_output:
        return sorted_output[0]
    return sorted_output


def get_input_file_and_output_folder(images_to_analyze, idx):
    """
    Returns the input file and output folder for a given index.

    Args:
        images_to_analyze (list): List of images to analyze.
        idx (int): Index of the image to retrieve.

    Returns:
        tuple: Input file path and output folder path.

    """
    file_path_0 = images_to_analyze[idx]
    filename0_without_ext = os.path.splitext(file_path_0)[0]
    return file_path_0, filename0_without_ext


def get_input_files_and_output_folders(images_to_analyze, start_idx, incr=1):
    """
    Returns the input files and output folders for consecutive indices.

    Args:
        images_to_analyze (list): List of images to analyze.
        start_idx (int): Starting index.
        incr (int, optional): Increment value. Defaults to 1.

    Returns:
        tuple: Input file paths and output folder paths.

    """
    file_path_0 = images_to_analyze[start_idx]
    file_path_1 = images_to_analyze[start_idx + incr]

    filename0_without_ext = os.path.splitext(file_path_0)[0]
    filename1_without_ext = os.path.splitext(file_path_1)[0]

    return file_path_0, file_path_1, filename0_without_ext, filename1_without_ext


def get_list_of_files(path):
    """
    Returns a list of files in the given path.

    Args:
        path (str or list): Path to the directory or a list of file paths.

    Returns:
        list: List of file paths.

    # Examples:
    #     >>> get_list_of_files('/path/to/files')
    #     ['/path/to/files/file1.txt', '/path/to/files/file2.txt', '/path/to/files/file3.txt']
    #
    #     >>> get_list_of_files(['/path/to/files/file1.txt', '/path/to/files/file2.txt'])
    #     ['/path/to/files/file1.txt', '/path/to/files/file2.txt']

    """
    if isinstance(path, list):
        return path

    if not path.lower().endswith('.txt') and not path.lower().endswith('.lst'):
        images_to_analyze = os.listdir(path)
        images_to_analyze = [os.path.join(path, f) for f in images_to_analyze if os.path.isfile(os.path.join(path, f))]
        images_to_analyze = natsorted(images_to_analyze)
    else:
        images_to_analyze = loadlist(path)

    return images_to_analyze


def first_image_tracking(mask_t0, labels_t0, regprps=None, assigned_ids=None, seed=None):
    """
    Performs tracking on the first image.

    Args:
        mask_t0 (ndarray): Binary mask of the first image.
        labels_t0 (ndarray): Labels of the first image.
        regprps (list, optional): List of region properties. Defaults to None.
        assigned_ids (list, optional): List of assigned IDs. Defaults to None.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        ndarray: Tracked image.

    """
    tracks_t0 = np.zeros_like(mask_t0, dtype=np.uint32)

    if regprps is None:
        regprps = regionprops(labels_t0)

    for iii, region in enumerate(regprps):
        # create a random color and store it
        if iii == 0:
            new_col = get_unique_random_color_int24(forbidden_colors=assigned_ids, seed_random=seed,
                                                    assign_new_col_to_forbidden=True)
        else:
            new_col = get_unique_random_color_int24(forbidden_colors=assigned_ids, assign_new_col_to_forbidden=True)

        tracks_t0[labels_t0 == iii + 1] = new_col

    tracks_t0[labels_t0 == 0] = 0xFFFFFF
    # save the image or return it
    return tracks_t0  # this is the image # need copy it


def deduplicate_duplicated_cells_using_random_ids(RGB24, labels=None, background=0xFFFFFF, assigned_ids=None):
    """
    Deduplicates duplicated cells using random IDs.

    Args:
        RGB24 (ndarray): RGB image.
        labels (ndarray, optional): Label image. Defaults to None.
        background (int, optional): Background color. Defaults to 0xFFFFFF.
        assigned_ids (list, optional): List of assigned IDs. Defaults to None.

    Returns:
        ndarray: Deduplicated RGB image.

    """
    if labels is None:
        labels = measure.label(RGB24, connectivity=1, background=background)

    cell_ids = []
    for region in regionprops(labels):
        cell_ids.append(RGB24[region.coords[0][0], region.coords[0][1]])

    cell_ids_n_count = Counter(cell_ids)
    duplicated_cells = {k: v for k, v in cell_ids_n_count.items() if v >= 2}

    duplicated_cells_keys = list(duplicated_cells.keys())
    for region in regionprops(labels):
        id = RGB24[region.coords[0][0], region.coords[0][1]]
        if id in duplicated_cells_keys:
            duplication_count = duplicated_cells[id]
            if duplication_count >= 2:
                duplicated_cells[id] = duplication_count - 1
                RGB24[labels == region.label] = get_unique_random_color_int24(forbidden_colors=assigned_ids,
                                                                              assign_new_col_to_forbidden=True)
            else:
                duplicated_cells_keys.remove(id)

    return RGB24


def assign_random_ID_to_missing_cells(RGB24img, labels, regprps=None, assigned_ids=None):
    """
    Assigns random IDs to missing cells in the image.

    Args:
        RGB24img (ndarray): RGB image.
        labels (ndarray): Label image.
        regprps (list, optional): List of region properties. Defaults to None.
        assigned_ids (list, optional): List of assigned IDs. Defaults to None.

    Returns:
        ndarray: RGB image with assigned random IDs.

    """
    if regprps is None:
        regprps = regionprops(labels)

    for iii, region in enumerate(regprps):
        color_of_first_pixel_of_potential_match = RGB24img[region.coords[0][0], region.coords[0][1]]

        if color_of_first_pixel_of_potential_match == 0:
            new_col = get_unique_random_color_int24(forbidden_colors=assigned_ids, assign_new_col_to_forbidden=True)
            RGB24img[labels == labels[region.coords[0][0], region.coords[0][1]]] = new_col

    return RGB24img


def get_lost_cells_between_first_and_second_set(first_set, second_set):
    """
    Returns the lost cells between the first and second set.

    Args:
        first_set (list or set): First set of cells.
        second_set (list or set): Second set of cells.

    Returns:
        set: Set of lost cells.

    Examples:
        >>> get_lost_cells_between_first_and_second_set([1, 2, 3, 4, 5], [2, 4, 6, 8])
        {1, 3, 5}

        >>> get_lost_cells_between_first_and_second_set({1, 2, 3, 4, 5}, {2, 4, 6, 8})
        {1, 3, 5}

    """
    set_t0 = first_set
    if not isinstance(set_t0, set):
        set_t0 = set(set_t0)
    set_t1 = second_set
    if not isinstance(set_t1, set):
        set_t1 = set(set_t1)
    return set_t0.difference(set_t1)


def get_common_cells(first_set, second_set):
    """
    Returns the common cells between the first and second set.

    Args:
        first_set (list or set): First set of cells.
        second_set (list or set): Second set of cells.

    Returns:
        set: Set of common cells.

    Examples:
        >>> get_common_cells([1, 2, 3, 4, 5], [2, 4, 6, 8])
        {2, 4}

        >>> get_common_cells({1, 2, 3, 4, 5}, {2, 4, 6, 8})
        {2, 4}

    """
    set_t0 = first_set
    if not isinstance(set_t0, set):
        set_t0 = set(set_t0)
    set_t1 = second_set
    if not isinstance(set_t1, set):
        set_t1 = set(set_t1)
    return set_t0.intersection(set_t1)


def get_cells_in_image(RGB24_img):
    """
    Returns the unique IDs of cells in the image.

    Args:
        RGB24_img (ndarray): RGB image.

    Returns:
        ndarray: Array of unique cell IDs.

    """
    unique_ids = np.unique(RGB24_img)
    return unique_ids


def get_cells_in_image_n_fisrt_pixel(RGB24_img):
    """
    Returns the unique IDs of cells in the image and their corresponding first pixel indices.

    Args:
        RGB24_img (ndarray): RGB image.

    Returns:
        ndarray: Array of unique cell IDs.
        ndarray: Array of indices of the first pixels of the cells.

    """
    u, indices = np.unique(RGB24_img, return_index=True)
    return u, indices


def get_cells_in_image_n_count(RGB24_img):
    """
    Returns the unique IDs of cells in the image and their corresponding counts.

    Args:
        RGB24_img (ndarray): RGB image.

    Returns:
        ndarray: Array of unique cell IDs.
        ndarray: Array of counts for each cell ID.

    """
    u, counts = np.unique(RGB24_img, return_counts=True)
    return u, counts


def plot_triangles(img, coords, tri, inner_color, perimeter_color=None):
    """
    Plot the triangles of a cell on an image.

    Args:
        img (ndarray): Image to plot on.
        coords (ndarray): Coordinates of the cell vertices.
        tri (Delaunay): Delaunay triangulation of the cell.
        inner_color (int): Inner color of the triangles.
        perimeter_color (int, optional): Perimeter color of the triangles. Defaults to None.

    """
    for simplex in tri.simplices:
        r = coords[simplex][:, 0]  # get the y coords of the simplex
        c = coords[simplex][:, 1]  # get the x coords of the simplex
        rr, cc = polygon(r, c)
        img[rr, cc] = inner_color
        if perimeter_color is not None:
            rr, cc = polygon_perimeter(r, c)
            img[rr, cc] = perimeter_color


if __name__ == '__main__':
    if True:
        tst_name = '/F/hiC_microC/haplotype_resolved_embryo/GSE121255_RAW/GSM3428927_2_4hrs-R2.dedup.pairs.txt.gz'

        print(smart_name_parser(tst_name,'short_pop_all_exts'))

        tst_name = '/F/hiC_microC/haplotype_resolved_embryo/GSE121255_RAW/GSM3428927_2_4hrs-R2'
        print(smart_name_parser(tst_name, 'short_pop_all_exts'))

        tst_name = '/F/hiC_microC/haplotype_resolved_embryo/GSE121255_RAW/GSM3428927_2_4hrs-R2.txt'
        print(smart_name_parser(tst_name, 'short_pop_all_exts'))

        import sys
        sys.exit(0)


    if True:
        print(smart_name_parser('Série 10kb X1 Spot196 mut2 TgX malex.tif.tif', 'tutu'))

        import sys
        sys.exit(0)

    if True:
        # these are the tests for the smart_name_parser

        print(smart_name_parser('/this/is/a/test/of/your/system.tif', 'parent', appenders=['toto', 'tutu',
                                                                                           'tata.tif']))  
        print(smart_name_parser('/this/is/a/test/of/your/system.tif', 'parent',
                                appenders='tata.tif'))  
        print(smart_name_parser('/this/is/a/test/of/your/system.tif', 'full_no_ext',
                                appenders='tata.tif'))  
        print(smart_name_parser('/this/is/a/test/of/your/system.tif', ['TA', 'parent'],
                                appenders='tata.tif'))  

        print('smart_name_parser', smart_name_parser(
            '/E/Sample_images/sample_images_PA/mini/focused_Series012.png'))  # ['/E/Sample_images/sample_images_PA/mini', '', 'focused_Series012', '/E/Sample_images/sample_images_PA/mini/focused_Series012', 'focused_Series012', '/E/Sample_images/sample_images_PA/mini/focused_Series012']
        print('smart_name_parser', smart_name_parser(
            '/E/Sample_images/sample_images_PA/mini/focused_Series012'))  # ['/E/Sample_images/sample_images_PA/mini', '', 'focused_Series012', '/E/Sample_images/sample_images_PA/mini/focused_Series012', 'focused_Series012', '/E/Sample_images/sample_images_PA/mini/focused_Series012']
        print('smart_name_parser', smart_name_parser(
            'focused_Series012.png'))  # smart_name_parser ['', '.png', 'focused_Series012.png', 'focused_Series012', 'focused_Series012', 'focused_Series012.png']
        print('smart_name_parser',
              smart_name_parser('D:\\Sample_images\\sample_images_PA\\trash_test_mem\\mini\\focused_Series012.png'))
        print('smart_name_parser', smart_name_parser(
            'D:\\Sample_images\\\\sample_images_PA/trash_test_mem\\mini\\focused_Series012.png'))  # hybrid widows linux names
        print('smart_name_parser',
              smart_name_parser('D:\\Sample_images\\\\sample_images_PA/trash_test_mem\\mini\\focused_Series012'))
        print('smart_name_parser', smart_name_parser(''))
        print('smart_name_parser', smart_name_parser('', replace_empty_by_none=True))
        print('smart_name_parser', smart_name_parser(None))
        print('smart_name_parser',
              smart_name_parser('/E/Sample_images/sample_images_PA/mini/focused_Series012.png',
                                ordered_output=[
                                    'tracked_cells_resized.png']))  # retruns a TA name --> it is really cool!!!
        tracked_cells_resized, TA_path = smart_name_parser('/E/Sample_images/sample_images_PA/mini/focused_Series012.png', ordered_output=['full_no_ext', 'tracked_cells_resized.png'])
        print('TADA', tracked_cells_resized, TA_path)
        tracked_cells_resized, TA_path = smart_name_parser('/E/Sample_images/sample_images_PA/mini/focused_Series012.png', ordered_output=['TA', 'tracked_cells_resized.png'])
        print('TADA2', tracked_cells_resized, TA_path)
        import sys
        sys.exit(0)

    if True:
        # test of getting n consecutive images
        # also see how to handle None --> TODO
        img_list = loadlist('/E/Sample_images/sample_images_PA/mini/list.lst')
        print(img_list)

        desired_images = get_n_input_files_and_output_folders(img_list, 1, -1, 1) # get one image before and after

        print(desired_images)
        import sys
        sys.exit(0)


    # if True:
    #     random_vx_images = (np.random.randint(2, size=(1024, 1024)) * 255).astype('uint8')
    #     cell_labels = (np.random.randint(255, size=(1024, 1024))).astype('uint8')
    #     vertices = np.where(random_vx_images == 255)  # c'est
    #     count_same = 0
    #     count_different = 0
    #
    #     start_all = timer()
    #     for i in range(len(vertices[0])):
    #         y = vertices[0][i]
    #         x = vertices[1][i]
    #         # ids = neighbors8((y, x), cell_labels).tolist() # celui ci est faux
    #         ids2 = neighbors8_2((y, x), cell_labels)
    #
    #         # if  ids!=ids2:
    #         #     print(ids, ids2, ids==ids2)
    #
    #     print('time', timer() - start_all)
    #     # start_all = timer()
    #     # for i in range(len(vertices[0])):
    #     #     y = vertices[0][i]
    #     #     x = vertices[1][i]
    #     #     ids = neighbors8((y, x), cell_labels).tolist()  # celui ci est faux # --> keep my old way but fix its errors
    #     #     ids2 = neighbors8_2((y, x), cell_labels)
    #     #
    #     #     if ids != ids2:
    #     #         print(ids, ids2, ids == ids2, y, x)
    #     #         count_different+=1
    #     #     else:
    #     #         count_same+=1
    #     #
    #     # print(count_same, count_different)
    #     # print('time', timer() - start_all)
    #
    #     # this one is much faster but why ???
    #     start_all = timer()
    #     for i in range(len(vertices[0])):
    #         y = vertices[0][i]
    #         x = vertices[1][i]
    #         ids = neighbors8((y, x), cell_labels).tolist()  # celui ci est faux # --> keep my old way but fix its errors
    #         # ids2 = neighbors8_2((y, x), cell_labels)
    #
    #         # if  ids!=ids2:
    #         #     print(ids, ids2, ids==ids2)
    #
    #     print('time', timer() - start_all)



    cells_at_t0 = [0, 1, 2, 3, 4, 5, 6, 7, 128]
    cells_at_t1 = [1, 3, 4, 5, 128, 33, 24, 52]

    print(get_lost_cells_between_first_and_second_set(cells_at_t0, cells_at_t1))
    print(get_lost_cells_between_first_and_second_set(cells_at_t1, cells_at_t0))

    img = Img(
        '/E/Sample_images/segmentation_assistant/ovipo_uncropped/200709_armGFP_suz_46hAPF_ON.lif - Series011/tracked_cells_resized.tif')
    img = RGB_to_int24(img)
    start_all = timer()
    print(len(get_cells_in_image(img)))  # ça marche et c'est super rapide en fait --> really good and fast
    print('time', timer() - start_all)
    # can get the first pixel of each using indices

    start_all = timer()
    cells, first_px = get_cells_in_image_n_fisrt_pixel(
        img)  # ça marche et c'est super rapide en fait --> really good and fast
    print(cells)
    print(first_px)  # this is the ravel index
    print('time', timer() - start_all)

    # --> 0.20 secs
    start_all = timer()
    cells, area = get_cells_in_image_n_count(img)
    print(cells)
    print(area)
    print('time', timer() - start_all)

    # full self replacement of one image using == --> 14 secs
    start_all = timer()
    for cell in cells:
        img[img == cell] = cell
    print('time', timer() - start_all)

    # start_all = timer()
    # for cell in cells:
    #
    #     img[img == cell] = cell
    # print('time', timer() - start_all)

    # encore plus lent et null --> 45 secs
    # start_all = timer()
    # for cell in cells:
    #     np.where(img == cell, cell, img)
    #     # img[img == cell] = cell
    # print('time', timer() - start_all)

    # super slow --> do not use np.where in fact cause super slow
    start_all = timer()
    for cell in cells:
        cond = np.where(img == cell)
        img[cond] = cell
    print('time', timer() - start_all)

    # print('neighbs length', len(ids))

    # print(vertices)
    # if vertices[0].size != 0:
    #     print('y', vertices[0][0])
    # if vertices[1].size != 0:
    #     print('x', vertices[1][0])
    # if vertices[0].size + vertices[1].size != 0:
    #     print('first vx', result[vertices[0][0], vertices[1][0]])
