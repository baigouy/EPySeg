# generic tools for tracking
import numpy as np
# from epyseg.img import Img, RGB_to_int24
# from scipy.spatial import Delaunay
from skimage.draw import polygon,polygon_perimeter
from skimage.measure import regionprops
from skimage import measure
from collections import Counter
import os
from natsort import natsorted
from epyseg.utils.loadlist import loadlist
from epyseg.ta.colors.colorgen import get_unique_random_color_int24
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()

def get_TA_file(file_name_without_ext,TA_file):
    return os.path.join(file_name_without_ext,TA_file)

# TODO faire un single file getter et l'etendre pr obtenir autant de fichiers que je veux!!!
def get_n_consecutive_input_files_and_output_folders(images_to_analyze, start_idx, nb_of_images_to_get, incr=1):
    # output_data = []
    # for i in range(0,nb_of_images_to_get,incr):
    #     output_data.append(get_input_file_and_output_folder(images_to_analyze, start_idx))
    # return output_data
    return get_n_input_files_and_output_folders(images_to_analyze, start_idx, 0, nb_of_images_to_get, incr=incr)

# this could be considered as being list tools
# could even just get the files then split the names only if necessary in fact does that bring anything compared to directly playing with the python list (no unless if the file of indices do not exist)
# can I get a compensation also for missing files ?????
def get_n_input_files_and_output_folders(images_to_analyze, start_idx, start_range, end_range_inclusive, incr=1):
    try:
        return images_to_analyze[start_idx+start_range: start_idx+end_range_inclusive:incr]
    except:
        # print('unsuccessful --> need be smarter')
        output_data = []
        for i in range(start_range, end_range_inclusive+1, incr):
            output_data.append(get_input_file_and_output_folder(images_to_analyze, start_idx+i))
        return output_data

# now it starts to be useful
# TODO do something to get names and or various name cutting procedures --> cause I use it all the time and can be useful
# maybe make the output customizable --> TODO
def get_n_files_from_list(images_to_analyze, start_idx, start_range, end_range_inclusive, incr=1):
    if images_to_analyze is None or not images_to_analyze:
        raise Exception('Empty list nothing to do...')
    try:
        if start_idx+start_range<0 or start_idx+end_range_inclusive>=len(images_to_analyze):
            raise IndexError("list index out of range")
        return images_to_analyze[start_idx+start_range: start_idx+end_range_inclusive+1:incr]
    except:
        # print('unsuccessful --> need be smarter')
        output_data = []
        for i in range(start_range, end_range_inclusive + 1, incr):
            if start_idx+i<0 or start_idx+i>=len(images_to_analyze):
                output_data.append(None)
                continue
            output_data.append(images_to_analyze[start_idx + i])
        return output_data

# will parse the name into smthg useful and in the desired format
# raw --> raw name
# parent --> returns parent path
# TA path --> returns TA path # --> name without the extension  # noext can be a synonym for that
# if unknown --> assume the guy wants a TA file from the path
# TODO maybe add some such as short_name and full_name

# TODO document that and parse it
# full_no_ext
# short_no_ext
# parent
# ext
# short
# full --> returun path

known_parsers = ['parent',
                'ext',
                'short',
                'full_no_ext',
                'TA', # same as full no ext
                'short_no_ext',
                 'full'
                ]
# smart_name_parser for /E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png --> ['/E/Sample_images/sample_images_PA/trash_test_mem/mini', '.png', 'focused_Series012.png', '/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012', 'focused_Series012', '/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png']



def replace_empty_string_by_none(input_string):
    if not input_string:
        return None
    return input_string

# def populate_files_within_the_ta_folder(TA_folder, extensions_to_list):
#     # qssqsdqqsdsq
#     # list files with the desired extension in the TA folder
#     files =
# sqdqsd
#     if extensions_to_list is None:# do list all files
#     pass

# TODO maybe convert that to pathlib some day as it seems much more powerful than os.path
def smart_name_parser(full_file_path, ordered_output=known_parsers, replace_empty_by_none=False): # known_parsers[1:3]
    if full_file_path is None:
        logger.error('Please specify an input file name')
        return
    if ordered_output is None:
        logger.error('Parsing unspecified --> nothing to do')
        return

    # ORDERED FILES

    fixed_path = full_file_path.replace('\\\\','/').replace('\\','/') # should work for most path
    filename0_without_path = os.path.basename(fixed_path)
    filename0_without_ext, ext = os.path.splitext(filename0_without_path)#[0] # TODO check if no ext how to do
    parent_dir_of_filename0 = os.path.dirname(fixed_path)

    single_output = False
    if isinstance(ordered_output, str):
        single_output = True
        ordered_output=[ordered_output]

    sorted_output = []
    for parser in ordered_output:
        if parser not in known_parsers:
            sorted_output.append(os.path.join(parent_dir_of_filename0,os.path.join(filename0_without_ext, parser)))
            continue
        if parser == 'parent':
            sorted_output.append(parent_dir_of_filename0)
            continue
        if  parser =='short_no_ext':
            sorted_output.append(filename0_without_ext)
            continue
        if  parser =='full_no_ext' or parser=='TA':
            sorted_output.append(os.path.join(parent_dir_of_filename0, filename0_without_ext))
            continue
        if  parser =='short':
            sorted_output.append(filename0_without_path)
            continue
        if parser == 'full':
            sorted_output.append(full_file_path) # TODO maybe still format it in the system way --> can be useful
            continue
        if parser == 'ext':
            sorted_output.append(ext) # TODO maybe still format it in the system way --> can be useful
            continue

    # TA_output_filename = os.path.join(parent_dir_of_filename0, filename0_without_ext,   'epyseg_raw_predict.tif')  # TODO allow custom na

    if replace_empty_by_none:
        sorted_output = [replace_empty_string_by_none(input_string) for input_string in sorted_output]
    # print('sorted_output',sorted_output)
    # return (*sorted_output,)+() # necessary for easy unpacking --> NO IN FACT NOT NECESSARY NOR USEFUL
    if single_output:
        return sorted_output[0]
    return sorted_output # ça marche --> pas besoin de tuple mais faut que ce soit de la bonne taille mais en fait devrait tjrs marcher

def get_input_file_and_output_folder(images_to_analyze, idx):
    file_path_0 = images_to_analyze[idx]
    filename0_without_ext = os.path.splitext(file_path_0)[0]
    return file_path_0, filename0_without_ext

def get_input_files_and_output_folders(images_to_analyze, start_idx, incr=1):
    file_path_0 = images_to_analyze[start_idx]
    file_path_1 = images_to_analyze[start_idx + incr]

    # print('files', file_path_1, file_path_0)

    # # weird to have that --> could change this...
    # if not file_path_1.endswith('.tif') or not file_path_0.endswith('.tif'):
    #     continue

    # print('files', file_path_1, file_path_0)

    filename0_without_ext = os.path.splitext(file_path_0)[0]
    filename1_without_ext = os.path.splitext(file_path_1)[0]

    return file_path_0, file_path_1, filename0_without_ext, filename1_without_ext

def get_list_of_files(path):
    if isinstance(path,list):
        return path
    if not path.lower().endswith('.txt') and not path.lower().endswith('.lst'):
        images_to_analyze = os.listdir(path)
        images_to_analyze = [os.path.join(path, f) for f in images_to_analyze if os.path.isfile(os.path.join(path, f))]
        images_to_analyze = natsorted(images_to_analyze)
    else:
        images_to_analyze = loadlist(path)

    return images_to_analyze

# do the first cell --> just a filling of the image
def first_image_tracking(mask_t0, labels_t0, regprps=None, assigned_ids=None, seed=None):
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
        # print(new_col)
        tracks_t0[labels_t0 == iii + 1] = new_col

    tracks_t0[labels_t0 == 0] = 0xFFFFFF
    # save the image or return it
    return tracks_t0  # this is the image # need copy it


def deduplicate_duplicated_cells_using_random_ids(RGB24, labels=None, background=0xFFFFFF, assigned_ids=None):
    if labels is None:
        labels = measure.label(RGB24, connectivity=1,
                              background=background)

    # plt.imshow(labels)
    # plt.show()

    cell_ids = []
    for region in regionprops(labels):
        cell_ids.append(RGB24[region.coords[0][0], region.coords[0][1]])

    cell_ids_n_count = Counter(cell_ids)
    duplicated_cells = {k: v for k, v in cell_ids_n_count.items() if v>=2}

    # for id, counts in duplicated_cells.items():
    #     print(id,'*-*', counts)

    duplicated_cells_keys = list(duplicated_cells.keys())
    for region in regionprops(labels):
        id = RGB24[region.coords[0][0], region.coords[0][1]]
        if id in duplicated_cells_keys:
            duplication_count =duplicated_cells[id]
            if duplication_count>=2:
                duplicated_cells[id]=duplication_count-1
                RGB24[labels == region.label] = get_unique_random_color_int24(forbidden_colors=assigned_ids,
                                                                                      assign_new_col_to_forbidden=True)
            else:
                # if id in duplicated_cells_keys:
                duplicated_cells_keys.remove(id)


    # ça marche --> get all elements that are above two and fix them
    # for id, counts in cell_ids_n_count.items():
    #     print(id,'*-*', counts)
    return RGB24

def assign_random_ID_to_missing_cells(RGB24img, labels, regprps=None,  assigned_ids=None): #background=255,
    if regprps is None:
        regprps = regionprops(labels)

    for iii, region in enumerate(regprps):
        color_of_first_pixel_of_potential_match = RGB24img[region.coords[0][0], region.coords[0][1]]

        if color_of_first_pixel_of_potential_match == 0:
                new_col = get_unique_random_color_int24(forbidden_colors=assigned_ids, assign_new_col_to_forbidden=True)
                RGB24img[labels == labels[region.coords[0][0], region.coords[0][1]]] = new_col

    return RGB24img


def get_lost_cells_between_first_and_second_set(first_set, second_set):
    # if not set convert to sets
    set_t0 = first_set
    if not isinstance(set_t0, set):
        set_t0 = set(set_t0)
    set_t1 = second_set
    if not isinstance(set_t1, set):
        set_t1 = set(set_t1)
    return set_t0.difference(set_t1)


def get_common_cells(first_set, second_set):
    # if not set convert to sets
    set_t0 = first_set
    if not isinstance(set_t0, set):
        set_t0 = set(set_t0)
    set_t1 = second_set
    if not isinstance(set_t1, set):
        set_t1 = set(set_t1)
    return set_t0.intersection(set_t1)


# this is super fast and just requires an RGB24 bit image as input
def get_cells_in_image(RGB24_img):
    unique_ids = np.unique(RGB24_img)
    return unique_ids


# can easily get cell area this way too

# indices are the ravel indices --> need be reconverted back to x and y coords but nice because that's fast
def get_cells_in_image_n_fisrt_pixel(RGB24_img):
    u, indices = np.unique(RGB24_img, return_index=True)
    return u, indices


def get_cells_in_image_n_count(RGB24_img):
    u, counts = np.unique(RGB24_img, return_counts=True)
    return u, counts


# # really the fastest algo --> replace all by that 
# def neighbors8(vx_coords, cell_labels):
#     min_y = vx_coords[0] - 1
#     max_y = vx_coords[0] + 2
#     min_x = vx_coords[1] - 1
#     max_x = vx_coords[1] + 2
#     min_x = min_x if min_x >= 0 else 0
#     max_x = max_x if max_x < cell_labels.shape[1] else cell_labels.shape[1]
#     min_y = min_y if min_y >= 0 else 0
#     max_y = max_y if max_y < cell_labels.shape[0] else cell_labels.shape[0]
#     return cell_labels[min_y:max_y, min_x:max_x].ravel()
# 
# 
# # TODO compare both algos and find fastest
# # VERY SLOW ALGO NEVER USE!!!
# def neighbors8_2(vx_coords, cell_labels):
#     coords = []
#     for i in [-1, 0, 1]:
#         for j in [-1, 0, 1]:
#             y = vx_coords[0] + i
#             x = vx_coords[1] + j
#             if y >= 0 and y < cell_labels.shape[0]:
#                 if x >= 0 and x < cell_labels.shape[1]:
#                     coords.append(cell_labels[vx_coords[0] + i, vx_coords[1] + j])
#     return coords
# def triangulate_a_cell(coords):
#     tri = Delaunay(coords)
#     return tri


# TODO maybe also directly support plotting tri without coords --> assume tri contains all the coords of the vertices --> hack this code to make it better but almost there
def plot_triangles(img, coords, tri, inner_color, perimeter_color=None):
    # plot the delaunay triangulation of a cell in a given color
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
        # these are the tests for the smart_name_parser

        print('smart_name_parser', smart_name_parser(
            '/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png'))  # ['/E/Sample_images/sample_images_PA/trash_test_mem/mini', '', 'focused_Series012', '/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012', 'focused_Series012', '/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012']
        print('smart_name_parser', smart_name_parser(
            '/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012'))  # ['/E/Sample_images/sample_images_PA/trash_test_mem/mini', '', 'focused_Series012', '/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012', 'focused_Series012', '/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012']
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
              smart_name_parser('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png',
                                ordered_output=[
                                    'tracked_cells_resized.png']))  # retruns a TA name --> it is really cool!!!
        tracked_cells_resized, TA_path = smart_name_parser('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png', ordered_output=['full_no_ext', 'tracked_cells_resized.png'])
        print('TADA', tracked_cells_resized, TA_path)
        tracked_cells_resized, TA_path = smart_name_parser('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png', ordered_output=['TA', 'tracked_cells_resized.png'])
        print('TADA2', tracked_cells_resized, TA_path)
        import sys
        sys.exit(0)

    if True:
        # test of getting n consecutive images
        # also see how to handle None --> TODO
        img_list = loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini/list.lst')
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
