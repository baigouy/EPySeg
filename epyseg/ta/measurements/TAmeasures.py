# TODO --> could store local neighbors like vertices and bonds for cells by using  border_cells_plus_one = get_border_cells_plus_one(get_cells_and_their_neighbors_from_image(lab_cells, vertices=np.stack(np.where(lab_cells == 0), axis=1),bg_color=0), border_cells, remove_border_cells=True) # this is as fast and detects nicely the cells having no vertices --> maybe I should by the way rather use that for cell neighborhood --> TODO # --> maybe a smart idea in fact 

# TODO --> 3D should now work but test it still

# fixé la polarite je pense globalement c'est presque les memes valeurs mais pas à 100% --> is that due to boundaries of the image or nee check the sorting of the pixels in both casee to see how they differ and if that may explain errors


# nb there might be a big in number of vertices computation too --> needs a fix --> seems be different between TA and pyTA --> fix
# also bug in is border cells --> all cells are border cells in pyta --> that sucks --> really need improve my algo and do more checks!!

# now connect and finalize tracking and the helper for correction --> TODO then edit the manuscript and do test of the install within conda or better miniconda and do so in an envirnoment to get it to work also need finalize plots as graphs or as images --> see how I can do that and improve things rapidly, should not be too hard actually then finalize and imporve the z depth ratio and the pixel width to make it automatically added to the ehight so that no action is required also detect automaticaly when a table needs be updated because the seg mask changed !!!
# then done and finalize the pipeline for the RNAseq by Babis!!!

# TODO store bond length in an array so that I can get the packing of the cell --> in fact should not be that hard to docs
# just get bond length at the same time as I get the cell and that would do the job --> then count nb of occurences below


# see how I can associate it !!!!
# associate all bonds to cells --> TODO and to finalize


# nb there seems to be an error with pixel_count of bonds_2D as it is not an integer --> see why --> but can easily be fixed I guess --> maybe linked to the perimeter approx of scipy
# almost all ok now !!!


# can I have all bonds of a cell
# --> see how to best do that


# could also remove vx1_x and vx2_x and replace it by the ID of the vertices and things can be recovered with a simple join
# TODO add primary key because it can be very useful
# TODO --> shall I store bonds --> ????


# I can easily get all bonds around by looking at the level of the cell by looking at the perimeter, one complexity is still to get the pairs of vertices
# ideally that would be cool to also get super tiny bonds of two pixels long --> see how I can compute that ???
# shall I store bonds and also
# I now miss --> I think I have all I need --> see if and how I can improve it
# and is_border_cell_plus_one and 'vx_1_z': [],
#                 'vx_2_z': [],
#
# --> not too hard to do I guess!!!


# compute average intensity of cytoplasm

# almost ok but just add all the missing ones !!!

# TODO --> now add the database saving part --> then almost all done
# nice sql tuto
# https://datatofish.com/create-database-python-using-sqlite3/

# bug is fixed

# TODO --> store all in the db ...
# now store all in the db then I'm almost done in a way
# see all the TA parameters to see which one I can recover
# instead of storing vx pos i could store local vx id --> gain of space and it's easy to get back to the coords anyway too
# finalize also the plotter

# how can I get bond length --> shall I store the data somewhere ???

# finally create the db --> TODO

# i really have evrything then do the TA code

# do plots for polarity or alike ???
# TODO add to all regionprops the intensity so that intensity can be computed, maybe do it manually for the

# TODO base myself on createCellDb of TA in UltimateDbsCreator --> TODO
# TODO need a code to detect border cells and border cells plus one because very useful!!!
# TODO make a stretch nematic also à la TA just to keep if for consistency --> TODO


# strecth à la TA -->  Point2D.Double S1S2 = flood.compute_stretch(1.)[2];
#                         Point2D.Double center = flood.getCenter2D();


# TODO try call IJ from python call(["./ImageJ-linux64", "myscript.py"]) or use the python imageJ thingy --> just the path need be defined somewhere
# can try both and if that does not work then let the user save the file and open it manually!!!


# Perimeter = No. of horizontal & vertical links + (No. of diagonal links * ) --> http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc --> smart in fact and fast because connectivity is not required
# in fact no need to sort just need links and compute the sum of every link --> in fact that is terribly fast I think --> still need think how to get just the links
# en fait ils calculent pixel par pixel en fonction de leur connectivité


# order is required though for the polarity measures
# TODO create a status table that gets updated when an image is changed or saved --> just a boolean up to date so that one can easily update things --> much simpler than the algos and what I do in TA --> very smart in fact
# test the stuff better!
# how to add missing vertices at the margin


# get an image and for each cell derive its contour and measure its perimeter

# https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.perimeter
import math
import os.path
from multiprocessing import Pool
from scipy import ndimage
import random
from scipy.spatial.distance import squareform
from skimage.measure._regionprops import RegionProperties
from epyseg.img import Img, pad_border_xy
from skimage.measure import label, regionprops
import numpy as np
import traceback
from epyseg.ta.measurements.nematic import compute_stretch_nematic
from epyseg.tools.early_stopper_class import early_stop
from epyseg.utils.loadlist import loadlist
from epyseg.ta.database.sql import TAsql, get_property
from epyseg.ta.segmentation.neo_wshed import wshed
# from epyseg.ta.clones.clone_fuser import distance_between_points
from epyseg.ta.measurements.nematic_measurements import compute_polarity_nematic
from epyseg.ta.measurements.measurements3D.get_point_on_surface_if_centroid_is_bad import point_on_surface
from epyseg.ta.measurements.measurements3D.measures3D import compute_3D_surfacearea, perimeter_3D
from epyseg.ta.measurements.measurements3D.triangulate import triangulate
from epyseg.ta.tracking.rapid_detection_of_vertices import detect_vertices_and_bonds, cantor_color_bonds, neighbors8
from timeit import default_timer as timer
from epyseg.ta.tracking.tools import smart_name_parser
from epyseg.ta.tracking.tracking_error_detector_and_fixer import get_border_cells, get_border_cells_plus_one, \
    get_cells_and_their_neighbors_from_image
from epyseg.ta.utils.rps_tools import pavlidis, compute_distance_between_consecutive_points
import math
from math import sqrt
from epyseg.tools.logger import TA_logger  # logging

logger = TA_logger()

__DEBUG__ = False


# nb for heightmap I need also the scaling factor for Z/(x or y) dim --> MEGA URGENT TODO --> BUT I'M almost there
# also make this a class that can do things
# save all to a db
# compute nematic polarity
# then all done
# make options to know if things need be computed or not ???

# do a mt class in the stuff
# maybe start soft with setting processors and alike --> TODO and good idea
# finalize tracking --> all can be done today
# need associte vertices to bonds --> quite easy also check all the things I have in TA and check if everything is there ?
# allow edit cell divs and death maybe later or at some point...
# compute intentity with or without vertices --> TODO

# __ parameters are for debug only --> do not USE!!!
# TODO start here also by finalizing the mask --> pb is if there is a change in the next step that will be a pb because masks will not match --> see how I can fix that
# min_cell_size

# that may work but I assume it will be slow --> how can I ensure there is no pb --> always add None and change the None to a value if possible??? --> maybe it's a good idea --> can I do that
# see how I can do that !!!!


# try it and add random error to it
# ça marche et ça ne coûte rien en fait --> fill les None values avec des None
# voir comment faire pr que les lignes soient tjrs remplies pareil --> sinon si error happens log it and don't add the line at all
# check for single pixels cells if things work especially for triangulation

def distance_between_points(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_sum_and_avg_intensity(original_image, reg, idx=None, y_shift_to_transfrom_local_coords_to_global_coords=0,
                              x_shift_to_transfrom_local_coords_to_global_coords=0):
    # print(type(reg))

    if len(original_image.shape) >= 3:
        axis = 0
    else:
        axis = None

    # if isinstance(reg, regionprops):
    if idx is not None:
        final_reg = reg[idx]
    else:
        final_reg = reg

    if isinstance(final_reg, RegionProperties):
        ys = final_reg.coords[:, 0]
        xs = final_reg.coords[:, 1]
    else:
        if isinstance(reg, list):
            final_reg = np.asarray(final_reg)
        #     ys = final_reg[:, 0]
        #     xs = final_reg[:, 1]
        # else:
        #     print('TADAT ', type(reg), print(reg), print(final_reg))
        ys = final_reg[:, 0]
        xs = final_reg[:, 1]

    sum_bd_intensity = np.sum(original_image[
                                  ys + y_shift_to_transfrom_local_coords_to_global_coords,
                                  xs + x_shift_to_transfrom_local_coords_to_global_coords],
                              axis=axis)
    avg_bd_intensity = sum_bd_intensity / float(len(xs))
    # else:
    #     if isinstance(reg,list):
    #         final_reg = np.asarray(reg)
    #     sum_bd_intensity = np.sum(original_image[
    #                                   final_reg[:, 0] + y_shift_to_transfrom_local_coords_to_global_coords,
    #                                   final_reg[:, 1] + x_shift_to_transfrom_local_coords_to_global_coords],
    #                               axis=axis)
    #     avg_bd_intensity = sum_bd_intensity / float(len(reg))

    return sum_bd_intensity, avg_bd_intensity


def get_orientation(first_vertex, second_vertex):
    bd_orientation = math.degrees(atan_RAD(first_vertex[0] - second_vertex[0], first_vertex[1] - second_vertex[1]))
    while bd_orientation >= 180.:
        bd_orientation -= 180.
    return bd_orientation


# TODO CHECK IF I REALLY NEED THAT / compare to atan2 for example but ok for now
# TODO --> probably smarter to use modulo as stated here https://stackoverflow.com/questions/37358016/numpy-converting-range-of-angles-from-pi-pi-to-0-2pi
def atan_RAD(y, x):
    # print(math.pi/2.)
    # same x -->
    if abs(x) == 0.:
        return math.pi / 2.
    res = math.atan(abs(y) / abs(x))
    # if abs(x) == 0:
    #     print('abs(y) / abs(x)', abs(y) / abs(x), abs(x), math.atan(abs(y) / abs(x)))
    res = math.pi - res if x <= 0. and y > 0. else res
    res = math.pi + res if x <= 0. and y <= 0. else res
    res = 2. * math.pi - res if x > 0. and y <= 0. else res
    return res
    # public static double calculate_angle_in_radians(double x, double y) {
    #     double res = Math.atan(Math.abs(y) / Math.abs(x));
    #     res = (x <= 0. && y > 0.) ? Math.PI - res : res;
    #     res = (x <= 0. && y <= 0.) ? Math.PI + res : res;
    #     res = (x > 0. && y <= 0.) ? 2. * Math.PI - res : res;
    #     return res;
    # }


def fill_missing_with_Nones(database):
    # ensure that all lists have the same nb of entries --> need add None before or after the current data --> in fact this is complex --> think how I can do that ???
    # need check how many entries are there and if the column existed before or not
    # in fact no pb because it is done image per image --> the only pb will be when I will try to merge data from different images which does not really make sense for databases that are very different --> see how to do that
    # check all have same length and if not --> append empty values to it to get the same size --> the sad thing is that it must be done after each entry --> maybe be slow
    max_length = 0
    for k, v in database.items():
        max_length = max(max_length, len(v))
    for k, v in database.items():
        if len(v) != max_length:
            missing = [None] * (max_length - len(v))
            v.extend(missing)
    return database


# do a remove things to small to be cells --> in fact need run the whsed on it
def TAMeasurements(file_or_list, __forced_orig=None, __forced_cells=None, __forced_heightmap=None,
                   measure_polarity=False, measure_3D=False, min_cell_size=10, progress_callback=None,
                   bond_cut_off=2, multi_threading=True):  # if less or equal to 2 --> is not a bond but is a vertex
    # we MT the whole stuff because it's fairly easy to do so
    if isinstance(file_or_list, list):
        start = timer()
        if multi_threading:
            from tqdm import tqdm
            from functools import partial
            import sys
            import multiprocessing
            # import platform
            #
            # if platform.system() == "Darwin":
            #     multiprocessing.set_start_method('spawn')

            nb_procs = multiprocessing.cpu_count()-1
            if nb_procs<=0:
                nb_procs = 1
            print('using',nb_procs, 'processors')

            pool = Pool(processes=nb_procs)
            process = partial(TAMeasurements, measure_polarity=measure_polarity, measure_3D=measure_3D,
                           min_cell_size=min_cell_size, bond_cut_off=bond_cut_off)
            for i, _ in enumerate(tqdm(pool.imap_unordered(process, file_or_list), total=len(file_or_list))):
                # pass
                # sys.stderr.write('\rdone {0:%}'.format(i / len(merge_names))) # --> I could use that to plot with the other progressbar --> ok
                # cool --> I can use that to display progress in the other progress bar --> the QT one
                # pass
                if early_stop.stop:
                    return
                if progress_callback is not None:
                    progress_callback.emit((i / len(file_or_list)) * 100)

            pool.close()
            pool.join()
            print('total time', timer() - start)
            return
        else:
            for iii, file in enumerate(file_or_list):
                try:
                    if early_stop.stop:
                        return
                    if progress_callback is not None:
                        progress_callback.emit((iii / len(file_or_list)) * 100)
                    else:
                        print(str((iii / len(file_or_list)) * 100) + '%')
                except:
                    pass
                try:
                    TAMeasurements(file, measure_polarity=measure_polarity, measure_3D=measure_3D,
                                   min_cell_size=min_cell_size, bond_cut_off=bond_cut_off)
                except:
                    if __DEBUG__:
                        print('an error occurred while processing file ' + str(file))
                    traceback.print_exc()
                    logger.error('an error occurred while processing file ' + str(file))
            print('total time', timer() - start)
            return

    start = timer()


    if __DEBUG__:
        print('processing file',file_or_list)
    #
    bonds_associated_to_a_given_cell = {}
    bond_lengths_associated_to_a_given_cell = {}
    local_bond_ID_and_px_count = {}

    TA_path = None
    db_path = None

    # original_image = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png')[..., 1]

    if early_stop.stop == True:
        # print('early stop')
        return

    # instantiate the different databases

    # some will be dynamically added such as the channels --> but will be a pb if several images have different channel nb --> think about an easy fix ???
    # things from TA that also need be added

    # TO BE ADDED vx_coords_cells sum_px_intensity_cells_ch1 S1_stretch_cells vx_coords_cells orientation_cells ALSO NEED CENTER OF MASS FOR EACH CHANNEL!!! sum_px_intensity_perimeter_ch1 P1_polarity_ch1 P1_polarity_ch1_normalized V1_polarity_ch1 x_centroid_intensity_weighted_ch1 local_id_of_bonds
    vertices_2D = {'local_ID': [],
                   'x': [],
                   'y': [],
                   'is_border': [],
                   'is_border_plus_one': [],
                   }
    #                axis_major_length
    #             axis_minor_length
    #             eccentricity
    #             orientation ranging from -pi/2 to pi/2 counter-clockwise.
    cells_2D = {'local_ID': [],
                'cytoplasmic_area': [],
                'area': [],
                # 'major_axis_length': [],
                # 'minor_axis_length': [],
                'orientation': [],
                'elongation': [],
                'S1': [],
                'S2': [],
                'pixel_within_cell_x': [],
                'pixel_within_cell_y': [],
                'centroid_x': [],
                'centroid_y': [],
                'first_pixel_x': [],
                'first_pixel_y': [],
                'perimeter': [],
                'perimeter_pixel_count': [],
                'vertices': [],
                'nb_of_vertices_or_neighbours': [],
                'nb_of_vertices_or_neighbours_cut_off': [],
                'bond_cut_off': [],  # --> same as bd_size_cut_off in TA
                'bonds': [],
                'is_border': [],
                # formerly is_border_cell --> smarter this way so that I can have the same code for all the databases
                'is_border_plus_one': [],  # TODO
                }  # local_id_cells --> do I need that instead --> think about it
    cells_3D = {'local_ID': [],
                # 'centroid_z': [], # in fact does that make sense to have that and there are several ways to get it --> wait and see if users or I need and implement it the smart way then
                # 'pixel_within_cell_z': [], # in fact does that make sense to have that and there are several ways to get it --> wait and see if users or I need and implement it the smart way then
                'area3D': [],
                # NB adding 3D is smarter for using proper natural join because otherwise it does not work well
                'area_flat_height': [],
                'perimeter3D': [],
                }
    # TO BE ADDED TO BONDS DYNAMICALLY: sum_px_int_vertices_excluded_ch1 sum_px_int_vertices_included_ch1
    bonds_2D = {'local_ID': [],
                'length': [],
                'pixel_count': [],
                'orientation': [],
                # 'vx_1_x': [],
                # 'vx_1_y': [],
                # 'vx_2_x': [],
                # 'vx_2_y': [],
                'vx_1': [],
                'vx_2': [],
                'first_pixel_x': [],
                'first_pixel_y': [],
                'cell_id1_around_bond': [],
                'cell_id2_around_bond': [],
                'is_border': [],  # is_border_bond # new addition but can be useful
                'is_border_plus_one': [],
                }
    bonds_3D = {'local_ID': [],
                'length3D': [],
                'vx_1_z': [],  # do I want that --> yes, may be useful in fact...
                'vx_2_z': [],
                # 'first_pixel_z': [],
                }
    # see what else I can add and how I can handle all of these things myself
    # cells_2D

    if __DEBUG__:
        print('debug #1')

    # TODO handle several channels for polarity and alike
    if __forced_orig is not None:
        original_image = __forced_orig  # if no shift max and min should always be 255 otherwise there is an error somewhere
    else:
        original_image = Img(file_or_list)
        TA_path = smart_name_parser(file_or_list, ordered_output='full_no_ext')
        db_path = os.path.join(TA_path, 'pyTA.db')

    # if this path is None then debug --> display the image otherwise save it then save stuff

    cells = None
    if __forced_cells is not None:
        cells = __forced_cells  # TODO --> hack this
        if len(cells.shape) >= 3:
            cells = cells[..., 0]
    else:
        handCorrection1, handCorrection2 = smart_name_parser(file_or_list,
                                                             ordered_output=['handCorrection.png',
                                                                             'handCorrection.tif'])
        if os.path.isfile(handCorrection2):
            cells = Img(handCorrection2)
        elif os.path.isfile(handCorrection1):
            cells = Img(handCorrection1)
        if cells is None:
            if __DEBUG__:
                print('File not found ' + str(handCorrection2) + ' please segment the images first')
            logger.error('File not found ' + str(handCorrection2) + ' please segment the images first')
            return
        if len(cells.shape) >= 3:
            cells = cells[..., 0]

    if __DEBUG__:
        print('debug #2')

    # I need add a border to this image otherwise there will be a bug --> must be done in any case
    if min_cell_size is not None and min_cell_size > 0:
        # plt.imshow(cells)
        # plt.show()
        cells = wshed(cells, seeds='mask', min_seed_area=min_cell_size)  # restore border
        # plt.imshow(cells)
        # plt.show()

    # add a fake border to the image --> that is absolutely necessary !!! --> KEEP
    cells = pad_border_xy(cells, mode=255)  # add a wite border to the seg mask
    # plt.imshow(cells)
    # plt.show()

    # need save the mask
    if TA_path is not None:
        Img(cells, dimensions='hw').save(os.path.join(TA_path,
                                                      'handCorrection.tif'))  # TODO --> replace this with a clean save function because that is a bit complicated what i do here!!!
    else:
        # plt.imshow(cells)
        # plt.show()
        pass


    if __DEBUG__:
        print('debug #3')

    # TODO also store vertices and bonds ids --> a quite good idea in fact --> can be used for plots --> fairly good idea

    height_map = None
    if measure_3D:
        if __forced_heightmap is not None:
            height_map = __forced_heightmap
        else:
            height_map_file = smart_name_parser(file_or_list, ordered_output=['height_map.tif'])[0]

            # ideally should apply the rescaling factor to the heightmap if any or assume rescaling factor is 1 --> see and make sure everything is done in there
            if os.path.isfile(height_map_file):
                try:
                    height_map = Img(height_map_file)
                    if db_path is not None:
                        try:
                            voxel_z_over_x_ratio = get_property(db_path, 'voxel_z_over_x_ratio')  # imp
                        except:
                            if __DEBUG__:
                                print('voxel_z_over_x_ratio property not found in the db "' + str(
                                db_path) + '", assuming scaling factor is 1, which is very unlikely.')
                            logger.warning('voxel_z_over_x_ratio property not found in the db "' + str(
                                db_path) + '", assuming scaling factor is 1, which is very unlikely.')
                    else:
                        voxel_z_over_x_ratio = 1.  # no change to the heightmap
                        # logger.warning('db not found, assuming scaling factor is 1, which is very unlikely.')
                    if voxel_z_over_x_ratio != 1.:
                        height_map = height_map * voxel_z_over_x_ratio
                        if __DEBUG__:
                            print(
                            'applying aspect ratio ' + str(voxel_z_over_x_ratio) + ' to the height map of file ' + str(
                                file_or_list))
                        logger.info(
                            'applying aspect ratio ' + str(voxel_z_over_x_ratio) + ' to the height map of file ' + str(
                                file_or_list))
                except:
                    # logger.error(
                    #     'height map' + str(height_map_file) + ' could not be loaded, 3D measurements are not possible')
                    traceback.print_exc()
                    pass
        if height_map is None:
            if __DEBUG__:
                print('height map' + str(height_map_file) + ' file is missing, 3D measurements are not possible')
            logger.error('height map' + str(height_map_file) + ' file is missing, 3D measurements are not possible')  # log that there is a missing height map and that not much can be done


    if __DEBUG__:
        print('debug #4')
    # height_map = None #Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png')[..., 1]
    # height_map = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png')[..., 0] # another good test because all is 0 there --> marche car aire est proche de l'aire 2D par contre pr handcorrection ça marche pas car tres different de l'aire 2D --> see why and how to fix it
    # height_map = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/handCorrection.png')[..., 0] # good for a test as all heights are the same... --> in fact that makes sense since some heigts are 0 and some are 255 since the centroid of the polygon is always 0

    # all seems to work now
    lab_cells = label(cells, connectivity=1, background=255)

    if __DEBUG__:
        print('debug #4a')
    if TA_path is not None:
        Img(lab_cells, dimensions='hw').save(os.path.join(TA_path,
                                                          'cells.tif'))  # Pretty much the same as cell_identity.tif in TA --> cool and more useful in a way!!!
    else:
        # plt.imshow(cells)
        # plt.show()
        pass
    if __DEBUG__:
        print('debug #4b')
    #
    # vertices = detect_vertices_and_bonds(lab_cells, boundary_color=0)
    vertices = detect_vertices_and_bonds(lab_cells, boundary_color=0)

    if __DEBUG__:
        print('debug #4c')

    width = vertices.shape[1]
    height = vertices.shape[0]


    if __DEBUG__:
        print('debug #4d')
    # ok whereas the other one gives a bug

    # c'est ce code qui donne une seg fault
    try:
        border_cells = get_border_cells(lab_cells, bg_color=0)
    except:
        traceback.print_exc()
        logger.error('Could not detect border cells')
        border_cells=[]
    # border_cells=[]
    # border_cells_plus_one = get_border_cells_plus_one(get_cells_and_their_neighbors_from_image(lab_cells, vertices=np.asarray(np.where(vertices==255))), border_cells, remove_border_cells=True)

    if __DEBUG__:
        print('debug #4e')

    # this is super slow --> 4 secs --> that really sucks --> but in fact the overhead is just for the first image then it's faster --> that probably would work
    # works as nicely with pixels of the boundary
    # border_cells_plus_one = get_border_cells_plus_one(get_cells_and_their_neighbors_from_image(lab_cells, vertices=np.stack(np.where(vertices != 0), axis=1),bg_color=0), border_cells, remove_border_cells=True)
    border_cells_plus_one = get_border_cells_plus_one(
        get_cells_and_their_neighbors_from_image(lab_cells, vertices=np.stack(np.where(lab_cells == 0), axis=1),
                                                 bg_color=0), border_cells,
        remove_border_cells=True)  # this is as fast and detects nicely the cells having no vertices --> maybe I should by the way rather use that for cell neighborhood --> TODO


    if __DEBUG__:
        print('debug #4f')
    # exclude border cells plus one for now and add it later --> need a bit of recoding and no big deal!!!

    # print(np.nonzero(vertices))

    border_vertices = []
    border_vertices_plus_one = []

    if __DEBUG__:
        print('debug #5')

    # this creates a local ID for vertices (same as a label for vertices) # may be numbaised to gain speed if needed
    non_zero_y, non_zero_x = np.nonzero(vertices)
    locally_labeled_vertices = np.zeros_like(vertices, dtype=lab_cells.dtype)
    for iii in range(len(non_zero_x)):
        locally_labeled_vertices[non_zero_y[iii], non_zero_x[iii]] = iii + 1
        # also we add it to the database
        vertices_2D['local_ID'].append(iii + 1)
        # print('non_zero_x[iii]',non_zero_x[iii])
        x = non_zero_x[iii]
        y = non_zero_y[iii]
        vertices_2D['y'].append(y)
        vertices_2D['x'].append(x)
        if x == 0 or y == 0 or x == width - 1 or y == height - 1:
            vertices_2D['is_border'].append(1)
            border_vertices.append(iii + 1)
        else:
            vertices_2D['is_border'].append(0)
        if x == 1 or y == 1 or x == width - 2 or y == height - 2:
            vertices_2D['is_border_plus_one'].append(1)
            border_vertices_plus_one.append(iii + 1)
        else:
            vertices_2D['is_border_plus_one'].append(0)

    if __DEBUG__:
        print('debug #6')

    if TA_path is not None:
        Img(locally_labeled_vertices, dimensions='hw').save(
            os.path.join(TA_path, 'vertices.tif'))  # similar but not identical to TA vertices.tif file
    else:
        # plt.imshow(locally_labeled_vertices)
        # plt.show()
        pass

    # plt.imshow(localy_labeled_vertices)
    # plt.show()

    # need color all vertices
    # vertices_coords = np.nonzero(vertices)

    # plt.imshow(vertices)
    # plt.show()
    # should I label the vertices ??? --> maybe because that could be useful but ok maybe for now

    # bonds = np.copy(lab_cells)
    bonds = cantor_color_bonds(lab_cells, vertices, boundary_color=0)  #
    bonds[vertices == 255] = 0  # really needs be done --> do a meta function that does all
    # in fact real bonds need remove vertices

    lab_bonds = label(bonds, connectivity=2,
                      background=0)  # nb I SHOULD NOT RELABEL THEM BECAUSE I MAY CREATE ERRORS BY DOING SO
    # lab_bonds = np.copy(bonds)

    if TA_path is not None:
        # print(lab_bonds.dtype)
        Img(lab_bonds, dimensions='hw').save(
            os.path.join(TA_path, 'bonds.tif'))  # pretty much the same as boundaryPlotter.tif in TA!!!
        # print('--->'*5,lab_bonds.dtype)
    else:
        # plt.imshow(lab_bonds)
        # plt.show()
        pass

    rps_bonds = regionprops(lab_bonds)


    if __DEBUG__:
        print('debug #7')
    # try to add vertices to bonds but in fact does not work well --> need an alternative method --> should be easy to do I guess !! --> TODO or I do have a mistake
    # or get feret points and find vertex for it or take points with 7 false around them and get the corresponding vertex on the image  --> maybe a good idea
    # take first and last and if not good scan all points to find real edges

    # then find pairs of vertices --> not hard --> gives new bonds
    # get coord in real image

    # can I get the vertices of every bond
    # need specify an ID per bond and increment for super tiny ones --> would be the local ID and can make it that there is table that can be joined that connects local and global ids --> easy todo I think

    # if any error occurs --> skip the whole stuff

    # if anything goes wrong with the bond skip the whole line from saving --> can I do that --> not really except if all is saved at the same time in the very end --> I could try that

    max_bond_id = 0
    for bond in rps_bonds:

        if early_stop.stop == True:
            # print('early stop')
            return
        # print('False perimeter', bond.perimeter)
        # --> ok

        bbox = bond.bbox

        # y_shift_to_transfrom_local_coords_to_global_coords = 0
        # x_shift_to_transfrom_local_coords_to_global_coords = 0

        if bbox[0] == 0 or bbox[0] == lab_bonds.shape[0] - 1:

            x_shift_to_transfrom_local_coords_to_global_coords = bbox[1] - 1
            y_shift_to_transfrom_local_coords_to_global_coords = bbox[0]
            # print(bbox[0],bbox)

            img = np.zeros((bond.image.shape[0], bond.image.shape[1] + 2), dtype=bond.image.dtype)
            img[:, 1:-1] = bond.image
            # print(bond.image.shape, img.shape)
            # if bbox[0]==0:
            img[0, 0] = True
            img[0, img.shape[1] - 1] = True
            # else:
            #     img[img.shape[0]-1, 0] = True
            #     img[img.shape[0]-1, img.shape[1]-1] = True
            # plt.imshow(img.astype(np.uint8)*255, cmap='gray', vmin=0)
            # plt.show()
            # if img.shape[0]!=1:
            #     plt.imshow(img.astype(np.uint8) * 255, cmap='gray', vmin=0)
            #     plt.show()
        elif bbox[1] == 0 or bbox[1] == lab_bonds.shape[1] - 1:
            x_shift_to_transfrom_local_coords_to_global_coords = bbox[1]
            y_shift_to_transfrom_local_coords_to_global_coords = bbox[0] - 1
            # print(bbox[0],bbox)
            img = np.zeros((bond.image.shape[0] + 2, bond.image.shape[1]), dtype=bond.image.dtype)
            img[1:-1, :] = bond.image
            # print(bond.image.shape, img.shape)
            # if bbox[1] == 0:
            img[0, 0] = True
            img[img.shape[0] - 1, 0] = True
            # else:
            #     img[img.shape[0] - 1, 0] = True
            #     img[img.shape[0] - 1, img.shape[0] - 1] = True
            # plt.imshow(img.astype(np.uint8)*255, cmap='gray', vmin=0)
            # plt.show()
            # if img.shape[1] != 1:
            #     plt.imshow(img.astype(np.uint8) * 255, cmap='gray', vmin=0)
            #     plt.show()
        else:
            x_shift_to_transfrom_local_coords_to_global_coords = bbox[1] - 1
            y_shift_to_transfrom_local_coords_to_global_coords = bbox[0] - 1
            img = np.pad(bond.image, pad_width=1, mode='constant', constant_values=False)
            # how can I get these coords

            # can use those coords of vertices to add vertices associated to bonds into a db --> I need correct their y and x but ok
            # coords = np.nonzero(vertices[bbox[0] - 1:bbox[2] + 1, bbox[1] - 1: bbox[
            #                                                                        3] + 1] == 255)  # TODO DO THIS FOR ALL CHANNELS --> recover vertices here --> clone this for all --> TODO

            # print('coords vertices', coords)  # --> gives me the coords of the vertices
            # print('zipped coords', np.c_[coords[0], coords[1]])  # this is how to zip numpy arrays -> in fact simpler than doing too complicated things --> VRAIMENT COOL
            # NB I could also use np.dstack((coords[0], coords[1]))

            # TODO add things to the DB if necessary
            # in tracks and in seg --> make sure to add a boundary around the image --> do this as an option -> but in fact really useful to do

            # --> can associate local bond id to that
            img[vertices[bbox[0] - 1:bbox[2] + 1, bbox[1] - 1: bbox[
                                                                   3] + 1] == 255] = True  # add vertices to the mask because they are not captured by the dilation

        # print('global coords', x_shift_to_transfrom_local_coords_to_global_coords, y_shift_to_transfrom_local_coords_to_global_coords)
        # associate a bond to a local id
        # very good idea in fact
        # also measure intensity

        # create bonds and find a way to transform local coords into global ones -> must not be hard --> can compute intensity for several channels if needed
        # store as NA depending on the channel
        # not bad and much better than before in fact

        # if negative values --> need a double fix so that same size --> see how to do that but probably not so hard
        # img[vertices[bbox[0] :bbox[2] + 2, bbox[1] : bbox[3] + 2] == 255] = True  # add vertices to the mask because they are not captured by the dilation

        # img[0][0]=True
        # first and last pixel
        # first_point = [bond.coords[0][0],bond.coords[0][1]]
        # last_point = [bond.coords[len(bond.coords[0])][0],bond.coords[len(bond.coords[0])][1]]
        #
        # print(first_point, last_point)

        # plt.imshow(img)
        # plt.show()

        # if __DEBUG__:
        #     print('debug #8')

        lab_perim = label(img, connectivity=2)
        reg = regionprops(lab_perim)
        idx = 0
        if len(reg) != 1:
            # print('error too many things for bonds')
            # plt.imshow(lab_perim)
            # plt.show()

            max_area = 0
            for ppp, perim in enumerate(reg):
                if perim.area > max_area:
                    max_area = perim.area
                    idx = ppp

            # print(idx, )
            # again need take biggest

            # print('fixed 000', idx)

        # make it measure intensity for the different channels of the image
        # do quantifs for all channels --> TODO

        # code seems ok but maybe I need to split it
        # see how to do
        # could check if it is a real bond for packing in fact --> TODO

        # print('real_perimeter bond', reg[idx].perimeter,reg[idx].area)  # the latter is pixel count reg[idx].perimeter_crofton,

        # juts plot it to be sure no error of vertices --> otherwise needs a fix

        # could do that for all the channels

        # could also do that before adding vertices in fact --> TODO

        vertices_tmp = locally_labeled_vertices[
            reg[idx].coords[:, 0] + y_shift_to_transfrom_local_coords_to_global_coords,
            reg[idx].coords[:, 1] + x_shift_to_transfrom_local_coords_to_global_coords]
        vertices_indices = np.nonzero(vertices_tmp)

        # get vertices in the region...

        # print('vertices_indices',vertices_indices)
        vertices_ids = vertices_tmp[vertices_indices]
        del vertices_tmp
        vertices_indices = vertices_indices[0]
        vertices_ids = vertices_ids.tolist()
        vertices_indices = vertices_indices.tolist()

        # print('vertices_ids', vertices_ids, vertices_indices)

        # can this be a bug --> I guess yes --> need a fix then --> maybe take the two closest bonds and remove one ???

        # VERY DIRTY FIX FOR CASE WHERE TOO MANY VERTICES ARE THERE --> SEE IF THERE IS NO SIMPLER SOLUTION WHEN ADDING THE VERTICES TO ONLY GET THE RIGHT ONES

        # if __DEBUG__:
        #     print('debug #9')
        # FORCES ME TO REPEAT A LOT OF CODE --> NOT A GOOD IDEA!!!
        if len(vertices_ids) > 2:
            try:
                # logger.error(                'error: wrong number of vertices --> please report this to baigouy@gmail.com' + str(vertices_ids))

                # for debug KEEP
                # vx_0 = (
                #     reg[idx].coords[vertices_indices[0], 0] ,
                #     reg[idx].coords[vertices_indices[0], 1] )
                # vx_1 = (reg[idx].coords[vertices_indices[1], 0] ,
                #         reg[idx].coords[vertices_indices[1], 1] )
                # vx_2 = (reg[idx].coords[vertices_indices[2], 0] ,
                #         reg[idx].coords[vertices_indices[2], 1] )
                #
                # print(vx_0,vx_1, vx_2)
                # lab_perim[vx_0[0], vx_0[1]]+=1
                # lab_perim[vx_1[0], vx_1[1]]+=1
                # lab_perim[vx_2[0], vx_2[1]]+=1

                # In fact need take the last vx encounter on the left and the last starting from the right
                tmp = list(zip(reg[idx].coords[:, 0], reg[idx].coords[:, 1]))
                vx_coords = [(reg[idx].coords[indx, 0], reg[idx].coords[indx, 1]) for indx in vertices_indices]

                tmp2 = list(set(tmp) - set(vx_coords))
                from scipy.spatial import distance

                C = distance.pdist(vx_coords)

                valid_idx = np.where(C > math.sqrt(2))[0]
                out = valid_idx[C[valid_idx].argmin()]

                coords = np.argwhere(squareform(C) == C[out])[0]

                vx_0 = vx_coords[coords[0]]
                vx_1 = vx_coords[coords[1]]

                # print('vx_0, vx_1', vx_0, vx_1)

                group1 = []
                group2 = []

                for coord in vx_coords:
                    if distance_between_points(vx_0, coord) <= math.sqrt(2):
                        group1.append(coord)
                    if distance_between_points(vx_1, coord) <= math.sqrt(2):
                        group2.append(coord)

                # print('groups', group1, group2)

                if len(group1) == 1:
                    vx_0 = group1[0]
                else:
                    # find closest vx to non vx pixels and that will always be my two real vertices!
                    min_dist = 1000
                    closest = group1[0]
                    for coord in group1:
                        dist = distance.cdist(tmp2, [coord]).min()
                        if dist < min_dist:
                            min_dist = dist
                            closest = coord
                            if dist == 1:
                                break
                    vx_0 = closest

                if len(group2) == 1:
                    vx_1 = group2[0]
                else:
                    # find closest vx to non vx pixels and that will always be my two real vertices!
                    min_dist = 1000
                    closest = group2[0]
                    for coord in group2:
                        dist = distance.cdist(tmp2, [coord]).min()
                        if dist < min_dist:
                            min_dist = dist
                            closest = coord
                            if dist == 1:
                                break
                    vx_1 = closest

                # this is now always ok --> I have finally my code !!!
                # need get the indices and finalize the stuff

                # idx1 = tmp.index(vx_0)
                # idx2 = tmp.index(vx_1)

                # print('idx1, idx2', idx1, idx2)

                # print('pre', vertices_indices, vertices_ids)

                # vertices_ids = [vertices_ids[vertices_indices.index(idx1)], vertices_ids[vertices_indices.index(idx2)]]

                # vertices_indices = [idx1, idx2]
                # print('corrected', vertices_ids, vertices_indices)

                # for debug --> keep
                # lab_perim[vx_0[0], vx_0[1]] += 1
                # lab_perim[vx_1[0], vx_1[1]] += 1

                # plt.imshow(lab_perim)
                # plt.show()

                # update mask and reg with the removed vertices so that all measures and alike are correct --> TOO BAD I HAVE TO DUPLICTAE SO MUCH CODE (SEE ALL THE LINES BELOW) BUT OK FOR NOW
                for vx in vx_coords:
                    img[vx[0], vx[1]] = False

                img[vx_0[0], vx_0[1]] = True
                img[vx_1[0], vx_1[1]] = True
                lab_perim = label(img, connectivity=2)
                reg = regionprops(lab_perim)
                idx = 0
                if len(reg) != 1:
                    # print('error too many things for bonds')
                    # plt.imshow(lab_perim)
                    # plt.show()

                    max_area = 0
                    for ppp, perim in enumerate(reg):
                        if perim.area > max_area:
                            max_area = perim.area
                            idx = ppp

                vertices_tmp = locally_labeled_vertices[
                    reg[idx].coords[:, 0] + y_shift_to_transfrom_local_coords_to_global_coords,
                    reg[idx].coords[:, 1] + x_shift_to_transfrom_local_coords_to_global_coords]
                vertices_indices = np.nonzero(vertices_tmp)

                # get vertices in the region...

                # print('vertices_indices',vertices_indices)
                vertices_ids = vertices_tmp[vertices_indices]
                del vertices_tmp
                vertices_indices = vertices_indices[0]
                vertices_ids = vertices_ids.tolist()
                vertices_indices = vertices_indices.tolist()

                # plt.imshow(lab_perim)
                # plt.show()
            except:
                traceback.print_exc()
                if __DEBUG__:
                    print('error correcting the number of vertices of the bond --> please report this to baigouy@gmail.com')
                logger.error(
                    'error correcting the number of vertices of the bond --> please report this to baigouy@gmail.com')

            # TODO see how to handle that because this is super frequent in fact !!!! but not easy to deal with...

        # if __DEBUG__:
        #     print('debug #10')
        # TODO DO THIS FOR ALL CHANNELS
        # maybe do that vertices included or not ??? MAYBE

        # print(type(reg), type(reg[idx]))

        sum_bd_intensity, avg_bd_intensity = get_sum_and_avg_intensity(original_image, reg, idx,
                                                                       y_shift_to_transfrom_local_coords_to_global_coords,
                                                                       x_shift_to_transfrom_local_coords_to_global_coords)

        # nb the vertices are the two points of the region coords to be white in the parent image with local transform --> how cna I find them

        # print('reg[idx]',reg[idx])
        # if not vertices_ids:
        #     plt.imshow(lab_perim)
        #     plt.show()

        # in some case the vertices_indices can be empty --> is that normal or a bug ??? --> probably a bug that needs be fixed --> this is because vertices ids is also empty --> how can I fix that wand why does this happens
        # print('neo vertices', vertices_indices) # this gives me the indices of the vertices of the bonds --> see how I can associate them with a cell at some point maybe if necessary

        # KEEP nb a cell can have 0 vertices --> indeed a floating cell embedded in another would have 0 vertex --> shall I give it a nb of vertices of 1 or 0??? --> maybe smart to set its vertex id to None/Null

        # In case the bond has no vertices do as follows
        first_vx = None
        second_vertex = None
        bd_orientation = None

        if vertices_indices:
            first_vertex = (
                reg[idx].coords[vertices_indices[0], 0] + y_shift_to_transfrom_local_coords_to_global_coords,
                reg[idx].coords[vertices_indices[0], 1] + x_shift_to_transfrom_local_coords_to_global_coords)
            second_vertex = (
                reg[idx].coords[vertices_indices[-1], 0] + y_shift_to_transfrom_local_coords_to_global_coords,
                reg[idx].coords[vertices_indices[-1], 1] + x_shift_to_transfrom_local_coords_to_global_coords)
            bd_orientation = get_orientation(first_vertex, second_vertex)

            # TODO action required if more than two vertices --> take the two furthest appart or the first and last --> MAYBE WILL NEED A CLEAN/SMART FIX SOME DAY
            # print('first_vertex, second_vertex',first_vertex, second_vertex)
            first_vx = vertices_ids[0]
            second_vertex = vertices_ids[-1]

        # print('TODO sum_bd_intensity,avg_bd_intensity',sum_bd_intensity,avg_bd_intensity)

        # keep for debug --> if the original mask is passed max and min should always be 255 --> the mask color --> seems to work perfectly --> and also avg should always be 255, if less --> bug somewhere --> needs a fix --> TODO
        # could do the same with the cells in fact
        # also need fix local centoid wit that in fact i can probably still take the centroid
        # print('max, min', original_image[reg[idx].coords[:, 0]+y_shift_to_transfrom_local_coords_to_global_coords, reg[idx].coords[:, 1]+x_shift_to_transfrom_local_coords_to_global_coords].max(), original_image[reg[idx].coords[:, 0]+y_shift_to_transfrom_local_coords_to_global_coords, reg[idx].coords[:, 1]+x_shift_to_transfrom_local_coords_to_global_coords].min()) # alw

        # required prior to quantification of 3D length of bonds

        # TODO NEED PATCH THE COORDS THERE IN FACT
        # print('sorted_bond', sorted_bond)

        x_shift_to_transfrom_local_coords_to_global_coords = bbox[1] - 1  # why -1 --> should in fact be the right stuff
        y_shift_to_transfrom_local_coords_to_global_coords = bbox[0] - 1

        # if __DEBUG__:
        #     print('debug #11')

        if measure_3D and height_map is not None:
            sorted_bond = pavlidis(reg[idx], closed_contour=False)
            sorted_bond = [
                [y + y_shift_to_transfrom_local_coords_to_global_coords,
                 x + x_shift_to_transfrom_local_coords_to_global_coords]
                for y, x in sorted_bond]

            bd_length3D = perimeter_3D(sorted_bond, heightmap=height_map, points_are_sorted=True,
                                       is_closed_contour=False)

            # print('length3D', bd_length3D)

        # we save all the data in the very end so that if anything goes wrong no data is added at all
        max_bond_id = max(max_bond_id, bond.label)
        bonds_2D['local_ID'].append(bond.label)
        bonds_3D['local_ID'].append(bond.label)
        bonds_2D['length'].append(reg[idx].perimeter)
        bonds_2D['pixel_count'].append(reg[idx].area)
        local_bond_ID_and_px_count[bond.label] = reg[idx].area

        bonds_2D['first_pixel_x'].append(bond.coords[0][0])
        bonds_2D['first_pixel_y'].append(bond.coords[0][1])

        # bug because some bonds lie at the margin and their neighb should not be found like that --> in fact that would still work I guess
        # if nothing --> border bond ???? --> think about it
        # cells_around_bond = set(lab_cells[bond.coords[0][0] - 1:bond.coords[0][0] + 2, bond.coords[0][1] - 1:bond.coords[0][1] + 2].ravel().tolist())# THIS CREATES A BUG WITH NEGATIVE SLICES BUT NO CLUE WHY --> IS THAT A SECURITY ANYWAY NOT WHAT I WANT !!!
        cells_around_bond = set(
            neighbors8(bond.coords[0], lab_cells))  # nb this is another way of detecting border cells!!!
        if 0 in cells_around_bond:
            cells_around_bond.remove(0)
        cells_around_bond = list(cells_around_bond)
        # print('cells_around_bond',cells_around_bond)

        # --> indeed border bonds ???

        # border bonds have just one neighbor --> easy to spot them
        # --> in fact should still be one
        # if len(cells_around_bond)==0:
        #
        #
        #
        #     print('bug')
        #     print(bond.coords[0][0], bond.coords[0][1])
        #     # somehow there is a bug here
        #     # print(lab_cells[int(bond.coords[0][0] - 1):int(bond.coords[0][0] + 2), int(bond.coords[0][1] - 1):int(bond.coords[0][1] + 2)], bond.coords[0][0] - 1,bond.coords[0][0] + 2, bond.coords[0][1] - 1,bond.coords[0][1] + 2)
        #     # print(neighbors8(bond.coords[0], lab_cells))
        #     plt.imshow(lab_cells)
        #     plt.show()

        # bonds_2D['vx_1_y'].append(first_vertex[0])
        # bonds_2D['vx_1_x'].append(first_vertex[1])
        # bonds_2D['vx_2_y'].append(second_vertex[0])
        # bonds_2D['vx_2_x'].append(second_vertex[1])

        bonds_2D['vx_1'].append(first_vx)
        bonds_2D['vx_2'].append(second_vertex)

        # how is it possible that some bonds have 0 id around them --> need think abount it
        if cells_around_bond:
            bonds_2D['cell_id1_around_bond'].append(cells_around_bond[0])
            bonds_2D['cell_id2_around_bond'].append(cells_around_bond[-1])
        else:
            bonds_2D['cell_id1_around_bond'].append(None)
            bonds_2D['cell_id2_around_bond'].append(None)
        is_border_bond = 0 if len(
            cells_around_bond) > 1 else 1  # a bit hacky but since anyways it is saved as an int in the sqlite database then no pb to have it directly written as 0 and 1 instead of True and False
        # i need the coords of these vertices --> where are they
        # if first_vx[0] == height-1 or first_vx[0] == 0 or first_vx[1] == 0 or first_vx[1]==width-1 or second_vertex[0] == height-1 or second_vertex[0] == 0 or second_vertex[1] == 0 or second_vertex[1]==width-1:
        #         is_border_bond = 1
        if first_vx in border_vertices or second_vertex in border_vertices:
            is_border_bond = 1

        is_border_bond_plus_one = 0
        # if first_vx[0] == height-2 or first_vx[0] == 1 or first_vx[1] == 1 or first_vx[1]==width-2 or second_vertex[0] == height-2 or second_vertex[0] == 1 or second_vertex[1] == 1 or second_vertex[1]==width-2:
        #     is_border_bond_plus_one = 1
        if first_vx in border_vertices_plus_one or second_vertex in border_vertices_plus_one:
            is_border_bond_plus_one = 1

        # print(type(is_border_bond))
        bonds_2D['is_border'].append(is_border_bond)
        bonds_2D['is_border_plus_one'].append(is_border_bond_plus_one)

        # get common ids around the two vertices --> recommended for pairs of vertices
        # qsqdqd
        # or get two ids around fisrt pixel --> would also work here

        bonds_2D['orientation'].append(bd_orientation)

        if len(original_image.shape) >= 3:
            for ccc in range(original_image.shape[-1]):
                # print('TODO sum_bd_intensity,avg_bd_intensity', sum_bd_intensity[ccc], avg_bd_intensity[ccc])
                if 'sum_px_int_vertices_included_ch' + str(ccc) not in bonds_2D:
                    bonds_2D['sum_px_int_vertices_included_ch' + str(ccc)] = []
                    bonds_2D['avg_px_int_vertices_included_ch' + str(ccc)] = []
                bonds_2D['sum_px_int_vertices_included_ch' + str(ccc)].append(sum_bd_intensity[ccc])
                bonds_2D['avg_px_int_vertices_included_ch' + str(ccc)].append(avg_bd_intensity[ccc])
        else:
            if 'sum_px_int_vertices_included_ch' + str(0) not in bonds_2D:
                bonds_2D['sum_px_int_vertices_included_ch' + str(0)] = []
                bonds_2D['avg_px_int_vertices_included_ch' + str(0)] = []
            bonds_2D['sum_px_int_vertices_included_ch' + str(0)].append(sum_bd_intensity)
            bonds_2D['avg_px_int_vertices_included_ch' + str(0)].append(avg_bd_intensity)

        if measure_3D and height_map is not None:
            bonds_3D['length3D'].append(bd_length3D)
        #
        # if __DEBUG__:
        #     print('debug #12')

        # count nb of neighbs
        # will need to remove as many bonds below cutoff --> whole bond then serves as a vertex obviously...

        # TODO get the vertices then almost all done in fact

        # can I get the vertices per image -> ce seront juste tow added points and then I can associate this ID there
        # tst = np.zeros_like(img, dtype=np.uint8)
        # tst = original_image[reg[idx].coords[:, 0], reg[idx].coords[:, 1]]

        # print(tst.shape) # ---> 41,

        # plt.imshow( tst)
        # plt.show()

        # print('sum_bd_intensity, avg_bd_intensity', sum_bd_intensity, avg_bd_intensity)

        # I am almost there this time!!!

        # when I will add bonds at the margin i will create errors --> need temporarily remove them then bring them back --> TODO

        # if bbox touches an extremity --> just remove stuff

        # en fait j'y suis presque

        # easy to sort the pixels as any pixel touching the border is my real starting point in fact --> easy to fix in fact and to sort
        # --> juts need transform fake coords into real ones --> should be easy --> just try but maybe not that hard

        # could also easily compute polarity

    # very good --> just need the cantor stuff here!!!

    # or scan along border and count neighbs if exactly 2 colors --> this is a bond --> how do I give it an id --> define a forbidden color that represent the border of the image and can then apply the cantor formula --> could be a way todo that
    # or increment

    # to find bonds at edges I need to scan in lines until I find vertices ??? then change idea
    # in fact those bonds are splitted --> think if that is true or not

    # need apply id to bonds to get things to work perfectly --> TODO maybe
    # all is ok soon

    # bonds = np.zeros_like(vertices)
    # bonds[vertices==128]=128 # make a bond specific file
    # vertices[vertices==128]=0 # make a vertex specific file
    # plt.imshow(vertices)
    # plt.show()

    # TODO just need the cantor coloring of bonds !!! --> TODO --> shall I do a generic filter or ignore in fact and just do cantor on the colored image directly
    # plt.imshow(bonds)
    # plt.show()

    # plt.imshow(lab_cells)
    # plt.show()

    # can I do that before cells as I need that for computing nb of neighbors !!! --> I think yes
    # I need associate to cell id in order to lower their counts

    # new position #############################

    lab_vertices = label(vertices, connectivity=2, background=0)
    rps_verts = regionprops(lab_vertices)

    if __DEBUG__:
        print('debug #14')

    # could simply get their ID and length between them --> quite easy in fact --> compute 2D distance in fact
    # pairs_of_vertices = []
    # detect tiny bonds that just consist of exactly two pixels --> TODO
    for vertex in rps_verts:
        if early_stop.stop == True:
            # print('early stop')
            return
        if vertex.area >= 2:
            # we have found a vertex pair --> need fix it
            # print('pair found', vertex.area, vertex.coords)

            # if more than 2 --> can have trouble (is that even possible ??? I don't think so but would really need to check
            if vertex.area > 2:  # in fact that is easy I could split it into n closest pairs of points --> that should work, but first let's see if it is possible or not ???
                # logger.error('error please contact baigouy@gmail.com to let him know that up to '+str(vertex.area)+ 'vertices can be adjacent to each other and provide him with a test image so that he can fix the bug'+str( vertex.coords))
                # sys.exit(0)

                # array([[154, 519],
                #        [155, 518],
                #        [156, 519]]))

                # need sort the three points by distance

                # more than ntwo vertices are found side by side --> these are series of four ways vertices adjacent to each others --> sort them and split them into pairs and parse them
                sorted = pavlidis(vertex)

                # print(sorted)
                # print(compute_distance_between_consecutive_points(sorted))
                # now that the list is sorted I need parse it 2 by 2 --> if I do so this is generic and i will have no pb

                # sqdsqdsqd
                # TODO need a fix for that too cause very important
                # sort vertices by distance --> just make sure they are all sorted
                # need sort them so that I minimize the

                lst = sorted
            else:
                # if bond_cut_off is not None:
                #     if vertex.area<=bond_cut_off:
                #         nb_of_vertices_or_neighbours_cut_off-=1

                # print(type(vertex.coords))  # --> np.ndarray
                # print(type(vertex.coords.tolist()))
                lst = vertex.coords.tolist()
                # print('sls lsts ', lst)
            # print(len(lst))  # --> 2 --> juste deux points que je pourrais ajouter à une table --> en fait ce serait assez facile
            # faudra tt sauver dans du vectoriel et tt plotter
            # permettre du plot as bonds ou alike
            # faire un image compositer basique et plus puissant, un plotteur de fleches et de nematic et de vecteur et de trait --> TODO then all will be done --> then I will have even something better than TA and simpler to use and that can be command line driven

            # peut etre faire un object qui calcule et ecrit ttes les propriétés des objets --> facile à handler je pense --> quite a good idea I think
            # call area cytoplasmic_area
            # area being (cytoplasmic area + 2 perimeter_area/2)
            # maybe have a header that explains it all what the measures are
            # maybe allow plots in ROIs and or in clones
            # TODO --> some day
            # pairs_of_vertices.append(lst)

            # need get the ids of the vertices and invent a local id --> max of previous --> TODO

            # print('ids sdqsdqs',ids)

            # print('pairs of vertices', lst)
            # print('perimeter2D',vertex.perimeter) # why always 0 ??? --> shall I compute it myself # compute distance 2D --> clearly the perimeter algo of scipy does not work in such case
            # print('length',distance_between_points(lst[0], lst[-1]))
            # print('pixel_count', len(lst))
            # print('vx_1', ids[0])
            # print('vx_2', ids[-1])

            # do it generically by pairs of points
            # convert a list to another list of pairs and execute the code below!!!
            # pair each element with the next in the list and loop over this!!

            pairs = list(zip(lst, lst[1:]))
            # if len(lst) == 3:
            #
            #     print('lst',lst)
            #     print('pairs', pairs)
            #     print(compute_distance_between_consecutive_points(lst))

            # print('before', locally_labeled_vertices[vertex.coords[:, 0], vertex.coords[:, 1]])

            for lst in pairs:
                # again changing this code will be a pain... because very long
                crds = np.asarray(lst)
                # need get the id locally
                ids = locally_labeled_vertices[crds[:, 0], crds[:, 1]]  # this works and is generic
                # print('after', ids)
                max_bond_id += 1

                # print('lst',lst)

                bonds_2D['local_ID'].append(max_bond_id)
                # bonds_3D['local_ID'].append(max_bond_id) # --> maybe add id --> also so that one can pair it with the other db --> see what I can do!!!
                bonds_2D['length'].append(
                    distance_between_points(lst[0], lst[-1]))  # pb is here I need more to be generic...
                bonds_2D['pixel_count'].append(len(lst))

                bonds_2D['vx_1'].append(ids[0])
                bonds_2D['vx_2'].append(ids[-1])

                # bonds_2D['cell_id1_around_bond'].append(cells_around_bond[0])
                # bonds_2D['cell_id2_around_bond'].append(cells_around_bond[-1])  # --> TODO but a bit more complex in fact !!!
                bd_orientation = get_orientation(lst[0], lst[-1])
                bonds_2D['orientation'].append(
                    bd_orientation)  # --> although            largely             meaningless...

                # TODO --> here too
                # bonds_2D['cell_id1_around_bond'].append(cells_around_bond[0])
                # bonds_2D['cell_id2_around_bond'].append(cells_around_bond[-1])  # --> TODO but a bit more complex in fact !!!

                # TODO need do that also with list because this is gonna be wrong

                # this will be wrong --> need fix it
                sum_bd_intensity, avg_bd_intensity = get_sum_and_avg_intensity(original_image,
                                                                               crds)  # this line will be wrong --> need fix it
                if len(original_image.shape) >= 3:
                    for ccc in range(original_image.shape[-1]):
                        # print('TODO sum_bd_intensity,avg_bd_intensity', sum_bd_intensity[ccc], avg_bd_intensity[ccc])
                        if 'sum_px_int_vertices_included_ch' + str(ccc) not in bonds_2D:
                            bonds_2D['sum_px_int_vertices_included_ch' + str(ccc)] = []
                            bonds_2D['avg_px_int_vertices_included_ch' + str(ccc)] = []
                        bonds_2D['sum_px_int_vertices_included_ch' + str(ccc)].append(sum_bd_intensity[ccc])
                        bonds_2D['avg_px_int_vertices_included_ch' + str(ccc)].append(avg_bd_intensity[ccc])
                else:
                    if 'sum_px_int_vertices_included_ch' + str(0) not in bonds_2D:
                        bonds_2D['sum_px_int_vertices_included_ch' + str(0)] = []
                        bonds_2D['avg_px_int_vertices_included_ch' + str(0)] = []
                    bonds_2D['sum_px_int_vertices_included_ch' + str(0)].append(sum_bd_intensity)
                    bonds_2D['avg_px_int_vertices_included_ch' + str(0)].append(avg_bd_intensity)

                    # not so hard to do I guess because I have id and bond id

                # CHECK NEIGHBORHOOD TO FIND id -> need get common ids
                # can this stuff be a border bond ??? --> I guess not but maybe --> need think further about it!!!

                cells_around_vx1 = set(
                    neighbors8(lst[0],
                               lab_cells))  # nb this is another way of detecting border cells!!!vertex.coords[0]
                cells_around_vx2 = set(
                    neighbors8(lst[-1],
                               lab_cells))  # nb this is another way of detecting border cells!!! #vertex.coords[1]
                # get common cells of the set --> these are the two ids of the cells sharing this mini bond
                cells_around_bond = cells_around_vx1.intersection(cells_around_vx2)
                if 0 in cells_around_bond:
                    cells_around_bond.remove(0)
                cells_around_bond = list(cells_around_bond)

                if len(cells_around_bond) > 2:
                    print('pairs of vertices have too many cells in common, please report error to baigouy@gmail.com')
                # print('cells_around_bond',cells_around_bond, cells_around_vx1, cells_around_vx2)
                # cells_around_bond

                if cells_around_bond:
                    bonds_2D['cell_id1_around_bond'].append(cells_around_bond[0])
                    bonds_2D['cell_id2_around_bond'].append(cells_around_bond[-1])
                else:
                    bonds_2D['cell_id1_around_bond'].append(None)
                    bonds_2D['cell_id2_around_bond'].append(None)

                # bonds_2D['cell_id1_around_bond'].append(cells_around_bond[0])
                # bonds_2D['cell_id2_around_bond'].append(cells_around_bond[-1])
                is_border_bond = 0 if len(
                    cells_around_bond) > 1 else 1  # a bit hacky but since anyways it is saved as an int in the sqlite database then no pb to have it directly written as 0 and 1 instead of True and False

                if first_vx in border_vertices or second_vertex in border_vertices:
                    is_border_bond = 1

                is_border_bond_plus_one = 0
                # if first_vx[0] == height-2 or first_vx[0] == 1 or first_vx[1] == 1 or first_vx[1]==width-2 or second_vertex[0] == height-2 or second_vertex[0] == 1 or second_vertex[1] == 1 or second_vertex[1]==width-2:
                #     is_border_bond_plus_one = 1
                if first_vx in border_vertices_plus_one or second_vertex in border_vertices_plus_one:
                    is_border_bond_plus_one = 1

                # print(type(is_border_bond))
                bonds_2D['is_border'].append(is_border_bond)
                bonds_2D['is_border_plus_one'].append(is_border_bond_plus_one)

                # print(type(is_border_bond))
                # bonds_2D['is_border'].append(is_border_bond)

                # check if real and really normal
                if cells_around_bond:
                    if cells_around_bond[0] not in bonds_associated_to_a_given_cell:
                        bonds_associated_to_a_given_cell[cells_around_bond[0]] = []
                        bond_lengths_associated_to_a_given_cell[cells_around_bond[0]] = []
                    bonds_associated_to_a_given_cell[cells_around_bond[0]].append(max_bond_id)
                    bond_lengths_associated_to_a_given_cell[cells_around_bond[0]].append(len(lst))
                    if cells_around_bond[-1] != cells_around_bond[0]:
                        if cells_around_bond[-1] not in bonds_associated_to_a_given_cell:
                            bonds_associated_to_a_given_cell[cells_around_bond[-1]] = []
                            bond_lengths_associated_to_a_given_cell[cells_around_bond[-1]] = []
                        bonds_associated_to_a_given_cell[cells_around_bond[-1]].append(max_bond_id)
                        bond_lengths_associated_to_a_given_cell[cells_around_bond[-1]].append(len(lst))

    # new position #############################

    if __DEBUG__:
        print('debug #15')

    rps = regionprops(lab_cells, intensity_image=original_image)

    # ça marche mais ne detecte qd meme pas ts les vertices
    # faire un detecteur de vertex à la boundary --> useful and put bonds there too --> simpler in fact

    # draw a small shape just to see
    # le code de perimeter de scipy est bon --> cool

    cells[cells != 0] = 1

    # in fact if image is passed in I can directly get avg intensity --> easy peasy
    # --> TODO
    # set in fact do on a channel by channel basis

    for cell in rps:
        if early_stop.stop == True:
            # print('early stop')
            return

        # print('wrong perimeter',cell.perimeter)  # --> 94.14213562373095 vs 95.799 in IJ --> no clue how they do that and need check the real one --> maybe this is ok in fact because I count 97 in fact for outer
        # manually computed for the circle:
        # print(math.sqrt(2) * 8 + 2 * 4)  # same perimeter but --> needs contour --> i need find the countour first then it is gonna be ok

        # as expected --> in IJ perimeter is completely wrong... and really makes no sense to me...

        # https://stackoverflow.com/questions/12747319/scipy-label-dilation --> maybe exactly what I want in fact

        # plt.imshow(cell.filled_image) # --> ok this is what I want in fact but it is cropped --> need pad and have bounding box to compute coords of dilation
        # plt.show()

        # what I could do is pas the image and dilate it and then just get the mask in the original image for that exact region and measure its perimeter
        # and or sort it and compute the 3D perimeter!!!
        # TODO

        # seems ok I think
        # plt.imshow(cell.image)
        # plt.title('toto')
        # plt.show()

        # maybe need change structuring element though
        # plt.imshow(ndimage.binary_dilation(cell.image))
        # plt.show()
        bbox = cell.bbox
        # print(cell.bbox)
        #

        # img = np.pad(cell.image, tuple(cell.image.shape[0]+2, cell.image.shape[1]+2), mode='constant', constant_values=False)
        # img = np.pad(cell.image, tuple(cell.image.shape[0]+2, cell.image.shape[1]+2), mode='constant', constant_values=False)
        img = np.pad(cell.image, pad_width=1, mode='constant', constant_values=False)
        # print(cell.perimeter_crofton) --> not good at all --> forget

        # struct2 = ndimage.generate_binary_structure(2, 2) # --> better because would get TA vertices because it gets the borders
        # print(struct2)
        # dilated_mask = ndimage.binary_dilation(img, structure=struct2)
        dilated_mask = ndimage.binary_dilation(img)

        dilated_mask ^= np.pad(cell.image, pad_width=1, mode='constant', constant_values=False)

        # seems exactly what I want in fact --> do I even need more ????

        # dilated_mask[vertices[bbox[0]:bbox[2] + 2, bbox[1]: bbox[3] + 2] == 255] = True  # add vertices to the mask because they are not captured by the dilation

        # x_shift_to_transfrom_local_coords_to_global_coords = bbox[1] - 1
        # y_shift_to_transfrom_local_coords_to_global_coords = bbox[0] - 1

        # ça marche c'est sans les vertices
        # if cell.label == 6:
        #     # the dialted mask seems be correct
        #     plt.imshow(dilated_mask)
        #     plt.show()

        dilated_mask[vertices[bbox[0] - 1:bbox[2] + 1, bbox[1] - 1: bbox[
                                                                        3] + 1] == 255] = True  # add vertices to the mask because they are not captured by the dilation
        # the pb here is that some vertices are outside the cell which does not make sense and is dangerous for the label

        # ça marche --> avec les vertices
        # if cell.label == 6:
        #     the dialted mask seems be correct
        # plt.imshow(dilated_mask)
        # plt.show()

        lab_perim = label(dilated_mask, connectivity=2)
        reg = regionprops(lab_perim)

        idx = 0
        if len(reg) != 1:
            # Too many cells detected --> needs a fix --> need take the biggest one or need a trick
            # print('error wrong length', cell.label)
            # plt.imshow(lab_perim)
            # plt.show()
            # need a fix if vertex is outside the cell

            max_area = 0
            for ppp, perim in enumerate(reg):
                if perim.area > max_area:
                    max_area = perim.area
                    idx = ppp

            # print('fixed index', idx, cell.label)

        # if cell.label == 6:
        #     # the bug is because of two cells due to the presence of a vertex --> if len is not 0 --> need take the biggest I guess
        #     plt.imshow(lab_perim)
        #     plt.show()

        # print('real_perimeter', reg[idx].perimeter, reg[idx].area)  # nb half of this area needs be added to the cell area in fact # , reg[0].perimeter_crofton) # --> probably ok but if I need compute intensities I need place it back --> can replace it in several of my codes

        # need sort pixels to recover the real polarity nematic of the cell

        # c'est plutot long (ça double le temps mais pas si mal car me permet de calculer le perimeter 3D et de
        # faire une table appelee measurements 3D
        # faire une table appelee measurements 3D
        # permettre des natural joins de pleins de tables en fait --> un peu plus flexible et modulaire qu'avant
        # faire un calcul de l'intensite du cytoplasme aussi --> pas trop dur en fait à faire

        # TODO make this optional --> required for polarity and also for 3D

        # print('reg[0].coords',reg[0].coords)
        #
        # print('sorted_contour', sorted_contour)

        # # measure 3D perimeter
        # y_shift_to_transfrom_local_coords_to_global_coords = bbox[0] - 1
        # x_shift_to_transfrom_local_coords_to_global_coords = bbox[1] - 1

        # need translate all the contour points
        y_shift_to_transfrom_local_coords_to_global_coords = bbox[0] - 1
        x_shift_to_transfrom_local_coords_to_global_coords = bbox[1] - 1

        if measure_3D or measure_polarity:
            sorted_contour = pavlidis(reg[idx], closed_contour=True)
            # need correct coords of sorted contour --> this must also be done before computing polarity!!!
            sorted_contour = [
                [y + y_shift_to_transfrom_local_coords_to_global_coords,
                 x + x_shift_to_transfrom_local_coords_to_global_coords]
                for y, x in sorted_contour]

        if measure_polarity:
            # compute polarity nematic:
            centoid_corrected = [reg[idx].centroid[0] + y_shift_to_transfrom_local_coords_to_global_coords,
                                 reg[idx].centroid[1] + x_shift_to_transfrom_local_coords_to_global_coords]

            # is that because the contour is not corrected ???
            polarity_nematic = compute_polarity_nematic(centoid_corrected, sorted_contour, original_image)

            # print(polarity_nematic, centoid_corrected) # centroid is correct but there is a bug in the polarity
            # need save the data for every channel in the db --> TODO

            # print('polarity nematic', polarity_nematic) # TODO DO THIS FOR ALL CHANNELS

        if measure_3D and height_map is not None:
            # measure 3D perimeter

            # print('sorted_bond perim3D', sorted_contour)
            perimeter3D = perimeter_3D(sorted_contour, heightmap=height_map, points_are_sorted=True,
                                       is_closed_contour=False)
            # print('perimeter3D', perimeter3D)

        # TODO allow intensity measurements here too

        # print('plenty of measurements')
        # print(cell.mean_intensity, cell.max_intensity, cell.min_intensity)
        # print(cell.weighted_centroid, cell.centroid)
        # print('first point', cell.coords[0][0], cell.coords[0][1])

        # get cell vertices here
        vertices_indices_cells = np.nonzero(vertices[
                                                reg[idx].coords[:,
                                                0] + y_shift_to_transfrom_local_coords_to_global_coords,
                                                reg[idx].coords[:,
                                                1] + x_shift_to_transfrom_local_coords_to_global_coords])[
            0].tolist()

        # if len(vertices_indices) > 2:
        #     logger.error(
        #         'error: wrong number of vertices --> please report this to baigouy@gmail.com' + str(vertices_indices))
        #
        # # print('neo vertices', vertices_indices) # this gives me the indices of the vertices of the bonds --> see how I can associate them with a cell at some point maybe if necessary
        # first_vertex = (reg[idx].coords[vertices_indices[0], 0] + y_shift_to_transfrom_local_coords_to_global_coords,
        #                 reg[idx].coords[vertices_indices[0], 1] + x_shift_to_transfrom_local_coords_to_global_coords)
        # second_vertex = (reg[idx].coords[vertices_indices[-1], 0] + y_shift_to_transfrom_local_coords_to_global_coords,
        #                  reg[idx].coords[vertices_indices[-1], 1] + x_shift_to_transfrom_local_coords_to_global_coords)

        # could in fact just get vertices local id and then can be recovered using joins
        # maybe sort the ids ????
        # print('neo vertices_indices_cells', vertices_indices_cells)

        # plt.imshow(vertices)
        # plt.show()
        # print('oubs',vertices_indices_cells, len( reg[idx].coords), idx)# why None

        vx_local_ids = []
        for cell_vertex in vertices_indices_cells:
            # print(idx, len(reg), cell_vertex, len(reg[idx].coords))
            # print(reg[idx])
            # print(vertices_indices_cells[cell_vertex])
            y = reg[idx].coords[cell_vertex, 0] + y_shift_to_transfrom_local_coords_to_global_coords,
            x = reg[idx].coords[cell_vertex, 1] + x_shift_to_transfrom_local_coords_to_global_coords,
            vx_local_ids.append(locally_labeled_vertices[y, x][0])
            # qsdqsdqsd

        bond_indices_cells = list(set(lab_bonds[
                                          reg[idx].coords[:,
                                          0] + y_shift_to_transfrom_local_coords_to_global_coords,
                                          reg[idx].coords[:,
                                          1] + x_shift_to_transfrom_local_coords_to_global_coords].tolist()))
        if 0 in bond_indices_cells:
            bond_indices_cells.remove(0)
        # bond_local_ID = []

        # print('bond_indices_cells',bond_indices_cells) # this contains all 'long' bonds of the current cell

        if cell.label not in bonds_associated_to_a_given_cell:
            bonds_associated_to_a_given_cell[cell.label] = []
            bond_lengths_associated_to_a_given_cell[cell.label] = []
        bonds_associated_to_a_given_cell[cell.label].extend(bond_indices_cells)
        # need add all the bond lengthes corresponding to these values --> TODO
        for bd_id in bond_indices_cells:
            bond_lengths_associated_to_a_given_cell[cell.label].append(local_bond_ID_and_px_count[bd_id])
        # if I now store the length of the bonds i can easily get the apcking

        # split by comma for example
        # print('vx_local_ids',vx_local_ids)

        # T0D0 NEED A 3D version of this
        point_awlays_inside_the_cell = point_on_surface(cell, lab_cells)
        # print('point inside cell', point_awlays_inside_the_cell)

        if measure_3D and height_map is not None:
            # all is ok now
            triangulation_centroid = point_awlays_inside_the_cell  # central point for triangulation
            # if int(ceil(cell.centroid[0]))==point_awlays_inside_the_cell[0] and int(ceil(cell.centroid[1]))==point_awlays_inside_the_cell[1]:
            # now just need to triangulate the cell

            # get the vertices of the contour and the triangulation centroid to triangulate a cell
            # alternatively could take all points of the contour --> maybe try both

            # compute_3D_surfacearea

            # print(reg[0].coords)
            # print('test', reg[0].coords[:,0])

            # there is a bug in the detection of vertices at least for the border cells --> needs a fix

            y_corrected = [y_shift_to_transfrom_local_coords_to_global_coords + y for y in reg[idx].coords[:,
                                                                                           0].tolist()]  # see how I handle the relative coords here --> do I have a bug???
            x_corrected = [x_shift_to_transfrom_local_coords_to_global_coords + x for x in
                           reg[idx].coords[:, 1].tolist()]

            verts = np.where(vertices[
                                 y_corrected, x_corrected] == 255)  # TODO need add this to the cell or maybe the ids of those ??? --> good idea

            # print(verts)

            # vertices not found properly --> why
            # if cell.label == 6:
            #     print('verts',verts, y_shift_to_transfrom_local_coords_to_global_coords,x_shift_to_transfrom_local_coords_to_global_coords)

            y_corrected = np.asarray(y_corrected)[verts[0]]
            x_corrected = np.asarray(x_corrected)[verts[0]]

            # if cell.label == 6:
            #     print('x_corrected,y_corrected',x_corrected,y_corrected)

            x_corrected = x_corrected.tolist()
            y_corrected = y_corrected.tolist()

            # y_corrected = [y_shift_to_transfrom_local_coords_to_global_coords + y for y in verts[:, 0].tolist()]
            # x_corrected = [x_shift_to_transfrom_local_coords_to_global_coords + x for x in verts[:, 1].tolist()]

            # just get coords where

            # I think that is 100% what I want now!!!

            # this is very fast just need heightmap and try compute 3D area just for fun, for a test try on a real sample

            # en fait c'est à peine plus long en 3D ...

            try:

                # je pense qu'il y a des bugs

                y_corrected.append(triangulation_centroid[0])
                x_corrected.append(triangulation_centroid[1])

                # why so few points -> bug here
                points_for_triangulation = list(zip(y_corrected, x_corrected))

                if len(points_for_triangulation) < 4:
                    if __DEBUG__:
                        print('not enough vertices for cell #' + str(
                        cell.label) + ' --> trying to add a subset of contour pixels instead, in order to triangulate')
                    logger.warning('not enough vertices for cell #' + str(
                        cell.label) + ' --> trying to add a subset of contour pixels instead, in order to triangulate')

                    # try add n random points from the table
                    # points_for_triangulation.append()

                    y_corrected = [y_shift_to_transfrom_local_coords_to_global_coords + y for y in reg[idx].coords[:,
                                                                                                   0].tolist()]  # see how I handle the relative coords here --> do I have a bug???
                    x_corrected = [x_shift_to_transfrom_local_coords_to_global_coords + x for x in
                                   reg[idx].coords[:, 1].tolist()]

                    # get x% random values from both
                    # print('y_corrected, x_corrected', len(y_corrected), len(x_corrected))
                    tmp = list(zip(y_corrected, x_corrected))

                    nb_of_point_to_get = 12  # could also take x percent of the contour --> TODO or take n points maybe as if the cell was an n sided cell --> maybe take 10 --> enough in fact
                    # if nb_of_point_to_get<4:
                    #     nb_of_point_to_get = 4
                    if nb_of_point_to_get > len(y_corrected):
                        nb_of_point_to_get = len(y_corrected)

                    seleted_points = random.sample(tmp, nb_of_point_to_get)
                    points_for_triangulation.extend(seleted_points)

                # print('points_for_triangulation',points_for_triangulation)

                # points_for_triangulation.append(triangulation_centroid)

                # can I do that ???

                # points_for_triangulation =[reg[0].coords]
                # there is a bug in there dunno where

                # nb rather just take vertices because a bit bigger than the cell and mention it is an approximation
                # --> pas mal mais rajoute encore deux seqs de plus mais pas mal qd meme
                # triangles = triangulate(np.asarray(points_for_triangulation), triangulated_object_ID=cell.label, labels_to_check=lab_cells)#,triangulated_object_ID=cell.label)  # need filter the triangulation for concave shapes of clones which is very frequent

                # nb si seulement deux points --> pas de triangulation --> la question c'est pkoi seulement deux points
                # if cell.label == 6:
                #     print('np.asarray(points_for_triangulation)',np.asarray(points_for_triangulation))
                # print('points_for_triangulation',points_for_triangulation)

                triangles = triangulate(np.asarray(
                    points_for_triangulation))  # ,triangulated_object_ID=cell.label)  # need filter the triangulation for concave shapes of clones which is very frequent

                # make it measure the 3D area of the clone --> TODO
                # if __DEBUG__:
                # print('area cell 3D', cell.label, compute_3D_surfacearea(triangles,
                #                                                              heightmap=height_map))  # --> ça marche enfin on dirait! --> cool

                # small hack debug to make centroid same height as vertices if handcorrection is passed in --> so that all points are the same height and area is 0 in fact

                # finally just do the length meaurement in 3D for the perimeter and bonds and i'll be done --> this will also be useful for Amrutha

                # maybe I could also save the area for flat 3D as it will not be the same as the real area  --> very good idea in fact --> quite smart in fact I really love it
                # height_map[triangulation_centroid[0], triangulation_centroid[1]]=255 # --> ça marche super en fait et pas de bugs
                # TODO define the 3D stuff

                # if measure_3D and height_map is not None:

                area3D_flat_3D = compute_3D_surfacearea(triangles, disable_2D_warning=True)
                area3D = compute_3D_surfacearea(triangles, heightmap=height_map)

                # print('area in 3D', area3D_flat_3D, area3D,cell.area)  # there has to be a bug ??? because since all at same height the area must be the same as in 2D --> check that


            # ferets_of_interest.append(area3D)

            # seems almost ok

            # plot_triangulation(triangles, out)
            # if __DEBUG__:
            # plot_triangulation(triangles, original_image)
            # plot_triangulation(triangles)
            except:
                traceback.print_exc()
                # clearly it misses most of the vertices of the cell --> big bug ---> see what is going wrong and why
                # it also fails for cells without vertices
                print('failed for cell2', cell.label, cell.area, cell.bbox, verts, points_for_triangulation,
                      cell.coords)
                # print('np.asarray(points_for_triangulation)', np.asarray(points_for_triangulation))


        # TODO try cell triangulation for 3D --> should be easy todo in fact !!!

        # plt.imshow(original_image)
        # plt.show()

        # see what's left to be done

        # now that I have the point inside the cell I can take it

        # on top of cell centroid I can take a point in the cell
        # first point
        # or could take a point always in the cell for sure --> smart
        # coords --> can be a table named coords

        # almost all ok --> see how long it can take to reimplement PCP --> I guess can be quite fast indeed

        #

        # try also triangulation and compute 3D area in fact using height map --> need be sure the centroid is in the cell

        # print('real cell area', cell.area + reg[idx].perimeter / 2)

        # qsdqsqsd
        # almost ok but then would need the vertices and the bonds --> see how to do
        # for bonds I have the perfect algo juts need to add pairs of vertices
        # and this does not require sorting bonds

        # place back on orig image --> easy if one dilation --> bbox -1 and +1 for all dims

        #
        # plt.imshow(dilated_mask) # seems ok then just need merge with the original mask and I'll have what I need
        # plt.show()
        #

        # plt.imshow(lab_perim)
        # plt.show()

        # is there a bug

        # that's very good but I just miss vertices at the border from this image --> rescue them maybe, see if with no border that works or not ???

        # the funny thing is this code labels all the vertices --> except at the border --> that is very funny in fact and also labels bonds --> really cool in fact I love it
        # en fait si les vertices at the border sont labeles par des 1 une foir fini

        # DO I NEED THAT ???? --> try nactivate this line then all ok !!!
        cells[bbox[0] - 1:bbox[2] + 1, bbox[1] - 1: bbox[3] + 1] += dilated_mask.astype(np.uint8)
        # cells[bbox[0]-1:bbox[2]+1, bbox[1]-1: bbox[3]+1] = cell.label+1

        # place back the mask on original maybe for quantification such as intensity --> TODO

        # seems like the perimeter is in fact what I want in 2D at least: Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.

        # probably not good in fact

        # break

        # again we save all the data in the very end so that if anything goes wrong the stuff is not added at all
        # in fact that trick may be a super code to label all the cells and all the bonds of it --> really cool in fact -->think about it and how to integrate it some day, almost there though
        cells_2D['local_ID'].append(cell.label)
        if cell.label in border_cells:
            cells_2D['is_border'].append(
                1)  # somehow there is a bug here because it is saved as a text and I wanted to have it as a boolean --> no clue why
        else:
            cells_2D['is_border'].append(0)
        # this is very slow --> is there no better way ??
        if cell.label in border_cells_plus_one:  # TODO
            cells_2D['is_border_plus_one'].append(
                1)  # somehow there is a bug here because it is saved as a text and I wanted to have it as a boolean --> no clue why
        else:
            cells_2D['is_border_plus_one'].append(0)
        cells_2D['bond_cut_off'].append(bond_cut_off)
        cells_3D['local_ID'].append(cell.label)
        cells_2D['pixel_within_cell_y'].append(point_awlays_inside_the_cell[0])
        cells_2D['pixel_within_cell_x'].append(point_awlays_inside_the_cell[1])
        cells_2D['first_pixel_y'].append(cell.coords[0][0])
        cells_2D['first_pixel_x'].append(cell.coords[0][1])
        cells_2D['cytoplasmic_area'].append(cell.area)
        # print('centroid', cell.centroid)
        centroid = cell.centroid
        # if len(centroid)==3:
        #     cells_3D['centroid_z']=centroid[0] # impossible cause 2D
        # image is 2D
        cells_2D['centroid_y'].append(centroid[-2])
        cells_2D['centroid_x'].append(centroid[-1])

        cells_2D['perimeter'].append(reg[idx].perimeter)# nb maybe this value is incorrect because it's an apporximation --> probably I should replace it even if it will be much slower
        cells_2D['perimeter_pixel_count'].append(reg[idx].area)
        cells_2D['area'].append(cell.area + reg[idx].perimeter / 2.)  # NB should I do that in 3D too???

        # scipy elongation and orientation of cells --> TODO --> maybe also add the TA one just for compatibility
        # cells_2D['major_axis_length'].append(cell.major_axis_length)
        # cells_2D['minor_axis_length'].append(cell.minor_axis_length)

        try:
            # stretch is slow --> so maybe offer this as an option ???
            stretch_nematic = compute_stretch_nematic(cell)
            cells_2D['orientation'].append(stretch_nematic.get_angle2())
            # cells_2D['eccentricity'].append(cell.eccentricity)
            cells_2D['elongation'].append(stretch_nematic.getS0())
            cells_2D['S1'].append(stretch_nematic.S1)
            cells_2D['S2'].append(stretch_nematic.S2)
        except:
        # if True:
            # traceback.print_exc()
            cells_2D['orientation'].append(None)
            # cells_2D['eccentricity'].append(cell.eccentricity)
            cells_2D['elongation'].append(None)
            cells_2D['S1'] .append( None)
            cells_2D['S2'].append( None)


        # sqsqdsq

        if measure_polarity:
            # print('polarity nematic', polarity_nematic)  # TODO DO THIS FOR ALL CHANNELS
            for ppp in range(len(polarity_nematic) - 1):
                if isinstance(polarity_nematic[0], list):
                    for ccc in range(len(polarity_nematic[0])):
                        # print('TODO sum_bd_intensity,avg_bd_intensity', sum_bd_intensity[ccc], avg_bd_intensity[ccc])
                        if 'Q' + str(ppp + 1) + '_polarity_ch' + str(ccc) not in cells_2D:
                            cells_2D['Q' + str(ppp + 1) + '_polarity_ch' + str(ccc)] = []
                            cells_2D['normalized_Q' + str(ppp + 1) + '_polarity_ch' + str(ccc)] = []
                        cells_2D['Q' + str(ppp + 1) + '_polarity_ch' + str(ccc)].append(polarity_nematic[ppp][ccc])
                        try:
                            cells_2D['normalized_Q' + str(ppp + 1) + '_polarity_ch' + str(ccc)].append(
                                polarity_nematic[ppp][ccc] / polarity_nematic[-1][ccc])
                        except:
                            # error division by 0 --> set it to 0
                            cells_2D['normalized_Q' + str(ppp + 1) + '_polarity_ch' + str(ccc)].append(0.)
                else:
                    if 'Q' + str(ppp + 1) + '_polarity_ch' + str(0) not in cells_2D:
                        cells_2D['Q' + str(ppp + 1) + '_polarity_ch' + str(0)] = []
                        cells_2D['normalized_Q' + str(ppp + 1) + '_polarity_ch' + str(0)] = []
                    cells_2D['Q' + str(ppp + 1) + '_polarity_ch' + str(0)].append(polarity_nematic[ppp])
                    try:
                        cells_2D['normalized_Q' + str(ppp + 1) + '_polarity_ch' + str(0)].append(
                            polarity_nematic[ppp] / polarity_nematic[-1])
                    except:
                        # error division by 0 --> set it to 0
                        cells_2D['normalized_Q' + str(ppp + 1) + '_polarity_ch' + str(0)].append(0.)

                    # qssqsqqssqsqsqdsqdsqd
            # print(type(polarity_nematic[0]))  # --> why is it an image ???
        # plot the intensities too!!!
        # plot cytoplasmic intensity
        # plot cortical intensity
        # plot nematics and I'll be done...

        # store all the cell vertices ids

        cells_2D['vertices'].append(vx_local_ids)  # TODO shall I add bonds --> also need create a vertices table
        cells_2D['nb_of_vertices_or_neighbours'].append(len(vx_local_ids))
        # nb_of_vertices_or_neighbours_cut_off = len(vx_local_ids)

        bond_length_of_the_cell = bond_lengths_associated_to_a_given_cell[cell.label]
        nb_of_vertices_or_neighbours_cut_off = len(bond_length_of_the_cell)
        if bond_cut_off is not None and bond_cut_off > 0:
            bond_length_of_the_cell = [bdl for bdl in bond_length_of_the_cell if bdl > bond_cut_off]
            nb_of_vertices_or_neighbours_cut_off = len(bond_length_of_the_cell)
        # del bond_lengths_associated_to_a_given_cell[cell.label]
        cells_2D['nb_of_vertices_or_neighbours_cut_off'].append(nb_of_vertices_or_neighbours_cut_off)
        bonds_of_cll = bonds_associated_to_a_given_cell[cell.label]
        cells_2D['bonds'].append(bonds_of_cll)

        if measure_3D and height_map is not None:
            cells_3D['area3D'].append(area3D)
            cells_3D['area_flat_height'].append(area3D_flat_3D)
            cells_3D['perimeter3D'].append(perimeter3D)

        # print('cytoplasmic area',cell.area)  # area is the area without the perimeter, see how the perim is computed --> same as in IJ

    if __DEBUG__:
        print('debug #17')
    # cells[vertices==255]=6

    # plt.imshow(cells)
    # plt.show()

    # Img(lab_cells).save('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/0_test_IJ_perimeter.tif')

    # all is ok I may just need to get bonds and their perimeter now etc and do also 3D measurements --> in same or in new table --> think about it
    # almost there and almost have a TA that is even working better than original and much simpler to code and maintain

    # need get bonds and measure them --> should be easy in fact
    # get the bonds, fill them and get their 2D perimeter then find their associated vertices and create the final version of the bond and measure its length --> not so hard then can give it an id --> TODO

    # skimage.measure.perimeter(image, neighbourhood=4)

    # finally just need get now the pairs of vertices and add them to bonds
    # maybe also smart to measure average intensity

    # detect vertex pairs
    # --> just detect them and pair them maybe --> TODO
    # see the most efficient way to do that

    # vertices_coords = np.where(vertices==255)
    #
    #
    # # print(vertices_coords)
    # for vvv in range(len(vertices_coords[0])):
    #     y = vertices_coords[0][vvv]
    #     x = vertices_coords[1][vvv]
    #
    #     print(y,x)
    #     # get its neighbors in a safe way
    #     for iii in range[-1,2,1]:
    #         for jjj in range[-1,2,1]:
    #             if vertices[]

    # easiest way is to flood the vertices and get those with area >= 2 --> a piece of cake TODO

    # TODO REALLY NEED ADD THAT TOO
    # I need add this to bonds 2D and remove missing ones

    # in fact that is easy I need add the stuff to the cell only in the very end
    # cells_2D['nb_of_vertices_or_neighbours_cut_off'].append(nb_of_vertices_or_neighbours_cut_off)

    # en fait marche pas --> see how I can do that

    # all is ok
    # print(pairs_of_vertices)
    # print(len(pairs_of_vertices))  # -->18
    # print(len(pairs_of_vertices[0]))  # -->2

    # final thing that must be done is the polarity --> but it requires sorting vertices and so will be slow --> do not do it by default

    # these can also be considered as fourway vertices in the same way as bonds below a cutoff --> I could do that and save those
    # shall I store local vx id too

    # almost all is saved and finally I need also do things in 3D --> not that hard by the way and can be done if I have the height map --> enable it by default if height map is there
    # just need triangulate all the cells
    # maybe offer control of the nb of CPUs to use at startup or in settings and try to MT things --> probably not hard and can be done at the level of the list --> should not be too hard
    # need also count the nb of vertices per cell to get nb of neighbours and need get
    # need create a viewer and something that can export to IJ from the command line --> à tester en fait mais devrait pas etre trop dur je pense et je peux facilement generer des stacks on demand --> TODO

    # do all of that and get rid of TA

    # try to really finalize something
    # see how much memory I would need in the end for that
    # do a nematic class and allow it to be plotted

    # see how much time it would take just for one case of one big image ????
    # do the parallelization
    # finalize tracking error correction --> TODO
    # then try to finalize a GUI, and really clean all too
    # try speed up also labels by having it to compute just the really necessary things...

    # print('bonds_associated_to_a_given_cell',bonds_associated_to_a_given_cell)
    # print('bond_lengths_associated_to_a_given_cell',bond_lengths_associated_to_a_given_cell)

    if __DEBUG__:
        print('debug #18')
    if TA_path is not None:
        db = TAsql(db_path)
        # print('cells_2D'*20)
        db.create_and_append_table(table_name='vertices_2D', datas=fill_missing_with_Nones(vertices_2D))
        db.create_and_append_table(table_name='cells_2D', datas=fill_missing_with_Nones(cells_2D))
        # print('cells_3D'*20)
        # print('bonds_2D' * 20)
        db.create_and_append_table(table_name='bonds_2D', datas=fill_missing_with_Nones(bonds_2D))
        if measure_3D and height_map is not None:
            db.create_and_append_table(table_name='cells_3D', datas=fill_missing_with_Nones(cells_3D))
            db.create_and_append_table(table_name='bonds_3D', datas=fill_missing_with_Nones(bonds_3D))
        elif measure_3D and height_map is None:
            db.drop_table('cells_3D')
            db.drop_table('bonds_3D')

        db.close()
    else:
        # plt.imshow(cells)
        # plt.show()
        # maybe return an error and always close the db

        print('cells_2D', cells_2D)
        print('cells_3D', cells_3D)
        print('bonds_2D', bonds_2D)
        print('bonds_3D', bonds_3D)
        print('vertices_2D', vertices_2D)

        for k, v in cells_2D.items():
            print(len(v))

        cells_2D = fill_missing_with_Nones(cells_2D)
        print('#' * 20)
        for k, v in cells_2D.items():
            print(len(v))

    if __DEBUG__:
        print('debug #19')
    print('total time', timer() - start)

    # do it a class and make it modular so that the minimal nb of things are executed

    # je pense 4-5 secs par image sans trier les contours --> not super slow but also not very fast
    # speed up computation of contour by using numpy array --> remember my distance code

    # should all be fast to do --> 3-4 days I think

    # see what's missing
    # 2.3 secs per stuff --> not bad, in fact maybe as fast as TA now
    # need sort contour
    # need get bonds and their vertices --> TODO
    # need all other parameters
    # if scaling parameter then one needs to specify
    # need scaling factors if heightmap is there

    # finalize everything and maybe retrain a model that already works very well for the projection

    # TODO do the db part
    # also do the sorting part

    # and also I need to do the intensity part --> TODO
    # also make it a class that is easy to use

    # also do the triangulation part for 3D --> should not be too hard to do


if __name__ == '__main__':
    from datetime import datetime
    print(datetime.now())


    if False:
        print(round(5.59))  # not good
        print(round(5.39))  # not good
        import sys

        sys.exit(0)

    if False:
        # original_image = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/handCorrection.png')[..., 1]  # if no shift max and min should always be 255 otherwise there is an error somewhere
        original_image = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png')[
            ..., 1]  # if no shift max and min should always be 255 otherwise there is an error somewhere
        # cells = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/handCorrection.png')[..., 0]
        cells = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/handCorrection.tif')
        # height_map = None  # Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png')[..., 1]
        height_map = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png')[..., 1]
        # height_map = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png')[..., 0] # another good test because all is 0 there --> marche car aire est proche de l'aire 2D par contre pr handcorrection ça marche pas car tres different de l'aire 2D --> see why and how to fix it
        # height_map = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/handCorrection.png')[..., 0] # good for a test as all heights are the same... --> in fact that makes sense since some heigts are 0 and some are 255 since the centroid of the polygon is always 0

        # for manual debug KEEP:
        # TAMeasurements(None,__forced_orig=original_image, __forced_cells=cells, __forced_heightmap=height_map)

        # TAMeasurements(None, __forced_orig=original_image, __forced_cells=cells, __forced_heightmap=height_map,measure_polarity=True, measure_3D=True)
        # import sys
        # sys.exit(0)

        # make it loop add nb of neighbors à la TA with cut off and also add support for lists of files--> in  the same self loop by the way --> no waste of time this way and add support for progress

        # TAMeasurements('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png', measure_polarity=True,                   measure_3D=True)  # --> almost there just need create one db per file now!!!
        # TAMeasurements('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012.png', measure_polarity=True,                   measure_3D=True)  # --> almost there just need create one db per file now!!!
        # TAMeasurements(loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/list.lst'),measure_polarity=True, measure_3D=True)  # --> almost there just need create one db per file now!!!
        # TAMeasurements(['/E/Sample_images/sample_images_pyta/surface_projection/210219.lif_t000.tif'],                   measure_polarity=True, measure_3D=True)  # --> almost there just need create one db per file now!!!
        # TAMeasurements(['/E/Sample_images/sample_images_pyta/surface_projection/exposures_1_P04.tif'],measure_polarity=True, measure_3D=True)  # --> almost there just need create one db per file now!!!
        # TAMeasurements(loadlist('/E/Sample_images/sample_images_pyta/surface_projection/list.lst'),measure_polarity=True, measure_3D=True)  # --> almost there just need create one db per file now!!!
        # TAMeasurements(loadlist('/E/Sample_images/sample_images_pyta/surface_projection/list.lst'),measure_polarity=False, measure_3D=False)  # causes a bug --> try see why ????
        # TAMeasurements(['/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/mini_focused_Series014_for_debug_polarity_nematic.tif'], measure_polarity=True,measure_3D=True)  # --> almost there just need create one db per file now!!!

        # test with list
        TAMeasurements(['/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png','/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series014.png'], measure_polarity=False, measure_3D=False) # --> almost there just need create one db per file now!!!


        # /E/Sample_images/sample_images_pyta/surface_projection/list.lst

        # TODO make it also save the vertices, cells and bonds ??? --> easy TODO in fact
        import sys
        sys.exit(0)

    if True:
        #  failed attempt to launch it in a thread

        start = timer()
        # try MT the finish all
        # import multiprocessing
        # merge_names = ['/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png','/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series014.png','/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series015.png','/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series016.png','/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series018.png','/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series019.png']
        # TRY MY THE MeA
        # pas mal mais il me faudrait une progress bar en fait
        # with multiprocessing.Pool(processes=8) as pool:
        #     results = pool.map(TAMeasurements, merge_names)

        # merge_names = loadlist('/E/Sample_images/trash/complete/liste.lst')
        # merge_names = loadlist('/E/Sample_images/sample_images_denoise_manue/211029_EcadKI_mel_40-54hAPF_ON/surface_projection/list.lst')
        merge_names = loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini_vide/list_simple.lst')

        from functools import partial
        from epyseg.ta.pyqt_test_threads_instead_of_threadpool import Worker2

        func = partial(TAMeasurements)
        thread = Worker2(func, merge_names, measure_polarity=False, measure_3D=False)

        # self.thread.result.connect(self.print_output)
        # self.thread.finished.connect(self.thread_complete)
        # this is specific of this method I must update the nb of inputs and outputs of the model # be CAREFUL IF COPYING THIS CODE THE FOLLOWING MUST BE REMOVED
        # worker.signals.finished.connect(self._set_model_inputs_and_outputs)
        # self.thread.progress.connect(self.progress_fn)
        thread.setTerminationEnabled(True)

        # Execute
        # if isinstance(self.thread, FakeWorker2):
        #     # no threading
        #     self.thread.run()
        # else:
        # threading
        # self.threadpool.start(worker)
        # self.threads.append(worker)
        # worker.moveToThread(self.thread)

        # self.thread.started.connect(self.thread.run)
        thread.start()

        # NON MT --> 1400secs

        # marche mais pas qd lancé depuis pyqt --> why -> can I simulate launch in a thread ????

        # MT --> 176secs --> really worth it!!!
        # from tqdm import tqdm
        # import sys
        # pool = Pool(processes=15)
        # for i, _ in enumerate(tqdm(pool.imap_unordered(TAMeasurements, merge_names), total=len(merge_names))):
        #     # pass
        #     # sys.stderr.write('\rdone {0:%}'.format(i / len(merge_names))) # --> I could use that to plot with the other progressbar --> ok
        #     # cool --> I can use that to display progress in the other progress bar --> the QT one
        #     pass
        #
        # pool.close()
        # pool.join()

        print('total execution time', timer() - start)
        # import sys
        # sys.exit(0)

        # why not running ???

