# TODO ADD PLOT AS VARIOUS SHAPES --> TODO SUCH AS POINTS OR DOTS OR CIRCLE OR SQUARES OR RECTANGLES OR ALIKE !!!
# TODO --> add this at some point ADD PLOT  AS POINT OR ANY SHAPE!!!!


# for points I would need to have two columns and maybe a radius or something alike --> THINK OF THAT
# maybe also extend that to 3D as I more and more need that !!!!










# can I add plot as CIRCLE OR PLOT AS DOT ???? --> check what plot as vertices do...
# NB I DO HAVE PLOT AS POINTS AND THAT SHOULD DO THE JOB !!!





# not bad --> almost there --> see what is really missing and add it for the MS --> TODO


# TODO I have a big bug in the polarity --> need fix it but maybe not very urgent in fact !!!
# do a small test on a small image --> easy to debug
# test of all

# examples of SQLite commands I'd like to reimplement rapidly
#           PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, area_cells FROM cells + LUT DNA
#             PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells WHERE is_border_cell=='false' + LUT DNA + OPACITY 35% + DILATATION 1
#             PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells NATURAL JOIN tracked_clone_022 + LUT DNA MIN=GLOBAL_MIN MAX=GLOBAL_MAX + OPACITY 35% + DILATATION 1
#             PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells FROM cells ORDER BY area_cells DESC LIMIT 10 + LUT DNA MIN=GLOBAL_MIN MAX=GLOBAL_MAX
#             PLOT AS ARROWS SELECT first_pixel_x_cells,first_pixel_y_cells , center_x_cells,center_y_cells, area_cells AS COLOR, '1' AS STROKE_SIZE, 'DEFAULT_ARROW' AS ARROWHEAD_TYPE, '5' AS ARROWHEAD_HEIGHT,'3' AS ARROWHEAD_HEIGHT, '0.3' AS STROKE_OPACITY FROM cells + LUT DNA
#             PLOT AS BONDS SELECT first_pixel_x_bonds, first_pixel_y_bonds, bond_orientation FROM bonds WHERE first_pixel_y_bonds != 'null' + LUT DNA


# some columns would require specific plots
# implement plot as nematic
# implement plot as cells


# offer  exclude border cells
# support quantiles
#


# https://matplotlib.org/2.0.2/faq/usage_faq.html --> check backends beacuse that is most likely what I need for my figure editor


# plots stuff as cell, bonds or vertices ideally from a db
# should not be too hard --> need load the image duplicate it and do the plots through conversions maybe need enable numba otherwise it will be too slow!!!
# from PIL.ImageDraw import ImageDraw
import traceback

from PIL import ImageDraw, Image

from epyseg.draw.shapes.image2d import Image2D
from epyseg.img import Img, PIL_to_numpy, invert
from skimage.measure import regionprops
import numpy as np
import matplotlib
from scipy.ndimage import grey_erosion, grey_dilation, generate_binary_structure
import os
from epyseg.ta.database.sql import TAsql, createMasterDB, sort_col_numpy
from epyseg.ta.luts.lut_minimal_test import PaletteCreator, apply_lut
from epyseg.ta.measurements.nematic import Nematic
from epyseg.ta.tracking.tools import smart_name_parser


# matplotlib.use('Qt5Agg') #, Qt5Cairo,
# matplotlib.use('Qt5Cairo') #, Qt5Cairo,
# matplotlib.use('WXAgg') #, Qt5Cairo,
# matplotlib.use('agg') #, Qt5Cairo,
# matplotlib.use('TkCairo') #, Qt5Cairo,
#  supported values are['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
# matplotlib.rcParams['backend'] = 'cairo'

# - interactive
# backends:
# GTK3Agg, GTK3Cairo, MacOSX, nbAgg,
# Qt4Agg, Qt4Cairo, Qt5Agg, Qt5Cairo,
# TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo
#
# - non - interactive
# backends:
# agg, cairo, pdf, pgf, ps, svg, template

# print(matplotlib.use)

import matplotlib.pyplot as plt
from epyseg.tools.logger import TA_logger  # logging
from epyseg.utils.loadlist import  loadlist
import re

logger = TA_logger()

# TODO finalize that
# need a type of plot and need data to be plotted that is associated to the id of the relevant table --> TODO
# whole data can also be obtained from a SQL command --> such as PLOT AS CELLS ...
# can I do blending directly at the qimage level ??? if so --> quite simple in a way!!!


# then need get a mapping of the colors --> TODO
# for the SQL plots see how I can do that --> need convert to number and do stuff with it
# faire un importeur de csv
# plus simple de matcher la cell ID avec une valeur que les coordonnées en fait dans cette nouvelle version et en plus il y a moins de données ce qui sera plus simple à gérer -> juste besoin de la local ID des cellules ou des vertex ou des bonds --> vraiment du gateau


'''
TODO to be ok with TA

//PLOT AS BONDS SQL
//PLOT AS RECTANGLE
//PLOT AS LINE
//PLOT AS NEMATIC
//PLOT AS VECTOR same as plot as line
//PLOT AS SQUARE
//PLOT AS POINTS
PLOT AS STRINGS 
'''

# check a set of TA commands to see how I can parse them
# in a way the SQL command is anything between select and the end of the line or a plus --> it is anything before the first encountered extra keyword --> should not be too hard to do!!!
# especially now that the command is cleaned
# if unknown keyword --> then return an error


extra_plot_commands_for_images = ['LUT', 'EROSION', 'DILATATION', 'DILATION', 'OPACITY', 'TRANSPARENCY']

def plot_SQL(SQL_command):
    """
    Plot SQL data based on the given SQL command.

    Args:
        SQL_command (str): The SQL command to plot.

    Examples:
        >>> plot_SQL("PLOT AS CELLS SELECT * FROM table")
        cleaned_SQL_command "PLOT AS CELLS SELECT * FROM table"
        plot_type "CELLS"
        SQL_command "SELECT * FROM table"
        extra_commands_to_parse "None"

    """
    if SQL_command is None:
        logger.error('Empty SQL command --> nothing to do')
        return

    if not isinstance(SQL_command, str):
        logger.error('SQL command must be a string, instead a ' + str(type(SQL_command)) + ' was passed...')
        return

    if SQL_command.strip().startswith('#') or SQL_command.strip().startswith('//'):
        return

    cleaned_SQL_command = _clean_string(SQL_command)

    print('cleaned_SQL_command "' + cleaned_SQL_command + '"')

    plot_type = None
    if cleaned_SQL_command.upper().startswith('PLOT AS CELLS'):
        plot_type = 'CELLS'
        cleaned_SQL_command = _strip_from_command(cleaned_SQL_command, 'PLOT AS CELLS')
    elif cleaned_SQL_command.upper().startswith('PLOT AS BONDS'):
        plot_type = 'BONDS'
        cleaned_SQL_command = _strip_from_command(cleaned_SQL_command, 'PLOT AS BONDS')
    elif cleaned_SQL_command.upper().startswith('PLOT AS VERTICES'):
        plot_type = 'VERTICES'
        cleaned_SQL_command = _strip_from_command(cleaned_SQL_command, 'PLOT AS VERTICES')
    else:
        logger.error('unsupported plot command: ' + str(cleaned_SQL_command))
        return

    print('plot_type "' + str(plot_type) + '"')

    first_idx = None
    for extra in extra_plot_commands_for_images:
        try:
            idx = cleaned_SQL_command.upper().index(extra)
            if first_idx is None:
                first_idx = idx
            else:
                first_idx = min(first_idx, idx)
        except:
            continue

    extra_commands_to_parse = None
    if first_idx is not None:
        SQL_command = cleaned_SQL_command[0:first_idx].strip()
        if SQL_command.endswith('+'):
            SQL_command = SQL_command[:-1].strip()
        extra_commands_to_parse = cleaned_SQL_command[first_idx:].strip()
    else:
        SQL_command = cleaned_SQL_command

    print('SQL_command "' + str(SQL_command) + '"')
    print('extra_commands_to_parse "' + str(extra_commands_to_parse) + '"')

    if extra_commands_to_parse:
        individual_extras = extra_commands_to_parse.split('+')

        for individual_extra in individual_extras:
            print('individual_extra "' + individual_extra.strip() + '"')

            extra_type = individual_extra.strip().split()

            if extra_type[0].strip().upper() not in extra_plot_commands_for_images:
                logger.error('unknown image extra ' + str(extra_type[0]) + ' --> ignoring')
            else:
                if extra_type[0].strip().upper() == extra_plot_commands_for_images[0]:
                    print('LUT to parse')
                    print('param', parse_lut(extra_type))
                elif extra_type[0].strip().upper() == extra_plot_commands_for_images[1]:
                    print('EROSION to parse')
                    print('param', parse_erosion_dilation(extra_type))
                elif extra_type[0].strip().upper() == extra_plot_commands_for_images[2] or extra_type[0].strip().upper() == extra_plot_commands_for_images[3]:
                    print('DILATION to parse')
                    print('param', parse_erosion_dilation(extra_type))
                elif extra_type[0].strip().upper() == extra_plot_commands_for_images[4]:
                    print('OPACITY to parse')
                    print('param', parse_opacity(extra_type))
                elif extra_type[0].strip().upper() == extra_plot_commands_for_images[5]:
                    print('TRANSPARENCY to parse')
                    print('param', 1. - parse_opacity(extra_type))

def parse_lut(extra_type):
    """
    Parse the LUT extra command.

    Args:
        extra_type (list): List of strings representing the LUT extra command.

    Returns:
        dict or str: The parsed LUT mapping as a dictionary or the name of the LUT.

    Examples:
        >>> parse_lut(['LUT', 'DNA'])
        'DNA'

    """

    if len(extra_type) == 2 and (not ':' in extra_type[1]):
        try:
            return extra_type[1]  # returns the name of the LUT
        except:
            pass
    else:
        if ':' in extra_type[1]:
            lut_mapping = {}
            for mapping in extra_type[1:]:
                split_mappings = mapping.split(':')
                split_mappings = [split_mapping.strip() for split_mapping in split_mappings]
                key = int(split_mappings[0])
                value = split_mappings[1]
                if '#' in value:
                    value = int(value.replace('#', ''), 16)
                else:
                    value = int(value)
                lut_mapping[key] = value
        else:
            logger.warning('LUT bounds not implemented yet')
            return parse_lut(extra_type=extra_type[0:2])
        return lut_mapping

    return None


def parse_opacity(extra_type):
    """
    Parse the opacity extra command.

    Args:
        extra_type (list): List of strings representing the opacity extra command.

    Returns:
        float or None: The parsed opacity value as a float or None if parsing fails.

    Examples:
        >>> parse_opacity(['OPACITY', '50'])
        50.0

        >>> parse_opacity(['OPACITY', '25%'])
        0.25

    """

    if len(extra_type) == 2:
        try:
            ispercent = '%' in extra_type[1]
            if ispercent:
                extra_type[1] = extra_type[1].replace('%', '')
            value = float(extra_type[1])
            if ispercent:
                value /= 100.
            return value
        except:
            pass
    return None


def parse_erosion_dilation(extra_type):
    """
    Parse the erosion or dilation extra command.

    Args:
        extra_type (list): List of strings representing the erosion or dilation extra command.

    Returns:
        int or None: The parsed erosion or dilation value as an integer or None if parsing fails.

    Examples:
        >>> parse_erosion_dilation(['EROSION', '2'])
        2

    """

    if len(extra_type) == 2:
        try:
            return int(extra_type[1])
        except:
            pass
    return None


def _clean_string(string):
    """
    Clean the string by removing leading/trailing whitespaces and extra whitespaces between words.

    Args:
        string (str): The input string.

    Returns:
        str: The cleaned string.

    Examples:
        >>> _clean_string('  Hello     world!   ')
        'Hello world!'

    """

    parsed_commands = string.split()
    parsed_commands = [command.strip() for command in parsed_commands]
    cleaned_string = ' '.join(parsed_commands)
    return cleaned_string


def _strip_from_command(command, things_to_strip):
    """
    Strip the specified things from the command.

    Args:
        command (str): The command string.
        things_to_strip (str): The things to strip from the command.

    Returns:
        str: The stripped command.

    Examples:
        >>> _strip_from_command('SELECT * FROM table', 'SELECT')
        '* FROM table'

        >>> _strip_from_command('HELLO world', 'hello')
        'world'

    """

    import re
    case_insensitive_replace = re.compile(re.escape(things_to_strip), re.IGNORECASE)
    stripped = case_insensitive_replace.sub('', command).strip()

    return stripped





# NB THERE IS A BUG THAT FORCES THE PLOT TWICE --> PROBABLY SOME ADJUSTMENT STUFF --> NEED CHANGE THIS
def plot_as_any(parent_image, SQL_command, plot_type='cells', return_mask=False, invert_mask=True, db=None,current_frame=None, **kwargs):
    """
    Plot image based on SQL command results.

    Args:
        parent_image (str): Path to the parent image.
        SQL_command (str): SQL command to run and get the results for plotting.
        plot_type (str, optional): Type of plot to generate. Defaults to 'cells'.
        return_mask (bool, optional): Whether to return the mask image. Defaults to False.
        invert_mask (bool, optional): Whether to invert the mask image. Defaults to True.
        db (TAsql, optional): TAsql object representing the database. Defaults to None.
        current_frame (int, optional): Current frame number. Defaults to None.
        **kwargs: Additional keyword arguments for extra parameters.

    Returns:
        None

    # Examples:
    #     # Example 1: Plotting as cells with default parameters
    #     plot_as_any('path/to/parent_image.tif', 'SELECT * FROM table')
    #
    #     # Example 2: Plotting as bonds with additional parameters
    #     plot_as_any('path/to/parent_image.tif', 'SELECT * FROM table', plot_type='bonds', return_mask=True, LUT='rainbow', opacity=0.5)
    #
    #     # Example 3: Plotting as vertices with database object
    #     db = TAsql('path/to/database.db')
    #     plot_as_any('path/to/parent_image.tif', 'SELECT * FROM table', plot_type='vertices', db=db, current_frame=1)

    """
    if parent_image is None:
        logger.error('No input image specified, cannot plot image!')
        return

    # print('current_frame inside', current_frame)

    TA_path = smart_name_parser(parent_image, ordered_output='TA')
    erosion_dilation = None
    lut = None
    opacity = None
    freq = None

    if kwargs:
        # parse extra parameters such as dilation...
        # ['LUT', 'EROSION', 'DILATATION','DILATION', 'OPACITY', 'TRANSPARENCY']
        if extra_plot_commands_for_images[0] in kwargs:
            lut = kwargs[extra_plot_commands_for_images[0]]
        if extra_plot_commands_for_images[1] in kwargs:
            erosion_dilation = -kwargs[extra_plot_commands_for_images[1]]
        if extra_plot_commands_for_images[
            3] in kwargs:  # elif extra_type[0].strip().upper() == extra_plot_commands_for_images[2]
            erosion_dilation = kwargs[extra_plot_commands_for_images[3]]
        if extra_plot_commands_for_images[4] in kwargs:
            opacity = kwargs[extra_plot_commands_for_images[4]]
        if extra_plot_commands_for_images[5] in kwargs:
            opacity = 1. - kwargs[extra_plot_commands_for_images[5]]
        if 'freq' in kwargs:
            freq = kwargs['freq']

    # those guys require the cell db
    if plot_type in ['cells', 'nematic', 'nematics']:
        # by default plot as cells
        logger.debug('plot as cells')
        image_plot = Img(os.path.join(TA_path,'cells.tif')).astype(np.uint64)
    elif plot_type == 'bonds':
        logger.debug('plot as bonds')
        image_plot = Img(os.path.join(TA_path,'bonds.tif')).astype(np.uint64)
    elif plot_type == 'vertices':
        logger.debug('plot as vertices')
        image_plot = Img(os.path.join(TA_path,'vertices.tif')).astype(np.uint64)
    elif plot_type == 'packing':
        logger.debug('plot as packing')
        image_plot = Img(os.path.join(TA_path, 'cells.tif')).astype(np.uint64)
    else:
        logger.error('Plot type unknonw: \'' + str(plot_type) + '\'')
        return

    if erosion_dilation and plot_type in ['packing', 'cells', 'bonds', 'vertices']:
        s = generate_binary_structure(2, 1)
        if erosion_dilation > 0:
            # do gray dilation on image
            for _ in range(abs(erosion_dilation)):
                image_plot = grey_dilation(image_plot, footprint=s)
        elif erosion_dilation < 0:
            # do gray erosion on image
            for _ in range(abs(erosion_dilation)):
                image_plot = grey_erosion(image_plot, footprint=s)

    # then run sql command and see what TODO exactly
    # map all to the first
    # if full then find image in full list by its index maybe -> do that at some point

    database_path = os.path.join(TA_path, 'pyTA.db')
    # database_path = os.path.join(TA_path, 'TA.db')
    if not os.path.isfile(database_path):
        logger.error('pyTA db not found, nothing to plot ' + str(database_path))
        return

    query_results = None
    # db = None
    if db is None:
        try:
            db = TAsql(database_path)
            headers, query_results = db.run_SQL_command_and_get_results(SQL_command, return_header=True)

            # need get the max and min too of that command --> see how I can do that

        except:
            pass
        finally:
            if db is not None:
                db.close()
                db=None
    else:
        try:
            # use and query the master db but do not close it --> need query it twice to get the max and min of the values
            # db = TAsql(database_path)
            if SQL_command.strip().endswith(';'):
                SQL_command.replace(';','')

            # print('new sql command for masted db', SQL_command + ' WHERE frame_nb == '+str(current_frame))
            headers, query_results = db.run_SQL_command_and_get_results(SQL_command + ' WHERE frame_nb == '+str(current_frame), return_header=True)
        except:
            pass

    if query_results is None:
        logger.error('Plot failed because of an erroneous SQL command: ' + str(SQL_command))
        return

    # faut que je reparse le truc
    # ça marche mais voir comment faire ça de maniere portable en fait
    # print(query_results, headers)

    # TODO replace by panda dataframe to handle many more things or do smart loopings to read as if it were cols from the data --> in a way that is also not that hard
    method1 = False
    method2 = False
    if method1:
        # add a 0,0 mapping for cells --> TODO
        # in case of clones I may miss so many cells !!! --> think about the fastest way to add them !!!
        query_results.insert(0, (0, 0))

        # is there anything more secure ???
        # maybe yes with a loop but will be much less efficient --> think about it

        # ça marche mais si il manque plein de cellules il faut que je les ajoute --> si differe de continu --> soit ajouter des valeurs

        # j'en suis pas loin mais faut reflechir à la facon la plus facile de faire ça
        # need get the max of the image

        # convert this to a mapping array
        # should not be too hard to do
        formatted = np.asarray(query_results)
        # print(formatted.shape, formatted.dtype)  # 337 2

        # In[153]: keyarray = np.array(['S', 'M', 'L', 'XL'])
        # In[158]: data = np.array([[0, 2, 1], [1, 3, 2]])
        # In[159]: keyarray[data]
        # Out[159]:

        keyarray = formatted[:, 1]

        # output = np.zeros_like(image_plot)
        output = keyarray[image_plot]

        # print('output', output.shape, output.dtype)

        # convert left values to that in right --> indexing

        # no do some magic plotting
        # and somehow apply a lut ???

        # do stuff for the cells --> such as a plot of the area, etc...
        # need get global max and min --> a good idea and very easy todo with sqlite

        # create an empty image that we can then fill

        # can easily replace that by values from a database

        # for plot in rps:
        #     # do stuff
        #     # cell_output[cell.coords[:, 0], cell.coords[:, 1]] = 255
        #     output[plot.coords[:, 0], plot.coords[:, 1]] = plot.area # easy area plot in fact

        # if opacity ... do stuff
        # if Lut do further stuff

        # todo --> DO A SECOND MAPPING FOR LUT --> TODO -->

        # maybe could use negative values to get excluded cells
        # the color coding would be better with me !!! in fact

        # plt.imshow(output, cmap='Pastel1')
        # # plt.imshow(output, cmap='rainbow')
        # # plt.imshow(output, cmap='viridis')
        # # plt.imshow(output, cmap='gray')
        # plt.show()
    elif method2:
        # marche pas car l'erosion peut faire perdre des IDs de cellules --> need a fix
        rps = regionprops(image_plot)
        formatted = np.asarray(query_results, dtype=object)
        output = np.zeros_like(image_plot)

        # can easily replace that by values from a database
        # another way of doing, it might ultimately be simpler

        for iii, id in enumerate(formatted[:, 0]):
            # for plot in rps:
            plot = rps[int(id) - 1]
            # do stuff
            # cell_output[cell.coords[:, 0], cell.coords[:, 1]] = 255



            output[plot.coords[:, 0], plot.coords[:, 1]] = formatted[iii, 1]  # easy area plot in fact

        # # plt.imshow(output, cmap='Pastel1')
        # # plt.imshow(output, cmap='rainbow')
        # plt.imshow(output, cmap='viridis')
        # # plt.imshow(output, cmap='gray')
        # plt.show()
    else:
        # ça marche aussi même avec l'erosion --> pas le plus propre ni le plus efficace mais super robuste --> donc probablement utiliser ça !!!
        # rps = regionprops(image_plot)
        # in fact that still does not work great --> cannot handle None
        formatted = np.asarray(query_results, dtype=object) # required for handling complex outputs that contain string Nones and ints and floats

        # print(formatted) # all ok here --> why not plotting properly

        # print('formatted.shape', formatted.shape)


        # print(formatted, '\n test sqdqdsq', db==None, SQL_command)

        # do an autoscale for polarity based on the value of polarity for the max Q1 and Q2 values and at some point offer a rescaling factor maybe
        # print formatted
        # plot as cells
        if plot_type in ['packing', 'cells', 'bonds', 'vertices']:
            output = np.zeros_like(image_plot, dtype=float) # super important to have float here otherwise loses data !!!
            if return_mask:
                mask = np.zeros_like(image_plot)
            # can easily replace that by values from a database
            # another way of doing, it might ultimately be simpler

            # print(formatted[:, 0])
            # if not formatted:
            #     return




            for iii, id in enumerate(formatted[:, 0]):
                # for plot in rps:
                # plot = rps[int(id) - 1]
                # do stuff
                # cell_output[cell.coords[:, 0], cell.coords[:, 1]] = 255
                # output[plot.coords[:, 0], plot.coords[:, 1]] = formatted[iii, 1]  # easy area plot in fact

                # print(id, formatted[iii, 1], type(id)) # numpy.str_ --> for id --> why is that ??????
                idx = image_plot == id
                output[idx] = formatted[iii, 1]  # easy area plot in fact




                # print(formatted[iii, 1])

                if return_mask:
                    mask[idx] = 255
                # FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
                #   output[image_plot == id] = formatted[iii, 1]  # easy area plot in fact

                # print(formatted[iii, 1]) # --> values are ok but I have a bug of conversion somewhere --> if values are very small  --> less than 1 they are rounded
            # plt.imshow(output)
            # plt.show()
            # up until here all is ok --> bug is in color coding
        elif plot_type.startswith('nematic'):
            # if return_mask:
            #     mask = np.zeros_like(image_plot)

            rps = regionprops(image_plot)
            # output = np.zeros_like(image_plot.shape, dtype=np.uint8)
            # im = numpy_to_PIL(output)

            # nb image is defined by with then height --> opposite of numpy!!!!
            # create a black image of the size of the image

            im = np.zeros_like(image_plot)
            # im = Image.new('RGB', (image_plot.shape[1],image_plot.shape[0]), (0,0,0))
            # draw = ImageDraw.Draw(im)

            has_scaling_factor = 'SCALING_FACTOR' in headers
            scaling_factor_idx = -1
            if has_scaling_factor:
                scaling_factor_idx = headers.index('SCALING_FACTOR')
                # print('scaling_factor_idx',scaling_factor_idx)
            has_colors = 'COLOR' in headers
            colors_idx = -1
            if has_colors:
                colors_idx= headers.index('COLOR')
            has_stroke_size = 'STROKE_SIZE' in headers
            stroke_idx = -1
            if has_stroke_size:
                stroke_idx =  headers.index('STROKE_SIZE')

            lines2ds = []
            # TODO --> implement nematic plot maybe with autoscaling depending on magnitude --> should not be too hard to do!!!
            for iii, id in enumerate(formatted[:, 0]):
                # need draw a line over the cell centroid
                # I need the rps corresponding to the cell to get its centroid
                # if
                plot = rps[int(id) - 1]
                # do stuff
                # cell_output[cell.coords[:, 0], cell.coords[:, 1]] = 255
                # output[plot.coords[:, 0], plot.coords[:, 1]] = formatted[iii, 1]  # easy area plot in fact
                centroid = plot.centroid


                # print(plot.centroid)
                # draw a nematic at this level
                # need

                # print('formatted[iii,1],formatted[iii,2]', formatted[iii,1],formatted[iii,2], formatted[iii,1]+1)



                nemat = Nematic(S1=formatted[iii, 1],S2=formatted[iii, 2], center = centroid)
                # nemat.draw(output)
                # nemat.draw(draw, stroke=2 if not has_stroke_size else formatted[iii, stroke_idx], rescaling_factor=None if not has_scaling_factor else formatted[iii, scaling_factor_idx], color= 0xFFFF00 if not has_colors else formatted[iii, colors_idx]) # todo scale it -> try that


                # TODO FINALIZE THE PLOTS WITH THE COLORS AND STUFF ALIKE




                l2d = nemat.toLine2D(rescaling_factor=1 if not has_scaling_factor else formatted[iii, scaling_factor_idx])
                # TODO add rescaling and line stroke
                if l2d is not None:
                    l2d.stroke = 2 if not has_stroke_size else formatted[iii, stroke_idx]
                    l2d.color =0xFFFF00 if not has_colors else formatted[iii, colors_idx]
                    lines2ds.append(l2d)

                    # print(l2d)


                # print(nemat)

                # would need to draw it as a line at some point but ok for now!!!
                # nemat

            im2d = Image2D(im)
            im2d.annotation.extend(lines2ds)

            im = im2d.save(None)
            im = im2d.convert_qimage_to_numpy(im)
            output = im

            # print('lines2ds', lines2ds)
            # output = PIL_to_numpy(im)



            if return_mask:
                mask = np.zeros_like(output)
                mask[output!=0]=255

            # print(output.shape)
            #
            # plt.imshow(output)
            # plt.show()

    # random_lut=random.choice(list(luts.keys()))
    # print('random.lut',random_lut)
    # plot.imshow(output)
    # plt.show()

    # print(output)


    # just get data normally and it is only for min and max that I will need the masterdb max and min
    # only for the LUT that I need that the rest can stay as such in fact
    if plot_type not in ['nematic','nematics']:
        # all seems ok --> just compare to TA to avoid issues
        # random_lut = 'HI_LO'
        lutcreator = PaletteCreator()
        luts = lutcreator.list
        # lut = lutcreator.create3(luts[lut])
        try:
            palette = lutcreator.create3(luts[lut])
        except:
            if lut is not None:
                logger.error('could not load the specified lut (' +str(lut)+') a gray lut is loaded instead') # --> ignoring or shall I default to gray
            # palette = None
            # default to grey palette
            palette = lutcreator.create3(luts['GRAY'])


        # maybe in the end also only get the modified pixels and plot only those --> especially for overlay or use transparency value to hide what should not be plotted --> set to fully transparent --> TODO!!!

        # lut = lutcreator.create3(luts['HI_LO'])

        # vraiment j'en suis pas loin --> juste voir comment finaliser les trucs
        # peut etre convertir l'output du truc de maniere simple
        # --> est-ce slow --> probablement pas
        # see how I can do that



        # ça marche presque --> juste tt finaliser maintenant !!!
        # output = apply_lut(output, lut, True, max=1000, min=900)
        # output = apply_lut(output, lut, True, max=6000, min=0) # max = 3*+ que le real max --> should all appear weak
        # output = apply_lut(output, lut, True, max=6000, min=100) # max = 3*+ que le real max --> should all appear weak # it seems to work but compare to TA
        # output = apply_lut(output, lut, True, max=3000, min=0) # 2*plus
        # is the bottom incorrect --> maybe
        # output = apply_lut(output, lut, True, max=1600, min=400) # 2*plus #a bit below real max # does it create a bug ???? --> maybe --> check it --> it looks weird to me
        # output = apply_lut(output, lut, True, max=0.9*output.max(), min=output.min()+(10.*(output.max()-output.min())/100.)) # 2*plus #a bit below real max # does it create a bug ???? --> maybe --> check it --> it looks weird to me


        # print(output) # no it's not rescaled
        # plt.imshow(output)
        # plt.show()

        # print(output.max())

        if plot_type == 'packing':
            # the trick to plot as packing is to maintain the data as it normally is and just bound it between 0 and 255 --> ignore all possible min and max
            # TODO see if I should not also add another value or other plots for example for bonds
            # or maybe plot as cell_id or alike and same stuff for bonds --> in such case apply no rescale whatever the data is... --> keep the raw values

            output = apply_lut(output, palette, True, min=0, max=255)
        else:

            # here I should get the max and min of the stuff to plot --> assumes there is a single value --> will that always be true????


            # need get real max and min here to apply a real global color code --> TODO


            # need get local or global max and min
            #
            # if SQL_command.strip().endswith(';'):
            #     SQL_command.replace(';', '')
            #
            # print(SQL_command + ' WHERE frame_nb == ' + str(current_frame))
            # headers, query_results = db.run_SQL_command_and_get_results(
            #     SQL_command + ' WHERE frame_nb == ' + str(current_frame), return_header=True)
            # probably need rerun the query

            # rerun the query

            min = None
            max = None
            # if db is
            if db is None:
                # print('getting local max and min TODO')
                try:
                    min, max = sort_col_numpy(formatted[:, -1], freq=freq)
                    logger.info('min is set to '+str(min)+ ' max is set to '+str(max))
                    # min is set to 1 max is set to 1 --> why
                except:
                    traceback.print_exc()
                    logger.error('local min/max would not be retrieved')

                # print()
            else:
                # print('getting global max and min TODO')
                # db.get_min_max()

                # print(formatted[:,-1]) # these are just the last values I would like to sort
                # seems ok now try to sort them

                # marche mais faut voir comment faire
                # if freq is not None:
                # is there any case where I should not do that ???

                headers, query_results = db.run_SQL_command_and_get_results(SQL_command, return_header=True)
                formatted = np.asarray(query_results, dtype=object)


                # need rerun the query without the frame nb

                # print(db.sort_col_numpy(formatted[:,-1]))

                # print(db.sort_col_numpy(formatted[:,-1], freq=(0.05, 0.05))) # ça marche pas mal enb fait
                try:
                    min, max = sort_col_numpy(formatted[:,-1], freq=freq)
                    logger.info('min is set to ' + str(min) + ' max is set to ' + str(max))
                except:
                    traceback.print_exc()
                    logger.error('global min/max would not be retrieved')

                # print(formatted.shape)


            output = apply_lut(output, palette, True, min=min, max=max) # 2*plus #a bit below real max # does it create a bug ???? --> maybe --> check it --> it looks weird to me

            # is that faster to get global max and min once for all

        # need check



        # je crois que tt marche mais faudrait vraiment tester en grandeur reelle
        # voir comment recuperer que les regions à vraiment plotter --> doit pas etre trop dur --> ignorer les trucs qui n'existent pas


        # if len(output.shape)==2:
        #     # plt.imshow(output, cmap='Pastel1')
        #     # plt.imshow(output, cmap='rainbow')
        #     plt.imshow(output, cmap='viridis')
        #     # plt.imshow(output, cmap='gray')
        # else:
        #     plt.imshow(output)
        # plt.show()
    if return_mask:
        if invert_mask:
            mask=invert(mask)
        # also we apply the mask to the output
        output[mask!=0]=0
        return mask, output
    return output


def plot_as_image_old(plot_type='cells'):
    if plot_type == 'cells':
        # by default plot as cells
        logger.debug('plot as cells')
        image_plot = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012/cells.tif').astype(
            np.uint64)

    elif plot_type == 'bonds':
        logger.debug('plot as bonds')
        image_plot = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012/bonds.tif').astype(
            np.uint64)
    elif plot_type == 'vertices':
        logger.debug('plot as vertices')
        image_plot = Img('/E/Sample_images/sample_images_PA/mini/focused_Series012/vertices.tif').astype(
            np.uint64)
    else:
        logger.error('Plot type unknown: \'' + str(plot_type) + '\'')
        return

    rps = regionprops(image_plot)

    # do stuff for the cells --> such as a plot of the area, etc...
    # need get global max and min --> a good idea and very easy todo with sqlite

    # create an empty image that we can then fill
    output = np.zeros_like(image_plot)

    # can easily replace that by values from a database

    for plot in rps:
        # do stuff
        # cell_output[cell.coords[:, 0], cell.coords[:, 1]] = 255
        output[plot.coords[:, 0], plot.coords[:, 1]] = plot.area  # easy area plot in fact

    plt.imshow(output)
    plt.show()
    # plt.show(block=True)


# --> this is all very good and simple
# could also make graphs easily now --> TODO too


# TO plot a cell I need a command that generate a local id as first col then the value to plot as second col then various parameters that can be used for plotting -> try finalize that!!!!


if __name__ == '__main__':

    print(matplotlib.get_backend())
    matplotlib.validate_backend(matplotlib.get_backend())
    matplotlib.use('TkAgg')  # dirty fix for erroneous display with Qt5Agg


    if True:
        file = '/E/Sample_images/trash/complete/focused_Series010.png'
        # file = '/E/Sample_images/sample_images_pyta/surface_projection/210219.lif_t000.tif'
        # SQL_command = "SELECT local_id_cells,P1_polarity_ch2, P2_polarity_ch2, '#FF0000' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells"   #for TA db
        # SQL_command = "SELECT local_id,nb_of_vertices_or_neighbours FROM cells_2D WHERE is_border == FALSE"  # a bit more complex just to see which method to choose --> very easy
        SQL_command = "SELECT local_id,S1, S2, 10 AS SCALING_FACTOR FROM cells_2D"  # a bit more complex just to see which method to choose --> very easy
        # extras = {'LUT': 'DNA'}
        # extras = {'LUT': 'None'}
        # mask, output = plot_as_any(file, SQL_command, plot_type='cells', return_mask=True, **extras)
        # mask, output = plot_as_any(file, SQL_command, plot_type='cells', return_mask=True, **extras)
        # pb cannot plot is border --> somehow cannot plot booleans --> need a special treatment to avoid issues ??? assume min is 0 then and max is another thing ??? --> shall I implement that or a plot as boolean
        mask, output = plot_as_any(file, SQL_command, plot_type='nematics', return_mask=True)
        # --> bug
        if len(output.shape) == 2:
            # plt.imshow(output, cmap='Pastel1')
            # plt.imshow(output, cmap='rainbow')
            plt.imshow(output, cmap='viridis')
            # plt.imshow(output, cmap='gray')
        else:
            plt.imshow(output)
            # plt.imshow(mask)
        plt.show()

    if False:
        # master db plot
        master_db = createMasterDB(loadlist('/E/Sample_images/sample_images_pyta/surface_projection/list.lst'))


        SQL_command = "SELECT local_id, area FROM cells_2D"  # a bit more complex just to see which method to choose --> very easy
        extras = {'LUT': 'DNA'}
        file = '/E/Sample_images/sample_images_pyta/surface_projection/210219.lif_t000.tif'
        mask, output = plot_as_any(file, SQL_command, plot_type='cells', return_mask=True, db=master_db, current_frame=0, **extras)
        if len(output.shape) == 2:
            # plt.imshow(output, cmap='Pastel1')
            # plt.imshow(output, cmap='rainbow')
            plt.imshow(output, cmap='viridis')
            # plt.imshow(output, cmap='gray')
        else:
            plt.imshow(output)
            # plt.imshow(mask)
        plt.show()
        master_db.close()



    if False:
        file = '/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012.png'
        # file = '/E/Sample_images/sample_images_pyta/surface_projection/210219.lif_t000.tif'
        # SQL_command = "SELECT local_id_cells,P1_polarity_ch2, P2_polarity_ch2, '#FF0000' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells"   #for TA db
        # SQL_command = "SELECT local_id,nb_of_vertices_or_neighbours FROM cells_2D WHERE is_border == FALSE"  # a bit more complex just to see which method to choose --> very easy
        SQL_command = "SELECT local_id, is_border FROM cells_2D"  # a bit more complex just to see which method to choose --> very easy
        # extras = {'LUT': 'DNA'}
        # extras = {'LUT': 'None'}
        # mask, output = plot_as_any(file, SQL_command, plot_type='cells', return_mask=True, **extras)
        # mask, output = plot_as_any(file, SQL_command, plot_type='cells', return_mask=True, **extras)
        # pb cannot plot is border --> somehow cannot plot booleans --> need a special treatment to avoid issues ??? assume min is 0 then and max is another thing ??? --> shall I implement that or a plot as boolean
        mask, output = plot_as_any(file, SQL_command, plot_type='cells', return_mask=True)
        # --> bug
        if len(output.shape) == 2:
            # plt.imshow(output, cmap='Pastel1')
            # plt.imshow(output, cmap='rainbow')
            plt.imshow(output, cmap='viridis')
            # plt.imshow(output, cmap='gray')
        else:
            plt.imshow(output)
            # plt.imshow(mask)
        plt.show()

    if False:
        file = '/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012.png'
        file = '/E/Sample_images/sample_images_pyta/surface_projection/210219.lif_t000.tif'
        # SQL_command = "SELECT local_id_cells,P1_polarity_ch2, P2_polarity_ch2, '#FF0000' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells"   #for TA db
        # SQL_command = "SELECT local_id,nb_of_vertices_or_neighbours FROM cells_2D WHERE is_border == FALSE"  # a bit more complex just to see which method to choose --> very easy
        SQL_command = "SELECT local_id, nb_of_vertices_or_neighbours FROM cells_2D"  # a bit more complex just to see which method to choose --> very easy
        extras = {'LUT': 'PACKING'}
        # mask, output = plot_as_any(file, SQL_command, plot_type='cells', return_mask=True, **extras)
        mask, output = plot_as_any(file, SQL_command, plot_type='packing', return_mask=True, **extras)
        if len(output.shape) == 2:
            # plt.imshow(output, cmap='Pastel1')
            # plt.imshow(output, cmap='rainbow')
            plt.imshow(output, cmap='viridis')
            # plt.imshow(output, cmap='gray')
        else:
            plt.imshow(output)
            # plt.imshow(mask)
        plt.show()

    if True:

        # file = '/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series014.png' # to see if the nematic bug is linked to too many channels --> maybe the case
        # SQL_command = "SELECT local_id,Q1_polarity_ch0, Q2_polarity_ch0, '#FF0000' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells_2D WHERE is_border == FALSE"  # a bit more complex just to see which method to choose --> very easy

        file = '/E/Sample_images/sample_images_PA/mini_different_nb_of_channels/focused_Series012.png'
        # SQL_command = "SELECT local_id_cells,P1_polarity_ch2, P2_polarity_ch2, '#FF0000' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells"   #for TA db
        SQL_command = "SELECT local_id,Q1_polarity_ch1, Q2_polarity_ch1, '#FF0000' AS COLOR, 2 AS STROKE_SIZE, 0.1 AS SCALING_FACTOR FROM cells_2D WHERE is_border == FALSE"  # a bit more complex just to see which method to choose --> very easy


        # file = '/E/Sample_images/sample_images_PA/mini/focused_Series012.png'
        # SQL_command = "PLOT AS CELLS SELECT local_id, area FROM cells_2D + LUT DNA + OPACITY 35% + DILATATION 1"
        # SQL_command = "PLOT AS NEMATICS SELECT center_x_cells,center_y_cells,P1_polarity_ch1, P2_polarity_ch1, '#FF0000' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells WHERE is_border=='false'"
        # SQL_command = "SELECT local_id, area FROM cells_2D"
        # SQL_command = "SELECT local_id, area FROM cells_2D WHERE is_border == FALSE"  # a bit more complex just to see which method to choose --> very easy
        # try plot a command that has a
        # SQL_command = "SELECT local_id, area FROM cells_2D WHERE is_border == FALSE"  # a bit more complex just to see which method to choose --> very easy
        # SQL_command = "SELECT local_id,Q1_polarity_ch1, Q2_polarity_ch1, '#FF0000' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells_2D WHERE is_border == FALSE"  # a bit more complex just to see which method to choose --> very easy


        # extras = {} # see how I can apply LUTS
        # extras = {'DILATION':1} # ça marche
        # extras = {'EROSION': 1}  # ça marche pas avec methode2 # seems to give me a bug --> why ??? --> shifts ids --> yes makes sense because some cells may be lost
        # extras = {'EROSION': 3}  # ça marche # seems to give me a bug --> why ??? --> shifts ids --> yes makes sense because some cells may be lost
        # extras = {'EROSION': 5}  # ça marche meme avec une strong erosion # seems to give me a bug --> why ??? --> shifts ids --> yes makes sense because some cells may be lost
        # extras = {'EROSION': 10}  # ça marche meme avec une strong erosion # seems to give me a bug --> why ??? --> shifts ids --> yes makes sense because some cells may be lost
        # extras = {'DILATION':22} # ça marche # dilation gives crappy results --> dialtion should be 1 not really more, just to fill the holes
        extras = {'LUT':'DNA'} # ça marche # dilation gives crappy results --> dialtion should be 1 not really more, just to fill the holes
        # extras = {'LUT':'copper'} # a cmap lut just to try # ça marche # dilation gives crappy results --> dialtion should be 1 not really more, just to fill the holes
        # output = plot_as_any(file, SQL_command, **extras)
        # output = plot_as_any(file, SQL_command, plot_type='cells',**extras)
        mask, output = plot_as_any(file, SQL_command, plot_type='nematics', return_mask=True,**extras)
        if len(output.shape)==2:
            # plt.imshow(output, cmap='Pastel1')
            # plt.imshow(output, cmap='rainbow')
            plt.imshow(output, cmap='viridis')
            # plt.imshow(output, cmap='gray')
        else:
            plt.imshow(output)
            # plt.imshow(mask)
        plt.show()

    # test all the stuff
    # TODO

    '''

    here are plenty of command line examples just to see how I can handle that
     titleNCommands.put("SQL command to select the 10 biggest cells (biggest area) (this will not generate a plot)", "SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells FROM cells ORDER BY area_cells DESC LIMIT 10");
            titleNCommands.put("plots 10 biggest cells (biggest area)", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells FROM cells ORDER BY area_cells DESC LIMIT 10");
            titleNCommands.put("plots 10 biggest cells (biggest area) and color code them by area (color code only applies to the current image, it is local not global)", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells FROM cells ORDER BY area_cells DESC LIMIT 10 + LUT DNA");
            titleNCommands.put("plots 10 biggest cells (biggest area) and color code them by area (use same color code/scaling parameters for all images)", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells FROM cells ORDER BY area_cells DESC LIMIT 10 + LUT DNA MIN=GLOBAL_MIN MAX=GLOBAL_MAX");
            titleNCommands.put("plots 10 biggest cells (biggest area) and color them in red", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells, '#FF0000' AS COLOR FROM cells ORDER BY area_cells DESC LIMIT 10");
            titleNCommands.put("plots 10 biggest cells (biggest area) and color them in red and plot them transparently (please select a background image)", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells, '#FF0000' AS COLOR FROM cells ORDER BY area_cells DESC LIMIT 10 + OPACITY 35%");
            titleNCommands.put("plots 10 biggest cells (biggest area) and color them in red, dilate them by one pixel and plot them transparently (please select a background image)", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells, '#FF0000' AS COLOR FROM cells ORDER BY area_cells DESC LIMIT 10 + DILATATION 1 + OPACITY 35%");
            titleNCommands.put("plots in red the elongation/stretch nematic scaled by a factor = 60, border cells are excluded", "PLOT AS NEMATICS SELECT center_x_cells,center_y_cells,S1_stretch_cells, S2_stretch_cells, '#FF0000' AS COLOR, '2' AS STROKE_SIZE, 60 AS SCALING_FACTOR FROM cells WHERE is_border_cell=='false'");
            titleNCommands.put("plots in red the polarity nematic of the image channel 1 scaled by a factor = 0.06, border cells are excluded, NB: will not display anything if the channel is empty", "PLOT AS NEMATICS SELECT center_x_cells,center_y_cells,P1_polarity_ch1, P2_polarity_ch1, '#FF0000' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells WHERE is_border_cell=='false'");
            titleNCommands.put("plots in green the polarity nematic of the image channel 2 scaled by a factor = 0.06, border cells are excluded, NB: will not display anything if the channel is empty", "PLOT AS NEMATICS SELECT center_x_cells,center_y_cells,P1_polarity_ch2, P2_polarity_ch2, '#00FF00' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells WHERE is_border_cell=='false'");
            titleNCommands.put("plots in blue the polarity nematic of the image channel 3 scaled by a factor = 0.06, border cells are excluded, NB: will not display anything if the channel is empty", "PLOT AS NEMATICS SELECT center_x_cells,center_y_cells,P1_polarity_ch3, P2_polarity_ch3, '#0000FF' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells WHERE is_border_cell=='false'");
            titleNCommands.put("plots in purple the polarity nematic of the 16 bits (single channel) image scaled by a factor = 0.0006, border cells are excluded", "PLOT AS NEMATICS SELECT center_x_cells,center_y_cells,P1_polarity_12bits, P2_polarity_12bits, '#FF00FF' AS COLOR, '2' AS STROKE_SIZE, 0.0006 AS SCALING_FACTOR FROM cells WHERE is_border_cell=='false'");
            titleNCommands.put("plots in red the intensity normalized polarity nematic of the image channel 1 scaled by a factor = 0.06, border cells are excluded, NB: will not display anything if the channel is empty", "PLOT AS NEMATICS SELECT center_x_cells,center_y_cells,P1_polarity_ch1_normalized, P2_polarity_ch1_normalized, '#FF0000' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells WHERE is_border_cell=='false'");
            titleNCommands.put("plots in green the intensity normalized polarity nematic of the image channel 2 scaled by a factor = 0.06, border cells are excluded, NB: will not display anything if the channel is empty", "PLOT AS NEMATICS SELECT center_x_cells,center_y_cells,P1_polarity_ch2_normalized, P2_polarity_ch2_normalized, '#00FF00' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells WHERE is_border_cell=='false'");
            titleNCommands.put("plots in blue the intensity normalized polarity nematic of the image channel 3 scaled by a factor = 0.06, border cells are excluded, NB: will not display anything if the channel is empty", "PLOT AS NEMATICS SELECT center_x_cells,center_y_cells,P1_polarity_ch3_normalized, P2_polarity_ch3_normalized, '#0000FF' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells WHERE is_border_cell=='false'");
            titleNCommands.put("plots in purple the intensity normalized polarity nematic of the 16 bits (single channel) image scaled by a factor = 0.0006, border cells are excluded", "PLOT AS NEMATICS SELECT center_x_cells,center_y_cells,P1_polarity_12bits_normalized, P2_polarity_12bits_normalized, '#FF00FF' AS COLOR, '2' AS STROKE_SIZE, 0.0006 AS SCALING_FACTOR FROM cells WHERE is_border_cell=='false'");
            titleNCommands.put("plots (-10, -5) yellow vectors centered on cells with a scaling factor 2, border cells are excluded", "PLOT AS VECTORS SELECT center_x_cells,center_y_cells, -10, -5, '#FFFF00' AS COLOR, '2' AS STROKE_SIZE, 2 AS SCALING_FACTOR FROM cells WHERE is_border_cell=='false'");
            //do the same for polarity for one channel in particular
            //maybe put all three channels separately
            titleNCommands.put("plots a line joining the centroid of cells and their first encountered pixel", "PLOT AS LINES SELECT first_pixel_x_cells,first_pixel_y_cells , center_x_cells,center_y_cells, '#0000FF' AS COLOR, '12' AS STROKE_SIZE, '0.3' AS STROKE_OPACITY FROM cells");
            titleNCommands.put("Plot cell geometry, triangles are in cyan, 4-sided green, 5-sided yellow, hexagons gray, 7-sided blue, cells 8 or 9 neighbours are red", "PLOT AS CELLS SELECT first_pixel_x_cells, first_pixel_y_cells, nb_of_vertices_cut_off FROM cells + LUT 3:#00FFFF 4:#00FF00 5:#FFFF00 6:#AAAAAA 7:#0000FF 8:#FF0000 9:#FF0000");
            titleNCommands.put("plots an arrow joining the centroid of cells and the first encountered cell pixel", "PLOT AS ARROWS SELECT center_x_cells,center_y_cells,first_pixel_x_cells , first_pixel_y_cells FROM cells");
            titleNCommands.put("plots an arrow joining the centroid of cells and the first encountered cell pixel, the arrow width is defined to 20 pixels and the arrowheight to 10", "PLOT AS ARROWS SELECT center_x_cells,center_y_cells,first_pixel_x_cells , first_pixel_y_cells, 20 AS ARROWHEAD_WIDTH, 10 AS ARROWHEAD_HEIGHT  FROM cells");
            titleNCommands.put("plots an half headed arrow (up) joining the centroid of cells and the first encountered cell pixel", "PLOT AS ARROWS SELECT center_x_cells,center_y_cells,first_pixel_x_cells , first_pixel_y_cells, 'HALF_HEAD_UP_ARROW' AS ARROWHEAD_TYPE FROM cells");
            titleNCommands.put("plots an half headed arrow (down) joining the centroid of cells and the first encountered cell pixel", "PLOT AS ARROWS SELECT center_x_cells,center_y_cells,first_pixel_x_cells , first_pixel_y_cells, 'HALF_HEAD_DOWN_ARROW' AS ARROWHEAD_TYPE FROM cells");
            titleNCommands.put("plots an double headed arrow joining the centroid of cells and the first encountered cell pixel", "PLOT AS ARROWS SELECT center_x_cells,center_y_cells,first_pixel_x_cells , first_pixel_y_cells, 'DOUBLE_HEAD_ARROW' AS ARROWHEAD_TYPE FROM cells");
            titleNCommands.put("plots an inhibition arrow joining the centroid of cells and the first encountered cell pixel", "PLOT AS ARROWS SELECT center_x_cells,center_y_cells,first_pixel_x_cells , first_pixel_y_cells, 'INIBITION_ARROW' AS ARROWHEAD_TYPE FROM cells");
            titleNCommands.put("plots a double headed inhibition arrow joining the centroid of cells and the first encountered cell pixel", "PLOT AS ARROWS SELECT center_x_cells,center_y_cells,first_pixel_x_cells , first_pixel_y_cells, 'DOUBLE_HEADED_INIBITION' AS ARROWHEAD_TYPE FROM cells");

            titleNCommands.put("plots a text containing the 'local id of cells' (i.e. their cell nb) centered over the cell centroid", "PLOT AS STRINGS SELECT center_x_cells, center_y_cells , local_id_cells, '#00FF00' AS COLOR, '#FF0000' AS BG_COLOR, '22' AS FONT_SIZE, 'ITALIC' AS FONT_STYLE FROM cells");
            titleNCommands.put("plots a square the width of which is proportional to the cell area/100 where the top left corner of the square is the centroid", "PLOT AS SQUARES SELECT center_x_cells, center_y_cells  , area_cells/100, '8016' AS COLOR,'30000' AS FILL_COLOR,'0.3' AS FILL_OPACITY FROM cells");
            //titleNCommands.put("plots a vector joining the first encountered pixel of the cell to its cenroid", "PLOT AS VECTORS SELECT first_pixel_x_cells,first_pixel_y_cells , center_x_cells,center_y_cells, '#00FF00' AS COLOR, '1' AS STROKE_SIZE, 'DEFAULT_ARROW' AS ARROWHEAD_TYPE, '5' AS ARROWHEAD_HEIGHT,'3' AS ARROWHEAD_HEIGHT, '0.3' AS STROKE_OPACITY FROM cells;");
            titleNCommands.put("plots all cells with an area between 1000 and 1200 pixels. NB may not show anything depending on your cell area", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, '#FFFF00' AS COLOR FROM cells WHERE area_cells > 1000 AND area_cells <1200  + OPACITY 35% + DILATATION 1");
            titleNCommands.put("color codes all cells according to their area", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, area_cells FROM cells + LUT DNA");
            titleNCommands.put("color codes (according to cell area) a vector joining the first encountered pixel of the cell to its cenroid", "PLOT AS ARROWS SELECT first_pixel_x_cells,first_pixel_y_cells , center_x_cells,center_y_cells, area_cells AS COLOR, '1' AS STROKE_SIZE, 'DEFAULT_ARROW' AS ARROWHEAD_TYPE, '5' AS ARROWHEAD_HEIGHT,'3' AS ARROWHEAD_HEIGHT, '0.3' AS STROKE_OPACITY FROM cells + LUT DNA");
            titleNCommands.put("color codes bonds according to their orientation (ignores bonds consisting only of two vertices)", "PLOT AS BONDS SELECT first_pixel_x_bonds, first_pixel_y_bonds, bond_orientation FROM bonds WHERE first_pixel_y_bonds != 'null' + LUT DNA");
            titleNCommands.put("draws cells as polygons (using vertex coordinates)", "PLOT AS POLYGONS SELECT vx_coords_cells, area_cells AS COLOR,area_cells AS FILL_COLOR,  '0.65' AS STROKE_SIZE FROM cells + LUT DNA + OPACITY 35%");
            titleNCommands.put("Test a SQL command (i.e. preview table output)", "SELECT * FROM CELLS");
            titleNCommands.put("color code cells accoding to their elongation", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells + LUT DNA + OPACITY 35% + DILATATION 1");
            titleNCommands.put("color code cells, except border cells, accoding to their elongation", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells WHERE is_border_cell=='false' + LUT DNA + OPACITY 35% + DILATATION 1");
            titleNCommands.put("color code cells, except border cells and cells immediately adjacent to border cells, accoding to their elongation", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells WHERE is_border_cell_plus_one=='false' + LUT DNA + OPACITY 35% + DILATATION 1");

            titleNCommands.put("(Global) color code cells belonging to clone #22 according to their amount of stretch (NB: virtual clone #22 must exist for the command to work)", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells NATURAL JOIN tracked_clone_022 + LUT DNA MIN=GLOBAL_MIN MAX=GLOBAL_MAX + OPACITY 35% + DILATATION 1");
            titleNCommands.put("(Global) color code cells belonging to clone #0 according to their amount of stretch (NB: virtual clone #0 must exist for the command to work)", "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells NATURAL JOIN tracked_clone + LUT DNA MIN=GLOBAL_MIN MAX=GLOBAL_MAX + OPACITY 35% + DILATATION 1");
    '''


    if False:
        plot_SQL('PLOT     AS  CELLS SELECT * FROM CELLS;')
        plot_SQL('PLOT     AS            BONDS SELECT * FROM CELLS;')
        plot_SQL('PLOT     AS  VERTices                      SELECT * FROM CELLS;')
        plot_SQL('PLOT     AS TEST SELECT * FROM CELLS;')
        plot_SQL('PLOT AS        POLYGON SELECT * FROM CELLS;')

        plot_SQL("PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells FROM cells ORDER BY area_cells DESC LIMIT 10")  # --> see how I can do that
        plot_SQL("PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells FROM cells ORDER BY area_cells DESC LIMIT 10 + LUT DNA")  # --> see how I can do that
        plot_SQL("PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells, '#FF0000' AS COLOR FROM cells ORDER BY area_cells DESC LIMIT 10 + DILATATION 1 + OPACITY 35%")  # in fact each extra is starting with a + --> quite easy to get
        # --> ignore pluses in extras
        plot_SQL("PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells FROM cells ORDER BY area_cells DESC LIMIT 10 + LUT DNA MIN=GLOBAL_MIN MAX=GLOBAL_MAX")  # --> maybe not that hard because the values if complex are connected to MIn and MAX --> if all are like that then I can parse all extras as keys and values and all is gonna be a piece of cake and easy to detect and report errors --> TODO

        plot_SQL("PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells + LUT DNA + OPACITY 35% + DILATATION 1")  # --> maybe not that hard because the values if complex are connected to MIn and MAX --> if all are like that then I can parse all extras as keys and values and all is gonna be a piece of cake and easy to detect and report errors --> TODO
        plot_SQL("PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells + LUT DNA + OPACITY 35% + DILATATION 1+BOUGA BOUU")  # --> maybe not that hard because the values if complex are connected to MIn and MAX --> if all are like that then I can parse all extras as keys and values and all is gonna be a piece of cake and easy to detect and report errors --> TODO
        plot_SQL("PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells + LUT DNA + OPACITY 35% + DILATATION 1+BOUGA")  # --> maybe not that hard because the values if complex are connected to MIn and MAX --> if all are like that then I can parse all extras as keys and values and all is gonna be a piece of cake and easy to detect and report errors --> TODO
        plot_SQL("PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells + LUT DNA + TRANSPARENCY 35% + DILATATION 1+BOUGA")  # --> maybe not that hard because the values if complex are connected to MIn and MAX --> if all are like that then I can parse all extras as keys and values and all is gonna be a piece of cake and easy to detect and report errors --> TODO
        plot_SQL("PLOT AS CELLS SELECT first_pixel_x_cells, first_pixel_y_cells, nb_of_vertices_cut_off FROM cells + LUT 3:#00FFFF 4:#00FF00 5:#FFFF00 6:#AAAAAA 7:#0000FF 8:#FF0000 9:#FF0000")  # --> maybe not that hard because the values if complex are connected to MIn and MAX --> if all are like that then I can parse all extras as keys and values and all is gonna be a piece of cake and easy to detect and report errors --> TODO

        # value and how to do that

        # or parse the extras between the pluses and accept all the data in between is data connected to the first keyword --> TODO
        # _strip_from_command("PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells FROM cells ORDER BY area_cells DESC LIMIT 10", "PLOT AS CELLS")
        # --> ok that seems to work

        print(parse_opacity("OPACITY 35%".split()))

        # for _ in range(abs(0)):
        #     print('hello')

    if False:
        plot_as_image_old(plot_type=None)
        plot_as_image_old()
