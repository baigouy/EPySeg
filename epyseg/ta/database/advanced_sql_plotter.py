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
from epyseg.img import Img, PIL_to_numpy
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

        plotCommandsTranslator.put("[Cc][Ee][Ll][Ll][Ss]{0,1}", "CELLS");
        plotCommandsTranslator.put("[Bb][Oo][Nn][Dd][Ss]{0,1}", "BONDS");
        plotCommandsTranslator.put("[Vv][Ee][Cc][Tt][Oo][Rr][Ss]{0,1}", "VECTORS");//vectors
        plotCommandsTranslator.put("[Nn][Ee][Mm][Aa][Tt][Ii][Cc][Ss]{0,1}", "NEMATICS");
        plotCommandsTranslator.put("[Ss][Qq][Uu][Aa][Rr][Ee][Ss]{0,1}", "SQUARES");
        plotCommandsTranslator.put("[Rr][Ee][Cc][Tt][Aa][Nn][Gg][Ll][Ee][Ss]{0,1}", "RECTANGLES");
        plotCommandsTranslator.put("[Ee][Ll][Ll][Ii][Pp][Ss][Ee][Ss]{0,1}", "ELLIPSES");
        plotCommandsTranslator.put("[Cc][Ii][Rr][Cc][Ll][Ee][Ss]{0,1}", "CIRCLES");//should we cast it as an ellipse
        plotCommandsTranslator.put("[Pp][Oo][Ll][Yy][Gg][Oo][Nn][Ss]{0,1}", "POLYGONS");
        plotCommandsTranslator.put("[Ll][Ii][Nn][Ee][Ss]{0,1}", "LINES");
        plotCommandsTranslator.put("[Aa][Rr][Rr][Oo][Ww][Ss]{0,1}", "ARROWS");//default arrow no rescaling allowed for this unlike for the nematics
        plotCommandsTranslator.put("[Pp][Oo][Ii][Nn][Tt][Ss]{0,1}|[Dd][Oo][Tt][Ss]{0,1}", "DOTS");//points or DOTS
        plotCommandsTranslator.put("[Ss][Tt][Rr][Ii][Nn][Gg][Ss]{0,1}|[Tt][Ee][Xx][Tt][Ss]{0,1}", "STRINGS");//points or DOTS


    need implement this too
         if (commandTypeAndCommand.containsKey("EROSION")) {
            parsedPlot[4] = "EROSION " + commandTypeAndCommand.get("EROSION");
        }
        if (commandTypeAndCommand.containsKey("DILATATION")) {
//            System.out.println("in " +commandTypeAndCommand.get("DILATATION"));
            parsedPlot[4] = "DILATATION " + commandTypeAndCommand.get("DILATATION");
        }
        //now we further parse lut flavours to do things with them
        if (commandTypeAndCommand.containsKey("LUT")) {
            parsedPlot[3] = "LUT " + commandTypeAndCommand.get("LUT");
        }
        if (commandTypeAndCommand.containsKey("OPACITY")) {
            parsedPlot[2] = "OPACITY " + commandTypeAndCommand.get("OPACITY");
        }
        if (commandTypeAndCommand.containsKey("TRANSPARENCY")) {
            parsedPlot[2] = "TRANSPARENCY " + commandTypeAndCommand.get("TRANSPARENCY");
        }
        
        
        # see how I can parse it smartly --> now
        
        
        
        
        some tests that need be implemented
        
           String testAdvancedCMD = "PLOT AS CELLS, area_cells+10 SELECT * FROM cells + OPACITY 55% + LUT (255*R,133+G,23+B)";

            //sinon juste faire des parsers de LUTS
            //styles LUT name ou formule des correspondances entre 
            //si formule --> entre parentheses
            HashMap<String, String> LUT_ParserNType = new HashMap<String, String>();

            String testPercentages = "TEST 55% 63% #FF00FF 33%";
            System.out.println(BrickPlotter.getPercent(testPercentages));

            String colorSplitterTest = "TEST #00FF00 #FF0000 #0000AA";
            System.out.println(BrickPlotter.parseHTMLColors(colorSplitterTest));

            System.out.println(BrickPlotter.parseHTMLColors("LUT #FF00FF 0:0 255:16000 12:#FF0000 16:1500"));



# manually force max and min too --> not sure it is very necessary 

 String maxParser = ".*([Mm][Aa][Xx]\\s{0,}[=]\\s{0,}(\\d{1,}[\\.,]{0,}\\d{0,})).*";//
            String minParser = ".*([Mm][Ii][Nn]\\s{0,}[=]\\s{0,}(\\d{1,}[\\.,]{0,}\\d{0,})).*";//
            String testCommand = "+ LUT DNA max=123.3 min=10.1";


    and I should also be able to parse such a complex line --> DOES A COMBINATION OF ALL
   String command = "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, area_cells FROM cells + LUT DNA +OPACITY 35% #FF0000\n"
                + "PLOT AS CELLS   SELECT * FROM cells \n"
                + "#SQL=SELECT * FROM bonds\n"
                + "CREATE TABLE TEST2 AS SELECT * FROM cells\n"
                + "#SQL=SELECT * FROM bonds\n"//ignore commented lines
                + "#SQL =  SELECT * FROM bonds\n"
                + "  PLOT   AS      BONDS  SELECT * FROM SELECT * FROM bonds   \n"
                + "  PLOT   AS      CRABOUNIASSE  SELECT * FROM cells  \n"
                + "PLOT AS  LINES SELECT * FROM cells\n"
                + "PLOT AS VECTORS   SELECT * FROM polarity    "
                + "PLOT AS VECTORS   SELECT * FROM polarity  + OPACITY 35% + LUT RAINBOW \n"
                + "PLOT AS VECTORS SELECT first_pixel_x_cells,first_pixel_y_cells , center_x_cells,center_y_cells, '16166' AS COLORS, '1' AS STROKE_SIZE, 'DEFAULT_ARROW' AS ARROWHEAD_TYPE, '5' AS ARROWHEAD_HEIGHT,'3' AS ARROWHEAD_HEIGHT, '0.3' AS STROKE_OPACITY FROM cells +DILATATION -10\n"
                + "PLOT AS VECTORS SELECT first_pixel_x_cells,first_pixel_y_cells , center_x_cells,center_y_cells, '16166' AS COLORS, '1' AS STROKE_SIZE, 'DEFAULT_ARROW' AS ARROWHEAD_TYPE, '5' AS ARROWHEAD_HEIGHT,'3' AS ARROWHEAD_HEIGHT, '0.3' AS STROKE_OPACITY FROM cells + OPACITY 100%\n"
                + "PLOT AS VECTORS SELECT first_pixel_x_cells,first_pixel_y_cells , center_x_cells,center_y_cells, area_cells AS COLOR, '1' AS STROKE_SIZE, 'DEFAULT_ARROW' AS ARROWHEAD_TYPE, '5' AS ARROWHEAD_HEIGHT,'3' AS ARROWHEAD_HEIGHT, '0.3' AS STROKE_OPACITY FROM cells + LUT DNA\n";
        
        
        
        
        HashMap<String, String> extraParser = new HashMap<String, String>();
        extraParser.put(LUT_flavour, "LUT");
        extraParser.put(OPACITY_flavour, "OPACITY");
        extraParser.put(TRANSPARENCY_flavour, "TRANSPARENCY");
        extraParser.put(DILATATION_flavour, "DILATATION");        

   public static ArrayList<Float> getPercent(String textToParse) {
        ArrayList<Float> percents = new ArrayList<Float>();
        String percentParser = "(\\d{1,})%";
        Pattern p = Pattern.compile(percentParser, Pattern.DOTALL);
        Matcher m = p.matcher(textToParse);
        while (m.find()) {
            percents.add(CommonClasses.String2Float(m.group(1)) / 100f);
        }
        return percents;
    }


public static ArrayList<Integer> parseHTMLColors(String textToParse) {
        /**
         * we parse colors except if preceded by a : stuff
         */
        String colorTAGS = "[^:](#[0-9a-fA-F]{6}|[0-9\\.]{1,})";
        Pattern p = Pattern.compile(colorTAGS, Pattern.DOTALL);
        Matcher m = p.matcher(textToParse);

        ArrayList<Integer> forbiddenColors = new ArrayList<Integer>();
        while (m.find()) {
            //System.out.println(m.group(1));
            forbiddenColors.add(CommonClasses.getColorFromHtmlColor(m.group(1)));
        }
        return forbiddenColors;
    }

   /**
     * deep parsing of LUT text files can do everything with that
     *
     * @param testLUT2
     * @return
     */
    public static HashMap<Integer, Integer> parseLUT(String testLUT2) {
        HashMap<Integer, Integer> LUTconversionTable = new HashMap<Integer, Integer>();
        String LutHumanSpecified = "(\\d{1,}):[']{0,1}([#A-Fa-f0-9]{1,}|[0-9\\.]{1,})[']{0,1}";

        Pattern p = Pattern.compile(LutHumanSpecified);
        Matcher m = p.matcher(testLUT2);

        while (m.find()) {
            try {
                String outputValue = m.group(2);
                if (!outputValue.contains("#")) {
                    LUTconversionTable.put(CommonClasses.String2Int(m.group(1)), CommonClasses.String2Int(outputValue));
                } else {
                    LUTconversionTable.put(CommonClasses.String2Int(m.group(1)), CommonClasses.getColorFromHtmlColor(outputValue));
                }
            } catch (Exception e) {
            }
        }
        return LUTconversionTable;
    }

    private String getSupportedPlotTypes() {
        String supportedPlots = "Supported plots are:";
        for (Map.Entry<String, String> entry : plotCommandsTranslator.entrySet()) {
            String value = entry.getValue();
            supportedPlots += "\n" + value;
        }
        return supportedPlots + "\n";
    }

    public String[] parsedSinglePlotLine(String individualCommmand) {
        if (individualCommmand == null || individualCommmand.isEmpty()) {
            return null;
        }
        /**
         * first we clean the commands
         */
        individualCommmand = individualCommmand.trim();

        /**
         * we ignore all commented lines
         */
        if (individualCommmand.startsWith("#") || individualCommmand.startsWith("//")) {
            return null;
        }
        /**
         * one is plot, two is SQL command, 3 is transparency, four is LUT
         */
        String[] parsedPlot = new String[5];
//        System.out.println(individualCommmand);
//        if (individualCommmand.matches(SQLPattern)) {
//            //parse SQL command
//            String SQLcommand = individualCommmand.replaceAll(SQLPattern, "$1").trim();
////            System.out.println("extracted SQLcommand='" + SQLcommand + "'");
//        }

        HashMap<String, String> commandTypeAndCommand = new HashMap<String, String>();
        //TODO do not cut if in between brackets --> allows formulas to be passed
        //only cuts if + is not in between brackets

        /**
         * first we strip TRANSPARENCY and anaestetic commands
         */
        String LUT_flavour = ".*\\s{0,}[Ll][Uu][Tt][Ss]{0,1}(.*)|.*\\s{0,}[Pp][Aa][Ll][Ee][Tt][Tt][Ee][Ss]{0,1}(.*)";
        String TRANSPARENCY_flavour = ".*\\s{0,}[Tt][Rr][Aa][Nn][Ss][Pp][Aa][Rr][Ee][Nn][Cc][Yy](.*\\d{1,}%.*)";
        String OPACITY_flavour = ".*\\s{0,}[Oo][Pp][Aa][Cc][Ii][Tt][Yy](.*\\d{1,}%.*)";
        String DILATATION_flavour = ".*\\s{0,}[Dd][Ii][Ll][Aa][Tt][Aa][Tt][Ii][Oo][Nn][Ss]{0,1}(.*)|.*\\s{0,}[Ee][Rr][Oo][Ss][Ii][Oo][Nn][Ss]{0,1}(.*)";
        //pour les LUTs faire aussi un parser manuel
        HashMap<String, String> extraParser = new HashMap<String, String>();
        extraParser.put(LUT_flavour, "LUT");
        extraParser.put(OPACITY_flavour, "OPACITY");
        extraParser.put(TRANSPARENCY_flavour, "TRANSPARENCY");
        extraParser.put(DILATATION_flavour, "DILATATION");

        String[] ExtraSplitted = individualCommmand.split("\\+(?![^()]*\\))", -1);
        //System.out.println(ExtraSplitted.length + " "+individualCommmand);
        for (String string : ExtraSplitted) {
            if (string.equals(individualCommmand)) {
                break;
            }
            for (Map.Entry<String, String> entry : extraParser.entrySet()) {
                String key = entry.getKey();
                String value = entry.getValue();
                //System.out.println(key + " "+string + " "+string.matches(key));

                if (string.matches(key)) {
                    commandTypeAndCommand.put(value, string.replaceAll(key, "$1").trim());
                    //            System.out.println(string);
                    individualCommmand = individualCommmand.replaceAll("\\+" + string, "").trim();
                }
            }
        }

        if (commandTypeAndCommand.containsKey("EROSION")) {
            parsedPlot[4] = "EROSION " + commandTypeAndCommand.get("EROSION");
        }
        if (commandTypeAndCommand.containsKey("DILATATION")) {
//            System.out.println("in " +commandTypeAndCommand.get("DILATATION"));
            parsedPlot[4] = "DILATATION " + commandTypeAndCommand.get("DILATATION");
        }
        //now we further parse lut flavours to do things with them
        if (commandTypeAndCommand.containsKey("LUT")) {
            parsedPlot[3] = "LUT " + commandTypeAndCommand.get("LUT");
        }
        if (commandTypeAndCommand.containsKey("OPACITY")) {
            parsedPlot[2] = "OPACITY " + commandTypeAndCommand.get("OPACITY");
        }
        if (commandTypeAndCommand.containsKey("TRANSPARENCY")) {
            parsedPlot[2] = "TRANSPARENCY " + commandTypeAndCommand.get("TRANSPARENCY");
        }

        //TODO further reparse them if needed
        // System.out.println("indi"+ individualCommmand);
        // System.out.println(commndTypeAndCommand);
        if (individualCommmand.matches(plotCommandPattern)) {
//            System.out.println("parsed command");
            //parse SQL command
            String plotType = individualCommmand.replaceAll(plotCommandPattern, "$1").trim();
            /**
             * we check if the plot type matches one of the possible plot types
             * I know
             */
            boolean matched = false;
            for (Map.Entry<String, String> entry : plotCommandsTranslator.entrySet()) {
                String key = entry.getKey();
                if (plotType.matches(key)) {
                    plotType = entry.getValue();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                System.err.println("UNKNOWN PLOT TYPE: '" + plotType + "'");
                System.err.println(getSupportedPlotTypes());
                plotType = "UNKNOWN";
            }
//            System.out.println("plot type='" + plotType + "'");
            String SQLcommand = individualCommmand.replaceAll(plotCommandPattern, "$2").trim();
//            System.out.println("SQLcommand='" + SQLcommand + "'");

            parsedPlot[0] = plotType;
            parsedPlot[1] = SQLcommand;
//                SQLCommands.add(SQLcommand);
//                plots.add(plotType);
            //try plottings 
        } else {
            /**
             * execute the unmatched pattern as an SQL command
             */
            parsedPlot[1] = individualCommmand;
        }
        return parsedPlot;
        //now we perform the extraction
    }

'''

# check a set of TA commands to see how I can parse them
# in a way the SQL command is anything between select and the end of the line or a plus --> it is anything before the first encountered extra keyword --> should not be too hard to do!!!
# especially now that the command is cleaned
# if unknown keyword --> then return an error


extra_plot_commands_for_images = ['LUT', 'EROSION', 'DILATATION', 'DILATION', 'OPACITY', 'TRANSPARENCY']


def plot_SQL(SQL_command):
    # just get the word after
    # or split and get all words --> very easy TODO in fact
    # if SQL_command.lower().starts_with()

    if SQL_command is None:
        logger.error('Empty SQL command --> nothing to do')
        return

    if not isinstance(SQL_command, str):
        logger.error('SQL command must be a string, instead a ' + str(type(SQL_command)) + ' was passed...')
        return

    if SQL_command.strip().startswith('#') or SQL_command.strip().startswith('//'):
        # assume this is a comment and then nothing TODO
        return

    cleaned_SQL_command = _clean_string(SQL_command)

    print('cleaned_SQL_command "' + cleaned_SQL_command + '"')

    plot_type = None
    # First we get the type of plot then we should get the SQL command
    if cleaned_SQL_command.upper().startswith('PLOT AS CELLS'):
        # plot as cells --> need map data to the local ID database of cells
        # print('plot as cells')
        plot_type = 'CELLS'
        # cleaned_SQL_command.replace()
        # get index and cut after

        # cleaned_SQL_command = cleaned_SQL_command.upper().index('PLOT AS CELLS')
        cleaned_SQL_command = _strip_from_command(cleaned_SQL_command, 'PLOT AS CELLS')
    elif cleaned_SQL_command.upper().startswith('PLOT AS BONDS'):
        # print('plot as bonds')
        plot_type = 'BONDS'
        cleaned_SQL_command = _strip_from_command(cleaned_SQL_command, 'PLOT AS BONDS')
    elif cleaned_SQL_command.upper().startswith('PLOT AS VERTICES'):
        # print('plot as vertices')
        plot_type = 'VERTICES'
        cleaned_SQL_command = _strip_from_command(cleaned_SQL_command, 'PLOT AS VERTICES')
    else:
        # unknown/unsupported plot --> raise an error
        logger.error('unsupported plot command: ' + str(cleaned_SQL_command))
        return

    print('plot_type "' + str(plot_type) + '"')
    # print('cleaned_SQL_command',cleaned_SQL_command)

    # now we need to get the pure SQL command --> see how I do that in TA
    # further parse it --> get the command between different stuff
    # SQL_command is actually anything before the first magic keyword encountered if any --> should be easy TODO

    # get extras

    first_idx = None
    for extra in extra_plot_commands_for_images:
        try:
            idx = cleaned_SQL_command.upper().index(extra)
            # if idx !=-1:
            if first_idx is None:
                first_idx = idx
            else:
                first_idx = min(first_idx, idx)
        except:
            # extra command not found --> ignoring
            continue

    # print('first_idx',first_idx)

    extra_commands_to_parse = None
    if first_idx is not None:
        # cut after and before --> before is the SQL command otherwise ignore
        # very good in fact

        SQL_command = cleaned_SQL_command[0:first_idx].strip()
        if SQL_command.endswith('+'):
            SQL_command = SQL_command[:-1].strip()
        extra_commands_to_parse = cleaned_SQL_command[first_idx:].strip()


    else:
        SQL_command = cleaned_SQL_command

    print('SQL_command "' + str(SQL_command) + '"')
    print('extra_commands_to_parse "' + str(
        extra_commands_to_parse) + '"')  # extras are indicated by a plus --> this way it is very simple to get them --> may remove plus if ends with it

    if extra_commands_to_parse:
        # need further parse these commands to get what I really need --> first find the identifier then find the values
        # print('TODO --> need parse extras')

        # we clean the extras and then get their values
        # extra_commands_to_parse = extra_commands_to_parse.replace('+','')# remove the pluses from extras

        if '+' in extra_commands_to_parse:
            individual_extras = extra_commands_to_parse.split('+')
        else:
            individual_extras = [extra_commands_to_parse]

        for individual_extra in individual_extras:
            print('individual_extra "' + individual_extra.strip() + '"')

            # ça marche super --> now I need to further parse the extras but I'm almost there in fact --> that is very easy TODO

            # split it to get the first keyword then see later how to handle the rest
            extra_type = individual_extra.strip().split()

            # print("extra_type",extra_type)

            if extra_type[0].strip().upper() not in extra_plot_commands_for_images:
                logger.error('unknown image extra ' + str(extra_type[0]) + ' --> ignoring')
            else:
                # known extra --> need parse it to make it meaningful --> TODO
                # need Call a different function depending on the data --> TODO
                if extra_type[0].strip().upper() == extra_plot_commands_for_images[0]:
                    # ['LUT', 'EROSION', 'DILATATION','DILATION', 'OPACITY', 'TRANSPARENCY']
                    print('LUT to parse')
                    print('param', parse_lut(extra_type))
                elif extra_type[0].strip().upper() == extra_plot_commands_for_images[1]:
                    print('EROSION to parse')
                    print('param', parse_erosion_dilation(extra_type))
                elif extra_type[0].strip().upper() == extra_plot_commands_for_images[2] or extra_type[
                    0].strip().upper() == extra_plot_commands_for_images[3]:
                    print('DILATION to parse')
                    print('param', parse_erosion_dilation(extra_type))
                elif extra_type[0].strip().upper() == extra_plot_commands_for_images[4]:
                    print('OPACITY to parse')
                    print('param', parse_opacity(extra_type))
                elif extra_type[0].strip().upper() == extra_plot_commands_for_images[5]:
                    print('TRANSPARENCY to parse')
                    print('param', 1. - parse_opacity(extra_type))

                # else:
                #     print('weird extra', extra_type[0])

        # --> very good idea

        # extra_commands_to_parse = _clean_string(extra_commands_to_parse)
        # print('cleaned_extras',extra_commands_to_parse)
        # now need parse it --> can do it in a consecutive way
        # if unknwon keyword --> say it and move on to next step
        # should not be too hard

        # then need even further parse it to get the real value of the parameter
        # extra_settings = extra_commands_to_parse.split()
        # if keyword is known and until next keyword --> get the values --> pb how can I warn about unknown keywords!!!

        # for extra_setting in extra_settings:
        #     print(extra_setting)


'''
    public static HashMap<Integer, Integer> parseLUT(String testLUT2) {
        HashMap<Integer, Integer> LUTconversionTable = new HashMap<Integer, Integer>();
        String LutHumanSpecified = "(\\d{1,}):[']{0,1}([#A-Fa-f0-9]{1,}|[0-9\\.]{1,})[']{0,1}";

        Pattern p = Pattern.compile(LutHumanSpecified);
        Matcher m = p.matcher(testLUT2);

        while (m.find()) {
            try {
                String outputValue = m.group(2);
                if (!outputValue.contains("#")) {
                    LUTconversionTable.put(CommonClasses.String2Int(m.group(1)), CommonClasses.String2Int(outputValue));
                } else {
                    LUTconversionTable.put(CommonClasses.String2Int(m.group(1)), CommonClasses.getColorFromHtmlColor(outputValue));
                }
            } catch (Exception e) {
            }
        }
        return LUTconversionTable;
    }
'''


def parse_lut(extra_type):
    # ['LUT', 'DNA', 'MIN=GLOBAL_MIN', 'MAX=GLOBAL_MAX'] # do I need to reimplement that ??? not sure it is very ueseful as the SQL command can easily be edited to avoid that
    # print(extra_type)
    # print(len(extra_type), not ':' in extra_type[1])
    if len(extra_type) == 2 and (not ':' in extra_type[1]):
        try:
            return extra_type[1]  # returns the name of the LUT
        except:
            pass
    else:
        if ':' in extra_type[1]:
            # get lutvalues and return them as a dict
            lut_mapping = {}
            for mapping in extra_type[1:]:
                split_mappings = mapping.split(':')
                split_mappings = [split_mapping.strip() for split_mapping in split_mappings]
                key = int(split_mappings[0])
                value = split_mappings[1]
                if '#' in value:
                    # convert it back to integer
                    value = int(value.replace('#', ''),
                                16)  # convert hexadecimal (base 16) back to int --> must remove #
                else:
                    value = int(value)
                lut_mapping[key] = value
        else:
            # need handle a stuff with max min or something hybrid in fact
            # TODO implement parsing of max and min but ignore for now ['LUT', 'DNA', 'MIN=GLOBAL_MIN', 'MAX=GLOBAL_MAX']
            logger.warning('LUT bounds not implemented yet')
            return parse_lut(extra_type=extra_type[0:2])
        return lut_mapping
    # TODO
    # something went wrong --> ignoring
    return None


def parse_opacity(extra_type):
    # TODO
    if len(extra_type) == 2:
        try:
            ispercent = '%' in extra_type[1]
            if ispercent:
                extra_type[1] = extra_type[1].replace('%', '')
            value = float(extra_type[1])
            if ispercent:
                value /= 100.
            return value  # returns the name of the LUT
        except:
            pass
    return None


def parse_erosion_dilation(extra_type):
    # TODO
    if len(extra_type) == 2:
        try:
            return int(extra_type[1])  # returns the name of the LUT
        except:
            pass
    return None


def _clean_string(str):
    parsed_SQL_commands = str.split()
    parsed_SQL_commands = [parsed_SQL_command.strip() for parsed_SQL_command in parsed_SQL_commands]
    # recreate it as a single sentence and do the tests
    cleaned_SQL_command = ' '.join(parsed_SQL_commands)
    return cleaned_SQL_command


def _strip_from_command(command, things_to_strip):
    # begin_strip = command.upper().index(things_to_strip)
    # print('begin_strip', begin_strip)
    # end_strip =len(things_to_strip)
    # print('end_strip',end_strip)

    # return command/
    # command = command[begin_strip:end_strip]
    # print('stripped "' +command+'"')

    # print(text.removesuffix('ly'))
    # print(text.removesuffix('World'))

    import re
    case_insensitive_replace = re.compile(re.escape(things_to_strip), re.IGNORECASE)
    stripped = case_insensitive_replace.sub('', command).strip()

    # print(stripped)
    return stripped

# **kwargs are the extra params --> TODO also ask for opacity
# indeed bg image must be provided there maybe also with a channel --> TODO--> see how I can handle that?

# make it store the max and the minb


# NB THERE IS A BUG THAT FORCES THE PLOT TWICE --> PROBABLY SOME ADJUSTMENT STUFF --> NEED CHANGE THIS
def plot_as_any(parent_image, SQL_command, plot_type='cells', return_mask=False, invert_mask=True, db=None, current_frame=None, **kwargs):
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
            mask=Img.invert(mask)
        # also we apply the mask to the output
        output[mask!=0]=0
        return mask, output
    return output


def plot_as_image_old(plot_type='cells'):
    if plot_type == 'cells':
        # by default plot as cells
        logger.debug('plot as cells')
        image_plot = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/cells.tif').astype(
            np.uint64)

    elif plot_type == 'bonds':
        logger.debug('plot as bonds')
        image_plot = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/bonds.tif').astype(
            np.uint64)
    elif plot_type == 'vertices':
        logger.debug('plot as vertices')
        image_plot = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/vertices.tif').astype(
            np.uint64)
    else:
        logger.error('Plot type unknonw: \'' + str(plot_type) + '\'')
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
        file = '/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012.png'
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
        file = '/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012.png'
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

        # file = '/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series014.png' # to see if the nematic bug is linked to too many channels --> maybe the case
        # SQL_command = "SELECT local_id,Q1_polarity_ch0, Q2_polarity_ch0, '#FF0000' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells_2D WHERE is_border == FALSE"  # a bit more complex just to see which method to choose --> very easy

        file = '/E/Sample_images/sample_images_PA/trash_test_mem/mini_different_nb_of_channels/focused_Series012.png'
        # SQL_command = "SELECT local_id_cells,P1_polarity_ch2, P2_polarity_ch2, '#FF0000' AS COLOR, '2' AS STROKE_SIZE, 0.06 AS SCALING_FACTOR FROM cells"   #for TA db
        SQL_command = "SELECT local_id,Q1_polarity_ch1, Q2_polarity_ch1, '#FF0000' AS COLOR, 2 AS STROKE_SIZE, 0.1 AS SCALING_FACTOR FROM cells_2D WHERE is_border == FALSE"  # a bit more complex just to see which method to choose --> very easy


        # file = '/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png'
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

        plot_SQL(
            "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells FROM cells ORDER BY area_cells DESC LIMIT 10")  # --> see how I can do that
        plot_SQL(
            "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells FROM cells ORDER BY area_cells DESC LIMIT 10 + LUT DNA")  # --> see how I can do that
        plot_SQL(
            "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells, '#FF0000' AS COLOR FROM cells ORDER BY area_cells DESC LIMIT 10 + DILATATION 1 + OPACITY 35%")  # in fact each extra is starting with a + --> quite easy to get
        # --> ignore pluses in extras
        plot_SQL(
            "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells , area_cells FROM cells ORDER BY area_cells DESC LIMIT 10 + LUT DNA MIN=GLOBAL_MIN MAX=GLOBAL_MAX")  # --> maybe not that hard because the values if complex are connected to MIn and MAX --> if all are like that then I can parse all extras as keys and values and all is gonna be a piece of cake and easy to detect and report errors --> TODO

        plot_SQL(
            "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells + LUT DNA + OPACITY 35% + DILATATION 1")  # --> maybe not that hard because the values if complex are connected to MIn and MAX --> if all are like that then I can parse all extras as keys and values and all is gonna be a piece of cake and easy to detect and report errors --> TODO
        plot_SQL(
            "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells + LUT DNA + OPACITY 35% + DILATATION 1+BOUGA BOUU")  # --> maybe not that hard because the values if complex are connected to MIn and MAX --> if all are like that then I can parse all extras as keys and values and all is gonna be a piece of cake and easy to detect and report errors --> TODO
        plot_SQL(
            "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells + LUT DNA + OPACITY 35% + DILATATION 1+BOUGA")  # --> maybe not that hard because the values if complex are connected to MIn and MAX --> if all are like that then I can parse all extras as keys and values and all is gonna be a piece of cake and easy to detect and report errors --> TODO
        plot_SQL(
            "PLOT AS CELLS SELECT first_pixel_x_cells,first_pixel_y_cells, S0_stretch_cells FROM cells + LUT DNA + TRANSPARENCY 35% + DILATATION 1+BOUGA")  # --> maybe not that hard because the values if complex are connected to MIn and MAX --> if all are like that then I can parse all extras as keys and values and all is gonna be a piece of cake and easy to detect and report errors --> TODO
        plot_SQL(
            "PLOT AS CELLS SELECT first_pixel_x_cells, first_pixel_y_cells, nb_of_vertices_cut_off FROM cells + LUT 3:#00FFFF 4:#00FF00 5:#FFFF00 6:#AAAAAA 7:#0000FF 8:#FF0000 9:#FF0000")  # --> maybe not that hard because the values if complex are connected to MIn and MAX --> if all are like that then I can parse all extras as keys and values and all is gonna be a piece of cake and easy to detect and report errors --> TODO

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
