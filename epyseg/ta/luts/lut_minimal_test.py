# TODO could easily do a lut browser and editor --> would be really easy and may be useful too --> some day do it but not now !!!
# this is a minimal but already quite good version of the TA LUT class
import random
import traceback
from matplotlib.colors import ListedColormap
import math
from epyseg.img import int24_to_RGB
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from epyseg.img import fig_to_numpy


def cmap_to_numpy(lut):
    """
    Converts a colormap to a NumPy array.

    Args:
        lut (Union[str, numpy.ndarray]): The colormap as a string name or a NumPy array.

    Returns:
        numpy.ndarray: The NumPy array representing the colormap.

    Examples:
        >>> colormap = 'hot'
        >>> result = cmap_to_numpy(colormap)
        >>> print(result[:1])
        [[10  0  0]]

    """
    if isinstance(lut, str):
        cm = plt.get_cmap(lut)
    else:
        cm = lut
    out = cm(range(256)) * 255
    out = out[..., 0:3].astype(np.uint8)
    return out


def numpy_to_cmap(lut):
    """
    Converts a NumPy array to a colormap.

    Args:
        lut (numpy.ndarray): The NumPy array representing the colormap.

    Returns:
        matplotlib.colors.ListedColormap: The colormap created from the NumPy array.

    # Examples:
    #     >>> colormap_array = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    #     >>> result = numpy_to_cmap(colormap_array)
    #     >>> print(result)
    #     <matplotlib.colors.ListedColormap object at 0x...>

    """
    lut_cp = np.copy(lut)
    if lut_cp.max() > 1:
        lut_cp = lut_cp / 255.

    if lut_cp.shape[-1] == 3:
        tmp = np.ones((256, 4), dtype=np.float64)
        tmp[..., 0:3] = lut_cp[..., :]
        lut_cp = tmp
    newcmp = ListedColormap(lut_cp)
    return newcmp


def create_ramp(min_val, max_val, lut, orientation='vertical'):
    """
    Creates a color ramp image using a colormap.

    Args:
        min_val (float): The minimum value of the color ramp.
        max_val (float): The maximum value of the color ramp.
        lut (Union[str, numpy.ndarray]): The colormap as a string name or a NumPy array.
        orientation (str, optional): The orientation of the color ramp. Defaults to 'vertical'.

    Returns:
        numpy.ndarray: The color ramp image.

    # Examples:
    #     >>> min_value = 0
    #     >>> max_value = 255
    #     >>> colormap = 'hot'
    #     >>> result = create_ramp(min_value, max_value, colormap)
    #     >>> print(result)


    """
    img = np.zeros((2, 1))
    img[0] = min_val
    img[1] = max_val
    fig = Figure(tight_layout=True)
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()

    cmap = lut
    if isinstance(lut, np.ndarray):
        cmap = numpy_to_cmap(lut)

    im = ax.imshow(img, cmap=cmap)
    fig.colorbar(im, ax=ax, orientation=orientation)
    ax.set_axis_off()
    ax.remove()
    canvas.draw()
    img = fig_to_numpy(fig)
    return img


def list_available_luts(return_matplotlib_lib_luts_in_separate_database=False):
    """
    Returns a dictionary of available colormaps.

    Args:
        return_matplotlib_lib_luts_in_separate_database (bool, optional): Whether to include matplotlib library colormaps separately. Defaults to False.

    Returns:
        dict: A dictionary containing the available colormaps.

    Examples:
        >>> result = list_available_luts()
        >>> print(result)
        {'DEFAULT': '22,13,-23', 'AFM_HOT': '34,35,36', 'GREEN': '0,3,0', 'RED': '3,0,0', 'BLUE': '0,0,3', 'GRAY': '3,3,3', 'GREY': '3,3,3', 'PACKING': 'PACKING', 'RED_HOT': '21,22,23', 'GREEN_HOT': '35,34,23', 'BLUE_HOT': '23,22,21', 'OCEAN': '23,28,3', 'GREEN_RED_VIOLET': '3,11,6', 'COLOR_PRINTABLE_ON_GRAY': '30,31,32', 'HI_LO': '3,3,3hl', 'DNA': '22,13,-23', 'RAINBOW_1': '22,13,-31', 'RAINBOW_2': '35,13,-35', 'RAINBOW_3': '33,13,10', 'RAINBOW_LIGHT': '33,13,-31', 'RAINBOW_ARTIFICIAL': '10,13,33', 'BLUE_WHITE_RED': '34,-13,-34', 'BROWN': '7,4,27', 'GREEN_FIRE_BLUE': '31,34,11', 'GREEN_FIRE_BLUE2': '31,34,29', 'ORANGE': '3,5,0', 'ORANGE_HOT': '21,-10,23', 'YELLOW_HOT': '34,-10,23', 'YELLOW': '3,3,0', 'YELLOW_PALE': '34,34,4', 'PURPLE_HOT': '34,36,34', 'PURPLE': '3,0,3', 'MAGENTA_HOT': '34,36,34', 'MAGENTA': '3,0,3', 'CYAN': '0,3,3', 'CYAN_HOT': '36,34,34', 'PM3D': '7,5,15', 'HSV': '3,2,2', 'HSV2': '7,5,15', 'YING_YANG': '13,13,13', 'YING_YANG2': '14,14,14', 'CYCLIC': '19,19,19', 'METEO1': '34,-35,-36', 'METEO2': '22,13,-31', 'DIABOLO_MENTHE': '-7,2,-7', 'DIABOLO_GRENADINE': '2,-7,-7', 'BLACK_PURPLE_PINK_WHITE': '30,31,32', 'RED_BLACK_BLUE': '-36,0,36', 'MATPLOTLIB_magma': 'magma', 'MATPLOTLIB_inferno': 'inferno', 'MATPLOTLIB_plasma': 'plasma', 'MATPLOTLIB_viridis': 'viridis', 'MATPLOTLIB_cividis': 'cividis', 'MATPLOTLIB_twilight': 'twilight', 'MATPLOTLIB_twilight_shifted': 'twilight_shifted', 'MATPLOTLIB_turbo': 'turbo', 'MATPLOTLIB_Blues': 'Blues', 'MATPLOTLIB_BrBG': 'BrBG', 'MATPLOTLIB_BuGn': 'BuGn', 'MATPLOTLIB_BuPu': 'BuPu', 'MATPLOTLIB_CMRmap': 'CMRmap', 'MATPLOTLIB_GnBu': 'GnBu', 'MATPLOTLIB_Greens': 'Greens', 'MATPLOTLIB_Greys': 'Greys', 'MATPLOTLIB_OrRd': 'OrRd', 'MATPLOTLIB_Oranges': 'Oranges', 'MATPLOTLIB_PRGn': 'PRGn', 'MATPLOTLIB_PiYG': 'PiYG', 'MATPLOTLIB_PuBu': 'PuBu', 'MATPLOTLIB_PuBuGn': 'PuBuGn', 'MATPLOTLIB_PuOr': 'PuOr', 'MATPLOTLIB_PuRd': 'PuRd', 'MATPLOTLIB_Purples': 'Purples', 'MATPLOTLIB_RdBu': 'RdBu', 'MATPLOTLIB_RdGy': 'RdGy', 'MATPLOTLIB_RdPu': 'RdPu', 'MATPLOTLIB_RdYlBu': 'RdYlBu', 'MATPLOTLIB_RdYlGn': 'RdYlGn', 'MATPLOTLIB_Reds': 'Reds', 'MATPLOTLIB_Spectral': 'Spectral', 'MATPLOTLIB_Wistia': 'Wistia', 'MATPLOTLIB_YlGn': 'YlGn', 'MATPLOTLIB_YlGnBu': 'YlGnBu', 'MATPLOTLIB_YlOrBr': 'YlOrBr', 'MATPLOTLIB_YlOrRd': 'YlOrRd', 'MATPLOTLIB_afmhot': 'afmhot', 'MATPLOTLIB_autumn': 'autumn', 'MATPLOTLIB_binary': 'binary', 'MATPLOTLIB_bone': 'bone', 'MATPLOTLIB_brg': 'brg', 'MATPLOTLIB_bwr': 'bwr', 'MATPLOTLIB_cool': 'cool', 'MATPLOTLIB_coolwarm': 'coolwarm', 'MATPLOTLIB_copper': 'copper', 'MATPLOTLIB_cubehelix': 'cubehelix', 'MATPLOTLIB_flag': 'flag', 'MATPLOTLIB_gist_earth': 'gist_earth', 'MATPLOTLIB_gist_gray': 'gist_gray', 'MATPLOTLIB_gist_heat': 'gist_heat', 'MATPLOTLIB_gist_ncar': 'gist_ncar', 'MATPLOTLIB_gist_rainbow': 'gist_rainbow', 'MATPLOTLIB_gist_stern': 'gist_stern', 'MATPLOTLIB_gist_yarg': 'gist_yarg', 'MATPLOTLIB_gnuplot': 'gnuplot', 'MATPLOTLIB_gnuplot2': 'gnuplot2', 'MATPLOTLIB_gray': 'gray', 'MATPLOTLIB_hot': 'hot', 'MATPLOTLIB_hsv': 'hsv', 'MATPLOTLIB_jet': 'jet', 'MATPLOTLIB_nipy_spectral': 'nipy_spectral', 'MATPLOTLIB_ocean': 'ocean', 'MATPLOTLIB_pink': 'pink', 'MATPLOTLIB_prism': 'prism', 'MATPLOTLIB_rainbow': 'rainbow', 'MATPLOTLIB_seismic': 'seismic', 'MATPLOTLIB_spring': 'spring', 'MATPLOTLIB_summer': 'summer', 'MATPLOTLIB_terrain': 'terrain', 'MATPLOTLIB_winter': 'winter', 'MATPLOTLIB_Accent': 'Accent', 'MATPLOTLIB_Dark2': 'Dark2', 'MATPLOTLIB_Paired': 'Paired', 'MATPLOTLIB_Pastel1': 'Pastel1', 'MATPLOTLIB_Pastel2': 'Pastel2', 'MATPLOTLIB_Set1': 'Set1', 'MATPLOTLIB_Set2': 'Set2', 'MATPLOTLIB_Set3': 'Set3', 'MATPLOTLIB_tab10': 'tab10', 'MATPLOTLIB_tab20': 'tab20', 'MATPLOTLIB_tab20b': 'tab20b', 'MATPLOTLIB_tab20c': 'tab20c', 'MATPLOTLIB_magma_r': 'magma_r', 'MATPLOTLIB_inferno_r': 'inferno_r', 'MATPLOTLIB_plasma_r': 'plasma_r', 'MATPLOTLIB_viridis_r': 'viridis_r', 'MATPLOTLIB_cividis_r': 'cividis_r', 'MATPLOTLIB_twilight_r': 'twilight_r', 'MATPLOTLIB_twilight_shifted_r': 'twilight_shifted_r', 'MATPLOTLIB_turbo_r': 'turbo_r', 'MATPLOTLIB_Blues_r': 'Blues_r', 'MATPLOTLIB_BrBG_r': 'BrBG_r', 'MATPLOTLIB_BuGn_r': 'BuGn_r', 'MATPLOTLIB_BuPu_r': 'BuPu_r', 'MATPLOTLIB_CMRmap_r': 'CMRmap_r', 'MATPLOTLIB_GnBu_r': 'GnBu_r', 'MATPLOTLIB_Greens_r': 'Greens_r', 'MATPLOTLIB_Greys_r': 'Greys_r', 'MATPLOTLIB_OrRd_r': 'OrRd_r', 'MATPLOTLIB_Oranges_r': 'Oranges_r', 'MATPLOTLIB_PRGn_r': 'PRGn_r', 'MATPLOTLIB_PiYG_r': 'PiYG_r', 'MATPLOTLIB_PuBu_r': 'PuBu_r', 'MATPLOTLIB_PuBuGn_r': 'PuBuGn_r', 'MATPLOTLIB_PuOr_r': 'PuOr_r', 'MATPLOTLIB_PuRd_r': 'PuRd_r', 'MATPLOTLIB_Purples_r': 'Purples_r', 'MATPLOTLIB_RdBu_r': 'RdBu_r', 'MATPLOTLIB_RdGy_r': 'RdGy_r', 'MATPLOTLIB_RdPu_r': 'RdPu_r', 'MATPLOTLIB_RdYlBu_r': 'RdYlBu_r', 'MATPLOTLIB_RdYlGn_r': 'RdYlGn_r', 'MATPLOTLIB_Reds_r': 'Reds_r', 'MATPLOTLIB_Spectral_r': 'Spectral_r', 'MATPLOTLIB_Wistia_r': 'Wistia_r', 'MATPLOTLIB_YlGn_r': 'YlGn_r', 'MATPLOTLIB_YlGnBu_r': 'YlGnBu_r', 'MATPLOTLIB_YlOrBr_r': 'YlOrBr_r', 'MATPLOTLIB_YlOrRd_r': 'YlOrRd_r', 'MATPLOTLIB_afmhot_r': 'afmhot_r', 'MATPLOTLIB_autumn_r': 'autumn_r', 'MATPLOTLIB_binary_r': 'binary_r', 'MATPLOTLIB_bone_r': 'bone_r', 'MATPLOTLIB_brg_r': 'brg_r', 'MATPLOTLIB_bwr_r': 'bwr_r', 'MATPLOTLIB_cool_r': 'cool_r', 'MATPLOTLIB_coolwarm_r': 'coolwarm_r', 'MATPLOTLIB_copper_r': 'copper_r', 'MATPLOTLIB_cubehelix_r': 'cubehelix_r', 'MATPLOTLIB_flag_r': 'flag_r', 'MATPLOTLIB_gist_earth_r': 'gist_earth_r', 'MATPLOTLIB_gist_gray_r': 'gist_gray_r', 'MATPLOTLIB_gist_heat_r': 'gist_heat_r', 'MATPLOTLIB_gist_ncar_r': 'gist_ncar_r', 'MATPLOTLIB_gist_rainbow_r': 'gist_rainbow_r', 'MATPLOTLIB_gist_stern_r': 'gist_stern_r', 'MATPLOTLIB_gist_yarg_r': 'gist_yarg_r', 'MATPLOTLIB_gnuplot_r': 'gnuplot_r', 'MATPLOTLIB_gnuplot2_r': 'gnuplot2_r', 'MATPLOTLIB_gray_r': 'gray_r', 'MATPLOTLIB_hot_r': 'hot_r', 'MATPLOTLIB_hsv_r': 'hsv_r', 'MATPLOTLIB_jet_r': 'jet_r', 'MATPLOTLIB_nipy_spectral_r': 'nipy_spectral_r', 'MATPLOTLIB_ocean_r': 'ocean_r', 'MATPLOTLIB_pink_r': 'pink_r', 'MATPLOTLIB_prism_r': 'prism_r', 'MATPLOTLIB_rainbow_r': 'rainbow_r', 'MATPLOTLIB_seismic_r': 'seismic_r', 'MATPLOTLIB_spring_r': 'spring_r', 'MATPLOTLIB_summer_r': 'summer_r', 'MATPLOTLIB_terrain_r': 'terrain_r', 'MATPLOTLIB_winter_r': 'winter_r', 'MATPLOTLIB_Accent_r': 'Accent_r', 'MATPLOTLIB_Dark2_r': 'Dark2_r', 'MATPLOTLIB_Paired_r': 'Paired_r', 'MATPLOTLIB_Pastel1_r': 'Pastel1_r', 'MATPLOTLIB_Pastel2_r': 'Pastel2_r', 'MATPLOTLIB_Set1_r': 'Set1_r', 'MATPLOTLIB_Set2_r': 'Set2_r', 'MATPLOTLIB_Set3_r': 'Set3_r', 'MATPLOTLIB_tab10_r': 'tab10_r', 'MATPLOTLIB_tab20_r': 'tab20_r', 'MATPLOTLIB_tab20b_r': 'tab20b_r', 'MATPLOTLIB_tab20c_r': 'tab20c_r'}

    """
    hash_pal = {
        "DEFAULT": "22,13,-23",
        "AFM_HOT": "34,35,36",
        "GREEN": "0,3,0",
        "RED": "3,0,0",
        "BLUE": "0,0,3",
        "GRAY": "3,3,3",
        "GREY": "3,3,3",
        "PACKING": "PACKING",
        "RED_HOT": "21,22,23",
        "GREEN_HOT": "35,34,23",
        "BLUE_HOT": "23,22,21",
        "OCEAN": "23,28,3",
        "GREEN_RED_VIOLET": "3,11,6",
        "COLOR_PRINTABLE_ON_GRAY": "30,31,32",
        "HI_LO": "3,3,3hl",
        "DNA": "22,13,-23",
        "RAINBOW_1": "22,13,-31",
        "RAINBOW_2": "35,13,-35",
        "RAINBOW_3": "33,13,10",
        "RAINBOW_LIGHT": "33,13,-31",
        "RAINBOW_ARTIFICIAL": "10,13,33",
        "BLUE_WHITE_RED": "34,-13,-34",
        "BROWN": "7,4,27",
        "GREEN_FIRE_BLUE": "31,34,11",
        "GREEN_FIRE_BLUE2": "31,34,29",
        "ORANGE": "3,5,0",
        "ORANGE_HOT": "21,-10,23",
        "YELLOW_HOT": "34,-10,23",
        "YELLOW": "3,3,0",
        "YELLOW_PALE": "34,34,4",
        "PURPLE_HOT": "34,36,34",
        "PURPLE": "3,0,3",
        "MAGENTA_HOT": "34,36,34",
        "MAGENTA": "3,0,3",
        "CYAN": "0,3,3",
        "CYAN_HOT": "36,34,34",
        "PM3D": "7,5,15",
        "HSV": "3,2,2",
        "HSV2": "7,5,15",
        "YING_YANG": "13,13,13",
        "YING_YANG2": "14,14,14",
        "CYCLIC": "19,19,19",
        "METEO1": "34,-35,-36",
        "METEO2": "22,13,-31",
        "DIABOLO_MENTHE": "-7,2,-7",
        "DIABOLO_GRENADINE": "2,-7,-7",
        "BLACK_PURPLE_PINK_WHITE": "30,31,32",
        "RED_BLACK_BLUE": "-36,0,36"
    }

    matplot_lib_luts = []
    try:
        # Get the available colormaps from matplotlib
        matplot_lib_luts = get_matplotlib_available_luts()

        # Add matplotlib colormaps to the dictionary
        for matplotlib in matplot_lib_luts:
            hash_pal['MATPLOTLIB_' + str(matplotlib)] = str(matplotlib)
    except:
        # no big deal if fails because I have enough luts already
        # traceback.print_exc()
        pass

    # Check if separate matplotlib colormaps are requested
    if return_matplotlib_lib_luts_in_separate_database:
        return matplot_lib_luts, hash_pal

    # Return the dictionary of colormaps
    return hash_pal


class PaletteCreator:
    PALETTE_IRFAN = 0
    PALETTE_IRFANVIEW = 0
    PALETTE_IMAGEJ_EXPORTED = 1
    PALETTE_IMAGEJ_RAW = 2
    PALETTE_UNKNOWN = 3
    currentPalette = "DEFAULT"

    def __init__(self):
        self.__list_palettes()

    def __list_palettes(self):
        """
        Retrieves the list of available palettes, including matplotlib library colormaps.
        """
        self.matplot_lib_luts, self.list = list_available_luts(return_matplotlib_lib_luts_in_separate_database=True)

    def get_packing_lut(self):
        """
        Returns the packing lookup table.

        Returns:
            numpy.ndarray: The packing lookup table.

        # Examples:
        #     >>> palette_creator = PaletteCreator()
        #     >>> result = palette_creator.get_packing_lut()
        #     >>> print(result)
        #     [[ 64  64  64]
        #      [100 190  40]
        #      [190 190  30]
        #      [128 128 128]
        #      [  0   0 175]
        #      [175   0   0]
        #      [100   0 170]
        #      [250  36   0]
        #      [150  75   0]
        #      ...
        #     ]
        """
        Sided3_or_less = (64, 64, 64)
        Sided4 = (100, 190, 40)
        Sided5 = (190, 190, 30)
        Sided6 = (128, 128, 128)
        Sided7 = (0, 0, 175)
        Sided8 = (175, 0, 0)
        Sided9 = (100, 0, 170)
        Sided10 = (250, 36, 0)
        Sided11_or_more = (150, 75, 0)

        packing_lut = np.zeros((256, 3), dtype=np.uint8)

        packing_lut[0] = Sided3_or_less
        packing_lut[1] = Sided3_or_less
        packing_lut[2] = Sided3_or_less
        packing_lut[3] = Sided3_or_less
        packing_lut[4] = Sided4
        packing_lut[5] = Sided5
        packing_lut[6] = Sided6
        packing_lut[7] = Sided7
        packing_lut[8] = Sided8
        packing_lut[9] = Sided9
        packing_lut[10] = Sided10
        packing_lut[11] = Sided11_or_more
        for iii in range(10, 256):
            packing_lut[iii] = Sided11_or_more

        return packing_lut

    def count(self, strg, char2count):
        """
        Counts the number of occurrences of a character in a string.

        Args:
            strg (str): The input string.
            char2count (str): The character to count.

        Returns:
            int: The count of the character in the string.

        Examples:
            >>> palette_creator = PaletteCreator()
            >>> result = palette_creator.count("Hello world!", "o")
            >>> print(result)
            2
        """
        if char2count is None:
            return 0
        if strg is None or strg == "":
            return 0
        nb = 0
        for c in strg:
            if c == char2count:
                nb += 1
        return nb

    def convertAbs(self, txt):
        """
        Converts absolute value notation in a string.

        Args:
            txt (str): The input string.

        Returns:
            str: The string with absolute value notation converted.

        # Examples:
        #     >>> palette_creator = PaletteCreator()
        #     >>> result = palette_creator.convertAbs("|-5| + 10")
        #     >>> print(result)
        #     "abs(-5) + 10"
        """
        try:
            while "|" in txt:
                size = len(txt)
                pos = txt.index("|")
                begin_pos = pos + 1
                end_pos = pos

                for i in range(pos + 1, size + 1):
                    c = txt[i]
                    if c == '|':
                        end_pos = i
                        break
                    end_pos = i + 1
                power_equa = txt[begin_pos:end_pos]
                beginning_of_word = txt[0:begin_pos - 1]
                end_of_word = txt[end_pos + 1]
                txt = beginning_of_word + "abs(" + power_equa + ")" + end_of_word
        except:
            traceback.print_exc()
        return txt

    def convertPowers(self, txt):
        """
        Converts power notation in a string.

        Args:
            txt (str): The input string.

        Returns:
            str: The string with power notation converted.

        # Examples:
        #     >>> palette_creator = PaletteCreator()
        #     >>> result = palette_creator.convertPowers("2^3 + 4^2")
        #     >>> print(result)
        #     "2**3 + 4**2"
        """
        try:
            while "^" in txt:
                size = len(txt)
                pos = txt.index("^")
                begin_pos = pos
                end_pos = pos
                for i in range(pos, size):
                    c = txt[i]
                    if c == '+' or c == '-' or c == '/' or c == '*':
                        end_pos = i
                        break
                    end_pos = i + 1
                for i in range(pos, 0, -1):
                    c = txt[i]
                    if c == '+' or c == '-' or c == '/' or c == '*':
                        begin_pos = i + 1
                        break
                    begin_pos = i
                power_equa = txt[begin_pos:end_pos]
                beginning_of_word = txt[0:begin_pos]
                end_of_word = txt[end_pos:]
                txt = beginning_of_word + self.power_parser(power_equa) + end_of_word
        except:
            traceback.print_exc()
        return txt

    # do cut left or right of pattern
    # TODO --> do all of there
    def power_parser(self, equation):
        nb, power = equation.split("^", 1)  # CommonClasses.strCutLeftFirst(equation, "^")
        # power = CommonClasses.strCutRightFisrt(equation, "^")
        return "math.pow(" + nb + "," + power + ")"

    def javaifyEquation(self, equation):
        """
        Converts an equation into a Java-friendly format.

        Args:
            equation (str): The input equation.

        Returns:
            str: The equation in a Java-friendly format.

        Raises:
            Exception: If the equation has an incorrect number of absolute value bars or brackets.

        # Examples:
        #     >>> palette_creator = PaletteCreator()
        #     >>> result = palette_creator.javaifyEquation("|x+y|")
        #     >>> print(result)
        """
        # print(equation)
        equation = equation.replace(" ", "")

        if "|" in equation:
            nb_of_norm_bars = self.count(equation, "|")
            if nb_of_norm_bars % 2 != 0:
                raise Exception("equation is incorrect, nb of norm bars is not even")
                return

            equation = self.convertAbs(equation)

        if "^" in equation:
            equation = self.convertPowers(equation)

        equation = equation.replace("[", "(")
        equation = equation.replace("]", ")")
        equation = equation.replace("{", "(")
        equation = equation.replace("}", ")")

        equation = equation.replace("math.sin(", "sin(")
        equation = equation.replace("math.sin(", "sin(")
        equation = equation.replace("math.cos(", "cos(")
        equation = equation.replace("math.cos(", "cos(")
        equation = equation.replace("math.tan(", "tan(")
        equation = equation.replace("math.tan(", "tan(")
        equation = equation.replace("math.sqrt(", "sqrt(")
        equation = equation.replace("math.sqrt(", "sqrt(")
        equation = equation.replace("sin(", "math.sin(")
        equation = equation.replace("cos(", "math.cos(")
        equation = equation.replace("tan(", "math.tan(")
        equation = equation.replace("sqrt(", "math.sqrt(")
        nb_of_open_brackets = self.count(equation, "(")
        nb_of_closing_brackets = self.count(equation, ")")

        if nb_of_open_brackets != nb_of_closing_brackets:
            raise Exception("equation is incorrect, nb of opened and closed brackets don't match")

        return equation

    def computeEquation(self, equation, variable, val):
        """
        Computes the result of an equation using the numexpr library.

        Args:
            equation (str): The equation to compute.
            variable (str): The variable name in the equation.
            val: The value for the variable.

        Returns:
            float: The computed result of the equation.

        # Examples:
        #     >>> palette_creator = PaletteCreator()
        #     >>> result = palette_creator.computeEquation("2*a*(b/c)**2", "a", [1, 2, 2])
        #     >>> print(result)
        #     array([0.88888889, 0.44444444, 4.])
        """
        import numexpr as ne
        var = {'a': np.array([1, 2, 2]), 'b': np.array([2, 1, 3]), 'c': np.array([3])}
        print('TOO', ne.evaluate('2*a*(b/c)**2', local_dict=var))
        try:
            for i in range(10):
                equation = equation.replace(str(i) + variable, str(i) + "*" + variable)
            for i in range(10):
                equation = equation.replace(variable + str(i), variable + "*" + str(i))

            equation = equation.replace(variable, "(double)" + variable)
            print(equation)
        except:
            traceback.print_exc()
            return 0

    def create5(self, red, green, blue):
        """
        Creates a palette using the RGB components.

        Args:
            red (float): The red component value.
            green (float): The green component value.
            blue (float): The blue component value.

        Returns:
            numpy.ndarray: The created palette.

        # Examples:
        #     >>> palette_creator = PaletteCreator()
        #     >>> result = palette_creator.create5(0.5, -0.2, 0.8)
        #     >>> print(result)
        #     array([[127,   0,   0],
        #            [136,   0,   0],
        #            [146,   0,   0],
        #            ...
        #            [  0,  76, 153],
        #            [  0,  76, 153],
        #            [  0,  76, 153]], dtype=uint8)
        """
        l = 0
        palette = np.zeros((256, 3), dtype=np.uint8)
        R = np.zeros((256,), dtype=np.uint8)
        G = np.zeros((256,), dtype=np.uint8)
        B = np.zeros((256,), dtype=np.uint8)

        for i in range(256):
            R[i] = int(self.RGB_formula(i, red) * 255.0)
            G[i] = int(self.RGB_formula(i, green) * 255.0)
            B[i] = int(self.RGB_formula(i, blue) * 255.0)

        if red < 0:
            R = self.reverse_ramp(R)

        if green < 0:
            G = self.reverse_ramp(G)

        if blue < 0:
            B = self.reverse_ramp(B)

        for i in range(len(B)):
            palette[l, 0] = R[i]
            palette[l, 1] = G[i]
            palette[l, 2] = B[i]
            l += 1

        return palette

    #  # /*
    #     #  * computes a color for each gray value between 0 and 255
    #     #  */
    def RGB_formula(self, i, equation):
        """
        Computes a color for each gray value between 0 and 255 based on the given equation.

        Args:
            i (int): The gray value.
            equation (int): The equation index.

        Returns:
            float: The computed color value.

        """
        val = 0
        cur_eq = abs(equation)

        if cur_eq == 0 or cur_eq == 37 or cur_eq == 38 or cur_eq == 39:
            # case 0:
            # case 37:
            # case 38:
            # case 39:
            val = 0
            # break
        elif cur_eq == 1:
            val = 0.5
        elif cur_eq == 2:
            val = 1.0
        elif cur_eq == 3:
            val = i / 255.
        elif cur_eq == 4:
            val = (i / 255.) * (i / 255.)
        elif cur_eq == 5:
            val = (i / 255.) * (i / 255.) * (i / 255.)
        elif cur_eq == 6:
            val = (i / 255.) * (i / 255.) * (i / 255.) * (i / 255.)
        elif cur_eq == 7:
            val = math.sqrt(i / 255.)
        elif cur_eq == 8:
            val = math.sqrt(math.sqrt(i / 255.))
        elif cur_eq == 9:
            val = math.sin(i / 255. * 90. * math.pi / 180.)
        elif cur_eq == 10:
            val = math.cos(i / 255. * 90. * math.pi / 180.)
        elif cur_eq == 11:
            val = abs((i / 255.) - 0.5)
        elif cur_eq == 12:
            val = (2. * ((i / 255.) - 1.)) * (2. * ((i / 255.) - 1.))
        elif cur_eq == 13:
            val = math.sin(i / 255. * 180. * math.pi / 180.)
        elif cur_eq == 14:
            val = abs(math.cos(i / 255. * 90. * math.pi / 180.))
        elif cur_eq == 15:
            val = math.sin(i / 255. * 360. * math.pi / 180.)
        elif cur_eq == 16:
            val = math.cos(i / 255. * 360. * math.pi / 180.)
        elif cur_eq == 17:
            val = abs(math.sin(i / 255. * 360. * math.pi / 180.))
        elif cur_eq == 18:
            val = abs(math.cos(i / 255. * 360. * math.pi / 180.))
        elif cur_eq == 19:
            val = abs(math.sin(i / 255. * 720. * math.pi / 180.))
        elif cur_eq == 20:
            val = abs(math.cos(i / 255. * 720. * math.pi / 180.))
        elif cur_eq == 21:
            val = 3. * (i / 255.)
        elif cur_eq == 22:
            val = 3. * (i / 255.0) - 1.
        elif cur_eq == 23:
            val = 3. * (i / 255.0) - 2.
        elif cur_eq == 24:
            val = abs(3. * (i / 255.0) - 1.)
        elif cur_eq == 25:
            val = abs(3. * (i / 255.0) - 2.)
        elif cur_eq == 26:
            val = (3. * (i / 255.) - 1.) / 2.
        elif cur_eq == 27:
            val = (3. * (i / 255.) - 2.) / 2.
        elif cur_eq == 28:
            val = abs((3. * (i / 255.) - 1.) / 2.)
        elif cur_eq == 29:
            val = abs((3. * (i / 255.) - 2.) / 2.)
        elif cur_eq == 30:
            val = (i / 255.) / 0.32 - 0.78125
        elif cur_eq == 31:
            val = 2. * (i / 255.) - 0.84
        elif cur_eq == 32:
            val = (i / 255.)
            # val = 4x1-2x+1.84x/0.08-11.5--> ? #show palette rgbformulae pr comprendre comment ils ont fait pr celui la ds gnuplot
        elif cur_eq == 33:
            val = abs(2. * (i / 255.) - 0.5)
        elif cur_eq == 34:
            val = 2. * (i / 255.)
        elif cur_eq == 35:
            val = 2. * (i / 255.) - 0.5
        elif cur_eq == 36:
            val = 2. * (i / 255.) - 1.

        if val < 0.0:
            val = 0.0

        if val > 1.0:
            val = 1.0

        return val

    def create_from_formula(self, red, green, blue):
        """
        Creates a color palette using formulas for the red, green, and blue channels.

        Args:
            red (str): The equation for the red channel.
            green (str): The equation for the green channel.
            blue (str): The equation for the blue channel.

        Returns:
            numpy.ndarray: The color palette as a NumPy array.

        """
        l = 0
        palette = np.zeros((256, 3), dtype=np.uint8)
        R = np.zeros((256,), dtype=np.uint8)
        G = np.zeros((256,), dtype=np.uint8)
        B = np.zeros((256,), dtype=np.uint8)

        if red != "":
            javaified_equation = self.javaifyEquation(red[0:1])
            for i in range(256):
                R[i] = (self.computeEquation(javaified_equation, "x", i / 255.) * 255.)

        if green != "":
            javaified_equation = self.javaifyEquation(green[0:1])
            for i in range(256):
                G[i] = (self.computeEquation(javaified_equation, "x", i / 255.) * 255.)

        if blue != "":
            javaified_equation = self.javaifyEquation(blue[0:1])
            for i in range(256):
                B[i] = (self.computeEquation(javaified_equation, "x", i / 255.) * 255.)

        for i in range(256):
            val = R[i]
            if val < 0:
                val = 0
            if val > 255:
                val = 255
            R[i] = val

            val = G[i]
            if val < 0:
                val = 0
            if val > 255:
                val = 255
            G[i] = val

            val = B[i]
            if val < 0:
                val = 0
            if val > 255:
                val = 255
            B[i] = val

        if red != "":
            if red[0] == '-':
                R = self.reverse_ramp(R)

        if green != "":
            if green[0] == '-':
                G = self.reverse_ramp(G)

        if blue != "":
            if blue[0] == '-':
                B = self.reverse_ramp(B)

        for i in range(len(B)):
            palette[l, 0] = R[i]
            palette[l, 1] = G[i]
            palette[l, 2] = B[i]
            l += 1

        return palette

    def reverse_ramp(self, pal):
        """
        Reverses the color ramp by taking the negative of the intensity (255 - intensity).

        Args:
            pal (numpy.ndarray): The color ramp to be reversed.

        Returns:
            numpy.ndarray: The reversed color ramp.

        """
        tmp = np.zeros_like(pal)
        for i in range(256):
            tmp[255 - i] = pal[i]
        return tmp

    def createEmptyPalette(self):
        """
        Creates an empty palette.

        Returns:
            numpy.ndarray: The empty palette array.

        """
        return np.zeros((256, 3), dtype=np.uint8)

    def create2(self, palette_name, equa1, equa2, equa3):
        """
        Creates a palette from a series of equations.

        Args:
            palette_name (str): The name of the palette.
            equa1 (str): The equation for the first color channel.
            equa2 (str): The equation for the second color channel.
            equa3 (str): The equation for the third color channel.

        Returns:
            numpy.ndarray: The created palette.

        # Examples:
        #     >>> palette_name = '37,38,-39'
        #     >>> equa1 = 'x + y'
        #     >>> equa2 = '2 * x - y'
        #     >>> equa3 = 'x - 2 * y'
        #     >>> result = PaletteCreator.create2(palette_name, equa1, equa2, equa3)
        #     >>> print(result)
        #     [[  0   0   0]
        #      [255   0   0]
        #      [  0 255   0]]

        """
        final_palette = None
        try:
            r = g = b = 3
            if "," in palette_name:
                formula_size = len(palette_name.split(","))
                if formula_size == 3:
                    r = int(palette_name.split(",")[0])
                    new_equa1 = self.create_compatible_equation(equa1, equa2, equa3, r)
                    g = int(palette_name.split(",")[1])
                    new_equa2 = self.create_compatible_equation(equa1, equa2, equa3, g)
                    b = int(palette_name.split(",")[2])
                    new_equa3 = self.create_compatible_equation(equa1, equa2, equa3, b)
                    equa1 = new_equa1
                    equa2 = new_equa2
                    equa3 = new_equa3

            if abs(r) < 37 and abs(g) < 37 and abs(b) < 37:
                return self.create3(palette_name)

            if abs(r) >= 37 and abs(g) >= 37 and abs(b) >= 37:
                palette1 = self.createEmptyPalette()
            else:
                palette1 = self.create(palette_name)

            palette2 = self.create_from_formula(equa1, equa2, equa3)
            final_palette = np.zeros((palette1.shape[0], palette1.shape[1]), dtype=np.uint8)
            l = 0
            l1 = 0
            l2 = 0
            for i in range(256):
                red1 = palette1[l1]
                green1 = palette1[l1]
                blue1 = palette1[l1]
                l1 += 1
                red2 = palette2[l2]
                green2 = palette2[l2]
                blue2 = palette2[l2]
                l2 += 1
                red = red2 if abs(r) >= 37 and abs(r) <= 39 else red1
                green = green2 if abs(g) >= 37 and abs(g) <= 39 else green1
                blue = blue2 if abs(b) >= 37 and abs(b) <= 39 else blue1

                final_palette[l, 0] = red
                final_palette[l, 1] = green
                final_palette[l, 2] = blue
                l += 1
        except:
            traceback.print_exc()

        return final_palette

    def create_compatible_equation(self, equa1, equa2, equa3, col):
        """
        Creates a compatible equation.

        Args:
            equa1 (str): The equation for the first color channel.
            equa2 (str): The equation for the second color channel.
            equa3 (str): The equation for the third color channel.
            col (int): The color value.

        Returns:
            str: The compatible equation.

        """
        equa = ""
        if col == 37:
            equa = "+" + equa1
        elif col == -37:
            equa = "-(" + equa1 + ")"
        elif col == 38:
            equa = "+" + equa2
        elif col == -38:
            equa = "-(" + equa2 + ")"
        elif col == 39:
            equa = "+" + equa3
        elif col == -39:
            equa = "-(" + equa3 + ")"

        return equa

    def create3(self, palette_formula):
        """
        Creates a palette from its formula.

        Args:
            palette_formula (str): The formula for creating the palette.

        Returns:
            numpy.ndarray: The created palette.

        Examples:
            >>> p = PaletteCreator()
            >>> result = p.create3('PACKING')
            >>> print(result[:4])
            [[64 64 64]
             [64 64 64]
             [64 64 64]
             [64 64 64]]

        """

        if palette_formula in self.matplot_lib_luts:
            return cmap_to_numpy(palette_formula)

        r = g = b = 3
        if palette_formula is None:
            palette = self.create5(r, g, b)
            return palette

        if "," in palette_formula.replace("hl", ""):
            formula_size = len(palette_formula.split(","))
            if formula_size == 3:
                r = int(palette_formula.replace("hl", "").split(",")[0])
                g = int(palette_formula.replace("hl", "").split(",")[1])
                b = int(palette_formula.replace("hl", "").split(",")[2])
            palette = self.create5(r, g, b)
        else:
            if palette_formula == 'PACKING':
                return self.get_packing_lut()

            palette = self.create5(r, g, b)

        if palette_formula == self.list['HI_LO']:
            palette[0, 0] = 0
            palette[0, 1] = 0
            palette[0, 2] = 255
            palette[255, 0] = 255
            palette[255, 1] = 0
            palette[255, 2] = 0

        return palette

def __show_lut(lut, title=None):
    """
    Displays the lookup table (LUT) as an image.

    Args:
        lut (numpy.ndarray): The lookup table.
        title (str, optional): The title of the displayed image.

    Returns:
        None

    # Examples:
    #     >>> lut = np.random.randint(0, 256, size=(256, 3), dtype=np.uint8)
    #     >>> __show_lut(lut, title='LUT Image')

    """

    image = np.linspace(0, 255, 256, dtype=np.uint8)
    image = np.tile(image, (32, 1))
    image = apply_lut(image, lut, convert_to_RGB=True)

    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

def lsm_LUT_to_numpy(lut):
    """
    Converts an LSM lookup table (LUT) to a NumPy array.

    Args:
        lut (list): The LSM lookup table.

    Returns:
        numpy.ndarray: The converted NumPy array representing the LUT.

    Examples:
        >>> lut = [255, 0, 0, 0]
        >>> result = lsm_LUT_to_numpy(lut)
        >>> print(result[:3])
        [[0 0 0]
         [1 0 0]
         [2 0 0]]

    """

    R = np.linspace(0, lut[0], 256, dtype=np.uint8)
    G = np.linspace(0, lut[1], 256, dtype=np.uint8)
    B = np.linspace(0, lut[2], 256, dtype=np.uint8)
    lut = np.stack([R, G, B], axis=-1)
    return lut

def R_G_B_lut_to_int24_lut(lut):
    """
    Converts an RGB lookup table (LUT) to an int24 LUT.

    Args:
        lut (numpy.ndarray): The RGB lookup table.

    Returns:
        numpy.ndarray: The converted int24 LUT.

    Examples:
        >>> lut = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        >>> result = R_G_B_lut_to_int24_lut(lut)
        >>> print(result)
        [16711680    65280      255]

    """

    if lut is None:
        return None

    if isinstance(lut, list):
        lut = lsm_LUT_to_numpy(lut)

    if len(lut.shape) == 1:
        return lut

    if lut.shape[-1] == 3:
        return lut[..., 0].astype(np.uint32) << 16 | lut[..., 1].astype(np.uint32) << 8 | lut[..., 2].astype(
            np.uint32)
    else:
        return lut[0, ...].astype(np.uint32) << 16 | lut[1, ...].astype(np.uint32) << 8 | lut[2, ...].astype(
            np.uint32)

def apply_lut(img, lut, convert_to_RGB=False, min=None, max=None):
    """
    Applies a lookup table (LUT) to an image.

    Args:
        img (numpy.ndarray): The input image.
        lut (numpy.ndarray or list): The lookup table.
        convert_to_RGB (bool, optional): Flag indicating whether to convert the image to RGB.
        min (float, optional): The minimum value for color coding.
        max (float, optional): The maximum value for color coding.

    Returns:
        numpy.ndarray: The image with the LUT applied.

    Examples:
        >>> img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        >>> lut = np.random.randint(0, 256, size=(256, 3), dtype=np.uint8)
        >>> result = apply_lut(img, lut, convert_to_RGB=True)
        >>> print(result.shape)
        (100, 100, 3)

    """

    real_max = img.max()
    real_min = img.min()

    if min is not None and max is not None:
        if img.max() > max:
            img[img > max] = max
        if img.min() < min:
            img[img < min] = min

        img = np.interp(img, (real_min if real_min > min else min, real_max if real_max < max else max), (
        real_min if real_min > min else min,
        real_max if real_max < max else max))

        img = np.interp(img, (min, max), (0, 255)).astype(np.uint8)

    else:
        img = np.interp(img, (img.min(), img.max()), (0, 255)).astype(np.uint8)

    int24_lut = R_G_B_lut_to_int24_lut(lut)

    image = int24_lut[img]
    if convert_to_RGB:
        image = int24_to_RGB(image)

    return image

def matplotlib_to_TA(lut_name='viridis'):
    """
    Converts a Matplotlib colormap to a lookup table (LUT) compatible with TA software.

    Args:
        lut_name (str, optional): The name of the Matplotlib colormap.

    Returns:
        numpy.ndarray: The converted lookup table (LUT).

    Examples:
        >>> lut = matplotlib_to_TA('viridis')
        >>> print(lut.shape)
        (256, 3)

    """

    cm = plt.get_cmap(lut_name)
    cm = cm(range(256)) * 255
    cm = cm[..., 0:3].astype(np.uint8)
    return cm


def get_matplotlib_available_luts():
    """
    Returns a list of available Matplotlib colormaps.

    Returns:
        list: The list of available colormaps.

    Examples:
        >>> colormaps = get_matplotlib_available_luts()
        >>> print(colormaps[:2])
        ['magma', 'inferno']

    """

    return plt.colormaps()


if __name__ == '__main__':
    # keep to avoid bugs
    import matplotlib
    matplotlib.use('TkAgg')

    if True:
        # lsm type of Lut --> may need be changed
        lut = [255,128,255]
        lut = lsm_LUT_to_numpy(lut)
        print(lut)
        print(type(lut))
        img = create_ramp(300, 1800, lut)
        plt.imshow(img)
        plt.show()
        import sys
        sys.exit(0)

    if True:
        lutcreator = PaletteCreator()
        lut = lutcreator.create3(lutcreator.list['MAGENTA_HOT']) #MAGENTA --> create it --> PURPLE instead

        img = create_ramp(300, 1800,lut)  # maybe ask at some point for the bg color --> TODO but almost all ok in fact
        plt.imshow(img)
        plt.show()
        import sys
        sys.exit(0)

    if True:
        lutcreator = PaletteCreator()
        lut = lutcreator.create3(lutcreator.list['PACKING'])

        img = create_ramp(300, 1800,lut)  # maybe ask at some point for the bg color --> TODO but almost all ok in fact
        plt.imshow(img)
        plt.show()
        import sys
        sys.exit(0)


    if False:
        # create a ramp
        lutcreator = PaletteCreator()
        lut = lutcreator.create3(lutcreator.list['DNA'])

        img = create_ramp(300, 1800,lut)  # maybe ask at some point for the bg color --> TODO but almost all ok in fact
        plt.imshow(img)
        plt.show()

    if False:

        print(cmap_to_numpy('gray')) # 'viridis' 'rainbow' 'gray'


    if True:
        lutcreator = PaletteCreator()

        luts = lutcreator.list

        # lut.create_from_formula('22','13','-23')
        lut = lutcreator.create2('34,35,36', 'x', 'x', 'x')  # bingo ça marche enfin!!!
        # print(lut)
        # print(lut.shape, lut.max(), lut.min())

        lut = lutcreator.create3(lutcreator.list['DNA'])
        # print(lut)
        # print(lut.shape, lut.max(), lut.min())

        # random lut each time
        # print(list(luts.keys()))

        random_lut = random.choice(list(luts.keys()))
        # print('random.lut', random_lut)

        # random_lut = 'HI_LO'
        lut = lutcreator.create3(luts[random_lut])

        # TODO maybe do a loop over all luts maybe also on an image --> easy way to select best luts

        __show_lut(lut, title=random_lut)
        # all seems ok now
        # plt.imshow(create_ramp(0,255, lut))
        # plt.show()


        # print(get_matplotlib_available_luts())  # --> y en a vraiment bcp --> je pourrais aussi les ajouter à ma db de luts

        # can i do a visual lut selector or something that applies luts to the image à la tinder so that one can select the most appropriate LUT --> in fact that is not so hard to do now that everything is done through indexing
        # img = create_ramp(0, 255, None)
        # img = create_ramp(-1, 1, None) # --> super easy instead


