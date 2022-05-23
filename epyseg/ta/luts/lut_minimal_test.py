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


# TODO do a lut that can be used for color coding packing

def cmap_to_numpy(lut):
    if isinstance(lut, str):
        cm = plt.get_cmap(lut)
    else:
        cm = lut
    out = cm(range(256)) * 255
    out = out[..., 0:3].astype(np.uint8)  # conversion from cmap of matplotlib to RGB 8bits np.ndarray
    return out

# based on https://matplotlib.org/3.1.1/tutorials/colors/colormap-manipulation.html
def numpy_to_cmap(lut):
    # can be used to plot any TA lut with matplotlib and get a ramp --> very useful!
    lut_cp = np.copy(lut)
    if lut_cp.max()>1:
        lut_cp =lut_cp/ 255.
        # print('in here')

    if lut_cp.shape[-1] == 3:
        tmp = np.ones((256, 4), dtype=np.float64)
        tmp[...,0:3]=lut_cp[...,:]
        lut_cp = tmp
    newcmp = ListedColormap(lut_cp)
    return newcmp


# orientation = 'vertical' #'horizontal'
# maybe ask at some point for the bg color --> TODO but almost all ok in fact
def create_ramp(min, max, lut, orientation='vertical'):
    # modified from https://matplotlib.org/stable/gallery/misc/agg_buffer_to_array.html
    img = np.zeros((2,1))
    img[0]=min
    img[1]=max
    fig = Figure(tight_layout=True)
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()

    # else assume it is a matplotlib lut
    cmap = lut
    if isinstance(lut, np.ndarray):
        cmap = numpy_to_cmap(lut)

    im = ax.imshow(img, cmap=cmap)
    fig.colorbar(im, ax=ax,orientation=orientation)
    ax.set_axis_off()
    ax.remove()
    canvas.draw()
    img = fig_to_numpy(fig)
    return img

def list_availbale_luts(return_matplot_lib_luts_in_separate_database=False):
    hash_pal = {}
    hash_pal["DEFAULT"] = "22,13,-23"  # DNA my favourite lut personally
    hash_pal["AFM_HOT"] = "34,35,36"
    hash_pal["GREEN"] = "0,3,0"
    hash_pal["RED"] = "3,0,0"
    hash_pal["BLUE"] = "0,0,3"
    hash_pal["GRAY"] = "3,3,3"
    hash_pal["GREY"] = "3,3,3"
    hash_pal["PACKING"] = "PACKING"
    hash_pal["RED_HOT"] = "21,22,23"
    hash_pal["GREEN_HOT"] = "35,34,23"  # "22,21,23"
    hash_pal["BLUE_HOT"] = "23,22,21"
    hash_pal["OCEAN"] = "23,28,3"
    hash_pal["GREEN_RED_VIOLET"] = "3,11,6"
    hash_pal["COLOR_PRINTABLE_ON_GRAY"] = "30,31,32"
    hash_pal["HI_LO"] = "3,3,3hl"
    hash_pal["DNA"] = "22,13,-23"
    hash_pal["RAINBOW_1"] = "22,13,-31"
    hash_pal["RAINBOW_2"] = "35,13,-35"
    hash_pal["RAINBOW_3"] = "33,13,10"
    hash_pal["RAINBOW_LIGHT"] = "33,13,-31"
    hash_pal["RAINBOW_ARTIFICIAL"] = "10,13,33"
    hash_pal["BLUE_WHITE_RED"] = "34,-13,-34"
    hash_pal["BROWN"] = "7,4,27"
    hash_pal["GREEN_FIRE_BLUE"] = "31,34,11"
    hash_pal["GREEN_FIRE_BLUE2"] = "31,34,29"
    hash_pal["ORANGE"] = "3,5,0"
    hash_pal["ORANGE_HOT"] = "21,-10,23"
    hash_pal["YELLOW_HOT"] = "34,-10,23"
    hash_pal["YELLOW"] = "3,3,0"
    hash_pal["YELLOW_PALE"] = "34,34,4"
    hash_pal["PURPLE_HOT"] = "34,36,34"
    hash_pal["PURPLE"] = "3,0,3"
    hash_pal["PM3D"] = "7,5,15"
    hash_pal["HSV"] = "3,2,2"
    hash_pal["HSV2"] = "7,5,15"
    hash_pal["YING_YANG"] = "13,13,13"
    hash_pal["YING_YANG2"] = "14,14,14"
    hash_pal["CYCLIC"] = "19,19,19"
    hash_pal["METEO1"] = "34,-35,-36"
    hash_pal["METEO2"] = "22,13,-31"  # is another rainbow LUT
    hash_pal["DIABOLO_MENTHE"] = "-7,2,-7"
    hash_pal["DIABOLO_GRENADINE"] = "2,-7,-7"
    hash_pal["BLACK_PURPLE_PINK_WHITE"] = "30,31,32"
    hash_pal["RED_BLACK_BLUE"] = "-36,0,36"


    # TODO also add all luts from matplotlib
    # def get_matplotlib_available_luts():
    #     return plt.colormaps()

    matplot_lib_luts = []
    try:
        matplot_lib_luts = get_matplotlib_available_luts()
        for matplotlib in matplot_lib_luts:
            hash_pal['MATPLOTLIB_' + str(matplotlib)] = str(matplotlib)
    except:
        # no big deal if fails because I have enough luts already
        # traceback.print_exc()
        pass
    # print(hash_pal)
    if return_matplot_lib_luts_in_separate_database:
        return matplot_lib_luts, hash_pal
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

    # private
    def __list_palettes(self):
        self.matplot_lib_luts, self.list = list_availbale_luts(return_matplot_lib_luts_in_separate_database=True)

    # lui faire aussi loader ttes les palettes exterieures se trouvant dans lib / luts ou luts.jar a voir
    # import_palettes_from_palette_folder(None)

    #     /**
    #      * returns the nb of letters that match the pattern in the word
    #      *
    #      * @param input string
    #      * @param char2count input character
    #      * @since <B>Packing Analyzer 1.0</B>
    #      */

    def get_packing_lut(self):
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
        #
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
            if char2count is None:
                return 0
            if strg is None or strg == "":
                return 0
            nb = 0
            # for i = 0; i < string.length(); i++):
            for c in strg:
                # char c = string.charAt(i)
                if (c == char2count):
                    nb += 1
            return nb

    def convertAbs(self, txt):
        try:
            while "|" in txt:
                size = len(txt)
                pos = txt.index("|")
                begin_pos = pos + 1
                end_pos = pos

                for i in range(pos + 1, size + 1):  # (i = pos + 1; i < size; i++):
                    c = txt[i]
                    if c == '|':
                        # case '|':
                        end_pos = i
                        break
                    end_pos = i + 1
                power_equa = txt[begin_pos, end_pos]
                beginning_of_word = txt[0, begin_pos - 1]
                end_of_word = txt[end_pos + 1]
                txt = beginning_of_word + "abs(" + power_equa + ")" + end_of_word
        except:
            traceback.print_exc()
        return txt

    def convertPowers(self, txt):
        try:
            while (txt.contains("^")):
                size = len(txt)
                pos = txt.index("^")
                begin_pos = pos
                end_pos = pos
                for i in range(pos, size):
                    c = txt[i]
                    if c == '+' or c == '-' or c == '/' or c == '*':
                        # switch (c):
                        #     case '+':
                        #     case '-':
                        #     case '/':
                        #     case '*':
                        end_pos = i
                        break
                    end_pos = i + 1
                for i in range(pos, 0, -1):
                    c = txt[i]
                    # switch (c):
                    #     case '+':
                    #     case '-':
                    #     case '/':
                    #     case '*':
                    if c == '+' or c == '-' or c == '/' or c == '*':
                        begin_pos = i + 1
                        break
                    begin_pos = i
                power_equa = txt.substring(begin_pos, end_pos)
                beginning_of_word = txt.substring(0, begin_pos)
                end_of_word = txt.substring(end_pos)
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

    #
    # /**
    #     * converts a into a java friendly equation
    #     *
    #     * @since <B>Packing Analyzer 3.0</B>
    #     */
    def javaifyEquation(self, equation):
        print(equation)
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

        if (nb_of_open_brackets != nb_of_closing_brackets):
            raise Exception("equation is incorrect, nb of opened and closed brackets don't match")

        return equation

    # /**
    #   * compute the result of an equation using the javascript engine
    #   *
    #   * @param equation
    #   * @param variables
    #   * @param vals
    #   * @return
    #   */
    def computeEquation(self, equation, variable, val):

        # print(equation,variable,val)
        # return
        # try:
        #     Compilable compiledEng = (Compilable) javaEngine;
        #     for (i = 0; i <= 9; i++) {
        #         equation = equation.replace(i + variable, i + "*" + variable);
        #     }
        #     for (i = 0; i <= 9; i++) {
        #         equation = equation.replace(variable + i, variable + "*" + i);
        #     }
        #     equation = equation.replace(variable, "(double)" + variable);
        #     CompiledScript script = compiledEng.compile(equation);
        #     javaEngine[variable, val);
        #     return String2Double(script.eval().toString());
        # except:
        #     # LogFrame2.printStackTrace(e);
        #     traceback.print_exc()
        #     return 0

        import numexpr as ne
        var = {'a': np.array([1, 2, 2]), 'b': np.array([2, 1, 3]), 'c': np.array([3])}
        print('TOO', ne.evaluate('2*a*(b/c)**2', local_dict=var))
        # In[146]: ne.evaluate('2*a*(b/c)**2', local_dict=var)
        # Out[146]: array([0.88888889, 0.44444444, 4.])
        # --> all seems ok in fact --> fianlize it

        try:
            for i in range(10):
                equation = equation.replace(str(i) + variable, str(i) + "*" + variable)
            for i in range(10):
                equation = equation.replace(variable + str(i), variable + "*" + str(i))

            equation = equation.replace(variable, "(double)" + variable)
            # CompiledScript script = compiledEng.compile(equation);
            # javaEngine[variable, val);
            # return String2Double(script.eval().toString());
            print(equation)
        except:
            # LogFrame2.printStackTrace(e);
            traceback.print_exc()
            return 0

        # TODO --> either do this way or do using eval to dynamically compute data
        # np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)

    def create5(self, red, green, blue):
        l = 0
        palette = np.zeros((256, 3), dtype=np.uint8)  # [256 * 3]
        R = np.zeros((256,), dtype=np.uint8)
        G = np.zeros((256,), dtype=np.uint8)
        B = np.zeros((256,), dtype=np.uint8)

        # /*
        #  * Loads RG and B component of the LUT.
        #  * I used the same RGB formula as in gnuplot
        #  */
        for i in range(256):
            R[i] = int(self.RGB_formula(i, red) * 255.0)
            G[i] = int(self.RGB_formula(i, green) * 255.0)
            B[i] = int(self.RGB_formula(i, blue) * 255.0)

        # /*
        #  * reverse ramp if number is negative
        #  */
        if red < 0:
            R = self.reverse_ramp(R)

        if green < 0:
            G = self.reverse_ramp(G)

        if blue < 0:
            B = self.reverse_ramp(B)

        for i in range(len(B)):  # i = 0 i <  i++):
            palette[l, 0] = R[i]
            palette[l, 1] = G[i]
            palette[l, 2] = B[i]
            l += 1

        return palette

    #  # /*
    #     #  * computes a color for each gray value between 0 and 255
    #     #  */
    def RGB_formula(self, i, equation):
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
            # break
        elif cur_eq == 2:
            val = 1.0
            # break
        elif cur_eq == 3:
            val = i / 255.
            # break
        elif cur_eq == 4:
            val = (i / 255.) * (i / 255.)
            # break
        elif cur_eq == 5:
            val = (i / 255.) * (i / 255.) * (i / 255.)
            # break
        elif cur_eq == 6:
            val = (i / 255.) * (i / 255.) * (i / 255.) * (i / 255.)
            # break
        elif cur_eq == 7:
            val = math.sqrt(i / 255.)
            # break
        elif cur_eq == 8:
            val = math.sqrt(math.sqrt((i / 255.)))
            # break
        elif cur_eq == 9:
            val = math.sin(i / 255. * 90. * math.pi / 180.)
            # break
        elif cur_eq == 10:
            val = math.cos(i / 255. * 90. * math.pi / 180.)
            # break
        elif cur_eq == 11:
            val = abs((i / 255.) - 0.5)
            # break
        elif cur_eq == 12:
            val = (2. * ((i / 255.) - 1.)) * (2. * ((i / 255.) - 1.))
            # break
        elif cur_eq == 13:
            val = math.sin(i / 255. * 180. * math.pi / 180.)
            # break
        elif cur_eq == 14:
            val = abs(math.cos(i / 255. * 90. * math.pi / 180.))
            # break
        elif cur_eq == 15:
            val = math.sin(i / 255. * 360. * math.pi / 180.)
            # break
        elif cur_eq == 16:
            val = math.cos(i / 255. * 360. * math.pi / 180.)
            # break
        elif cur_eq == 17:
            val = abs(math.sin(i / 255. * 360. * math.pi / 180.))
            # break
        elif cur_eq == 18:
            val = abs(math.cos(i / 255. * 360. * math.pi / 180.))
            # break
        elif cur_eq == 19:
            val = abs(math.sin(i / 255. * 720. * math.pi / 180.))
            # break
        elif cur_eq == 20:
            val = abs(math.cos(i / 255. * 720. * math.pi / 180.))
            # break
        elif cur_eq == 21:
            val = 3. * (i / 255.)
            # break
        elif cur_eq == 22:
            val = 3. * (i / 255.0) - 1.
            # break
        elif cur_eq == 23:
            val = 3. * (i / 255.0) - 2.
            # break
        elif cur_eq == 24:
            val = abs(3. * (i / 255.0) - 1.)
            # break
        elif cur_eq == 25:
            val = abs(3. * (i / 255.0) - 2.)
            # break
        elif cur_eq == 26:
            val = (3. * (i / 255.) - 1.) / 2.
            # break
        elif cur_eq == 27:
            val = (3. * (i / 255.) - 2.) / 2.
            # break
        elif cur_eq == 28:
            val = abs((3. * (i / 255.) - 1.) / 2.)
            # break
        elif cur_eq == 29:
            val = abs((3. * (i / 255.) - 2.) / 2.)
            # break
        elif cur_eq == 30:
            val = (i / 255.) / 0.32 - 0.78125
            # break
        elif cur_eq == 31:
            val = 2. * (i / 255.) - 0.84
            # break
        elif cur_eq == 32:
            val = (i / 255.)
            # val = 4x1-2x+1.84x/0.08-11.5--> ? #show palette rgbformulae pr comprendre comment ils ont fait pr celui la ds gnuplot
            # break
        elif cur_eq == 33:
            val = abs(2. * (i / 255.) - 0.5)
            # break
        elif cur_eq == 34:
            val = 2. * (i / 255.)
            # break
        elif cur_eq == 35:
            val = 2. * (i / 255.) - 0.5
            # break
        elif cur_eq == 36:
            val = 2. * (i / 255.) - 1.
            # break

        if val < 0.0:
            val = 0.0

        if val > 1.0:
            val = 1.0

        return val

    def create_from_formula(self, red, green, blue):
        l = 0
        palette = np.zeros((256, 3), dtype=np.uint8)  # [256 * 3]
        R = np.zeros((256,), dtype=np.uint8)
        G = np.zeros((256,), dtype=np.uint8)
        B = np.zeros((256,), dtype=np.uint8)

        if red != "":
            javaified_equared = self.javaifyEquation(red[0:1])
            for i in range(256):
                # /*
                #  * Let javascript compute the colors from equations for us
                #  */
                R[i] = (self.computeEquation(javaified_equared, "x", i / 255.) * 255.)

        if green != "":
            javaified_equagreen = self.javaifyEquation(green[0:1])  # green.substring(1))
            for i in range(256):  # = 0 i < 256 i++:
                # /*
                #  * Let javascript compute the colors from equations for us
                #  */
                G[i] = (self.computeEquation(javaified_equagreen, "x", i / 255.) * 255.)

        if blue != "":
            javaified_equablue = self.javaifyEquation(blue[0:1])  # blue.substring(1))
            for i in range(256):  # i = 0 i < 256 i++:
                # /*
                #  * Let javascript compute the colors from equations for us
                #  */
                B[i] = (self.computeEquation(javaified_equablue, "x", i / 255.) * 255.)

        # /*
        #  * check bounds (colors should be <=255 and >=0)
        #  */
        for i in range(256):  # = 0 i < 256 i++:
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

        # /*
        #  * check if the palette has to be inverted and if so invert it
        #  */
        if red != "":
            if red.charAt(0) == '-':
                R = self.reverse_ramp(R)

        if green != "":
            if green[0] == '-':
                G = self.reverse_ramp(G)

        if blue != "":
            if blue.charAt(0) == '-':
                B = self.reverse_ramp(B)

        # /*
        #  * load the lut into an array (PA compatible)
        #  */
        for i in range(len(B)):  # i = 0 i <  i++):
            palette[l, 0] = R[i]
            palette[l, 1] = G[i]
            palette[l, 2] = B[i]
            l += 1

        return palette

        # /*
        #  * takes the negative of a LUT (255-Intensity R,G and B)
        #  */

    def reverse_ramp(self, pal):
        tmp = np.zeros_like(pal)
        for i in range(256):  # (i = 0; i < pal.length; i++):
            tmp[255 - i] = pal[i]
        return tmp

        # /*
        #  * creates an empty LUT
        #  */

    def createEmptyPalette(self):
        return np.zeros((256, 3), dtype=np.uint8)

    # /*
    #  * creates a palette from a series of equations
    #  */
    def create2(self, palette_name, equa1, equa2, equa3):
        final_palette = None
        try:
            r = g = b = 3
            if "," in palette_name:
                formula_size = len(palette_name.split(","))
                if formula_size == 3:  # should always be true --> remove it ???
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
            final_palette = int[palette1.length]
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
                #
                # final_palette[l] = red
                # final_palette[l] = green
                # final_palette[l] = blue
                l += 1
        except:
            traceback.print_exc()

        return final_palette

        # /*

    #  * creates a compatible equation
    #  */
    def create_compatible_equation(self, equa1, equa2, equa3, col):
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

    # /*    
    #  * creates a palette from its formula
    #  */
    def create3(self, palette_formula):

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


# ask horiz or vert and allow save it maybe
# see also how to handle max and min from real data --> probably not hard but need think about it!!!
# deprecated --> use create_ramp_instead
def __show_lut(lut, title=None):
    # image = np.zeros((256,), dtype=np.uint8)
    # create ramp image

    image = np.linspace(0, 255, 256, dtype=np.uint8)  # ça c'est ok mais si je tile alors le perd le truc
    # print('image.dtype',image.dtype)
    image = np.tile(image, (32, 1))
    # print('image.shape', image.shape, image.dtype)

    image = apply_lut(image, lut, convert_to_RGB=True)


    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()


def R_G_B_lut_to_int24_lut(lut):
    if lut is None:
        return None
    return lut[..., 0].astype(np.uint32) << 16 | lut[..., 1].astype(np.uint32) << 8 | lut[..., 2].astype(
        np.uint32).astype(np.uint32)


# probably need change the conversion for global to be sure that I fit within the bounds --> TODO
# could force max and min global to have data better bounded


# pas mal --> is there a way I can get the real LUT with real image data
# --> somehow need create a gradient with min max of the real image and apply the color code to it without changing the range
# --> in that case let matplotlib do the job

# ask whether LUT should be saved or not ??? maybe append it to the image but will change image shape --> maybe return LUT as a separate object --> add an option
# if user wants it maybe create an image directly using matplotlib --> à tester car plus simple d'ajouter la lut
def apply_lut(img, lut, convert_to_RGB=False, min=None, max=None):
    # first need get real max and min
    real_max = img.max()
    real_min = img.min()

    # image is 0,0 here min max already for the stuff -> pb for the nematic --> needs a fix
    # print(real_min, real_max)

    # min =0
    # max = 255

    if min is not None and max is not None:
        # allows for global color coding !!!

        # I think this is the real conversion I should do

        # img = np.interp(img, (img.min(), img.max()), (min, max))
        # https://stackoverflow.com/questions/36000843/scale-numpy-array-to-certain-range

        # https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range/35993695#35993695 --> comapre both by the way
        # img = np.interp(img, (min, max), (min, max)) # pas la peine de s'embeter --> ça fait exactement ce que je veux -> en plus c'est super propre

        # i want to convert

        # store all values between 0 and 1 --> TODO then convert to 0 -255 for LUt to be applied

        # shall I preprocess it if some of its values are above the range --> maybe -> need a clip

        # pre clip the image --> for a real LUT display I will need the real data ??? think about it cause not sure
        if img.max() > max:
            img[img > max] = max
        if img.min() < min:
            img[img < min] = min

        # non en fait c'est bien ça je pense
        # img = np.interp(img, (min if min <real_min else real_min, max if max > real_max else real_max), (min, max)) # pas la peine de s'embeter --> ça fait exactement ce que je veux -> en plus c'est super propre

        # voilà comment on fait!!! --> tt est converti en 0 255
        img = np.interp(img, (real_min if real_min > min else min, real_max if real_max < max else max), (
        real_min if real_min > min else min,
        real_max if real_max < max else max))  # pas la peine de s'embeter --> ça fait exactement ce que je veux -> en plus c'est super propre

        # img = np.interp(img, (real_min if real_min >min else min, real_max if real_max <max else max), (0, 255)).astype(np.uint8) # pas la peine de s'embeter --> ça fait exactement ce que je veux -> en plus c'est super propre
        # img = np.interp(img, (min if min <real_min else real_min, max if max > real_max else real_max), (0, 255)).astype(np.uint8) # pas la peine de s'embeter --> ça fait exactement ce que je veux -> en plus c'est super propre
        # img = np.interp(img, (min, max), (min, max)) # pas la peine de s'embeter --> ça fait exactement ce que je veux -> en plus c'est super propre
        # img = np.interp(img, (img.min(), img.max()), (min, max)) # pas la peine de s'embeter --> ça fait exactement ce que je veux -> en plus c'est super propre

        # TODO also check how I do in TA!!!

        # img/=max
        # img*=255
        # img = img.astype(np.uint8)

        # https://docs.scipy.org/doc/scipy/reference/interpolate.html --> TODO --> maybe also check that

        # NumPy provides numpy.interp for 1-dimensional linear interpolation. In this case, where you want to map the minimum element of the array to −1 and the maximum to +1, and other elements linearly in-between, you can write:
        # np.interp(a, (a.min(), a.max()), (-1, +1))
        # https://codereview.stackexchange.com/questions/185785/scale-numpy-array-to-certain-range

        # https://numpy.org/doc/stable/reference/generated/numpy.interp.html

        # see exactly what it does in order not to do crap but i'm almost there I think, maybe if I use real values all will be ok

        # is that really ok in fact

        # MEGA TODO ce truc marche vraiment bien --> remplacer mon code de partout par ça en fait --> très bonne idee

        # ce truc a vraiment converti mes valeurs dans les bounds min max mais en fait --> c'est pas ce que je veux --> reflechir mais ça va vachement m'aider et dans bcp d'endroits --> TODO
        # print('real_max', real_max, 'vs', max, 'cur max', img.max())
        # print('real_min', real_min, 'vs', min, 'cur min', img.min())
        img = np.interp(img, (min, max), (0, 255)).astype(np.uint8)

        # plt.imshow(img)
        # plt.show()

        # eps = 1e-20
        # img = (img - min)/(max-min+ eps)

        # pb --> va creer des bug si superieur
        # img = np.interp(img, (0, int(255*real_max/max)), (0, int(255*real_max/max))).astype(np.uint8)

        # print(img.min(), img.max()) # 0.03 9.223372036854776e+16 --> bug due to numerical instabilities

        # this is clip
        # img[img <= 0] = 0.
        # img[img>=1]=1.
    else:
        # local color coding
        img = np.interp(img, (img.min(), img.max()), (0, 255)).astype(np.uint8)


        # if I put that the packing lut is applied properly
        # print(img.max())
        # img = img.astype(np.uint8)



        pass


        # print('in')
        # img = img()

    # should it always be done ??? maybe think further
    # if img.max() > 256:
    #     # need normalize it
    #     # img = (img - img.min())/(img.max()-img.min())
    #     img = np.interp(img, (img.min(), img.max()), (0, 255)).astype(np.uint8)
    #     print('2', img.max(), img.min(), img.dtype)

    # if img.max() <= 1 and img.min() >= 0:
    #     img = (img * 255).astype(np.uint8)

    # print(img.max(), img.min()) # --> this stuff was converted

    # print('lut',lut) # bug cause all values are the same --> ok indeed makes sense --> my equation is incorrectly handled
    # apply LUT
    int24_lut = R_G_B_lut_to_int24_lut(lut)

    # print('int24_lut',int24_lut)

    # image = int24_to_RGB(int24_lut[img])
    image = int24_lut[img]
    if convert_to_RGB:
        image = int24_to_RGB(image)

    # print('int24_lut[image]',int24_lut[image])
    return image


def matplotlib_to_TA(lut_name='viridis'):
    cm = plt.get_cmap(lut_name)  # 'viridis' 'rainbow'
    # print(cm.)
    # print(cm(range(256)) * 255)  # --> gives the cmap as an RGBA array noamrlized 0 1 --> convert it to 255 maybe

    # --> not bad in fact and easy todo

    cm = cm(range(256)) * 255
    cm = cm[..., 0:3].astype(np.uint8)  # conversion from cmap of matplotlib to RGB 8bits
    # print(cm)
    return cm


def get_matplotlib_available_luts():
    return plt.colormaps()

    # ça marche vraiment bien --> facile de gerer des luts en fait peut etre ajouter la possibilité de les sauver et de les exporter aussi vers matplotlib!!!!




if __name__ == '__main__':
    # keep to avoid bugs
    import matplotlib
    matplotlib.use('TkAgg')


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


