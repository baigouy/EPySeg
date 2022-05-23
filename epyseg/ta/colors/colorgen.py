# can be used to generate pseudo random or random colors
# TODO maybe allow it to generate slices according to dimensions --> shouldn't be too hard to do...

import random
# need seed it or reset seed
import sys


def get_forbidden_colors():
    # return [r_g_b_to_rgb(0, 0, 0), r_g_b_to_rgb(255, 255, 255)]
    return [(0, 0, 0),(255, 255, 255)]

def get_forbidden_colors_int24():
    # return [r_g_b_to_rgb(0, 0, 0), r_g_b_to_rgb(255, 255, 255)]
    return [0,0xFFFFFF]


def r_g_b_to_rgb(r, g, b):
    RGB = (r << 16) + (g << 8) + b
    return RGB


def r_g_b_from_rgb(rgb):
    b = rgb & 255
    g = (rgb >> 8) & 255
    r = (rgb >> 16) & 255
    return r, g, b

# self calling stack --> maybe not a good idea ....
def get_unique_random_color_int24(forbidden_colors=None, seed_random=None, assign_new_col_to_forbidden=False):
    if seed_random is not None:
        random.seed(seed_random)
    # r = random.getrandbits(8)
    # b = random.getrandbits(8)
    # g = random.getrandbits(8)
    # color = r_g_b_to_rgb(r, g, b)
    color = random.getrandbits(24)
    # color = (r, g, b)
    if forbidden_colors is not None:
        if color in forbidden_colors:
            while color in forbidden_colors:
                # can I hack this in order to prevent stack error --> probably yes
                # hack to prevent stack recursion
                color = random.getrandbits(24) #get_unique_random_color_int24(forbidden_colors=forbidden_colors)s
        if assign_new_col_to_forbidden:
            forbidden_colors.append(color)
    return color

# if I seed the random then it will work to generate random colors
def get_unique_random_color(forbidden_colors=None, seed_random=None, assign_new_col_to_forbidden=False):
    if seed_random is not None:
        random.seed(seed_random)
    r = random.getrandbits(8)
    b = random.getrandbits(8)
    g = random.getrandbits(8)
    # color = r_g_b_to_rgb(r, g, b)
    color = (r, g, b)
    if forbidden_colors is not None:
        if color in forbidden_colors:
            while color in forbidden_colors:
                color = get_unique_random_color(forbidden_colors=forbidden_colors)
        if assign_new_col_to_forbidden:
            forbidden_colors.append(color)
    return color

# TODO --> add alpha
def r_g_b_image_to_rgb_image(img):
    if img.shape[-1]!=3:
        print('Error only RGB images are supported')
        return img
    return img

# ça marche en fait --> pas mal du tout
if __name__ == '__main__':




    if True:
        random.seed(10)
        r = random.getrandbits(8)
        b = random.getrandbits(8)
        g = random.getrandbits(8)



        color = r_g_b_to_rgb(r, g, b)

        print(color, random.getrandbits(24),  random.getrandbits(32))

        forbidden = get_forbidden_colors_int24()

        color = get_unique_random_color_int24(forbidden)
        print(color)

        sys.exit(0)



    print(r_g_b_to_rgb(255, 255, 255))
    print(r_g_b_from_rgb(r_g_b_to_rgb(255, 255, 255)))

    # ça marche et ça doit donner la meme chose qu'en java et c'est bcp plus facile à coder...
    forbidden_colors = get_forbidden_colors()
    print(forbidden_colors)
    forbidden_colors.append((216, 194, 98))
    forbidden_colors.append((227, 10, 107))

    for i in range(100):
        if i == 0:
            color = get_unique_random_color(forbidden_colors=forbidden_colors, seed_random=0)
        else:
            color = get_unique_random_color(forbidden_colors=forbidden_colors)
        forbidden_colors.append(color)
        print(i, '-->', color)

