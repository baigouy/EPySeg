# can be used to generate pseudo random or random colors

import random
# need seed it or reset seed
import sys

def get_forbidden_colors():
    """
    Retrieves a list of forbidden colors represented in RGB format.

    Returns:
        list: A list of forbidden colors in RGB format.

    Examples:
        >>> get_forbidden_colors()
        [(0, 0, 0), (255, 255, 255)]
    """
    # Define the forbidden colors using RGB values
    forbidden_colors = [(0, 0, 0), (255, 255, 255)]

    return forbidden_colors


def get_forbidden_colors_int24():
    """
    Retrieves a list of forbidden colors represented in 24-bit integer format.

    Returns:
        list: A list of forbidden colors in 24-bit integer format.

    Examples:
        >>> get_forbidden_colors_int24()
        [0, 16777215]
    """
    # Define the forbidden colors using 24-bit integer values
    forbidden_colors = [0, 0xFFFFFF]

    return forbidden_colors


def r_g_b_to_rgb(r, g, b):
    """
    Converts individual RGB color components to a combined RGB value.

    Args:
        r (int): The red color component (0-255).
        g (int): The green color component (0-255).
        b (int): The blue color component (0-255).

    Returns:
        int: The combined RGB value.

    Examples:
        >>> r_g_b_to_rgb(255, 0, 0)
        16711680
        >>> r_g_b_to_rgb(0, 255, 0)
        65280
        >>> r_g_b_to_rgb(0, 0, 255)
        255
    """
    RGB = (r << 16) + (g << 8) + b
    return RGB



def r_g_b_from_rgb(rgb):
    """
    Extracts individual RGB color components from a combined RGB value.

    Args:
        rgb (int): The combined RGB value.

    Returns:
        tuple: A tuple containing the red, green, and blue color components.

    Examples:
        >>> r_g_b_from_rgb(16711680)
        (255, 0, 0)
        >>> r_g_b_from_rgb(65280)
        (0, 255, 0)
        >>> r_g_b_from_rgb(255)
        (0, 0, 255)
    """
    b = rgb & 255
    g = (rgb >> 8) & 255
    r = (rgb >> 16) & 255
    return r, g, b


# self calling stack --> maybe not a good idea ....
def get_unique_random_color_int24(forbidden_colors=None, seed_random=None, assign_new_col_to_forbidden=False):
    """
    Generates a unique random color represented in a 24-bit integer format.

    Args:
        forbidden_colors (list, optional): A list of forbidden colors in RGB format. Defaults to None.
        seed_random (int, optional): An integer value to seed the random number generator. Defaults to None.
        assign_new_col_to_forbidden (bool, optional): Specifies whether to assign the new color to the forbidden colors list. Defaults to False.

    Returns:
        int: A unique random color represented as a 24-bit integer.

    Examples:
        >>> get_unique_random_color_int24(forbidden_colors=[16777215])!=16777215
        True
    """
    if seed_random is not None:
        random.seed(seed_random)

    color = random.getrandbits(24)

    if forbidden_colors is not None:
        if color in forbidden_colors:
            while color in forbidden_colors:
                color = random.getrandbits(24)

        if assign_new_col_to_forbidden:
            forbidden_colors.append(color)

    return color


# if I seed the random then it will work to generate random colors
def get_unique_random_color(forbidden_colors=None, seed_random=None, assign_new_col_to_forbidden=False):
    """
    Generates a unique random color represented in RGB format.

    Args:
        forbidden_colors (list, optional): A list of forbidden colors in RGB format. Defaults to None.
        seed_random (int, optional): An integer value to seed the random number generator. Defaults to None.
        assign_new_col_to_forbidden (bool, optional): Specifies whether to assign the new color to the forbidden colors list. Defaults to False.

    Returns:
        tuple: A unique random color represented as an RGB tuple.

    Examples:
        >>> get_unique_random_color(forbidden_colors=[(255, 255, 255)]) != (255,255,255)
        True
    """
    if seed_random is not None:
        random.seed(seed_random)

    r = random.getrandbits(8)
    g = random.getrandbits(8)
    b = random.getrandbits(8)
    color = (r, g, b)

    if forbidden_colors is not None:
        if color in forbidden_colors:
            while color in forbidden_colors:
                color = get_unique_random_color(forbidden_colors=forbidden_colors)

        if assign_new_col_to_forbidden:
            forbidden_colors.append(color)

    return color



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

