import numpy as np
from functools import partial
# from PIL import Image
from scipy.ndimage import generic_filter
from epyseg.img import Img, pad_border_xy
import matplotlib.pyplot as plt

def associate_vertices_to_cells(cell_labels, vertices, forbidden_colors=(0, 0xFFFFFF),
                                output_vertices_and_associated_cells=False, output_cells_and_their_vertices=False,
                                output_cells_and_their_neighbors=False):
    """
    Associates vertices with corresponding cells based on cell labels.

    Args:
        cell_labels (numpy.ndarray): Array containing cell labels.
        vertices (tuple or numpy.ndarray): Tuple of vertex coordinates or vertex array.
        forbidden_colors (tuple, optional): Colors to ignore when associating vertices to cells. Defaults to (0, 0xFFFFFF).
        output_vertices_and_associated_cells (bool, optional): Whether to output vertices and their associated cells. Defaults to False.
        output_cells_and_their_vertices (bool, optional): Whether to output cells and their associated vertices. Defaults to False.
        output_cells_and_their_neighbors (bool, optional): Whether to output cells and their neighboring cells. Defaults to False.

    Returns:
        dict or list: Output based on the selected options.

    """
    outputs = []
    cell_id_n_vx_coords = {}
    vertices_n_associated_cells = {}

    if isinstance(vertices, tuple):
        vx_coords = vertices
    else:
        vx_coords = np.where(vertices == 255)

    for i in range(len(vx_coords[0])):
        y = vx_coords[0][i]
        x = vx_coords[1][i]
        ids = neighbors8((y, x), cell_labels)

        ids = set(ids)

        if forbidden_colors:
            for col in forbidden_colors:
                if col in ids:
                    ids.remove(col)

        vertices_n_associated_cells[(y, x)] = ids

        for id in ids:
            if id in cell_id_n_vx_coords:
                cell_id_n_vx_coords[id].append((y, x))
            else:
                cell_id_n_vx_coords[id] = [(y, x)]

    if output_cells_and_their_vertices:
        outputs.append(cell_id_n_vx_coords)

    if output_vertices_and_associated_cells:
        outputs.append(vertices_n_associated_cells)

    if output_cells_and_their_neighbors:
        cell_ids_n_neighbor_ids = {}
        for key, value in cell_id_n_vx_coords.items():
            neighbs = []
            for vx in value:
                neighbs += vertices_n_associated_cells[vx]

            neighbs = set(neighbs)
            neighbs.remove(key)

            cell_ids_n_neighbor_ids[key] = neighbs

        outputs.append(cell_ids_n_neighbor_ids)

    if len(outputs) == 1:
        return outputs[0]
    else:
        if outputs:
            return outputs

    return cell_id_n_vx_coords


def detect_neighbors_of_current_cell(cell_id, cell_vertices, cell_labels, forbidden_colors=(0,0xFFFFFF)):
    """
    Detects the neighbors of a current cell based on its vertices.

    Args:
        cell_id (int): ID of the cell.
        cell_vertices (list): List of cell vertices.
        cell_labels (numpy.ndarray): Array containing cell labels.
        forbidden_colors (tuple, optional): Colors to ignore when detecting neighbors. Defaults to (0, 0xFFFFFF).

    Returns:
        set: Set of neighboring cell IDs.

    """
    neighbs = []
    for vx in cell_vertices:
        neighbs += neighbors8(vx, cell_labels).tolist()

    neighbs = set(neighbs)
    neighbs.remove(cell_id)

    if forbidden_colors:
        for col in forbidden_colors:
            if col in neighbs:
                neighbs.remove(col)

    return neighbs

def neighbors8(vx_coords, cell_labels):
    """
    Calculates the 8 neighbors surrounding a given vertex in a matrix of cell labels.

    Args:
        vx_coords (tuple): The coordinates of the vertex (y, x).
        cell_labels (ndarray): The matrix of cell labels.

    Returns:
        ndarray: The cell labels of the neighboring cells as a 1D array.

    """
    min_y = vx_coords[0] - 1
    max_y = vx_coords[0] + 2
    min_x = vx_coords[1] - 1
    max_x = vx_coords[1] + 2

    min_y = min_y if min_y >= 0 else 0
    max_y = max_y if max_y < cell_labels.shape[0] else cell_labels.shape[0]
    min_x = min_x if min_x >= 0 else 0
    max_x = max_x if max_x < cell_labels.shape[1] else cell_labels.shape[1]

    return cell_labels[min_y:max_y, min_x:max_x].ravel()


def neighbors8_shift_mapping(vx_coords, cell_labels):
    """
    Calculates the coordinates of the 8 neighbors surrounding a given vertex in a matrix of cell labels.

    Args:
        vx_coords (tuple): The coordinates of the vertex (y, x).
        cell_labels (ndarray): The matrix of cell labels.

    Returns:
        list: The coordinates of the neighboring vertices.

    """
    min_y = vx_coords[0] - 1
    max_y = vx_coords[0] + 2
    min_x = vx_coords[1] - 1
    max_x = vx_coords[1] + 2

    min_y = min_y if min_y >= 0 else 0
    max_y = max_y if max_y < cell_labels.shape[0] else cell_labels.shape[0]
    min_x = min_x if min_x >= 0 else 0
    max_x = max_x if max_x < cell_labels.shape[1] else cell_labels.shape[1]

    coords = []
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            coords.append([y, x])

    return coords

def cantor_color_bonds(RGB24_or_lab, vertices, boundary_color=0xFFFFFF, copy=True):
    """
    Applies the Cantor pairing function to color the bonds between vertices in an image.

    Args:
        RGB24_or_lab (ndarray): The input image in RGB or LAB format.
        vertices (ndarray): The matrix of vertices in the image.
        boundary_color (int or list): The color(s) used for boundary detection. Defaults to 0xFFFFFF.
        copy (bool): Specifies whether to create a copy of the image. Defaults to True.

    Returns:
        ndarray: The matrix of colored bonds.

    """
    if isinstance(boundary_color, int):
        boundary_color = [boundary_color]

    method = partial(_cantor_color_bonds, boundary_colors=boundary_color)

    if copy:
        RGB24 = np.copy(RGB24_or_lab)
    else:
        RGB24 = RGB24_or_lab

    bonds = generic_filter(RGB24, method, (3, 3))
    bonds[vertices != 0] = 0

    max_id = bonds.max() + 1

    for iii in range(RGB24.shape[1]):
        if vertices[0, iii] == 255:
            max_id += 1
            continue
        bonds[0, iii] = max_id

    for iii in range(RGB24.shape[1]):
        if vertices[RGB24.shape[0] - 1, iii] == 255:
            max_id += 1
            continue
        bonds[RGB24.shape[0] - 1, iii] = max_id

    for jjj in range(RGB24.shape[0]):
        if vertices[jjj, 0] == 255:
            max_id += 1
            continue
        bonds[jjj, 0] = max_id

    for jjj in range(RGB24.shape[0]):
        if vertices[jjj, RGB24.shape[1] - 1] == 255:
            max_id += 1
            continue
        bonds[jjj, RGB24.shape[1] - 1] = max_id

    return bonds


def detect_vertices_and_bonds(RGB24_or_lab, detect_bonds=False,
                              boundary_color=0xFFFFFF, split_bonds_and_vertices=False, copy=True,
                              pad_to_detect_vertices=True):
    """
    Detects vertices and bonds in an image.

    Args:
        RGB24_or_lab (ndarray): The input image in RGB or LAB format.
        detect_bonds (bool): Specifies whether to detect bonds. Defaults to False.
        boundary_color (int or list): The color(s) used for boundary detection. Defaults to 0xFFFFFF.
        split_bonds_and_vertices (bool): Specifies whether to split bonds and vertices into separate matrices.
            Defaults to False.
        copy (bool): Specifies whether to create a copy of the image. Defaults to True.
        pad_to_detect_vertices (bool): Specifies whether to pad the image for vertex detection. Defaults to True.

    Returns:
        ndarray or tuple: The matrix of vertices, or a tuple of matrices if split_bonds_and_vertices is True.

    """
    if isinstance(boundary_color, int):
        boundary_color = [boundary_color]

    method = partial(_detect_vertices_and_bonds, boundary_colors=boundary_color)

    if copy and not pad_to_detect_vertices:
        RGB24 = np.copy(RGB24_or_lab)
    else:
        RGB24 = RGB24_or_lab

    if pad_to_detect_vertices:
        RGB24 = np.pad(RGB24, pad_width=1, mode='constant', constant_values=False)
        RGB24 = pad_border_xy(RGB24, mode=RGB24.max() + 1)

    vertices = generic_filter(RGB24, method, (3, 3))

    if pad_to_detect_vertices:
        vertices = vertices[1:-1, 1:-1]
        vertices[0, 0] = 255
        vertices[-1, 0] = 255
        vertices[0, -1] = 255
        vertices[-1, -1] = 255

        RGB24 = RGB24[1:-1, 1:-1]
    else:
        for jjj in range(1, RGB24.shape[0] - 1):
            neighbs = RGB24[jjj - 1:jjj + 2, 0:2]
            different_neighbours = set(neighbs.ravel().tolist())
            if len(different_neighbours) == 3:
                vertices[jjj][0] = 255
            neighbs = RGB24[jjj - 1:jjj + 2, RGB24.shape[0] - 2:RGB24.shape[0]]
            different_neighbours = set(neighbs.ravel().tolist())
            if len(different_neighbours) == 3:
                vertices[jjj][RGB24.shape[0] - 1] = 255

        for iii in range(1, RGB24.shape[1] - 1):
            neighbs = RGB24[0:2, iii - 1:iii + 2]
            different_neighbours = set(neighbs.ravel().tolist())
            if len(different_neighbours) == 3:
                vertices[0][iii] = 255
            neighbs = RGB24[RGB24.shape[1] - 2:RGB24.shape[1], iii - 1:iii + 2]
            different_neighbours = set(neighbs.ravel().tolist())
            if len(different_neighbours) == 3:
                vertices[RGB24.shape[0] - 1][iii] = 255

        vertices[0][0] = 255
        vertices[RGB24.shape[0] - 1][0] = 255
        vertices[RGB24.shape[0] - 1][RGB24.shape[1] - 1] = 255
        vertices[0][RGB24.shape[1] - 1] = 255

    if detect_bonds:
        vertices[np.where((RGB24 == boundary_color) & (vertices != 255))] = 128

    if split_bonds_and_vertices and detect_bonds:
        bonds = np.zeros_like(vertices)
        bonds[vertices == 128] = 128
        vertices[vertices == 128] = 0
        return vertices, bonds

    return vertices

# see also https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
def _sudzik_pairing(id1, id2, _sort_points=True):
    """
    Compute a pairing value for two integer IDs using Sudzik's pairing function.

    Sudzik's pairing function combines two integers (IDs) into a unique value.
    The function can optionally sort the input IDs before computing the pairing value.

    Parameters:
    - id1 (int): The first integer ID.
    - id2 (int): The second integer ID.
    - _sort_points (bool, optional): If True (default), the input IDs are sorted before pairing.

    Returns:
    int: The computed pairing value.

    Example:
    >>> _sudzik_pairing(3, 2)
    11
    >>> _sudzik_pairing(2, 3)
    11
    >>> _sudzik_pairing(128, 256)
    65664
    >>> _sudzik_pairing(256, 128) # I guess I want always the sorted version for my purpose
    65664
    >>> _sudzik_pairing(256, 128, _sort_points=False)
    65920
    >>> _sudzik_pairing(128, 256, _sort_points=False)
    65664
    """
    if _sort_points and id1 > id2:
        id2, id1 = id1, id2
    return id1 * id1 + id1 + id2 if id1 >= id2 else id1 + id2 * id2

# TODO --> extensively test all the cantor and sudzik parirings stuff and move them to color class (maybe always use sudzik, check with big numbers just to see)
# TODO also add the other color based cantors because this may be very useful !!!

def _cantor_pairing(id1, id2, _sort_points=True):
    """
    Applies the Cantor pairing function to two IDs to generate a unique bond ID.

    Args:
        id1 (int): The first ID.
        id2 (int): The second ID.
        _sort_points (bool): Specifies whether to sort the IDs before applying the function. Defaults to True.

    Returns:
        int: The bond ID.

    Examples:
        >>> _cantor_pairing(3, 4)
        31
        >>> _cantor_pairing(4, 3)
        31
        >>> _cantor_pairing(10, 5)
        125
    """
    x = id1
    y = id2
    if _sort_points and id1 < id2:
        x = id2
        y = id1
    return int(((x + y) * (x + y + 1) / 2) + y)



def _detect_vertices_and_bonds_old_doing_useless_stuff(P, detect_bonds=False, boundary_colors=(0xFFFFFF, 0)):
    if boundary_colors:
        if P[4] not in boundary_colors:  # not pure white
            return 0

    size = len(set(P))  # 4-way vertices only but why not
    if detect_bonds and size == 3:
        return 128
    return 255 if size >= 4 else 0


def _detect_vertices_and_bonds(P, boundary_colors=(0xFFFFFF, 0)):
    """
    Detects vertices in a neighborhood and assigns IDs to them.

    Args:
        P (ndarray): Neighborhood array.
        boundary_colors (tuple): Tuple of boundary colors. Defaults to (0xFFFFFF, 0).

    Returns:
        int: The vertex ID.

    """
    if boundary_colors:
        if P[4] not in boundary_colors:  # not pure white
            return 0

    size = len(set(P))  # 4-way vertices only but why not
    return 255 if size >= 4 else 0


def _identify_vertices_and_bonds(P, detect_bonds=False, boundary_colors=(0xFFFFFF, 0)):
    """
    Identifies vertices and assigns IDs to them. Optionally detects bonds.

    Args:
        P (ndarray): Neighborhood array.
        detect_bonds (bool): Specifies whether to detect bonds. Defaults to False.
        boundary_colors (tuple): Tuple of boundary colors. Defaults to (0xFFFFFF, 0).

    Returns:
        int: The vertex ID or bond ID.

    """
    if boundary_colors:
        if P[4] not in boundary_colors:  # not pure white
            return 0

    corrected_ids = set(P)
    size = len(corrected_ids)  # 4-way vertices only but why not

    if detect_bonds and size == 3:
        corrected_ids = list(corrected_ids.difference(boundary_colors))
        return _cantor_pairing(corrected_ids[0], corrected_ids[1])

    return 255 if size >= 4 else 0


def _cantor_color_bonds(P, boundary_colors=(0xFFFFFF, 0)):
    if boundary_colors:
        if P[4] not in boundary_colors:  # not pure white
            return 0

    corrected_ids = set(P)
    size = len(corrected_ids)  # 4-way vertices only but why not

    if size == 3:
        corrected_ids = list(corrected_ids.difference(boundary_colors))
        return _cantor_pairing(corrected_ids[0], corrected_ids[1])

    return 0


if __name__ == '__main__':

    if False:
        import sys
        print(_cantor_pairing(0xFFFFFF,0xFFFFFE)) #562949382996104 562949886312449
        print(_cantor_pairing(0xFFFFFF,0xFFFFFF)) #562949382996104 562949886312449
        print(_cantor_pairing(0xFFFFFFFF,0xFFFFFFFE)) #3.6893488130239234e+19 --> very big
        print(_cantor_pairing(0xFFFFFFFF,0xFFFFFFFF)) #3.689348813882917e+19 --> very big too # try avoid such big number but ok if can't do better...
        print(_cantor_pairing(0, 0))
        print(_cantor_pairing(0, 1)) # 0,1 and 1, 0 differ --> need sort the points before passing them or after passing them
        print(_cantor_pairing(1, 0))
        print(_cantor_pairing(10, 23)) # ça marche variment bien mais faire +1 pr eviter d'avoir 0 --> en fait je m'en fous car pourra jamais etre = 0
        print(_cantor_pairing(23, 10))
        print(_cantor_pairing(1, 32))
        print(_cantor_pairing(32, 1))

        # --> I love this because this is new and elegant


        # very good idea
        # I love this cantor formula
        sys.exit(0)


    # Open image and make into Numpy array
    # PILim = Image.open('patches.png').convert('RGB')
    RGBim = Img('/E/Sample_images/segmentation_assistant/ovipo_uncropped/200709_armGFP_suz_46hAPF_ON.lif - Series011/tracked_cells_resized.tif')[0:64, 0:64]
    # RGBim = np.array(PILim)

    # Make a single channel 24-bit image rather than 3 channels of 8-bit each

    print(RGBim.shape)

    # there is a bug but almost there still
    RGB24 = (RGBim[..., 0].astype(np.uint32) << 16) | (RGBim[..., 1].astype(np.uint32) << 8) | RGBim[..., 2].astype(
        np.uint32)

    print(RGB24.max(), RGB24.min())
    # RGB24 = (RGBim[...,0].astype(np.uint32)<<16) | (RGBim[...,1].astype(np.uint32)<<8) | RGBim[...,2].astype(np.uint32)

    # Run generic filter counting unique colours in neighbourhood
    result = detect_vertices_and_bonds(RGB24)


    # then could recolor them if needed --> could try that
    # give a unique id to bonds --> very easy in facts --> maybe replace old code ??? --> pb is that I need a uint64 image that will be quite big in mem but ok I guess or need relabel the image afterwards --> maybe by the way...
    result = generic_filter(RGB24.astype(np.uint64), partial(_identify_vertices_and_bonds,detect_bonds=True), (3, 3))
    # try to see how I can do that better!!!


    # result = generic_filter(RGB24.astype(float), partial(_identify_vertices_and_bonds,detect_bonds=True), (3, 3))
    print(result.dtype)
    # result = result.astype(np.uint8)

    # Save result
    # Image.fromarray(result.astype(np.uint8)).save('result.png')
    plt.imshow(result)
    plt.show()

    # plt.imshow(RGB24)
    # plt.show()

    indices = (result == 255)
    print('indices', indices)

    vertices = np.where(result == 255) #c'est
    print(vertices)
    if vertices[0].size != 0:
        print('y', vertices[0][0])
    if vertices[1].size != 0:
        print('x', vertices[1][0])
    if vertices[0].size + vertices[1].size != 0:
        print('first vx', result[vertices[0][0], vertices[1][0]])

    print(associate_vertices_to_cells(RGB24,
                                      result))  # ça marche en fait --> j'ai associé les vertex à chaque cellule --> cool
    # pb need add vertices at the corners too --> like the first or second pixel away from the border
