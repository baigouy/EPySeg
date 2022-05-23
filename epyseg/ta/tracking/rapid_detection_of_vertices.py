# cantor formula
# maybe the simplest way to give a unique id is that In mathematics, a pairing function is a process to uniquely encode two natural numbers into a single natural number. https://en.wikipedia.org/wiki/Pairing_function
# pairing function = ((x+y)*(x+y+1)/2)+y --> try that... --> but will that work with big ints as mine ??? and --> try with pure white and if ok then I have it # or use the local id to give a global id using this formula -->

# pi(k1, k2) = 1/2(k1 + k2)(k1 + k2 + 1) + k2
# pi(k1, k2) = 1/2(k1 + k2)(k1 + k2 + 1) + (k1*k2)


import numpy as np
from functools import partial
# from PIL import Image
from scipy.ndimage import generic_filter
from epyseg.img import Img, pad_border_xy
import matplotlib.pyplot as plt


# TODO have a look here to see if super fast or not https://stackoverflow.com/questions/10996769/pixel-neighbors-in-2d-array-image-using-python


# could create a file with vertices and bonds maybe with forbidden colors in fact
# ça y est j'ai les vertices exactement comme dans TA
# could ask for the color of the bonds --> can be black or white

# ask for what should be returned

# NB MEGA IMPORTANT IC OULD DEFINE BONDS BY VERTICES --> IF TWO DISTINCT VERTICES ARE ASSOCIATED TO THE SAME TWO CELL IT MEANS THEY ARE CONNECTED --> easy way of getting bonds and in addition also nicely detects bonds consisting of just two pixels --> smarter than what I used to do in TA!!! --> THINK ABOUT IT

# use the other method here because this is so slow


# MEGA TODO --> REPLACE TUPLE BY LIST for the coords of vertices !!!
def associate_vertices_to_cells(cell_labels, vertices, forbidden_colors=(0, 0xFFFFFF),
                                output_vertices_and_associated_cells=False, output_cells_and_their_vertices=False,
                                output_cells_and_their_neighbors=False):

    outputs = []
    # for all vertices detect the corresponding cell
    cell_id_n_vx_coords = {}
    # contains for each vertex all the associated cells --> can be used for rescue of neighbours for example or things alike
    vertices_n_associated_cells = {}

    if isinstance(vertices, tuple):
        vx_coords = vertices
    else:
        vx_coords = np.where(vertices == 255)  # assume vertices are labeled pure white...

    # print('dobs',type(vx_coords), type(vertices)) #dobs <class 'tuple'> <class 'numpy.ndarray'>
    # then need look around it and compute

    # print('x', vertices[0][0])
    # print('y', vertices[1][0])

    # look around each vx for ids and add the vx to all the corresponding ids
    # print(vx_coords[0], vx_coords[...,0])

    # the pb is that if vertices are missing definitely cells would be missing --> to avoid this I would need to get t
    # TODO FIX THIS TO DETECT ALL VERTICES (SEE THE TRIANGULATE CLASS FOR A FAILING EXAMPLE)!!!

    for i in range(len(vx_coords[0])):
        y = vx_coords[0][i]
        x = vx_coords[1][i]
        ids = neighbors8((y, x), cell_labels)
        # ids = neighbors8_2((y, x), cell_labels)

        # print('neighbs length', len(ids))

        ids = set(ids)  # --> ok but need remove pure white and black from that or ignore them # en fait je peux aussi faire ça...

        if forbidden_colors:
            for col in forbidden_colors:
                if col in ids:
                    ids.remove(col)

        # print(x,y, ids)
        # add all the neighbs of the current vx
        vertices_n_associated_cells[(y, x)] = ids
        # print('vx asssoc ids', ids)  #seems ok --> almost all 3

        for id in ids:
            if id in cell_id_n_vx_coords:
                # raw = cell_id_n_vx_coords[id]
                cell_id_n_vx_coords[id].append((y, x))
            else:
                cell_id_n_vx_coords[id] = [(y, x)]




    if output_cells_and_their_vertices:
        outputs.append(cell_id_n_vx_coords)

    if output_vertices_and_associated_cells:
        outputs.append(vertices_n_associated_cells)

    # TODO also offer the connexion of cells and all of its neighbors --> TODO
    # in fact I need bonds to get cells really connected or i need vertices

    if output_cells_and_their_neighbors:
        cell_ids_n_neighbor_ids = {}
        for key, value in cell_id_n_vx_coords.items():
            # print(key, '->', value)
            neighbs = []
            for vx in value:
                neighbs += vertices_n_associated_cells[vx]

            neighbs = set(neighbs)
            neighbs.remove(key)

            # if key == 8787781:
            #     print('souppa', len(neighbs), len(value), neighbs, value)

            cell_ids_n_neighbor_ids[key] = neighbs
        # print('final connection cell to cells', cell_ids_n_neighbor_ids)
        outputs.append(cell_ids_n_neighbor_ids)

    if len(outputs) == 1:
        return outputs[0]
    else:
        if outputs:
            return outputs

    # mega TODO --> return also the vertices and the cells --> offer this as an offer because can also be very useful and not very smart doing this several times because may be needed several times --> TODO
    return cell_id_n_vx_coords


# detects the neighbors of the lost cell so that I can reidentify the lost cells by their neighborhood
def detect_neighbors_of_current_cell(cell_id, cell_vertices, cell_labels, forbidden_colors=(0,0xFFFFFF)):
    # loop over cell vertices and detect neighbors
    # then find the cell that best matches this and assign it

    neighbs = []
    for vx in cell_vertices:

        # print('vx', vx)
        # cell_ids_n_neighbor_ids = {}
        # for key, value in cell_id_n_vx_coords.items():
            # print(key, '->', value)

            # for vx in value:

        # print('neighbors8', neighbors8(vx, cell_labels), neighbors8_2(vx, cell_labels)) # TODO replace method some day...
        neighbs += neighbors8(vx, cell_labels).tolist()

    neighbs = set(neighbs)
    neighbs.remove(cell_id)

    if forbidden_colors:
        for col in forbidden_colors:
            if col in neighbs:
                neighbs.remove(col)

            # if key == 8787781:
            #     print('souppa', len(neighbs), len(value), neighbs, value)

            # cell_ids_n_neighbor_ids[key] = neighbs
        # print('final connection cell to cells', cell_ids_n_neighbor_ids)
        # outputs.append(cell_ids_n_neighbor_ids)
    return neighbs

# eight_neighb = [[(-1,-1), (0,-1), (1,-1)],
#                 [(-1,0), 0, 1],
#                 [-1, 0, 1]]

def neighbors8(vx_coords, cell_labels):

    # print('vx_coords',vx_coords)

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
    min_y = vx_coords[0] - 1
    max_y = vx_coords[0] + 2
    min_x = vx_coords[1] - 1
    max_x = vx_coords[1] + 2

    # print('0min_x, max_x, min_y, max_y, vx_coords', min_x, max_x, min_y, max_y, vx_coords) # --> ok --> error in correc below

    min_y = min_y if min_y >= 0 else 0
    max_y = max_y if max_y < cell_labels.shape[0] else cell_labels.shape[0]
    min_x = min_x if min_x >= 0 else 0
    max_x = max_x if max_x < cell_labels.shape[1] else cell_labels.shape[1]

    # if min_x>max_x:
    #     min_x, max_x = max_x, min_x
    # if min_y>max_y:
    #     min_y, max_y = max_y, min_y

    # why not sorted
    # print('min_x, max_x, min_y, max_y, vx_coords',min_x, max_x, min_y, max_y, vx_coords)

    coords = []
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
                coords.append([y,x])

    return coords

    # [min_y:max_y, min_x:max_x]

    # return cell_labels[min_y:max_y, min_x:max_x].ravel()



# TODO compare both algos and find fastest
# DO NOT USE THIS ALGO AS IT IS MUCH SLOWER THAN THE PREVIOUS ONE!!!
# def neighbors8_2(vx_coords, cell_labels):
#     coords = []
#     for i in [-1, 0, 1]:
#         for j in [-1, 0, 1]:
#             if vx_coords[0]+i>=0 and vx_coords[0]+i<cell_labels.shape[0]:
#                 if vx_coords[1] + j >= 0 and vx_coords[1] + j < cell_labels.shape[1]:
#                     coords.append((vx_coords[0]+i, vx_coords[1]+j))
#     return coords
    # return cell_labels[min_y:max_y, min_x:max_x].ravel()



def cantor_color_bonds(RGB24_or_lab, vertices, boundary_color=0xFFFFFF,copy=True): #  # TODO use partial to send parameters to the method
    if isinstance(boundary_color, int):
        boundary_color = [boundary_color]
    method = partial(_cantor_color_bonds, boundary_colors=boundary_color) # bug if boundary color is set --> why --> most likely a bug in their code
    # vertices = generic_filter(RGB24, method, (3, 3))
    # return generic_filter(RGB24, method, (3, 3))

    # print(method)
    if copy:
        # copy image
        RGB24 = np.copy(RGB24_or_lab)
    else:
        # in place
        RGB24 = RGB24_or_lab
    # return generic_filter(RGB24, _cantor_color_bonds, (3, 3))

    # TODO handle border bonds that are wrong yet need be handled:
    # scan line by line and change id depending on


    bonds = generic_filter(RGB24, method, (3, 3))
    bonds[vertices != 0]=0


    # handle border bonds
    max_id = bonds.max()+1
    for iii in range(RGB24.shape[1]):
        if vertices[0,iii]==255:
            max_id+=1
            continue
        bonds[0,iii]=max_id
    for iii in range(RGB24.shape[1]):
        if vertices[RGB24.shape[0]-1,iii]==255:
            max_id+=1
            continue
        bonds[RGB24.shape[0]-1,iii]=max_id

    for jjj in range(RGB24.shape[0]):
        if vertices[jjj,0]==255:
            max_id+=1
            continue
        bonds[jjj,0]=max_id
    for jjj in range(RGB24.shape[0]):
        if vertices[jjj,RGB24.shape[1]-1]==255:
            max_id+=1
            continue
        bonds[jjj,RGB24.shape[1]-1]=max_id


    # return generic_filter(RGB24, method, (3, 3))
    return bonds


def detect_vertices_and_bonds(RGB24_or_lab, detect_bonds=False,
                              boundary_color=0xFFFFFF, split_bonds_and_vertices = False, copy=True, pad_to_detect_vertices=True):  # TODO use partial to send parameters to the method
    # print('inner shape',RGB24.shape) --> n'a pas marché

    if isinstance(boundary_color, int):
        boundary_color=[boundary_color]
    # method = partial(_identify_vertices_and_bonds, detect_bonds=detect_bonds)
    # return generic_filter(RGB24, _detect_vertices_and_bonds, (3, 3))
    # return generic_filter(RGB24, method, (3, 3))
    if copy and not pad_to_detect_vertices:
        # copy image
        RGB24= np.copy(RGB24_or_lab)
    else:
        # in place
        RGB24 = RGB24_or_lab


    # method = partial(_detect_vertices_and_bonds, detect_bonds=detect_bonds)
    method = partial(_detect_vertices_and_bonds, boundary_colors=boundary_color)
    # vertices = generic_filter(RGB24, method, (3, 3))

    if pad_to_detect_vertices:
        # do pad to get the right vertices --> TODO
        RGB24 = np.pad(RGB24, pad_width=1, mode='constant', constant_values=False)
        RGB24 = pad_border_xy(RGB24, mode=RGB24.max() + 1)
        # print(padded_cells.shape)

    # print('test', RGB24.shape, RGB24.dtype)
    vertices = generic_filter(RGB24, method, (3, 3))

    # up until here is ok

    if pad_to_detect_vertices:
        # add the few missing vertices to the paded image
        vertices = vertices[1:-1, 1:-1]  # unpad
        # add a vertex to each image corner
        vertices[0, 0] = 255
        vertices[-1, 0] = 255
        vertices[0, -1] = 255
        vertices[-1, -1] = 255

        RGB24 = RGB24[1:-1, 1:-1]# unpad RGB image too
    else:
        # old code, probably do not use because misses some vertices and is not very smart
        # detect vertices at image boundaries
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

        # add a vertex to each image corner
        vertices[0][0] = 255
        vertices[RGB24.shape[0] - 1][0] = 255
        vertices[RGB24.shape[0] - 1][RGB24.shape[1] - 1] = 255
        vertices[0][RGB24.shape[1] - 1] = 255

    # NB TODO if detect bonds --> detect bonds at the edges too

    # fix bonds at the border of the image that are not detected
    if detect_bonds:
        # print('in here')
        # pb si trop de couleurs --> à fixer un jour en fait
        vertices[np.where((RGB24==boundary_color) & (vertices!=255))] = 128

    if split_bonds_and_vertices and detect_bonds:
        bonds = np.zeros_like(vertices)
        bonds[vertices == 128] = 128  # make a bond specific file
        vertices[vertices == 128] = 0  # make a vertex specific file
        return vertices, bonds




    return vertices


def _cantor_pairing(id1, id2, _sort_points=True):
    x = id1
    y = id2
    if _sort_points and id1<id2:
        x=id2
        y=id1
    # return ((x + y) * (x + y + 1) / 2) + y
    return int(((x + y) * (x + y + 1) / 2) + y) #



# def identify_bonds(RGB24):
#     # the idea here is to give to each bond a unique ID and also associate the vertices to each bond --> create the equivalent of the TA file --> TODO
#     # loop for every white pixel and if two ids around it give it a unique id
#     # if three or more ids it is a vertex --> could also give it an id maybe to be consitent and should associate it to some bonds if needed --> very close to the previous code but I would pass to this a dict on top of that where I would add the id
#
#     # try cantor formula to avoid having to store/pass a map and search it
#
#     pass

def _detect_vertices_and_bonds_old_doing_useless_stuff(P, detect_bonds=False, boundary_colors=(0xFFFFFF, 0)):
    if boundary_colors:
        if P[4] not in boundary_colors:  # not pure white
            return 0

    size = len(set(P))  # 4 way vertices only but why not
    if detect_bonds and size == 3:
        return 128
    # if size >= 4:
    #     print(set(P))
    return 255 if size >= 4 else 0

# this thing detects bonds and vertices but does not give an id to bonds
# TODO convert to vertices the things touching the boundary too ??? or not???
def _detect_vertices_and_bonds(P, boundary_colors=(0xFFFFFF, 0)):
    if boundary_colors:
        if P[4] not in boundary_colors:  # not pure white
            return 0

    size = len(set(P))  # 4 way vertices only but why not
    # if detect_bonds and size == 3:
    #     return 128
    # if size >= 4:
    #     print(set(P))
    return 255 if size >= 4 else 0


# unique_bond_identifiers --> would be a dict having an id and a corresponding pair of cells --> maybe sort them or add the entry twice --> maybe sorting is more efficient


# nb I could use np.unique to recolor all of the cells to something in the range 1-whatever --> good idea in fact and could be fast --> try that
def _identify_vertices_and_bonds(P, detect_bonds=False, boundary_colors=(0xFFFFFF, 0)):

    if boundary_colors:
        if P[4] not in boundary_colors:  # not pure white
            return 0

    corrected_ids = set(P)
    size = len(corrected_ids)  # 4 way vertices only but why not

    # ça a l'air de marcher
    if detect_bonds and size == 3:
        # print('in')
        corrected_ids = list(corrected_ids.difference(boundary_colors))
        # TODO use cantor pairing here to get a unique id from two bonds
        # print(corrected_ids[0], corrected_ids[1])
        # print(_cantor_pairing(corrected_ids[0], corrected_ids[1]))
        return _cantor_pairing(corrected_ids[0], corrected_ids[1])
        # return 128
    # if size >= 4:
    #     print(set(P))
    return 255 if size >= 4 else 0


def _cantor_color_bonds(P, boundary_colors=(0xFFFFFF, 0)):
    if boundary_colors:
        # print('in')
        if P[4] not in boundary_colors:  # not pure white
            return 0
    corrected_ids = set(P)
    size = len(corrected_ids)  # 4 way vertices only but why not
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
