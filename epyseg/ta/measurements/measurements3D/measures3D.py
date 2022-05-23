import numpy as np
from epyseg.tools.logger import TA_logger # logging
from epyseg.ta.utils.rps_tools import pavlidis, compute_distance_between_consecutive_points

logger = TA_logger()

# this would actually work for 2D and 3D area --> heightmap is optional but it makes no sense to run this with 2D coords and no heightmaps
# if a heightmap is provided the z coords of every point will be replaced by height map height
# nb if several triangles are provided, it is assumed they are all traingulation of the same object, hence their area is summed and the sum is returned
def compute_3D_surfacearea(triangles, heightmap=None, stop_on_2D=False, disable_2D_warning=False):
    '''

    :param triangles: cell/object triangulation as input (as list or ndarray)
    :param heightmap: a heightmap for the cell
    :param stop_on_2D: break if 2D data is provided
    :return: the surface area of the triangulated object
    '''
    if not isinstance(triangles, np.ndarray):
        triangles = np.asarray(triangles)

    if triangles.shape[-1] == 2 and heightmap is None:
        if not disable_2D_warning:
            logger.error(
            'Error, triangle coords are purely 2D and no heightmap is provided, hence it is not possible to compute 3D area. To continue, the software will assume all heights are the same (= 0)  (i.e. the image is really a pure 2D image)')  # assuming 2D maybe --> just stack

        if stop_on_2D:
            return

        flat_height = np.zeros((*triangles.shape[0:-1], (1)))
        triangles = np.append(triangles, flat_height, -1)

    # MEGA TODO NB SHOULD I MAKE Z THE FIRST DIMENSION ACTUALLY ????, probably does not change anything
    if heightmap is not None:
        if triangles.shape[-1] == 3:
            b = triangles[:, :, :-1]  # get rid of z before adding it again
        else:
            b=triangles
        search_coords = (b[..., 0].ravel(), b[..., 1].ravel())  # 'np.where' like coords
        heights = heightmap[search_coords]

        # print('heights', heights) # en fait c'est bon car des triangles sont out -> modifie l'aire

        heights = np.reshape(heights, (*b.shape[0:-1], (1)))
        triangles = np.append(b, heights, -1)  # add the height column to the coords

    # compute a, b and c bond lengths for 3D triangles
    a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 + (
            triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2 +
         (triangles[:, 0, 2] - triangles[:, 1, 2]) ** 2) ** 0.5
    b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 + (
            triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2 +
         (triangles[:, 1, 2] - triangles[:, 2, 2]) ** 2) ** 0.5
    c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 + (
            triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2 +
         (triangles[:, 2, 2] - triangles[:, 0, 2]) ** 2) ** 0.5



    # compute the triangle area using Heron's formula
    # https://en.wikipedia.org/wiki/Triangle
    s = (a + b + c) / 2.0  # semi perimeter (half of the triangle's perimeter)
    areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5

    return areas.sum()



# def compute3D_curved_distance(point1, point2, heightmap=None, three_points_mode=False):
#     if heightmap is None:
#         # cannot compute 3D dist
#         print('no valid height map --> cannot compute curved distance')
#         return distance_between_points(point1, point2)
#
#     pt1_y = int(point1[0])
#     pt1_x = int(point1[1])
#     # pt1_z = point1[2]
#     pt2_y = int(point2[0])
#     pt2_x = int(point2[1])
#     # pt2_z = point1[2]
#     rr, cc = draw.line(pt1_y, pt1_x,
#                        pt2_y, pt2_x)
#     # for every point I need get the height of it and compute the real 3D distance --> not hard in fact I think
#
#     dist3D = 0
#
#     # print(rr, cc,'-->')
#
#     if three_points_mode:
#         # dirty hack to speed up things but ok to find the stuff then really recompute the real distance --> should be ok
#         rr = [rr[0], rr[int(len(rr) / 2.)], rr[len(rr) - 1]]
#         cc = [cc[0], cc[int(len(cc) / 2.)], cc[len(cc) - 1]]
#
#     # print(rr,cc)
#     # compute distance of current point with prev one --> TODO
#     for iii in range(1, len(rr), 1):
#         y1 = rr[iii - 1]
#         x1 = cc[iii - 1]
#         z1 = heightmap[y1, x1]
#         y2 = rr[iii]
#         x2 = cc[iii]
#         z2 = heightmap[y2, x2]
#         # print(y1, x1, 'height', z1)
#         # print(y2, x2, 'height', z2)
#
#         # print(y1, x1, z1, y2, x2, z2)
#         # print('--> ', distance3D(y1, x1, z1, y2, x2, z2))
#         # NB POINTS NEED BE SORTED --> CHECK THAT THIS IS THE CASE --> EASY MAX DIST BETWEEN CONSECUTIVE 2D POINTS CAN ONLY BE SQRT 2
#
#         dist3D += distance3D(y1, x1, z1, y2, x2, z2)
#     # print(dist3D)
#     return dist3D

# TODO --> do that --> need sorted stuff
# if points are not sorted then need pavlidis sort them
# do pretty much the same as in the other
def perimeter_3D(points, heightmap,points_are_sorted=False,is_closed_contour=False):
    if not points_are_sorted:
        sorted_points =pavlidis(points, closed_contour=is_closed_contour)
    else:
        sorted_points = points

    if not isinstance(sorted_points, np.ndarray):
        sorted_points = np.asarray(sorted_points)

    # NOW NEED GET HEIGHT FOR ALL POINTS --> SHOULD BE DOABLE

    search_coords = (sorted_points[..., 0].ravel(), sorted_points[..., 1].ravel())  # 'np.where' like coords
    # print('search_coords',search_coords)
    heights = heightmap[search_coords]


    # print(sorted_points.shape)
    # print(heights.shape)
    # print(heights)
    heights=heights[..., np.newaxis]
    # sorted_points = np.append(sorted_points, heights, -1) # add the heights dimension to the sorted points
    sorted_points = np.append(heights,sorted_points, -1) # add the heights dimension to the sorted points
    # sorted_points = np.hstack((heights, sorted_points))  # add the heights dimension to the sorted points

    # print('sorted_points',sorted_points)
    consecutive_distance = compute_distance_between_consecutive_points(sorted_points)

    # print(consecutive_distance)



    perimeter3D = np.sum(consecutive_distance)


    # then compare to the pixel by pixel method of it and should be the same result in fact
    # then for every point sum the distance from one point to the next...



    # also find a way to order pixels and get the perimeter right
    # si > sqrt 2 alors need a connection --> put this as a section and try connect to first or last of every list --> maybe not so hard to implement and in the end all the points should be connected but try on a real example !!!
    # TODO
    # do it both for 2D and 3D
    # the ordering of the pixels is key --> really need do that
    # then I'll have a fairly good clone of TA that would be great and fully open source!!!
    # raise NotImplementedError


    return perimeter3D


if __name__ == '__main__':

        if False:
            perimeter_3D()
            import sys
            sys.exit(0)

        # it all seems to work but now just compute the area in a smarter way

        # triangles3D= [[[2.37749672, -0.99445809,0], [1.20416474, -0.9998551,1], [2.87830002, -0.99819343,2]],[[2.37749672, -0.99445809,0], [1.20416474, -0.9998551,0], [2.87830002, -0.99819343,0]]]
        # the second triangle is a line --> so area is 0, but 1 is the same but 3D --> in that sense it has an area
        triangles3D = [[[2, 1, 0], [1, 1, 1], [3, 1, 2]], [[2, 1, 0], [1, 1, 0], [3, 1, 0]],
                       [[2, 1, 0], [1, 1, 0], [3, 2, 0]], [[2, 1, 0], [1, 1, 0], [2, 2, 0]],
                       [[0, 0, 0], [3, 0, 0], [3, 3, 0]]]  # why 0 for the second triangle --> do I have a bug ???

        triangles2D = [[[2, 1], [1, 1], [3, 1]], [[2, 1], [1, 1], [3, 1]],
                       [[2, 1], [1, 1], [3, 2]], [[2, 1], [1, 1], [2, 2]],
                       [[0, 0], [3, 0], [3, 3]]]  # why 0 for the second triangle -

        heightmap_flat = np.asarray([[0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0]])

        # print(heightmap_flat)

        heightmap_graded = np.asarray([[1, 2, 3, 4, 5],
                                       [6, 7, 8, 9, 10],
                                       [11, 12, 14, 15, 16],
                                       [17, 18, 19, 20, 21]])

        # print(heightmap_graded)

        # triangles2D= [[[2.37749672, -0.99445809], [1.20416474, -0.9998551], [2.87830002, -0.99819343]],[[2.37749672, -0.99445809], [1.20416474, -0.9998551], [2.87830002, -0.99819343]]]
        # if heightmap is provided --> get the coords from heightmap otherwise assume what is passed is already 3D
        # check and compare 2D and 3D
        # check also no dupes in the triangles!!!
        print(compute_3D_surfacearea(triangles3D, heightmap=None))  # heightmap_graded # None # heightmap_flat)z
        print(compute_3D_surfacearea(triangles2D, heightmap=None))
        print(compute_3D_surfacearea(triangles2D, heightmap=None,stop_on_2D=True))

        # if height map is provided then take it
