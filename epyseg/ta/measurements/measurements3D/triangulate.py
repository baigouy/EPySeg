# this class will be used to triangulate a cell in order to compute its 3D properties such as area
import random

from scipy.spatial import Delaunay
from skimage.draw import polygon_perimeter

from epyseg.img import Img, RGB_to_int24

from epyseg.ta.measurements.measurements3D.get_point_on_surface_if_centroid_is_bad import point_on_surface
from epyseg.ta.measurements.measurements3D.measures3D import compute_3D_surfacearea
from epyseg.ta.tracking.rapid_detection_of_vertices import detect_vertices_and_bonds, associate_vertices_to_cells
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage import measure
import traceback


# print(unprocessed_clone.shape)
# labels = label(RGB_to_int24(unprocessed_clone), connectivity=1, background=0xFFFFFF)
# vertices = detect_vertices_and_bonds(labels)

# check what I really need and add it there
# instead of region I could have the coords of the image

# is there a way to make it generic for cells and for clones --> think about it --> probably yes by just using the coords
# maybe check there is a single ID below
#
# coords_etdited = region.coords.tolist()
# coords_etdited.append(region.centroid)
#
# verts = np.where((vertices == 255) & (
#             clone_id == cell.label))  # if image has too many labels then use that if there are several clones inside the stuff
#
# tmpsss = np.zeros_like(labels)
# tmpsss[verts] = 255
# plt.imshow(tmpsss)
# plt.show()
#
# verts = list(zip(verts[0], verts[1]))
# # print(np.where(vertices == 255))
# coords_etdited.extend(verts)
# # coords_etdited = verts
#
# coords_etdited = np.asarray(coords_etdited)

# this is for filtering of the coords --> I could keep a light version of this in fact --> just make it generic

# from shapely.geometry import Polygon
#
# for triangle in triangles:
#     centroid_y, centroid_x = sum(triangle) / len(triangle)
#
#     # print('me', centroid_y, centroid_x)
#     if clone_id[int(centroid_y), int(centroid_x)] == cell.label:
#         poly = Polygon(triangle)
#
#         plt.plot(*poly.exterior.xy)
#
#         # parfait --> maintenant voir comment faire ça mieux et si je peux utiliser shapely ou pas ???
#
# # figure()
# # axis('equal')
# # plot(coords_etdited[:, 0], coords_etdited[:, 1], '.')
# # for i, j in edges:
# #     plot(coords_etdited[[i, j], 0], coords_etdited[[i, j], 1])
# # plt.triplot(coords_etdited[:, 0], coords_etdited[:, 1], tris)
# plt.show()

# labels, vertices,
def triangulate(coords_to_triangulate, triangulated_object_ID=None, labels_to_check=None):
    # TODO export all these triangulations so that they are more simply done
    # nb triangulation would be simpler if using just the vertices instead of the full contour --> maybe rely on vertices also for the edges and that is simple to do in a way
    # all this triangulation is only useful provided the image is 3D/has heightmap because otherwise the metrics are much simpler to handle
    # print('oubsi qsdsqdqsdqsd q')

    # this is slow here so it must be done once and much before in fact!!!

    # ça marche pas non plus faut vraiment plus de points

    # plt.imshow(cells2)
    # plt.show()

    # plt.imshow(vertices)
    # plt.show()
    # ça marche --> essayer d'ajouter des vertices

    tri = Delaunay(coords_to_triangulate)
    triangles = coords_to_triangulate[tri.vertices]

    # TODO also need compute the 3D area of the shape

    # print(triangles)

    # Polygon([(3.0, 0.0), (2.0, 0.0), (2.0, 0.75), (2.5, 0.75), (2.5, 0.6), (2.25, 0.6), (2.25, 0.2), (3.0, 0.2),
    #          (3.0, 0.0)])

    # ça marche par contre bug si inner hole --> faudrait faire qq chose de plus smart en fait
    # --> en fait devrait pas arriver --> en fait just warner l'utilisateur
    # sinon checker les triangles par rapport à l'image originale en fait c'est facile dans mon cas les points des interbonds doivent se trouver dans le clone --> pas trop dur en fait
    # mieux le centroid du triangle doit se trouver dans le clone et ça c'est vraimet super facile à faire et en plus j'ai pas besoin de truc compliques --> 100 % codable en python et TA compat à 100%

    # indices_to_delete

    # TODO check this code and verify how generic it is!!!
    if labels_to_check is not None and triangulated_object_ID is not None:
        # print('entering delete phase', triangles.shape)
        indices_to_delete = []
        for iii, triangle in enumerate(triangles):
            centroid_y, centroid_x = sum(triangle) / len(triangle)

            # print('me', centroid_y, centroid_x)
            # can I do it more smartly ??? using another ID
            if labels_to_check[int(centroid_y), int(centroid_x)] != triangulated_object_ID:
                # poly = Polygon(triangle)

                # plt.plot(*poly.exterior.xy)

                # parfait --> maintenant voir comment faire ça mieux et si je peux utiliser shapely ou pas ???

                # can I delete it
                indices_to_delete.append(iii)

        # print('indices_to_delete',indices_to_delete) # --> ok --> seems to work
        # ça marche bien ça --> à réutiliser!!!
        triangles = np.delete(triangles, indices_to_delete, axis=0)

        # print('after delete', triangles.shape)
    # check if that work

    return triangles


# if image is not plot as shapely if not then plot on the image --> very good idea
# TODO allow maybe fill and chose a color or generate random colors as in TA !!! --> would be a very good idea and that would make my life easier!!!
def plot_triangulation(triangles, img_to_plot_triangles_over=None):
    if img_to_plot_triangles_over is None:
        from shapely.geometry import Polygon
        for triangle in triangles:
            poly = Polygon(triangle)
            plt.plot(*poly.exterior.xy)
            plt.axis('equal')
        plt.show()
    else:
        for triangle in triangles:
            r = triangle[:, 0]  # get the y coords of the simplex
            c = triangle[:, 1]  # get the x coords of the simplex

            # print(np.asarray(r))
            # print(c)

            # TODO maybe plot the poygon and its contour too
            rr, cc = polygon_perimeter(r, c, shape=img_to_plot_triangles_over.shape)

            # print(rr, cc)
            #
            # print(type(rr), print(rr.dtype), rr.shape)
            # print(rr)
            img_to_plot_triangles_over[rr, cc] = random.randint(128,255) # not bad
        return img_to_plot_triangles_over

if __name__ == '__main__':


    # everything is fine --> just need finalize the thing and also measure the area and things like that
    # also find a way to order pixels and get the perimeter right
    # si > sqrt 2 alors need a connection --> put this as a section and try connect to first or last of every list --> maybe not so hard to implement and in the end all the points should be connected but try on a real example !!!


    # TODO !!!

    # do several tests
    # do tests with classical cells and with clones just to see how it behaves
    if True:
        # cell triangulation test
        # get any image with cells get vertices for cells
        # get centroid if inside the cell or a random point
        # launch the triangulation and display it

        CHECK_CONCAVE_CELLS_TRIANGULATION = True # maybe not really necessary for epithelial cells but really key for clones

        cells = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/tracked_cells_resized.png')
        cells = RGB_to_int24(cells)
        vertices = detect_vertices_and_bonds(cells)
        vertices = np.where(vertices == 255)
        t0_cells_n_vertices = associate_vertices_to_cells(cells, vertices, forbidden_colors=(0, 0xFFFFFF))
        # now for every cell do the triangulation
        # --> TODO
        labels = measure.label(cells, connectivity=1, background=0xFFFFFF)  # FOUR_CONNECTED
        # plt.imshow(labels)
        # plt.show()
        counter = 0
        out = np.zeros_like(cells)

        for region in regionprops(labels):
            # region.bbox
            # counter += 1
            # if counter == 60:
                cell_ID = cells[region.coords[0][0], region.coords[0][1]]
                # region.coords
                # print(type(np.asarray(t0_cells_n_vertices[cell_ID])))
                # print(type(region.coords))

                try:

                    points_for_triangulation = t0_cells_n_vertices[cell_ID] # why not all the cells are also entered in there --> do I have a bug and if so where ???
                    # if centroid is in the cell --> add it to the cells and that would be ok!!! otherwise take a random point in the cell
                    points_for_triangulation.append(point_on_surface(region, labels))
                except:
                    traceback.print_exc()
                    print('no vertices detected for this cell! --> need fix the algorithm')
                    # --> indeed it fails at detecting cells at the edges!!! -> need fix that --> see how I was doing that in TA and do the same here
                    continue


                try:
                    if not CHECK_CONCAVE_CELLS_TRIANGULATION:
                        triangles = triangulate(np.asarray(points_for_triangulation))
                    else:
                        triangles = triangulate(np.asarray(points_for_triangulation),triangulated_object_ID=cell_ID, labels_to_check=cells)

                    plot_triangulation(triangles, out)
                except:
                    traceback.print_exc()
                    print('failed for cell', region.area,
                      region.bbox)  # area 64 but not enough vertices --> I probably have a bug in my vertices code --> TODO --> fix that --> most likely at the edges
                    # TODO fix the algo

                # print(triangles)
                # break
        plt.imshow(out)
        plt.show()



    if True:

        from personal.pyTA.clones.clone_fuser import get_clone_contour
        heightmap = None # but for a real case this must be done

        # triangulate a clone
        # clone = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/tracked_clone.tif')
        # clone = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/tracked_clone_003.png')
        # clone = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/tracked_clone.png')
        clone = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012/tracked_clone_001.png')
        # ça marche super --> mettre ce code de partout!!!

        clone_id, clone_contour = get_clone_contour(fused_clone=None, unprocessed_clone=clone,
                                                    return_clone_id=True)


        # print(clone.shape)
        # nb can I use that also for clones ???

        # both are ok
        if False:
            plt.imshow(clone_id)
            plt.show()

            plt.imshow(clone_contour)
            plt.show()

        clone = RGB_to_int24(clone)
        cell_labels = measure.label(clone, connectivity=1, background=0xFFFFFF)
        #
        vertices = detect_vertices_and_bonds(cell_labels)

        # ce truc est ok
        if False:
            plt.imshow(vertices)
            plt.show()

        # why no cells here --> see what the pb is I must have a bug in my code somewhere though

        # du coup ça ne marche pas ici
        t0_cells_n_vertices = associate_vertices_to_cells(clone_id, vertices, forbidden_colors=(0, 0xFFFFFF))

        # print(t0_cells_n_vertices) # pourtant c'est bon --> j'ai une tonne de vx pr la cellule --> pkoi pas trouvée alors

        # get the clone and try to triangulate it, then I'll be done !!!


        # labels = measure.label(clone, connectivity=1, background=0xFFFFFF)  # FOUR_CONNECTED
        out = np.zeros_like(clone_id)

        for region in regionprops(clone_id):
                # region.bbox
                # counter += 1
                # if counter == 60:
                cell_ID = clone_id[region.coords[0][0], region.coords[0][1]]
                # region.coords
                # print(type(np.asarray(t0_cells_n_vertices[cell_ID])))
                # print(type(region.coords))

                try:

                    points_for_triangulation = t0_cells_n_vertices[
                        cell_ID]  # why not all the cells are also entered in there --> do I have a bug and if so where ???
                    # if centroid is in the cell --> add it to the cells and that would be ok!!! otherwise take a random point in the cell
                    # here that may even be smart to add the centoid of all cells contained in the clone --> TODO because smart
                    points_for_triangulation.append(point_on_surface(region, labels))
                except:
                    traceback.print_exc()
                    print('no vertices detected for this cell2! --> need fix the algorithm')
                    # --> indeed it fails at detecting cells at the edges!!! -> need fix that --> see how I was doing that in TA and do the same here
                    continue

                try:
                    # there is a bug in there dunno where
                    triangles = triangulate(np.asarray(points_for_triangulation), triangulated_object_ID=cell_ID, labels_to_check=clone_id) # need filter the triangulation for concave shapes of clones which is very frequent

                    # make it measure the 3D area of the clone --> TODO
                    print('area cell', cell_ID, compute_3D_surfacearea(triangles, heightmap=heightmap)) # --> ça marche enfin on dirait! --> cool

                    plot_triangulation(triangles, out)
                except:
                    traceback.print_exc()
                    print('failed for cell2', region.area,
                          region.bbox)  # area 64 but not enough vertices --> I probably have a bug in my vertices code --> TODO --> fix that --> most likely at the edges
                    # TODO fix the algo

                # print(triangles)
                # break
        plt.imshow(out)
        plt.show()

        # plt.imshow(labels)
        # plt.show()

        # tt marche --> plutot cool en fait


