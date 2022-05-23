from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, generate_binary_structure
from scipy.signal import convolve2d
from skimage.color import gray2rgb
from skimage.feature import peak_local_max
from skimage.segmentation import quickshift, watershed, find_boundaries
from skimage.morphology import remove_small_objects
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from epyseg.img import Img


def get_optimized_mask2(img, sauvola_mask=None, use_quick_shift=False, __VISUAL_DEBUG=False,
                        __DEBUG=False, score_before_adding=False, return_seeds=False):

    final_image = None
    if use_quick_shift:
        rotations = [0, 2, 3]
        kernels = [1, 1.33, 1.66, 2]

        for kern in kernels:
            for rot in rotations:
                segments_quick = getQuickseg(img, nb_of_90_rotation=rot, kernel_size=kern)
                segments_quick = segments_quick.astype(np.uint8)
                if final_image is None:
                    final_image = segments_quick
                else:
                    final_image = final_image + segments_quick

        if __VISUAL_DEBUG:
            plt.imshow(final_image)
            plt.title("avg")
            plt.show()

        final_image[final_image < final_image.max()] = 0
        final_image[final_image >= final_image.max()] = 1

    if sauvola_mask is None:
        from epyseg.postprocess.edmshed import sauvola
        t = sauvola(img, min_threshold=0.1, window_size=25)

        img[img >= t] = 1
        img[img < t] = 0
    else:
        img = sauvola_mask
        img[img > 0] = 1

    if __DEBUG:
        Img(img, dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/sauvola_mask.tif')

    raw_sauvola = img.copy()

    if use_quick_shift:
        final_image[raw_sauvola != 0] = 1
        if __VISUAL_DEBUG:
            plt.imshow(final_image)
            plt.show()

        if __DEBUG:
            Img(final_image.astype(np.uint8) * 255, dimensions='hw').save(
                '/home/aigouy/Bureau/trash/trash4/corrected_stuff.tif')

        final_image = ~remove_small_objects(~final_image.astype(bool), min_size=5, connectivity=1)
        final_image = skeletonize(final_image)

        if __DEBUG:
            Img(final_image.astype(np.uint8), dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/skel_quick.tif')
        if __VISUAL_DEBUG:
            plt.imshow(final_image)
            plt.title("binary")
            plt.show()

        vertices_quick, cut_bonds_quick = split_into_vertices_and_bonds(final_image)
        if __DEBUG:
            Img(vertices_quick.astype(np.uint8), dimensions='hw').save(
                '/home/aigouy/Bureau/trash/trash4/vertices_quick.tif')
            Img(cut_bonds_quick.astype(np.uint8), dimensions='hw').save(
                '/home/aigouy/Bureau/trash/trash4/cut_bonds_quick.tif')

    img = skeletonize(img)
    if __DEBUG:
        Img(img, dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/skel_sauvola_mask.tif')

    img = remove_small_objects(img, min_size=6, connectivity=2)
    if __DEBUG:
        Img(img, dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/skel_sauvola_mask_deblobed.tif')

    if __VISUAL_DEBUG:
        plt.imshow(img)
        plt.show()

    image = img.copy()
    image = image.astype(np.uint8) * 255
    image = Img.invert(image)

    distance = distance_transform_edt(image)

    if __VISUAL_DEBUG:
        plt.imshow(distance)
        plt.show()

    # local_maxi = peak_local_max(distance, indices=False, # old code changed due to deprecation
    #                             footprint=np.ones((8, 8)),
    #                             labels=image)
    tmp = peak_local_max(distance,
                         # indices=False, #change due to deprecation:  The indices argument in skimage.feature.peak_local_max has been deprecated. Indices will always be returned. (#4752)
                         footprint=np.ones((8, 8)),
                         labels=image)
    local_maxi = np.zeros_like(distance, dtype=bool)
    local_maxi[tuple(tmp.T)] = True

    distance = -distance

    markers = ndimage.label(local_maxi, structure=generate_binary_structure(2, 2))[0]

    if __VISUAL_DEBUG:
        plt.imshow(markers)
        plt.show()

    if __DEBUG:
        Img(markers, dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/markers_0.tif')

    labels = watershed(distance, markers, watershed_line=True)  # --> maybe implement that too
    labels[labels != 0] = 1
    labels[labels == 0] = 255
    labels[labels == 1] = 0
    labels[labels == 255] = 1

    if __VISUAL_DEBUG:
        plt.imshow(labels)
        plt.title('raw wshed')
        plt.show()

    if __DEBUG:
        Img(labels.astype(np.uint8), dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/wshed_edm.tif')

    vertices_edm, cut_bonds_edm = split_into_vertices_and_bonds(labels)
    if __DEBUG:
        Img(vertices_edm.astype(np.uint8), dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/vertices_edm.tif')
        Img(cut_bonds_edm.astype(np.uint8), dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/cut_bonds_edm.tif')

    if use_quick_shift:

        if __DEBUG:
            Img(img, dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/skel_sauvola_mask_deblobed.tif')
        unconnected = detect_unconnected_bonds(img)

        if __DEBUG:
            Img(unconnected, dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/unconnected.tif')

        labels_quick = label(cut_bonds_quick, connectivity=2, background=0)
        labels_quick_vertices = label(vertices_quick, connectivity=2, background=0)
        props_labels_quick = regionprops(labels_quick)

        labels_pred = label(unconnected, connectivity=2, background=0)

        raw_sauvola = rescue_bonds(labels_pred, labels_quick, raw_sauvola, labels_quick_vertices,
                                       props_labels_quick, score_before_adding=score_before_adding)

        if __DEBUG:
            Img(raw_sauvola.astype(np.uint8), dimensions='hw').save(
                '/home/aigouy/Bureau/trash/trash4/corrected_bonds_sauvola.tif')

    labels_edm = label(cut_bonds_edm, connectivity=2, background=0)
    labels_edm_vertices = label(vertices_edm, connectivity=2, background=0)
    props_labels_edm = regionprops(labels_edm)

    img = skeletonize(raw_sauvola)
    if __DEBUG:
        Img(raw_sauvola, dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/skel_sauvola_mask2.tif')
    unconnected = detect_unconnected_bonds(img)

    if __DEBUG:
        Img(unconnected, dimensions='hw').save('/home/aigouy/Bureau/trash/trash4/unconnected.tif')

    labels_pred = label(unconnected, connectivity=2, background=0)

    raw_sauvola = rescue_bonds(labels_pred, labels_edm, raw_sauvola, labels_edm_vertices, props_labels_edm,
                                   score_before_adding=score_before_adding)

    raw_sauvola = connect_unconnected(labels_pred, labels_edm, raw_sauvola, props_labels_edm,
                                          labels_edm_vertices)

    if return_seeds:
        return markers, labels, raw_sauvola

    return raw_sauvola


def split_into_vertices_and_bonds(skel):
    kernel = np.ones((3, 3))
    mask = convolve2d(skel, kernel, mode='same', fillvalue=1)

    mask[mask < 4] = 0
    mask[mask >= 4] = 1  # vertices

    mask = np.logical_and(mask, skel).astype(np.uint8)

    # bonds without vertices
    bonds_without_vertices = skel - mask

    return mask, bonds_without_vertices

def connect_unconnected(labels_pred, labels_quick, raw_pahansalkar, props_labels_quick, labels_quick_vertices):
    final_bonds_to_restore_because_they_connect_unconnected = []
    for region in regionprops(labels_pred):
        ids = []
        for coordinates in region.coords:
            for i in range(-2, 3, 1):
                for j in range(-2, 3, 1):
                    try:
                        if labels_quick[coordinates[0] + i, coordinates[1] + j] != 0:
                            ids.append(labels_quick[coordinates[0] + i, coordinates[1] + j])
                    except:
                        pass
        if ids:
            ids = list(dict.fromkeys(ids))
            for id in ids:
                final_bonds_to_restore_because_they_connect_unconnected.append(id)

    final_bonds_to_restore_because_they_connect_unconnected = Counter(
        final_bonds_to_restore_because_they_connect_unconnected)

    id_of_vertices_to_restore = []

    for key, count in final_bonds_to_restore_because_they_connect_unconnected.items():
        if count == 2:
            # we are treating the vx and really reconnecting it then we remove it
            for coordinates in props_labels_quick[key - 1].coords:
                raw_pahansalkar[coordinates[0], coordinates[1]] = 1
                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        try:
                            if labels_quick_vertices[coordinates[0] + i, coordinates[1] + j] != 0:
                                id_of_vertices_to_restore.append(
                                    labels_quick_vertices[coordinates[0] + i, coordinates[1] + j])
                        except:
                            pass
    if id_of_vertices_to_restore:
        id_of_vertices_to_restore = list(dict.fromkeys(id_of_vertices_to_restore))
        props_labels_vertices_quick = regionprops(labels_quick_vertices)
        for id in id_of_vertices_to_restore:
            for coordinates in props_labels_vertices_quick[id - 1].coords:
                raw_pahansalkar[coordinates[0], coordinates[1]] = 1
    return raw_pahansalkar

def rescue_bonds(labels_pred, labels_quick, raw_pahansalkar, labels_quick_vertices, props_labels_quick,
                 score_before_adding=False):
    minimum_scores = 0.5
    minimal_area_for_scoring = 4
    restore_vertices = True

    bonds_to_rescue = {}
    id_of_vertices_to_restore = [] # there was a bug here --> variable was local

    for region in regionprops(labels_pred):
        ids = []
        last_pos = None



        for coordinates in region.coords:
            for i in range(-2, 3, 1):
                for j in range(-2, 3, 1):
                    try:
                        if i == 0 and j == 0:
                            last_pos = (coordinates[0] + i, coordinates[1] + j)
                        if labels_quick[coordinates[0] + i, coordinates[1] + j] != 0:
                            ids.append(labels_quick[coordinates[0] + i, coordinates[1] + j])

                        if restore_vertices:
                            if labels_quick_vertices[coordinates[0] + i, coordinates[1] + j] != 0:
                                id_of_vertices_to_restore.append(
                                    labels_quick_vertices[coordinates[0] + i, coordinates[1] + j])
                    except:
                        pass

        if ids:
            ids = list(dict.fromkeys(ids))
            bonds_to_rescue[last_pos] = ids

    bonds_to_rescue_to_remove = []

    # redo connected or unconnected
    for key, ids in bonds_to_rescue.items():
        for id in ids:
            if score_before_adding and props_labels_quick[id - 1].area > minimal_area_for_scoring:
                count = 0
                total = 0
                for coordinates in props_labels_quick[id - 1].coords:
                    total += 1
                    if raw_pahansalkar[coordinates[0], coordinates[1]] != 0:
                        count += 1
                if total != 0:
                    score = count / total
                    if score < minimum_scores:
                        bonds_to_rescue_to_remove.append(id)
                        continue

            # we are treating the vx and really reconnecting it then we remove it
            labels_pred[key[0], key[1]] = 0

            for coordinates in props_labels_quick[id - 1].coords:
                raw_pahansalkar[coordinates[0], coordinates[1]] = 1
                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        try:
                            if labels_quick_vertices[coordinates[0] + i, coordinates[1] + j] != 0:
                                id_of_vertices_to_restore.append(
                                    labels_quick_vertices[coordinates[0] + i, coordinates[1] + j])
                        except:
                            pass
    # ça marche mais faut aussi restaurer les vertices de ces bonds --> à refaire --> pas trop dur je pense --> faire pareil pr autre --> recup
    if id_of_vertices_to_restore:
        id_of_vertices_to_restore = list(dict.fromkeys(id_of_vertices_to_restore))
        props_labels_vertices_quick = regionprops(labels_quick_vertices)
        for id in id_of_vertices_to_restore:
            for coordinates in props_labels_vertices_quick[id - 1].coords:
                raw_pahansalkar[coordinates[0], coordinates[1]] = 1

    return raw_pahansalkar


def detect_unconnected_bonds(skel):
    kernel = np.ones((3, 3))
    mask = convolve2d(skel, kernel, mode='same', fillvalue=1)

    mask[mask == 0] = 255
    mask[mask > 2] = 0
    mask[mask == 2] = 1

    mask = np.logical_and(skel, mask).astype(np.uint8)
    return mask

def getQuickseg(img, nb_of_90_rotation=0, kernel_size=2):
    if nb_of_90_rotation == 0:
        return find_boundaries(quickshift(gray2rgb(img), kernel_size=kernel_size, max_dist=6, ratio=3))
    else:
        return np.rot90(find_boundaries(
            quickshift(gray2rgb(np.rot90(img, nb_of_90_rotation)), kernel_size=kernel_size, max_dist=6, ratio=3)),
            4 - nb_of_90_rotation)


if __name__ == '__main__':
    from timeit import default_timer as timer

    start = timer()

    #
    # image = Img('/D/final_folder_scoring/20190924_ecadGFP_400nM20E_000.tif')
    #
    # print(image.shape)
    # print(image.has_c())

    # img = Img('D:/Dropbox/mini_test.tif').astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/5.tif')[...,0].astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/122.tif')[...,0].astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/11.tif')[...,0].astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/cellpose_img22.tif')[...,0].astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/cellpose_img22.tif')[...,5].astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/cellpose_img22.tif')[...,1].astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/cellpose_img22.tif')[...,3].astype(float)
    # img = Img.invert(img)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/focused_Series010.tif')[...,0].astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/focused_Series194.tif')[...,0].astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/Series019.tif')[...,0].astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/image_plant_best-zoomed.tif')[...,0].astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/100708_png06.tif')[...,0].astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/MAX_160610_test_ocelli_ok_but_useless_cause_differs_a_lot_from_ommatidia.lif - test_visualization_head_ommatidia_32h_APF_ok_2.tif')[...,0].astype(float)
    # img = Img('D:/Dropbox/stuff_for_the_new_figure/old/predict_avg_hq_correction_ensemble_wshed/proj0016.tif')[...,0].astype(float)

    # img = Img('D:/Dropbox/mini_test.tif').astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/5.tif')[...,0].astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/5.tif')[...,1].astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/5.tif')[...,2].astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/122.tif')[...,0].astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/11.tif')[...,0].astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/11.tif')[...,1].astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/11.tif')[...,2].astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/cellpose_img22.tif')[...,0].astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/cellpose_img22_bg_subtracted_ij.tif')[...,0].astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/focused_Series010.tif')[...,0].astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/focused_Series194.tif')[..., 0].astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/focused_Series194.tif')[..., 1].astype(float)
    # img = Img('/D/final_folder_scoring/predict_avg_hq_correction_ensemble_wshed/focused_Series194.tif')[..., 2].astype(float)

    img = Img('/D/final_folder_scoring/predict/11.tif')[..., 0]
    raw_sauvola = get_optimized_mask2(img, __VISUAL_DEBUG=True, __DEBUG=True)
    Img(raw_sauvola.astype(np.uint8), dimensions='hw').save(
        '/home/aigouy/Bureau/trash/trash4/corrected_bonds_sauvola2.tif')

    print('total time', timer() - start)
