import traceback
from scipy import ndimage
from skimage.filters import threshold_otsu
# from skimage.segmentation import watershed 
from skimage.segmentation import watershed
from epyseg.img import Img
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from timeit import default_timer as timer
import os
import numpy as np
import statistics
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()

def simpleFilter(img_orig, threshold=None, **kwargs): # threshold=None is autofilter
    if img_orig.has_c():
        for c in range(img_orig.shape[-1]):
            if threshold is not None:
                current = img_orig[...,c]
                current = current/current.max()
                current[current<=threshold]=0
                current[current>threshold]=255
                img_orig[...,c]=current
            else:
                current = img_orig[..., c]
                cur_threshold = threshold_otsu(current)
                current[current <= cur_threshold] = 0
                current[current > cur_threshold] = 255
                img_orig[..., c] = current
        return img_orig.astype(np.uint8)
    else:
        if threshold is not None:
            img_orig[img_orig <= threshold] = 0
            img_orig[img_orig > threshold] = 255
            return img_orig.astype(np.uint8)
        else:
            cur_threshold = threshold_otsu(img_orig)
            img_orig[img_orig <= cur_threshold] = 0
            img_orig[img_orig > cur_threshold] = 255
            return img_orig.astype(np.uint8)

def FilterMask(img_orig, final_wshed, filter='local median',  correction_factor=2,
             _DEBUG=False,
             _VISUAL_DEBUG=False,
             **kwargs):

    labs = label(Img.invert(final_wshed.astype(np.uint8)), connectivity=1, background=0)

    start = timer()

    output_folder = '/home/aigouy/Bureau/trash/test_new_seeds_seg_stuff/'
    if filter is None or filter == 0:
        return final_wshed.astype(np.uint8)
    else:
        if isinstance(filter, int):
            filter_by_size = filter
        else:
            filter_by_size = None
        avg_area = 0
        count = 0
        if _DEBUG:
            Img(final_wshed, dimensions='hw').save(os.path.join(output_folder, 'extras', 'test_size_cells.tif'))

        final_seeds = Img.invert(final_wshed)
        final_seeds = label(final_seeds, connectivity=1, background=0)

        if _VISUAL_DEBUG:
            plt.imshow(final_seeds)
            plt.show()

        removed_seeds = []
        keep_seeds = []

        labels_n_bbox = {}
        labels_n_area = {}
        border_cells = []
        ids_n_local_median = {}

        if isinstance(filter, str) and 'local' in filter:
            rps = regionprops(final_seeds)

            for region in rps:
                labels_n_bbox[region.label] = region.bbox
                labels_n_area[region.label] = region.area
                if (region.bbox[0] <= 3 or region.bbox[1] <= 3 or region.bbox[2] >= final_seeds.shape[-2] - 5 or
                        region.bbox[
                            3] >= \
                        final_seeds.shape[-1] - 5):
                    border_cells.append(region.label)

            _, tiles = Img.get_2D_tiles_with_overlap(final_seeds, overlap=64, dimension_h=-2, dimension_w=-1)

            for r in tiles:
                for tile in r:
                    rps2 = regionprops(tile)
                    for region in rps2:
                        if region.label in border_cells:
                            continue

                        if (region.bbox[0] <= 3 or region.bbox[1] <= 3 or region.bbox[2] >= final_seeds.shape[
                            -2] - 5 or
                                region.bbox[
                                    3] >= \
                                final_seeds.shape[-1] - 5):
                            continue

                        area_of_neighboring_cells = []
                        for region2 in rps2:
                            if region2.label == region.label:
                                continue
                            # find all cells with
                            if rect_distance(region.bbox, region2.bbox) <= 1:
                                area_of_neighboring_cells.append(labels_n_area[region2.label])

                        if area_of_neighboring_cells:
                            median = statistics.median_low(area_of_neighboring_cells)
                            ids_n_local_median[
                                region.label] = median / correction_factor
                            if region.area <= median / correction_factor:
                                removed_seeds.append(region.label)
                            else:
                                keep_seeds.append(region.label)
            removed_seeds = [x for x in removed_seeds if x not in keep_seeds]

            # TODO offer the things below as an option --> prevent removal of sure seeds or something like that
        else:
            areas = []

            for region in regionprops(final_seeds):
                if (region.bbox[0] <= 3 or region.bbox[1] <= 3 or region.bbox[2] >= final_seeds.shape[-2] - 5 or
                        region.bbox[3] >= final_seeds.shape[-1] - 5):
                    continue
                avg_area += region.area
                count += 1
                areas.append(region.area)
            avg_area /= count

            median = statistics.median_low(areas)

            if isinstance(filter, int):
                filter_by_size = filter
            elif 'avg' in filter:
                filter_by_size = avg_area / correction_factor
            elif 'median' in filter:
                filter_by_size = median / correction_factor
            # TODO maybe use stdev or alike to see if cell should really be removed
            if _DEBUG:
                print('filter cells below=', filter_by_size, 'avg cell area=', avg_area, 'median=',
                      median)  # , 'median', median

            if filter_by_size is not None and filter_by_size != 0:

                if _VISUAL_DEBUG:
                    plt.imshow(final_seeds)
                    plt.show()

                for region in regionprops(final_seeds):
                    labels_n_bbox[region.label] = region.bbox
                    labels_n_area[region.label] = region.area
                    if region.area < filter_by_size:
                        if (region.bbox[0] <= 2 or region.bbox[1] <= 2 or region.bbox[2] >= labs.shape[
                            -2] - 3 or
                                region.bbox[
                                    3] >= \
                                labs.shape[
                                    -1] - 3):
                            continue
                        removed_seeds.append(region.label)

        if _VISUAL_DEBUG:
            plt.imshow(final_seeds)
            plt.show()

        for region in regionprops(final_seeds):
            if region.label in removed_seeds:
                for coordinates in region.coords:
                    final_seeds[coordinates[0], coordinates[1]] = 0
        if _VISUAL_DEBUG:
            plt.imshow(final_seeds)
            plt.show()

        final_wshed = watershed(img_orig, markers=final_seeds, watershed_line=True)

        final_wshed[final_wshed != 0] = 1  # remove all seeds
        final_wshed[final_wshed == 0] = 255  # set wshed values to 255
        final_wshed[final_wshed == 1] = 0  # set all other cell content to
        if _VISUAL_DEBUG:
            plt.imshow(final_wshed)
            plt.show()

        duration = timer() - start
        if _DEBUG:
            print('final duration wshed in secs', duration)

        return final_wshed.astype(np.uint8)

def rect_distance(bbox1, bbox2):
    width1 = abs(bbox1[3] - bbox1[1])
    width2 = abs(bbox2[3] - bbox2[1])
    height1 = abs(bbox1[2] - bbox1[0])
    height2 = abs(bbox2[2] - bbox2[0])
    return max(abs((bbox1[1] + width1 / 2) - (bbox2[1] + width2 / 2)) - (width1 + width2) / 2,
               abs((bbox1[0] + height1 / 2) - (bbox2[0] + height2 / 2)) - (height1 + height2) / 2)


if __name__ == '__main__':
    pass
