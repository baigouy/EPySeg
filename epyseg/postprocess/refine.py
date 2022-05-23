import glob
import traceback
from scipy import ndimage
# from skimage.segmentation import watershed 
from skimage.segmentation import watershed
from epyseg.img import Img
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from timeit import default_timer as timer
import os
import numpy as np
from natsort import natsorted  # sort strings as humans would do
import statistics
from epyseg.tools.logger import TA_logger # logging

logger = TA_logger()

class EPySegPostProcess():
    stop_now = False

    def __init__(self):
        pass

    def process(self, input=None, output_folder=None, progress_callback=None, filter=None,
                 correction_factor=2,
                 cutoff_cell_fusion=None,
                 restore_safe_cells=False,
                 _DEBUG=False,
                 _VISUAL_DEBUG=False, **kwargs):

        start = timer()
        # filename0 = path
        # filename0_without_path = os.path.basename(filename0)
        # filename0_without_ext = os.path.splitext(filename0_without_path)[0]
        # parent_dir_of_filename0 = os.path.dirname(filename0)
        # TA_output_filename = os.path.join(parent_dir_of_filename0, filename0_without_ext,
        #                                   'handCorrection.tif')  # TODO allow custom names here to allow ensemble methods
        # non_TA_final_output_name = os.path.join(output_folder, filename0_without_ext + '.tif')
        #
        # filename_to_use_to_save = non_TA_final_output_name
        # if TA_mode:
        #     filename_to_use_to_save = TA_output_filename
        #
        # if TA_mode:
        #     # try also to change path input name
        #     if os.path.exists(
        #             os.path.join(parent_dir_of_filename0, filename0_without_ext, 'raw_epyseg_output.tif')):
        #         path = os.path.join(parent_dir_of_filename0, filename0_without_ext, 'raw_epyseg_output.tif')

        # img_orig = Img(path)
        # print('analyzing', path, self.stop_now)
        # try:
        #     if self.progress_callback is not None:
        #         self.progress_callback.emit((iii / len(list_of_files)) * 100)
        #     else:
        #         logger.info(str((iii / len(list_of_files)) * 100) + '%')
        # except:
        #     traceback.print_exc()
        #     pass

        # DO A DILATION OF SEEDS THEN AN EROSION TO JOIN CLOSE BY SEEDS

        img_orig = input

        img_has_seeds = True
        # mask with several channels
        if img_orig.has_c():
            if restore_safe_cells:
                img_seg = img_orig[..., 0].copy()

            seeds_1 = img_orig[..., img_orig.shape[-1] - 1]
            seeds_1 = Img.invert(seeds_1)
            # seeds_1[seeds_1 >= 0.5] = 255
            # seeds_1[seeds_1 < 0.5] = 0
            seeds_1[seeds_1 >= 0.2] = 255  # TODO maybe be more stringent here
            seeds_1[seeds_1 < 0.2] = 0

            s = ndimage.generate_binary_structure(2, 1)
            seeds_1 = ndimage.grey_dilation(seeds_1, footprint=s)
            seeds_1 = ndimage.grey_dilation(seeds_1, footprint=s)
            seeds_1 = ndimage.grey_dilation(seeds_1, footprint=s)
            seeds_1 = ndimage.grey_erosion(seeds_1, footprint=s)
            seeds_1 = ndimage.grey_erosion(seeds_1, footprint=s)
            # seeds_1 = ndimage.grey_erosion(seeds_1, footprint=s)
            # seeds_1 = ndimage.grey_erosion(seeds_1, footprint=s)

            # for debug
            if _DEBUG:
                Img(seeds_1, dimensions='hw').save(
                    os.path.join(output_folder, 'extras', 'wshed_seeds.tif'))  # not bad

            lab_seeds = label(seeds_1.astype(np.uint8), connectivity=2, background=0)
            #
            for region in regionprops(lab_seeds):
                if region.area < 10:
                    for coordinates in region.coords:
                        lab_seeds[coordinates[0], coordinates[1]] = 0

            if _DEBUG:
                Img(seeds_1, dimensions='hw').save(
                    os.path.join(output_folder, 'extras', 'wshed_seeds_deblobed.tif'))

            img_orig[..., 3] = Img.invert(img_orig[..., 3])
            img_orig[..., 4] = Img.invert(img_orig[..., 4])

            # seems to work --> now need to do the projection
            for c in range(1, img_orig.shape[-1] - 2):
                img_orig[..., 0] += img_orig[..., 1]

            img_orig[..., 0] /= img_orig.shape[-1] - 2
            img_orig = img_orig[..., 0]

        else:
            # mask with single channel
            img_has_seeds = False
            if restore_safe_cells:
                img_seg = img_orig.copy()

        if restore_safe_cells:
            if _DEBUG:
                print(os.path.join(output_folder, 'extras', 'img_seg.tif'))
                Img(img_seg, dimensions='hw').save(
                    os.path.join(output_folder, 'extras', 'img_seg.tif'))

        # for debug
        if _DEBUG:
            Img(img_orig, dimensions='hw').save(os.path.join(output_folder, 'extras', 'avg.tif'))

        img_saturated = img_orig.copy()
        if img_has_seeds:
            img_saturated[img_saturated >= 0.5] = 255
            img_saturated[img_saturated < 0.5] = 0
            if restore_safe_cells:
                # TODO maybe do a safe image
                img_seg[img_seg >= 0.3] = 255
                img_seg[img_seg < 0.3] = 0
                secure_mask = img_seg
        else:
            img_saturated[img_saturated >= 0.3] = 255
            img_saturated[img_saturated < 0.3] = 0
            if restore_safe_cells:
                img_seg[img_seg >= 0.95] = 255
                img_seg[img_seg < 0.95] = 0
                secure_mask = img_seg

        # convert it to seeds and make sure they are all present in there
        # if pixel is not labeled then read it
        if restore_safe_cells:
            labels_n_area_rescue_seeds = {}
            rescue_seeds = label(Img.invert(secure_mask), connectivity=1, background=0)
            for region in regionprops(rescue_seeds):
                labels_n_area_rescue_seeds[region.label] = region.area
            if _DEBUG:
                Img(secure_mask, dimensions='hw').save(os.path.join(output_folder, 'extras', 'secure_mask.tif'))
        # loop over those seeds to rescue

        # for debug
        if _DEBUG:
            Img(img_saturated, dimensions='hw').save(
                os.path.join(output_folder, 'extras', 'handCorrection.tif'))

        deblob = True
        if deblob:
            image_thresh = label(img_saturated, connectivity=2, background=0)
            # for debug
            if _DEBUG:
                Img(image_thresh, dimensions='hw').save(
                    os.path.join(output_folder, 'extras', 'before_deblobed.tif'))
            # deblob
            min_size = 200
            for region in regionprops(image_thresh):
                # take regions with large enough areas
                if region.area < min_size:
                    for coordinates in region.coords:
                        image_thresh[coordinates[0], coordinates[1]] = 0

            image_thresh[image_thresh > 0] = 255
            img_saturated = image_thresh
            # for debug
            if _DEBUG:
                Img(img_saturated, dimensions='hw').save(
                    os.path.join(output_folder, 'extras', 'deblobed.tif'))
            del image_thresh

        # for debug
        if _DEBUG:
            Img(img_saturated, dimensions='hw').save(
                os.path.join(output_folder, 'extras', 'deblobed_out.tif'))

        extra_dilations = True
        if extra_dilations:
            # do a dilation of 2 to close bonds
            s = ndimage.generate_binary_structure(2, 1)
            dilated = ndimage.grey_dilation(img_saturated, footprint=s)
            dilated = ndimage.grey_dilation(dilated, footprint=s)
            # Img(dilated, dimensions='hw').save(os.path.join(os.path.splitext(path)[0], 'filled_one_px_holes.tif'))

            # other_seeds = label(invert(np.grey_dilation(dilated, footprint=s).astype(np.uint8)), connectivity=1, background=0)

            labs = label(Img.invert(img_saturated.astype(np.uint8)), connectivity=1, background=0)
            for region in regionprops(labs):
                seeds = []

                # exclude tiny cells form dilation because they may end up completely closed
                if region.area >= 10 and region.area < 350:
                    for coordinates in region.coords:
                        dilated[coordinates[0], coordinates[1]] = 0
                    continue
                else:
                    # pb when big cells around cause connections are not done
                    # preserve cells at edges because they have to e naturally smaller because they are cut
                    # put a size criterion too
                    if region.area < 100 and (
                            region.bbox[0] <= 1 or region.bbox[1] <= 1 or region.bbox[2] >= labs.shape[-2] - 2 or
                            region.bbox[
                                3] >= \
                            labs.shape[-1] - 2):
                        # edge cell detected --> removing dilation
                        for coordinates in region.coords:
                            dilated[coordinates[0], coordinates[1]] = 0
                        continue

            img_saturated = dilated
            # for debug
            if _DEBUG:
                Img(img_saturated, dimensions='hw').save(
                    os.path.join(output_folder, 'extras', 'dilated_further.tif'))
            del dilated

        list_of_cells_to_dilate = []
        labs = label(Img.invert(img_saturated.astype(np.uint8)), connectivity=1, background=0)

        # c'est cette correction qui fixe bcp de choses mais recree aussi des choses qui n'existent pas... --> voir à quoi sont dus ces lignes blobs
        # faudrait redeblober
        if img_has_seeds:
            for region in regionprops(labs, intensity_image=img_orig):
                seeds = []

                if not extra_dilations and region.area < 10:
                    continue

                # if small and no associated seeds --> remove it ??? maybe or not
                for coordinates in region.coords:
                    id = lab_seeds[coordinates[0], coordinates[1]]
                    if id != 0:
                        seeds.append(id)

                seeds = set(seeds)

                if len(seeds) >= 2:
                    # we may have found an undersegmented cell --> try segment it better
                    list_of_cells_to_dilate.append(region.label)

        if len(list_of_cells_to_dilate) != 0:
            props = regionprops(labs, intensity_image=img_orig)
            for run in range(10):
                something_changed = False  # early stop

                for region in props:
                    if region.label not in list_of_cells_to_dilate:
                        continue

                    # TODO recheck those values and wether it makes sense
                    threshold_values = [80 / 255, 60 / 255, 40 / 255, 30 / 255,
                                        20 / 255,
                                        10 / 255]  # 160 / 255, 140 / 255, 120 / 255, 100 / 255,  1 / 255 , 2 / 255, , 5 / 255

                    try:
                        for threshold in threshold_values:
                            mask = region.image.copy()
                            image = region.image.copy()
                            image[region.intensity_image > threshold] = True
                            image[region.intensity_image <= threshold] = False
                            final = Img.invert(image.astype(np.uint8))
                            final[final < 255] = 0
                            final[mask == False] = 0
                            new_seeds = label(final, connectivity=1, background=0)
                            props2 = regionprops(new_seeds)
                            if len(props2) > 1:  # cell was resplitted into smaller
                                for r in props2:
                                    if r.area < 20:
                                        raise Exception

                                region.image[mask == False] = False
                                region.image[mask == True] = True
                                region.image[new_seeds > 0] = False
                                something_changed = True
                                for coordinates in region.coords:
                                    img_saturated[coordinates[0], coordinates[1]] = 255
                            region.image[mask == False] = False
                            region.image[mask == True] = True
                            del final
                            del new_seeds
                    except:
                        traceback.print_exc()
                        pass

                if not something_changed:
                    # print('no more changes anymore --> quitting')
                    break

        # for debug
        if _DEBUG:
            Img(img_saturated, dimensions='hw').save(
                os.path.join(output_folder, 'extras', 'saturated_mask4.tif'))

        final_seeds = label(Img.invert(img_saturated), connectivity=1,
                            background=0)  # keep like that otherwise creates tiny cells with erroneous wshed

        # for debug
        if _DEBUG:
            Img(final_seeds, dimensions='hw').save(
                os.path.join(output_folder, 'extras', 'final_seeds_before.tif'))
        final_seeds = label(Img.invert(img_saturated), connectivity=2, background=0)  # is that needed ???
        # for debug
        if _DEBUG:
            Img(final_seeds, dimensions='hw').save(
                os.path.join(output_folder, 'extras', 'final_seeds_before2.tif'))

        final_seeds[img_saturated == 255] = 0
        final_wshed = watershed(img_orig, markers=final_seeds,
                                watershed_line=True)

        final_wshed[final_wshed != 0] = 1  # remove all seeds
        final_wshed[final_wshed == 0] = 255  # set wshed values to 255
        final_wshed[final_wshed == 1] = 0  # set all other cell content to

        # filename0 = os.path.basename(path)
        # parent_path = os.path.dirname(os.path.dirname(path))

        if filter is None or filter == 0:
            # TODO maybe offer the choice between saving wshed on predict or on orig
            # Img(final_wshed, dimensions='hw').save(os.path.join(output_folder, os.path.splitext(filename0)[
            #     0]) + '.tif')  # need put original name here  TODO put image default name here
            # print('saving', filename_to_use_to_save)
            # Img(final_wshed.astype(np.uint8), dimensions='hw').save(filename_to_use_to_save)
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
            correspondance_between_cur_seeds_and_safe_ones = {}

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
                    if restore_safe_cells:
                        for coordinates in region.coords:
                            if rescue_seeds[coordinates[0], coordinates[1]] != 0:  # do r
                                correspondance_between_cur_seeds_and_safe_ones[region.label] = rescue_seeds[
                                    coordinates[0], coordinates[1]]
                                break
                            break

                _, tiles = Img.get_2D_tiles_with_overlap(final_seeds, overlap=64, dimension_h=-2, dimension_w=-1)

                for r in tiles:
                    for tile in r:
                        rps2 = regionprops(tile)
                        for region in rps2:
                            if self.stop_now:
                                return

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
                                if self.rect_distance(region.bbox, region2.bbox) <= 1:
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
                if restore_safe_cells:
                    removed_seeds_to_restore = []
                    for region in regionprops(final_seeds):
                        if region.label in removed_seeds:
                            first = True
                            for coordinates in region.coords:
                                if first and rescue_seeds[coordinates[0], coordinates[1]] != 0:
                                    percent_diff = min(labels_n_area[region.label], labels_n_area_rescue_seeds[
                                        rescue_seeds[coordinates[0], coordinates[1]]]) / max(
                                        labels_n_area[region.label], labels_n_area_rescue_seeds[
                                            rescue_seeds[coordinates[0], coordinates[1]]])

                                    if (percent_diff >= 0.7 and percent_diff < 1.0) or (
                                            labels_n_area[region.label] <= 200 and (
                                            percent_diff >= 0.3 and percent_diff < 1.0)):
                                        if _DEBUG:
                                            print('0 finally not removing seed, safe seed', region.label,
                                                  percent_diff,
                                                  labels_n_area[region.label],
                                                  labels_n_area_rescue_seeds[
                                                      rescue_seeds[coordinates[0], coordinates[1]]],
                                                  labels_n_area[region.label] / labels_n_area_rescue_seeds[
                                                      rescue_seeds[coordinates[0], coordinates[1]]],
                                                  region.centroid)
                                        removed_seeds_to_restore.append(region.label)
                                        break
                                    break
                    removed_seeds = [x for x in removed_seeds if x not in removed_seeds_to_restore]
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

            if cutoff_cell_fusion is not None and cutoff_cell_fusion > 1:
                cells_to_fuse = []

                for idx, removed_seed in enumerate(removed_seeds):
                    current_cells_to_fuse = set()
                    closest_pair = None
                    smallest_distance = None

                    for idx2 in range(idx + 1, len(removed_seeds)):
                        removed_seed2 = removed_seeds[idx2]

                        if closest_pair is None:
                            if self.rect_distance(labels_n_bbox[removed_seed], labels_n_bbox[removed_seed2]) <= 1:
                                closest_pair = removed_seed2
                                smallest_distance = self.rect_distance(labels_n_bbox[removed_seed],
                                                                       labels_n_bbox[removed_seed2])
                        elif self.rect_distance(labels_n_bbox[removed_seed],
                                                labels_n_bbox[removed_seed2]) <= smallest_distance:
                            closest_pair = removed_seed2
                            smallest_distance = self.rect_distance(labels_n_bbox[removed_seed],
                                                                   labels_n_bbox[removed_seed2])

                        if self.rect_distance(labels_n_bbox[removed_seed], labels_n_bbox[removed_seed2]) <= 1:
                            current_cells_to_fuse.add(removed_seed)
                            current_cells_to_fuse.add(removed_seed2)

                    if current_cells_to_fuse:
                        cells_to_fuse.append(current_cells_to_fuse)

                cells_to_fuse = [frozenset(i) for i in cells_to_fuse]
                cells_to_fuse = list(dict.fromkeys(cells_to_fuse))

                cells_to_keep = []
                if cutoff_cell_fusion is not None and cutoff_cell_fusion > 0:
                    superfuse = []

                    copy_of_cells_to_fuse = cells_to_fuse.copy()
                    for idx, fuse in enumerate(copy_of_cells_to_fuse):
                        current_fusion = set(fuse.copy())
                        changed = True
                        while changed:
                            changed = False
                            for idx2 in range(len(copy_of_cells_to_fuse) - 1, idx, -1):
                                fuse2 = copy_of_cells_to_fuse[idx2]
                                if idx2 == idx:
                                    continue
                                if fuse2.intersection(current_fusion):
                                    current_fusion.update(fuse2)
                                    del copy_of_cells_to_fuse[idx2]
                                    changed = True
                        superfuse.append(current_fusion)

                    for sf in superfuse:
                        if len(sf) > cutoff_cell_fusion:
                            for val in sf:
                                cells_to_keep.append(val)

                seeds_to_fuse = []

                cells_to_fuse = sorted(cells_to_fuse, key=len)
                for fuse in cells_to_fuse:
                    cumulative_area = 0
                    for _id in fuse:
                        if _id in cells_to_keep:
                            if _id in removed_seeds:
                                removed_seeds.remove(_id)
                            continue
                        cumulative_area += labels_n_area[_id]
                    if filter_by_size is not None:
                        if cumulative_area >= filter_by_size:  #: #1200: #filter_by_size: # need hack this to get local area
                            seeds_to_fuse.append(fuse)
                            for _id in fuse:
                                if _id in removed_seeds:
                                    removed_seeds.remove(_id)
                    else:
                        if cumulative_area >= ids_n_local_median[_id]:
                            seeds_to_fuse.append(fuse)
                            for _id in fuse:
                                if _id in removed_seeds:
                                    removed_seeds.remove(_id)

                # need recolor all the seeds in there with the new seed stuff
                for fuse in seeds_to_fuse:
                    for _id in fuse:
                        break
                    for region in regionprops(final_seeds):
                        if region.label in fuse:
                            for coordinates in region.coords:
                                final_seeds[coordinates[0], coordinates[1]] = _id

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
            # print('saving', filename_to_use_to_save)
            # Img(final_wshed.astype(np.uint8), dimensions='hw').save(filename_to_use_to_save)

            duration = timer() - start
            if _DEBUG:
                print('final duration wshed in secs', duration)

            return final_wshed.astype(np.uint8)  # is indeed a 2D image

    def rect_distance(self, bbox1, bbox2):
        width1 = abs(bbox1[3] - bbox1[1])
        width2 = abs(bbox2[3] - bbox2[1])
        height1 = abs(bbox1[2] - bbox1[0])
        height2 = abs(bbox2[2] - bbox2[0])
        return max(abs((bbox1[1] + width1 / 2) - (bbox2[1] + width2 / 2)) - (width1 + width2) / 2,
                   abs((bbox1[0] + height1 / 2) - (bbox2[0] + height2 / 2)) - (height1 + height2) / 2)


if __name__ == '__main__':
    # get the image invert what needs to be inverted

    # input = '/home/aigouy/Bureau/final_folder_scoring/predict_centroid_n_inverted/'
    # input = '/home/aigouy/Bureau/final_folder_scoring/predict_vgg16_shells/'
    # input = '/home/aigouy/Bureau/final_folder_scoring/predict/'
    # input = '/home/aigouy/Bureau/final_folder_scoring/predict_vgg16_light_divided_by_2/'
    # input = '/home/aigouy/Bureau/final_folder_scoring/predict_paper/'
    input = '/home/aigouy/Bureau/final_folder_scoring/predict_Linknet-seresnext101-smloss-256x256-ep0099-l0.158729/'  # 1
    # input = '/home/aigouy/Bureau/final_folder_scoring/predict_Linknet-seresnext101-smloss-256x256-ep0099-l0.158729_rot_HQ_only/' #2
    # input = '/home/aigouy/Bureau/final_folder_scoring/predict_linknet-vgg16-sigmoid-ep0191-l0.144317/'  # 3
    # input = '/home/aigouy/Bureau/final_folder_scoring/predict_linknet-vgg16-sigmoid-ep0191-l0.144317/' #3
    # input = '/home/aigouy/Bureau/final_folder_scoring/predict_linknet-vgg16-sigmoid-ep0191-l0.144317_rot_HQ_only/' #4

    # everything seems to work now finalize the GUI...
    post_proc = EPySegPostProcess(input=input, output_folder='/home/aigouy/Bureau/final_folder_scoring/epyseg_tests',
                                  filter='local median',
                                  correction_factor=2, TA_name=None, cutoff_cell_fusion=3,
                                  restore_safe_cells=True,
                                  _DEBUG=False, _VISUAL_DEBUG=False)

    # pure size based stuff
    # post_proc = EPySegPostProcess(input=input, output_folder='/home/aigouy/Bureau/final_folder_scoring/epyseg_tests',
    #                               filter=150,
    #                               correction_factor=2, TA_name=None, cutoff_cell_fusion=None,
    #                               restore_safe_cells=False,
    #                               _DEBUG=False, _VISUAL_DEBUG=False)


