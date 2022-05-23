from scipy import ndimage
from skimage.filters import threshold_otsu
# from skimage.segmentation import watershed 
from skimage.segmentation import watershed

from epyseg.img import Img
from skimage.measure import label, regionprops
import os
import numpy as np
from epyseg.tools.logger import TA_logger # logging
import tempfile

from epyseg.postprocess.filtermask import FilterMask
from epyseg.postprocess.edmshed import segment_cells

logger = TA_logger()


class RefineMaskUsingSeeds:

    def __init__(self):
        pass

    def process(self, input=None, mode=None, _DEBUG=False, _VISUAL_DEBUG=False, output_folder=tempfile.gettempdir(),
                output_name='handCorrection.tif', threshold=None,
                filter=None,
                correction_factor=2,
                **kwargs):

        if input is None:
            logger.error('no input image --> nothing to do')
            return

        # TODO test it with several images just to see if that works
        if isinstance(mode, str) and 'first' in mode:
            # return first channel only # shall I had a channel axis to it to avoid issues
            out = input[..., 0]
            # I do this to keep the ...hwc format...
            return out[..., np.newaxis]

        img_orig = input

        if not img_orig.has_c() or img_orig.shape[-1] != 7:
            # TODO in fact could do the fast mode still on a single image --> may be useful
            logger.error('image must have 7 channels to be used for post process')
            return img_orig

        if _DEBUG:
            Img(img_orig, dimensions='hwc').save(os.path.join(output_folder, 'raw_input.tif'))

        bckup_img_wshed = img_orig[..., 0].copy()
        if mode is not None and isinstance(mode, str):
            if 'ast' in mode:
                logger.debug('fast mode')
                img_orig[..., 0] += img_orig[..., 1]
                img_orig[..., 0] += img_orig[..., 2]
                img_orig = img_orig[..., 0] / 3
                img_orig = np.reshape(img_orig, (*img_orig.shape, 1))
            else:
                logger.debug('normal mode')
        else:
            logger.debug('normal mode')

        differing_bonds = np.zeros_like(img_orig)

        img_orig[..., 0] = segment_cells(img_orig[..., 0], min_threshold=0.02, min_unconnected_object_size=3)

        if img_orig.shape[-1] >= 5:
            img_orig[..., 1] = segment_cells(img_orig[..., 1], min_threshold=0.06, min_unconnected_object_size=6)
            img_orig[..., 2] = segment_cells(img_orig[..., 2], min_threshold=0.15, min_unconnected_object_size=12)
            img_orig[..., 3] = Img.invert(img_orig[..., 3])
            img_orig[..., 3] = segment_cells(img_orig[..., 3], min_threshold=0.06, min_unconnected_object_size=6)
            img_orig[..., 4] = Img.invert(img_orig[..., 4])
            img_orig[..., 4] = segment_cells(img_orig[..., 4], min_threshold=0.15, min_unconnected_object_size=12)

        if img_orig.shape[-1] == 7:
            img_orig[..., 5] = self.binarise(img_orig[..., 5], threshold=0.15)
            img_orig[..., 6] = Img.invert(img_orig[..., 6])
            img_orig[..., 6] = self.binarise(img_orig[..., 6], threshold=0.1)

        if _DEBUG:
            Img(img_orig, dimensions='hwc').save(os.path.join(output_folder, 'thresholded_masks.tif'))

        # get watershed mask for all images
        for i in range(img_orig.shape[-1]):
            if i < 5:
                final_seeds = label(Img.invert(img_orig[..., i]), connectivity=1, background=0)
            else:
                final_seeds = label(img_orig[..., i], connectivity=None, background=0)
            final_wshed = watershed(bckup_img_wshed, markers=final_seeds, watershed_line=True)
            final_wshed[final_wshed != 0] = 1
            final_wshed[final_wshed == 0] = 255
            final_wshed[final_wshed == 1] = 0

            differing_bonds[..., i] = final_wshed

            del final_seeds
            del final_wshed

        if _DEBUG:
            print(os.path.join(output_folder, 'differences.tif'))
            Img(differing_bonds, dimensions='hwc').save(os.path.join(output_folder, 'differences.tif'))
            Img(bckup_img_wshed, dimensions='hw').save(os.path.join(output_folder, 'orig_img.tif'))

        avg = np.mean(differing_bonds, axis=-1)
        avg = avg / avg.max()

        if _DEBUG:
            Img(avg, dimensions='hw').save(os.path.join(output_folder, output_name + str('avg.tif')))

        if threshold is None:
            threshold = self.autothreshold(avg)

        logger.debug('threshold used for producing the final mask=' + str(threshold))

        final_mask = avg.copy()
        final_mask = self.binarise(final_mask, threshold=threshold)

        if _DEBUG:
            Img(final_mask, dimensions='hw').save(os.path.join(output_folder, 'binarized.tif'))

        # close wshed mask to fill super tiny holes
        s = ndimage.generate_binary_structure(2, 1)
        final_mask = ndimage.grey_dilation(final_mask, footprint=s)

        # remove super tiny artificial cells (very small value cause already dilated)
        mask = label(Img.invert(final_mask), connectivity=1, background=0)
        for region in regionprops(mask):
            if region.area < 5:
                for coordinates in region.coords:
                    final_mask[coordinates[0], coordinates[1]] = 255
        del mask

        final_mask = label(Img.invert(final_mask), connectivity=1, background=0)
        final_mask = watershed(bckup_img_wshed, markers=final_mask, watershed_line=True)

        final_mask[final_mask != 0] = 1
        final_mask[final_mask == 0] = 255
        final_mask[final_mask == 1] = 0

        if filter is None or filter == 0:
            return final_mask.astype(np.uint8)
        else:
            logger.debug('Further filtering image')
            return FilterMask(bckup_img_wshed, final_mask, filter=filter, correction_factor=correction_factor)

    def autothreshold(self, single_2D_img):
        try:
            return threshold_otsu(single_2D_img)
        except ValueError:
            logger.error('Image is just one color, thresholding cannot be done')
            return single_2D_img

    def binarise(self, single_2D_img, threshold=0.5, bg_value=0, fg_value=255):
        # TODO may change this to >= and < try it
        single_2D_img[single_2D_img > threshold] = fg_value
        single_2D_img[single_2D_img <= threshold] = bg_value
        return single_2D_img
