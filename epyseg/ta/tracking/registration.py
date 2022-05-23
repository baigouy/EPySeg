# seems perfect --> gives no swapping --> just errors due to dupes and in that case I should just take the most likely cell --> cell with closest area or closest shape based on its vertices --> I'm almost done!!!!

# I think this shit is now fully coded...


# https://github.com/voxelmorph/voxelmorph --> is that what I was looking for ???
# maybe what I was looking for https://colab.research.google.com/drive/1WiqyF7dCdnNBIANEY80Pxw_mVz4fyV-S?usp=sharing --> bingo I found you THIS IS JUST EXACTLY WHAT I WANTED / NEEDED AND IT'S TENSORFLOW 2.0!!!!!!!!!!


# nb the IJ sift align works very well too but couldn't find an easy python equivalent...


# apparement qq un y a deja pensé https://pubmed.ncbi.nlm.nih.gov/18267377/
# http://bigwww.epfl.ch/publications/thevenaz9801.html


# the idea is to crop further and further the image in order to get pyramidal correction of seefd position in order to get the correct tracking done

# get an image chop it into tiles then chop each registered tile further and register it on the big image or change the stuff


# or do smaller things
# or detect a swap and if a swap is detected then try recompute the translation

# or chop new image into small tiles and try to match each of these tiles on the bigger one

# try combinations of these things

# think about it


# try register a small region on the big --> how would that work if images don't have the same size

# ask for depth of tiling


# from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation # change in scipy 0.19
# image = data.camera()
from skimage import transform
import numpy as np

# really very good

# parameters

# threshold
# threshold_translation = 64  # all the further translations should be small --> if that is not the case then there has to be a pb

__DEBUG__ = False




def apply_translation(img, y,x):
    afine_tf = transform.AffineTransform(translation=(x, y))
    translated = transform.warp(img, inverse_map=afine_tf, order=0, preserve_range=True)
    return translated

def pre_register_images(orig_t0, orig_t1, apply_to_orig_t0=False):
    global_translation, error, diffphase = phase_cross_correlation(orig_t0, orig_t1)
    gloabl_t0 = -global_translation[0]
    gloabl_t1 = -global_translation[1]
    if not apply_to_orig_t0:
        return gloabl_t0, gloabl_t1
    else:
        return apply_translation(orig_t0, -gloabl_t0, -gloabl_t1)

def get_pyramidal_registration(orig_t0, orig_t1, depth=2, threshold_translation=20):
    # orig_t0 = Img('/E/Sample_images/tracking_test/leg_cropped/100807_leg4_male.lei - leg4_male_Series031.tif')#[455:455+122,544:544+127] #[384:512,512:640]
    # orig_t1 = Img('/E/Sample_images/tracking_test/leg_cropped/100807_leg4_male.lei - leg4_male_Series033.tif')#[455:455+122,544:544+127] #[384:512,512:640]

    translation_matrix = np.zeros((*orig_t0.shape, 2))

    # global_translation, error, diffphase = register_translation(orig_t0, orig_t1)
    #
    # if __DEBUG__:
    #     print('error', error,
    #           diffphase)  # if error is bigger upon further registration --> try ignore it or use a ROI to register only the region of interest --> similar to what I had in TA...
    #
    # gloabl_t0 = -global_translation[0]
    # gloabl_t1 = -global_translation[1]

    gloabl_t0, gloabl_t1 = pre_register_images(orig_t0, orig_t1)

    translation_matrix[..., 0] += gloabl_t0
    translation_matrix[..., 1] += gloabl_t1

    # small bug :
    '''
    /home/aigouy/.local/lib/python3.7/site-packages/skimage/feature/register_translation.py:104: RuntimeWarning: invalid value encountered in true_divide
      (src_amp * target_amp)
    '''

    height = orig_t0.shape[0]
    width = orig_t0.shape[1]

    # Nb i need get parent translation and further apply it before retracking the stuff --> need get the cropping

    block_size = 256

    for i in range(depth):
        try:
            # final_value = i + (pyramidal_skip if i > 0 else 0)
            tile_height = block_size // (i * 2)
            tile_width = block_size // (i * 2)
        except:
            tile_height = block_size
            tile_width = block_size

        if __DEBUG__:
            print('size of stuff', ((i + 1) * 2), tile_width)
        # trans_y = []
        for y in range(0, height, tile_height):
            # trans_x = []
            for x in range(0, width, tile_width):
                if __DEBUG__:
                    print(y, x, '-->', y + tile_height, x + tile_width)
                # take bigger for one and not for the other --> include surrounding inj one and black in the other
                # tile_t0=mask_t0[y:y+tile_height,x:x+tile_width]
                # tile_t1=mask_t1[y:y+tile_height,x:x+tile_width]
                # translate image 0 before further registering it
                afine_tf = transform.AffineTransform(
                    translation=(-translation_matrix[y, x, 1], -translation_matrix[y, x, 0]))
                translated = transform.warp(orig_t0, inverse_map=afine_tf, order=0, preserve_range=True)

                # plt.imshow(translated)
                # plt.show()

                tile_t0 = translated[y:y + tile_height, x:x + tile_width]
                tile_t1 = orig_t1[y:y + tile_height, x:x + tile_width]
                # t0, t1 = translation(tile_t0, tile_t1)

                # can I create a patchwork of the registered image...


                # shift, error, diffphase = register_translation(tile_t0, tile_t1)
                # change for v 0.19 of scipy
                shift, error, diffphase = phase_cross_correlation(tile_t0, tile_t1)
                t0 = -shift[0]
                t1 = -shift[1]
                # imshow(mask_t0, mask_t1, im2)

                # print('translation', t0, t1, scale, angle)  # seems to be what I want --> just try default TA algo on that --> TODO
                if __DEBUG__:
                    print('translation', t0, t1)

                # ignore too big translations because they are most likely artifacts
                if abs(t0) > threshold_translation or abs(t1) > threshold_translation:
                    if __DEBUG__:
                        print('most likely an erroneous translation detected --> ignoring', t0, t1)
                    continue

                translation_matrix[y:y + tile_height, x:x + tile_width, 0] += t0
                translation_matrix[y:y + tile_height, x:x + tile_width, 1] += t1
                # trans_x.append((t0, t1))
            # trans_y.append(trans_x)

        # print('trans_y', trans_y)
        # in a way it does compute a flowfield too...

    return translation_matrix
