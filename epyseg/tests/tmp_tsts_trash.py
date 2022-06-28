import math
from random import gauss

import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.testutils import assert_array_equal

from epyseg.img import Img, to_stack, is_binary, convolve, create_2D_linear_gradient, save_as_tiff, apply_2D_gradient, \
    create_random_liner_gradient, gaussian_intensity_2D, create_random_gaussian_gradient, \
    create_random_intensity_graded_perturbation, auto_scale, blend, \
    blend_stack_channels_color_mode, pop
from epyseg.utils.loadlist import loadlist
import random

def invert(img):
    # should take the negative of an image should always work I think but try and see if not wise making a version that handles channels # does it even make sense ??? need to think a bit about it
    max = img.max()
    min = img.min()
    # print(np.negative(img))
    if not img.dtype == bool:
        img = np.negative(img) + max + min
    else:
        img = ~img
    return img


if __name__ == '__main__':
    if True:
        img = Img('/E/Sample_images/sample_images_FIJI/150707_WTstack.lsm')
        # auto_scale(img)
        pop(img)

        img = Img('/E/Sample_images/sample_images_FIJI/AuPbSn40.jpg')
        pop(img)

        # ok but then where is the bug???

        import sys
        sys.exit(0)

    if False:

        # lsm format of Luts --> need create it
        # I guess it's just linear until max is reached
        # increment so that the value fits
        # shall I ignore alpha
        # generate it linearly from 0 to max
        lut = [255, 0, 128, 0] # ça marche !!!
        R = np.linspace(0, lut[0], 256, dtype=np.uint8)
        G = np.linspace(0, lut[1], 256, dtype=np.uint8)
        B = np.linspace(0, lut[2], 256, dtype=np.uint8)
        # A = np.linspace(0, lut[3], 256, dtype=np.uint8)
        lut = np.stack([R,G,B], axis=-1)
        print(lut.shape)
        print(lut)


        import sys
        sys.exit(0)
    if True:
        img = Img('/E/Sample_images/sample_images_different_sizes_and_bit_depth_and_nb_of_channels/Rat_Hippocampal_Neuron_5channels.tif')
        img = Img('/E/Sample_images/sample_images_different_sizes_and_bit_depth_and_nb_of_channels/real_imageJ_image.tif')
        img = Img('/E/Sample_images/sample_images_different_sizes_and_bit_depth_and_nb_of_channels/210219.lif_t000.tif')


        # nb lsm luts are not recovered properly --> see how I can fix that
        img = Img('/E/Sample_images/sample_images_FIJI/150707_WTstack.lsm')
        # img = np.average(img, axis=-1)
        # img = np.max(img, axis=-1)
        print(img.metadata['LUTs'])

        # img.metadata['LUTs']=None

        # ça marche --> ok
        img = blend_stack_channels_color_mode(img) # TODO --> do a blend with several images

        # colors are ok --> so where is my bug ???
        try:
            plt.imshow(img/img.max())
            plt.show()
        except:
            try:
                plt.imshow(img[int(img.shape[0]/2)] / img[int(img.shape[0]/2)].max())
                plt.show()
            except:
                pass

        import sys
        sys.exit(0)

    if True:
        img = Img('/E/Sample_images/sample_images_FIJI/AuPbSn40.jpg')
        print(img.shape)
        img = np.stack([img, img,img],axis=-1)
        print(img.shape)
        plt.imshow(img)
        plt.show()

        import sys
        sys.exit(0)

    if True:
        tst = Img('/E/Sample_images/fluorescent_wings_spots_charroux/909dsRed/0.tif')
        tst = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png')
        tst = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/ON 290119.lif - Series002.tif') # this is super slow with this one --> make it literally unusable --> I need just get the displayed image only

        # too slow --> never do this for such a big image

        try:
            plt.imshow(tst)
            plt.show()
        except:
            pass


        tst = auto_scale(tst)

        # print(tst)

        print(tst.shape)
        # print(tst.metadata)
        try:
            plt.imshow(tst)
            plt.show()
        except:
            pass
        import sys
        sys.exit(0)

    if False:
        for i in range(10):
            gaussian = gaussian_intensity_2D(sigma=random.uniform(0.2, 5.), mu=random.uniform(-2., 2.))
            print(gaussian.min(), gaussian.max())
            plt.imshow(gaussian)
            plt.show()

    img = Img('/E/Sample_images/sample_images_PA/trash_test_mem/mini/focused_Series012.png')[...,1] #2D image
    pos_h = 0
    img = Img('/E/Sample_images/sample_images_pyta/hela-cells_48bits_RGB.tif') # 2D RGB image
    pos_h = 0
    img =Img('/E/Sample_images/adult_wings/pigmented_wings_benjamin_all/N01_YwrGal4_males_wing0_projected.tif') # 2D RGB image
    pos_h = 0

    # img = Img('/E/Sample_images/sample_images_pyta/Image4.lsm') # 3D image single channel
    # pos_h = 1
    # img=img[...,None] # add a new axis
    # if pos h is specified --> then use it
    print(img.shape)
    # grad = create_2D_gradient(img, horizontal=False)
    # grad = create_2D_gradient(img, min=0.2, max=1, horizontal=True, min_is_top_or_left=False)

    # grad = create_2D_linear_gradient(img, min=0.2, max=1, horizontal=True, min_is_top_or_left=False, pos_h=pos_h)
    # grad = create_random_liner_gradient(img, pos_h=pos_h)
    # grad = create_random_gaussian_gradient(img, pos_h=pos_h)
    grad = create_random_intensity_graded_perturbation(img, pos_h=pos_h, off_centered=True)



    print(grad.min(), grad.max(), grad.shape)
    # plt.imshow(grad)
    # plt.show()
    try:
        plt.imshow(img / img.max())
        plt.show()
    except:
        pass
    # TO apply I need add missing dimensions around the matching ones

    # tsts= np.outer(img, grad)
    # print(tsts.shape)

    # final = apply_2D_gradient(img[..., np.newaxis], grad)
    final = apply_2D_gradient(img, grad)
    Img(final, dimensions='hw').save('/E/Sample_images/sample_images_pyta/graded2.tif')
    # print(final.shape)
    # final = apply_2D_gradient(img, grad)

    # save_as_tiff(final, '/E/Sample_images/sample_images_pyta/graded.tif')
    # print(final.shape) # --> (25, 1128, 2048) --> dhw --> ok -->

    # print(type(final))
    # Img(final[..., np.newaxis], dimensions='dhwc').save('/E/Sample_images/sample_images_pyta/graded.tif')

    # img[...,:] = img[...,:]*grad[...,  np.newaxis] # ça marche # need do that for every channel before and after teh dims of interest --> TODO
    # img[...,:] = img[...,:]*grad[None,...,  np.newaxis] # ça marche # need do that for every channel before and after teh dims of interest --> TODO



    # for ch in range(img.shape[-1]):
    #     img = img * grad[..., np.newaxis]

    # not great because only appluies it to one of the channels
    # how to apply it to all dimensions
    # need loop over all dimensions until I reach the final ones --> TODO
    try:
        plt.imshow(img/img.max())
        plt.show()
    except:
        pass





    # apply the gradient now --> also offer a 3D gradient