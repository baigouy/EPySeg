# the idea is to read the height map image and create a max proj out of it --> TODO
from epyseg.img import Img
import numpy as np
import matplotlib.pyplot as plt

def surface_projection_from_height_map(original_image, height_map , integer_mode = False, channel=None):
    # to make it simple first just cast it
    height_map_int = height_map.astype(np.uint8)

    if channel is not None:
        if len(original_image.shape)>2:
            original_image = original_image[..., channel]

    surface_projection = np.zeros_like(height_map)


    # [os.path.join(path, f) for f in images_to_analyze if
    #                          os.path.isfile(os.path.join(path, f))[]
    # surface_projection = original_image[height_map,:,:]




    min = height_map_int.min()
    max = height_map_int.max()


    for val in range(min, max):
        surface_projection[height_map_int==val] = original_image[val, height_map_int==val]





    # plt.imshow(difference_to_int)
    # plt.show()
    #
    # plt.imshow(one_minus_difference)
    # plt.show()

    # plt.imshow(surface_projection)
    # plt.show()

    # Img(surface_projection,dimensions='hw').save('/E/Sample_images/sample_images_denoise_manue/trashme_test_robot/predict/surface_proj_int.tif')

    if integer_mode:
        return surface_projection

    weights_for_layer_1 = height_map - height_map_int
    weights_for_layer_0 = 1 - weights_for_layer_1
    # NB try indexed_probs = np.take_along_axis(probs, np.expand_dims(indices, axis=-1), axis=-1)[..., 0]

    for val in range(min, max-1):
        layer_0 = original_image[val, height_map_int==val] * weights_for_layer_0[height_map_int == val]
        layer_1 = original_image[val+1, height_map_int == val] * weights_for_layer_1[height_map_int == val]
        # do weighted average of the layers --> TODO
        # surface_projection[height_map_int == val] = (layer_0+layer_1)/2
        # for each pixel I need to do the difference to the layer


        # surface_projection[height_map_int == val] = (layer_0+layer_1)
        surface_projection[height_map_int == val] = layer_0+layer_1

    # pas mal mais voir comment gerer le fait que ce soit du float ???

    # for floats get the rest and compute an average --> that is most likely not easy TODO --> that should be fairly easy todo in fact


    # plt.imshow(surface_projection)
    # plt.show()


    # Img(surface_projection,dimensions='hw').save('/E/Sample_images/sample_images_denoise_manue/trashme_test_robot/predict/surface_proj_float.tif')
    return surface_projection


if __name__ == '__main__':
    height_map = Img('/E/Sample_images/sample_images_denoise_manue/trashme_test_robot/predict/Image7.tif')
    original_image = Img('/E/Sample_images/sample_images_denoise_manue/trashme_test_robot/Image7.tif')
    surface_proj = surface_projection_from_height_map(original_image, height_map)
    plt.imshow(surface_proj)
    plt.show()