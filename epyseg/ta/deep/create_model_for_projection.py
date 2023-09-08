# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.layers import Lambda
# from tensorflow.python.keras.models import Sequential
from epyseg.deeplearning.deepl import *
import tensorflow as tf
# import tensorflow.python.keras.backend as K

# somehow seg models require keras
# nb I could replace from tensorflow.python.keras.models import Model
# tf.keras.models.Model --> and this is the only proper way of doing it
# from tensorflow import keras # this is also the only way of using this --> fix it everywhere once for good
# shall I move all to torch some day --> yes maybe
# so shall I install keras also by default ???

def create_surface_projection_denoise_and_height_map_combinatorial_model(surface_proj_model, denoiser_model, HEIGHT_MAP_MODE='probability', use_cpu=False, save_file_name=None, __VERBOSE=False):
    """
    Creates a combined model for surface projection, denoising, and height map estimation.

    Args:
        surface_proj_model (str): Path to the surface projection model file.
        denoiser_model (str): Path to the denoiser model file.
        HEIGHT_MAP_MODE (str, optional): Height map mode. Defaults to 'probability'.
        use_cpu (bool, optional): Flag indicating whether to use CPU for computation. Defaults to False.
        save_file_name (str, optional): Name of the file to save the model. Defaults to None.
        __VERBOSE (bool, optional): Flag indicating verbosity. Defaults to False.

    Returns:
        tf.keras.Model: The combined model.

    Examples:
        surface_proj_model = '/E/models/my_model/bckup/CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/surface_projection_model.h5'
        denoiser_model = '/E/models/my_model/bckup/CARESEG_masked_data_and_masked_loss_to_prevent_identity_v1_210107_good/2D_denoiser_model.h5'
        model = create_surface_projection_denoise_and_height_map_combinatorial_model(surface_proj_model, denoiser_model)
        print(model.summary(line_length=250))
    """

    deepTA = EZDeepLearning(use_cpu=use_cpu)

    # print(surface_proj_model)

    deepTA.load_or_build(model=surface_proj_model)
    custom_model = deepTA.model

    if __VERBOSE:
        print('Surface projection model')
        print(custom_model.summary(line_length=250))

    deepTA.load_or_build(model=denoiser_model)
    custom_model2 = deepTA.model

    if __VERBOSE:
        print('2D denoiser model')
        print(custom_model2.summary(line_length=250))

    input_image_3D = tf.keras.Input(shape=(None, None, None, 1), name="Z-stack")

    model_3D_projection = tf.keras.models.Sequential()
    model_3D_projection.add(custom_model)
    model_3D_projection.add(custom_model2)

    if __VERBOSE:
        print('model_3D_projection')
        print(model_3D_projection.summary())

    if __VERBOSE:
        print('Height map')

    if HEIGHT_MAP_MODE == 'probability':
        model_height_map = tf.keras.models.Model(custom_model.input, custom_model.layers[-3].output)
    else:
        model_height_map = tf.keras.models.Model(custom_model.input, custom_model.layers[-2].output)

    model_3D_height_map = model_height_map(input_image_3D)

    if __VERBOSE:
        print('model_height_map', model_height_map.summary(line_length=250))

    real_height_map = tf.keras.layers.Lambda(lambda x: tf.keras.backend.cast(tf.keras.backend.argmax(x, axis=-4), dtype='float32'), name='height_map')(model_3D_height_map)

    real_denoised_3D_image = model_3D_projection.layers[0].layers[-2]([input_image_3D, model_3D_height_map])
    surface_projection_before_running_2D_denoising_model = model_3D_projection.layers[0].layers[-1](real_denoised_3D_image)
    result_of_CARE_or_CARESEG_denoising_model = model_3D_projection.layers[1](surface_projection_before_running_2D_denoising_model)
    first_channel_of_denoising_model_output = tf.keras.layers.Lambda(lambda x: x[..., 0:1], name='keep_first_channel_only')(result_of_CARE_or_CARESEG_denoising_model)

    model = tf.keras.Model(inputs=[input_image_3D], outputs=[first_channel_of_denoising_model_output, surface_projection_before_running_2D_denoising_model, real_height_map])

    if __VERBOSE:
        model.summary(line_length=250)

    if save_file_name is not None:
        model.save(save_file_name, include_optimizer=False, overwrite=True, save_format='h5')

    return model


if __name__ == '__main__':
    surface_proj_model = '/E/models/my_model/bckup/CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/surface_projection_model.h5'
    denoiser_model = '/E/models/my_model/bckup/CARESEG_masked_data_and_masked_loss_to_prevent_identity_v1_210107_good/2D_denoiser_model.h5'  # not great on CARE great on others --> really need different denoisers depending on DATA --> ok --> though I doubt people will use CARE like data as input...
    # denoiser_model_path = '/E/models/my_model/bckup/CARESEG_another_normal_training_201216_colab_but_gave_crap/2D_denoiser_model.h5'# great for CARE not great for the others
    # denoiser_model_path = '/E/models/my_model/bckup/CARESEG_retrained_normally_201216/2D_denoiser_model.h5'# great for CARE not great for the others
    # denoiser_model_path = '/E/models/my_model/bckup/CARESEG_trained_on_CARE_data_with_dilation_good/2D_denoiser_model.h5'# great for CARE not great for the others

    save_file_name = '/E/Sample_images/sample_images_pyta/test_merged_model.h5'

    # now using pretrained models
    surface_proj_model = 'SURFACE_PROJECTION'
    denoiser_model = '2D_DENOISER'

    surface_proj_model = 'SURFACE_PROJECTION_4'  # this one was not pushed neither --> need do it too
    # surface_proj_model = 'SURFACE_PROJECTION_5' # this one was not pushed neither --> need do it too
    # surface_proj_model = 'SURFACE_PROJECTION_6' # this one was not pushed neither --> need do it too --> sometimes better with folds, otherwise the 3 or 4 are better
    # denoiser_model = '2D_DENOISER'
    denoiser_model = '2D_DENOISEG'

    model = create_surface_projection_denoise_and_height_map_combinatorial_model(surface_proj_model, denoiser_model)
    print(model.summary(line_length=250))

    final_separated_denoiser_keep_one_channel = tf.keras.models.Sequential()
    final_separated_denoiser_keep_one_channel.add(model.layers[-3])
    final_separated_denoiser_keep_one_channel.add(tf.keras.layers.Lambda(lambda x: x[..., 0:1]))

    print('final denoiser only',final_separated_denoiser_keep_one_channel.summary(line_length=250))#perfect --> that is the 2D denoiser to which I should apply the