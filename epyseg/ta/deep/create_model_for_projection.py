# create a model for projection that I can use --> à tester

# the model should take a denoiser optionally and a surface proj model and then have the heightmap added to it and the conversion of any type of output to a single output model --> TODO --> no need for keep first channel as can be done by post processing --> and easier this way

# bug fix --> don't mix  tensorflow.keras with tensorflow.python.keras
from tensorflow.keras.models import Model
# from tensorflow.keras.backend import argmax
# from tensorflow.keras.layers.convolutional import ZeroPadding3D, Conv2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential

from epyseg.deeplearning.deepl import *
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K

#HEIGHT_MAP_MODE = 'probability' #'max_intensity' #'probability'
def create_surface_projection_denoise_and_height_map_combinatorial_model(surface_proj_model, denoiser_model, HEIGHT_MAP_MODE ='probability', use_cpu=False, save_file_name=None,__VERBOSE=False):
    # TODO make this a class --> easy to call
    # maybe just get it to return the model ??? --> since model is small I could keep it as it is or make it flexible with some weight loading

    # TODO find a way to ask which model should be saved or not and which whether or not the extra_black frames should be added
    # hack the CARE model to extend its use and to train better the different components of it
    # TODO modify the models also to only export the first channel
    # to find what to save --> just use the other stuff
    # se baser sur le model splitter pr faire ça en fait --> à faire

    deepTA = EZDeepLearning(use_cpu=use_cpu)

    deepTA.load_or_build(model=surface_proj_model)
    custom_model = deepTA.model


    if __VERBOSE:
        print('Surface projection model')
        print(custom_model.summary(line_length=250))

    # CARE 2D denoiser model
    deepTA.load_or_build(model=denoiser_model)
    custom_model2 = deepTA.model

    if __VERBOSE:
        print('2D denoiser model')
        print(custom_model2.summary(line_length=250))

    # create a model 3D proj from both and a 3D input then say it is model_3D_projection

    input_image_3D = tf.keras.Input(
        shape=(None, None, None, 1), name="Z-stack"
    )


    model_3D_projection = Sequential()
    # model_3D_projection.add(input_image_3D)
    model_3D_projection.add(custom_model)
    model_3D_projection.add(custom_model2)

    # are the weights preserved ???? there

    if __VERBOSE:
        print('model_3D_projection')
        print(model_3D_projection.summary())
        # reconstituted model 3D proj



    # # MODEL FIRST CHANNEL ONLY (CAN BE USEFUL FOR CARESEG)
    # model_channel_removed = Sequential()
    # model_channel_removed.add(model_3D_projection)
    # model_channel_removed.add(Lambda(lambda x: x[..., 0:1], name='keep_first_channel_only'))
    # # try create a careseg where only one layer is kept using a lambda
    # # y = Lambda(lambda x: x[..., 0], output_shape=input_shape[0:-1]+(1,))(x)
    # # y = Lambda(lambda x: x[:, 0, :, :], output_shape=(1,) + input_shape[2:])(x)
    # print(model_channel_removed.summary(line_length=250))
    # if first_channel_only_model is not None:
    #     model_channel_removed.save(os.path.join(main_path, first_channel_only_model), include_optimizer=True,
    #                                overwrite=True, save_format='h5')

    # try:
    # I did create a bug here !!!
    # CARE height map (hack of CARE to generate an heightmap)
    if __VERBOSE:
        print('Height map')
    if HEIGHT_MAP_MODE == 'probability':
        model_height_map = Model(custom_model.input, custom_model.layers[-3].output) # globally this is the best idea for height map --> the other one is a bit more shajy and can focus on regions that are not correct # also the second one is more sensitive to noise but there is an idea there I think!!!
    else:
        # 'max_intensity'
        model_height_map = Model(custom_model.input, custom_model.layers[-2].output)# another possible height map --> take max of intensity but not necessarily the best idea --> None is perfect and one is weird maybe --> maybe could also take average of both ??? --> maybe a pixel with low int should still contribute to final

    # neo test


    # print('model_height_map', model_height_map.summary(line_length=250))
    # model_height_map.save(os.path.join(main_path, 'height_map_model.h5'), include_optimizer=True,
    #                       overwrite=True, save_format='h5')

    # print('real Height map')


    # print('model_height_map.summary(',model_height_map.summary())
    #
    # model_real_height_map = Sequential()
    # model_real_height_map.add(model_height_map)
    # # KEEP MEGA NB the reason why I do get floats instead of integers for the height map is simply due to my average code and that means that the augmented data likely give different heights --> also need be careful because I can lose the softmax effect of CARE in some cases, but this has some advantages for the height map
    # model_real_height_map.add(Lambda(lambda x: K.cast(K.argmax(x, axis=-4), dtype='float32'), name='height_map'))
    # # print('model_real_height_map', model_real_height_map.summary(line_length=250))
    # # if height_map_model is not None:
    # #     model_real_height_map.save(os.path.join(main_path, height_map_model), include_optimizer=True,
    # #                                overwrite=True, save_format='h5')
    # # except:
    # #     traceback.print_exc()


    # # add extra frames to the model # --> maybe put this at the very beginning maybe # maybe should be first thing TO DO
    # padded_model = Sequential()
    # padded_model.add(ZeroPadding3D(padding=(4, 0, 0), input_shape=(
    #     None, None, None, 1)))  # add Z black frames above and below --> can improve the surface proj
    # padded_model.add(model_3D_projection)
    #
    # # model_channel_removed.add(ZeroPadding3D(padding=(4, 0, 0), input_shape=(224, 224, 224, 3))) # ça marche c'est comme ça que je peux modifier un modele --> ça rajoute 4 pads en haut et en bas à la premiere dimension
    # # model_channel_removed.add(ZeroPadding3D(padding=(4, 0, 0), input_shape=(None, None, None, 1))) # ça marche c'est comme ça que je peux modifier un modele --> ça rajoute 4 pads en haut et en bas à la premiere dimension
    # print(padded_model.summary(line_length=250))
    # if extra_Z_model is not None:
    #     padded_model.save(os.path.join(main_path, extra_Z_model), include_optimizer=True,
    #                       overwrite=True, save_format='h5')

    # model = Model(inputs=model_3D_projection.layers[2].input, outputs=[model_3D_projection.layers[2].output],
    #               name='my customized model')  # ça a l' de marcher --> ajouter ttes les sorties possibles #ok
    # # par contre j'arrive pas a recup le reste
    # print(model.summary(line_length=250))
    # tf.keras.utils.plot_model(model, "my_first_model.png")
    # tf.keras.utils.plot_model(model_3D_projection, "my_second_model.png")

    # TODO generate a model that has 3 outputs --> a denoised image,a non denoised surface projection and an height map --> these are the 3 things I would mostly be using --> TRY THAT


    # if generate_multi_output_model:
    # ça c'est les multi output models --> may or may not be useful!!!
    # TODO here --> would also be nice to have the height map and the surface proj too


    # MODEL that contains not denoised image and the denoised one --> both can be useful -->
    # title_input = tf.keras.Input(
    #     shape=(None, None, None, 1), name="Z-stack"
    # )
    #
    # output_model1 = model_3D_projection.layers[1](title_input)
    # output_model2 = model_3D_projection.layers[2](output_model1)
    #
    # model = Model(inputs=title_input, outputs=[output_model1, output_model2])  # , output_stack_model_1
    # # print(model.summary(line_length=250))
    # # if non_denoised_and_denoised_surface_projection_model is not None:
    # #     model.save(os.path.join(main_path, non_denoised_and_denoised_surface_projection_model), include_optimizer=True,
    # #                overwrite=True, save_format='h5')
    #
    # # print(model_3D_projection.layers[1].name)
    # print(model_3D_projection.layers[1].summary(line_length=250))  # ça marche car c'est un modèle
    # print(model_3D_projection.layers[1].layers[
    #           -3])  # ça marche aussi car c'est un modèle --> une des layers du modèle --> c'est le truc pour générer le height map

    # Other various multi output models


    # on dirait qu'il y a un bug avec le lambda du height map --> PKOI!!!!

    model_3D_height_map = model_height_map(
        input_image_3D)  # en fait c'est pas ce que je veux je pense faudrait que j'aille encore un plus haut pr avoir une proba --> est-ce la sortie du first lambda faudrait que je vérifie en fait

    if __VERBOSE:
        print('model_height_map', model_height_map.summary(line_length=250))

    # in fact what I want is not highest proba but highest intensity --> could try that and compare the outputs --> could in fact take argmax of the intensity corrected 3D image --> see where and how I should do that
    real_height_map = Lambda(lambda x: K.cast(K.argmax(x, axis=-4), dtype='float32'), name='height_map')(
        model_3D_height_map)

    # returns AttributeError: 'ModuleWrapper' object has no attribute 'layers' on new tensorflows ???
    # shall I force it sequential again

    # <tensorflow.python.keras.engine.functional.ModuleWrapper object at 0x7fb7d55d8d10> Traceback (most recent call last): in tf 2.7.2 but not in
    print('model_3D_projection.layers[0]',model_3D_projection.layers[0])
    # when working it is a <tensorflow.python.keras.engine.functional.Functional object at 0x7f5823bb0390>

    real_denoised_3D_image = model_3D_projection.layers[0].layers[-2]([input_image_3D,
                                                                       model_3D_height_map])  # MEGA NB DON'T FORGET THAT THIS LAYER NEEDS TWO INPUTS!!!! # en fait c'est ça mais ce truc a besoin de 2 inputs --> les lui fournir --> facile mais faut qd meme reflechir un peu
    surface_projection_before_running_2D_denoising_model = model_3D_projection.layers[0].layers[-1](
        real_denoised_3D_image)
    result_of_CARE_or_CARESEG_denoising_model = model_3D_projection.layers[1](
        surface_projection_before_running_2D_denoising_model)
    first_channel_of_denoising_model_output = Lambda(lambda x: x[..., 0:1], name='keep_first_channel_only')(
        result_of_CARE_or_CARESEG_denoising_model)

    # nb could do one just with two outputs --> would be much more useful in a way!!!
    # could also make one that returns the height map, the denoised image and the raw surface projection --> in a way that is much more uesful than that, because I will most likeley never use that!!!
    model = tf.keras.Model(
        inputs=[input_image_3D],
        outputs=[first_channel_of_denoising_model_output, surface_projection_before_running_2D_denoising_model,
                 real_height_map, real_denoised_3D_image],  #
        # real_height_map, first_channel_of_denoising_model_output, real_denoised_3D_image
    )  # currently the model only outputs the first output of the model --> see how I can fix and improve that...

    # ça ne marche pas --> le height map est en fait pas bon
    if __VERBOSE:
        model.summary(line_length=250)
    # if non_denoised_and_denoised_surface_projection_and_3D_stack_and_height_map_model is not None:
    #     model.save(os.path.join(main_path, non_denoised_and_denoised_surface_projection_and_3D_stack_and_height_map_model),
    #                include_optimizer=True,
    #                overwrite=True, save_format='h5')

    # more useful model than the previous because no 3D map here --> very good
    model = tf.keras.Model(
        inputs=[input_image_3D],
        outputs=[first_channel_of_denoising_model_output, surface_projection_before_running_2D_denoising_model,
                 real_height_map],
        # real_height_map, first_channel_of_denoising_model_output, real_denoised_3D_image
    )
    if __VERBOSE:
        model.summary(line_length=250)
    # if non_denoised_and_denoised_surface_projection_and_height_map_model is not None:
    #     model.save(os.path.join(main_path, non_denoised_and_denoised_surface_projection_and_height_map_model),
    #                include_optimizer=True,
    #                overwrite=True, save_format='h5')

    # os.path.join(main_path, non_denoised_and_denoised_surface_projection_and_height_map_model)
    if save_file_name is not None:
        model.save(save_file_name,
                   include_optimizer=False,
                   overwrite=True, save_format='h5')

    # try save this model and see what it outputs !!!

    # --> parafit --< first is denoised second is non denoised and last is height map --> can easily handle that
    # else:
    return model

    # TODO --> convert this as a method so that I can easily create it from within pyta

if __name__ == '__main__':
    surface_proj_model = '/E/models/my_model/bckup/CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/surface_projection_model.h5'
    denoiser_model = '/E/models/my_model/bckup/CARESEG_masked_data_and_masked_loss_to_prevent_identity_v1_210107_good/2D_denoiser_model.h5'  # not great on CARE great on others --> really need different denoisers depending on DATA --> ok --> though I doubt people will use CARE like data as input...
    # denoiser_model_path = '/E/models/my_model/bckup/CARESEG_another_normal_training_201216_colab_but_gave_crap/2D_denoiser_model.h5'# great for CARE not great for the others
    # denoiser_model_path = '/E/models/my_model/bckup/CARESEG_retrained_normally_201216/2D_denoiser_model.h5'# great for CARE not great for the others
    # denoiser_model_path = '/E/models/my_model/bckup/CARESEG_trained_on_CARE_data_with_dilation_good/2D_denoiser_model.h5'# great for CARE not great for the others

    save_file_name = '/E/Sample_images/sample_images_pyta/test_merged_model.h5'

    # create_surface_projection_denoise_and_height_map_combinatorial_model(surface_proj_model, denoiser_model, save_file_name=save_file_name)
    # model = create_surface_projection_denoise_and_height_map_combinatorial_model(surface_proj_model, denoiser_model)

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



    # try get just the denoiser model and see
    # print(model.layers[-3].summary(line_length=250))#perfect --> that is the 2D denoiser to which I should apply the



    final_separated_denoiser_keep_one_channel = Sequential()
    final_separated_denoiser_keep_one_channel.add(model.layers[-3])
    final_separated_denoiser_keep_one_channel.add(Lambda(lambda x: x[..., 0:1]))

    print('final denoiser onmly',final_separated_denoiser_keep_one_channel.summary(line_length=250))#perfect --> that is the 2D denoiser to which I should apply the



    # that seems to work --> could really do it


    # TODO maybe test all the possible projections