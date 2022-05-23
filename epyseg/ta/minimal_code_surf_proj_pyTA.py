# tt est pas mal j'y suis presque mais juste demander ce qu'on doit sauver en fait et aussi juste sauver les weights --> TODO
# TODO add more models to come


# in fact --> can do a combinatorial test to see which combin is best
# all is ok
# need to do model testing --> the difficulty is the combination of denoiser and surface proj, th rest is easy
#


# here define the minimal code I need to do surface proj, with and without heightmap generation, etc...

# need load a model that contains surface proj and then need load a denoiser
# could also do the projection based on height map by taking average image
# --> TODO and should not be too hard in fact!!!

# TODO rather use the code in single predict as I can do more stuff and even extend the stack
# if single predict does not exist --> do it!!!!
import traceback

import matplotlib.pyplot as plt
import numpy as np
from epyseg.deeplearning.deepl import EZDeepLearning
import os
from epyseg.img import Img, has_metadata
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential

# img = Img('/E/Sample_images/sample_images_pyta/Image49.lsm')

# TODO --> maybe do a code that selects out of 10 models --> these models are anyway so small that maybe I can store everything on githib no pb
from epyseg.ta.database.sql import TAsql, _to_dict, update_db_properties_using_image_properties
from epyseg.ta.deep.create_model_for_projection import \
    create_surface_projection_denoise_and_height_map_combinatorial_model
from epyseg.tools.early_stopper_class import early_stop
from epyseg.ta.tracking.tools import smart_name_parser

# nb in the augs I could also invert the Z axis in fact not because that would lose height
from epyseg.ta.measurements.measurements3D.height_map_to_projection import surface_projection_from_height_map


# TODO I need to test that!!!

# TODO ADD CHANNEL SUPPORT HERE THEN DONE!!

# need get the images processed and to add them to the next list

# TODO --> put the recursion as a parameter and do a code to test all possible combinations to see if any would be better

# can almost reach the best score of CARE with that --> very good in fact --> do flip my flies
def surface_projection_pyta(deepTA, input_file, progress_callback=None, cur_progress=None, save_raw_image=False,
                            channel=None, recursion_for_denoising=1): # 4 ou 5 --> super good for CARE crappy images # --> 4 is good for CARE with super crappy images  one recursion is better for the other --> only makes sens to increase the nb of recursions if image is noisy otherwise ignore
    if isinstance(input_file, list):
        processed_files = []
        for lll, file in enumerate(input_file):
            cur_progress = (lll / len(input_file)) * 100
            try:
                if early_stop.stop == True:
                    return processed_files
                result = surface_projection_pyta(deepTA, file, progress_callback=progress_callback,
                                                 cur_progress=cur_progress, save_raw_image=save_raw_image,
                                                 channel=channel, recursion_for_denoising=recursion_for_denoising)
                if result is not None:
                    processed_files.append(result)
            except:
                traceback.print_exc()
                print('Error: could not process file: "' + str(file) + '"')
        return processed_files
    parent, filename = smart_name_parser(
        input_file,
        ordered_output=['parent', 'short_no_ext'])

    if early_stop.stop == True:
        return

    if progress_callback is not None and cur_progress is not None:
        progress_callback.emit(cur_progress)

    Z_FRAMES_TO_ADD = None  # 5 #None #5 --> often None is better --> much better quality than the others --> but not always true!!! --> in fact need test all the images directly without adding extra frames in fact!!!
    # --> so my ensemble testing sucks
    # ideally always pass in the model with height map and surface proj, so that the output results are awlays the same
    result = predict_single_image(deepTA, input_file, Z_FRAMES_TO_ADD=Z_FRAMES_TO_ADD,
                                  input_normalization={'method': Img.normalization_methods[7], 'range': [2, 99.8],
                                                       'individual_channels': True, 'clip': False})


    # could then repredict based on a smaller model

    surface_proj_file_path = os.path.join(os.path.join(parent, 'surface_projection'), filename + '.tif')

    # is there really a difference or am I doing smthg wrong ???
    # plt.imshow(result[0])
    # plt.show()
    # plt.imshow(result[1])
    # plt.show()

    if not save_raw_image:
        # old
        # denoised_surface_proj = result[0]
        # Img(np.squeeze(denoised_surface_proj), dimensions='hw').save(surface_proj_file_path)

        raw_surface_proj = result[1]
        # reprocess it

        Img(np.squeeze(raw_surface_proj), dimensions='hw').save(surface_proj_file_path)

        model = deepTA.model
        try:
            # very dirty hack to apply normalization to surface proj before running the denoiser again --> totally crap and sub optimal but kinda working --> try improve that some day!!!
            tmp_model = Sequential()# get the sub model here
            tmp_model.add(model.layers[-3])
            tmp_model.add(Lambda(lambda x: x[..., 0:1]))
            deepTA.model = tmp_model
            for rrr in range(recursion_for_denoising):
                result2 = predict_single_image(deepTA, surface_proj_file_path, Z_FRAMES_TO_ADD=Z_FRAMES_TO_ADD, input_channel_of_interest=None,
                                              input_normalization={'method': Img.normalization_methods[7], 'range': [2, 99.8],
                                                                   'individual_channels': True, 'clip': False})
                denoised = result2[0]
                Img(np.squeeze(denoised), dimensions='hw').save(surface_proj_file_path)
                del result2
                del denoised
            del tmp_model
            # print('result[0]', result2[0], result2              )
        except:
            traceback.print_exc()
            print('error denoising')
        finally:
            deepTA.model = model

        # then reprocess this image with the submodel and save it


        # c'est aussi là que je dois creer ou bien mettre à jour la db de TA idealment sans casser les colonnes existantes
    else:
        raw_surface_proj = result[1]
        # get parent dir and save there in the appropriate folder
        # surface_proj_file_path = os.path.join(os.path.join(parent,'surface_projection'), filename+'.raw.tif')

        # print(surface_proj.shape, surface_proj.dtype, surface_proj_file_path)
        Img(np.squeeze(raw_surface_proj), dimensions='hw').save(surface_proj_file_path)
        # ta_path = smart_name_parser(surface_proj_file_path,ordered_output=['TA'])

    ta_path = smart_name_parser(surface_proj_file_path, ordered_output='TA')

    height_map = result[2]

    # TODO compute heigh map quality (if out of frame --> need try yet another surface projection algorithm)
    # lui faire computer et sauver ce fichier

    if Z_FRAMES_TO_ADD is not None:
        height_map = height_map - Z_FRAMES_TO_ADD  # correct height map for added black frames --> then ideally I need also compute the heightmap quality that is
    Img(np.squeeze(height_map), dimensions='hw').save(os.path.join(ta_path, 'height_map.tif'))

    # rewind the stuff and do proj
    height_map_quality_test = surface_projection_from_height_map(Img(input_file), np.squeeze(height_map),
                                                                 channel=channel)  # TODO would need hack this to support the channel to be read for all images in case they are multichannel
    Img(height_map_quality_test, dimensions='hw').save(os.path.join(ta_path, 'height_map_quality_test.tif'))

    # save parent image properties in the db in the properties table so that they can be easily reused by TA
    update_db_properties_using_image_properties(input_file, ta_path)

    return surface_proj_file_path  # this is the file that needs be loaded in the other


def surface_projection_pyta_bckup(deepTA, input_file, progress_callback=None, cur_progress=None, save_raw_image=False,
                            channel=None):
    if isinstance(input_file, list):
        processed_files = []
        for lll, file in enumerate(input_file):
            cur_progress = (lll / len(input_file)) * 100
            try:
                if early_stop.stop == True:
                    return processed_files
                result = surface_projection_pyta(deepTA, file, progress_callback=progress_callback,
                                                 cur_progress=cur_progress, save_raw_image=save_raw_image,
                                                 channel=channel)
                if result is not None:
                    processed_files.append(result)
            except:
                traceback.print_exc()
                print('Error: could not process file: "' + str(file) + '"')
        return processed_files
    parent, filename = smart_name_parser(
        input_file,
        ordered_output=['parent', 'short_no_ext'])

    if early_stop.stop == True:
        return

    if progress_callback is not None and cur_progress is not None:
        progress_callback.emit(cur_progress)

    Z_FRAMES_TO_ADD = None  # 5 #None #5 --> often None is better --> much better quality than the others --> but not always true!!! --> in fact need test all the images directly without adding extra frames in fact!!!
    # --> so my ensemble testing sucks
    # ideally always pass in the model with height map and surface proj, so that the output results are awlays the same
    result = predict_single_image(deepTA, input_file, Z_FRAMES_TO_ADD=Z_FRAMES_TO_ADD,
                                  input_normalization={'method': Img.normalization_methods[7], 'range': [2, 99.8],
                                                       'individual_channels': True, 'clip': False})

    # could then repredict based on a smaller model

    surface_proj_file_path = os.path.join(os.path.join(parent, 'surface_projection'), filename + '.tif')

    # is there really a difference or am I doing smthg wrong ???
    plt.imshow(result[0])
    plt.show()
    plt.imshow(result[1])
    plt.show()

    if not save_raw_image:
        denoised_surface_proj = result[0]
        # get parent dir and save there in the appropriate folder

        # print(surface_proj.shape, surface_proj.dtype, surface_proj_file_path)

        # could take the raw image and reprocess it separately with a subset of the model --> TODO

        Img(np.squeeze(denoised_surface_proj), dimensions='hw').save(surface_proj_file_path)

        # c'est aussi là que je dois creer ou bien mettre à jour la db de TA idealment sans casser les colonnes existantes
    else:
        raw_surface_proj = result[1]
        # get parent dir and save there in the appropriate folder
        # surface_proj_file_path = os.path.join(os.path.join(parent,'surface_projection'), filename+'.raw.tif')

        # print(surface_proj.shape, surface_proj.dtype, surface_proj_file_path)
        Img(np.squeeze(raw_surface_proj), dimensions='hw').save(surface_proj_file_path)
        # ta_path = smart_name_parser(surface_proj_file_path,ordered_output=['TA'])

    ta_path = smart_name_parser(surface_proj_file_path, ordered_output='TA')

    height_map = result[2]

    # TODO compute heigh map quality (if out of frame --> need try yet another surface projection algorithm)
    # lui faire computer et sauver ce fichier

    if Z_FRAMES_TO_ADD is not None:
        height_map = height_map - Z_FRAMES_TO_ADD  # correct height map for added black frames --> then ideally I need also compute the heightmap quality that is
    Img(np.squeeze(height_map), dimensions='hw').save(os.path.join(ta_path, 'height_map.tif'))

    # rewind the stuff and do proj
    height_map_quality_test = surface_projection_from_height_map(Img(input_file), np.squeeze(height_map),
                                                                 channel=channel)  # TODO would need hack this to support the channel to be read for all images in case they are multichannel
    Img(height_map_quality_test, dimensions='hw').save(os.path.join(ta_path, 'height_map_quality_test.tif'))

    # save parent image properties in the db in the properties table so that they can be easily reused by TA
    update_db_properties_using_image_properties(input_file, ta_path)

    return surface_proj_file_path  # this is the file that needs be loaded in the other

# in fact denoise will not be most often on a proj image but will rather be
def denoise(deepTA, input_file):
    if isinstance(input_file, list):
        for file in input_file:
            try:
                denoise(deepTA, file)
            except:
                traceback.print_exc()
                print('Error: could not process file: "' + str(file) + '"')
        return
        # return
    parent, filename = smart_name_parser(
        input_file,
        ordered_output=['parent', 'short_no_ext'])
    # apparently it's a terrible idea to add frames and it makes sense btw, even though some models do lose data --> exclude those models!!!
    Z_FRAMES_TO_ADD = None
    result = predict_single_image(deepTA, input_file, Z_FRAMES_TO_ADD=Z_FRAMES_TO_ADD,
                                  input_normalization={'method': Img.normalization_methods[7], 'range': [2, 99.8],
                                                       'individual_channels': True, 'clip': False})
    denoised = result[0]
    Img(denoised, dimensions='hw').save(
        os.path.join(os.path.join(parent, 'denoised'), filename + '.tif'))  # store in pyta folder --> TODO


def predict_single_image(deepTA, input_file, TILE_WIDTH=256, TILE_HEIGHT=256, TILE_OVERLAP=32,
                         input_channel_of_interest=None, input_normalization=None, Z_FRAMES_TO_ADD=None,
                         SIZE_FILTER=None, hq_predictions='mean'):
    input_shape = deepTA.get_inputs_shape()
    output_shape = deepTA.get_outputs_shape()

    predict_parameters = {}

    predict_parameters["input_channel_of_interest"] = input_channel_of_interest
    predict_parameters["default_input_tile_width"] = TILE_WIDTH
    predict_parameters["default_input_tile_height"] = TILE_HEIGHT
    predict_parameters["default_output_tile_width"] = TILE_WIDTH
    predict_parameters["default_output_tile_height"] = TILE_HEIGHT
    predict_parameters["tile_width_overlap"] = TILE_OVERLAP
    predict_parameters["tile_height_overlap"] = TILE_OVERLAP
    # predict_parameters["hq_pred_options"] = "Use all augs (pixel preserving + deteriorating) (Recommended for segmentation)"  # not yet supported on current versions of epyseg
    predict_parameters[
        "hq_pred_options"] = 'Only use pixel preserving augs (Recommended for CARE-like models/surface extraction)'  # not yet supported on current versions of epyseg #None #'Only use pixel preserving augs (Recommended for CARE-like models/surface extraction)'  # not yet supported on current versions of epyseg
    predict_parameters["post_process_algorithm"] = None  # 'Keep first channel only'# None
    predict_parameters["input_normalization"] = input_normalization
    predict_parameters["filter"] = SIZE_FILTER

    # print('Z_FRAMES_TO_ADD',Z_FRAMES_TO_ADD)
    # print(input_normalization)

    predict_generator = deepTA.get_predict_generator(
        inputs=[input_file], input_shape=input_shape,
        output_shape=output_shape,
        clip_by_frequency=None,
        z_frames_to_add=Z_FRAMES_TO_ADD,  # addition especially for this one
        **predict_parameters)  # before was 0.05 which is bad especially with restored images

    # run for all the datagen
    # deepTA.predict(predict_generator, output_shape, predict_output_folder=predict_output_folder, batch_size=1,**predict_parameters)

    # hq_predictions = None
    # run one by one
    final_predict_generator = predict_generator.predict_generator()
    results = deepTA.predict_single(final_predict_generator, output_shape, batch_size=1, hq_predictions=hq_predictions,
                                    **predict_parameters)

    # print(results)
    # print(len(results), results[0].shape)
    #
    # plt.imshow((np.squeeze(results[0])))
    # plt.show()

    # fairly easy now --> I can have everything

    # could make this a class and get it to save the desired image where needed
    return results


# there is a big bug --> not same output as with the default GUI
if __name__ == '__main__':

    if False:
        # input_file = '/E/Sample_images/sample_images_pyta/surface_projection/210219.lif_t000.tif'
        input_file = '/E/Sample_images/sample_images_pyta/210219.lif_t000.tif'
        ta_path = smart_name_parser(input_file, 'TA')
        update_db_properties_using_image_properties(input_file,ta_path) # should do the job
        # try:
        #     tmp = Img(input_file)
        #     if has_metadata(tmp):
        #         db_path = os.path.join(ta_path, 'pyta.db')
        #         db = TAsql(db_path)
        #         try:
        #             # these are all the variables of the db I may want to keep...
        #             voxel_size_x = None
        #             voxel_size_y = None
        #             voxel_size_z = None
        #             voxel_z_over_x_ratio = None
        #             time = None
        #             creation_time = None
        #
        #             if 'vx' in tmp.metadata:
        #                 voxel_size_x = tmp.metadata['vx']
        #             if 'vy' in tmp.metadata:
        #                 voxel_size_y = tmp.metadata['vy']
        #             if 'vz' in tmp.metadata:
        #                 voxel_size_z = tmp.metadata['vz']
        #             if 'AR' in tmp.metadata:
        #                 voxel_z_over_x_ratio = tmp.metadata['AR']
        #             if 'time' in tmp.metadata:
        #                 time = tmp.metadata['time']
        #             if 'creation_time' in tmp.metadata:
        #                 creation_time = tmp.metadata['creation_time']
        #
        #             neo_data = {'voxel_size_x': voxel_size_x,
        #                         'voxel_size_y': voxel_size_y,
        #                         'voxel_size_z': voxel_size_z,
        #                         'voxel_z_over_x_ratio': voxel_z_over_x_ratio,
        #                         'time': time,
        #                         'creation_time': creation_time}
        #
        #             if db.exists('properties'):
        #                 # db exists --> update it rather than recreating everything
        #                 header, cols = db.run_SQL_command_and_get_results('SELECT * FROM properties', return_header=True)
        #                 cols = cols[0]
        #                 data = _to_dict(header, cols)
        #                 for key in list(neo_data.keys()):
        #                     if neo_data[key] is None:
        #                         if not key in data:
        #                             data[key] = neo_data[key]
        #                     else:
        #                         data[str(key)] = neo_data[key]
        #             else:
        #                 data = neo_data
        #
        #             data = {k: [v] for k, v in data.items()}
        #             db.create_and_append_table('properties', data)
        #         except:
        #             traceback.print_exc()
        #             print('An error occurred while reading properties from the image or writing them to the database')
        #         finally:
        #             try:
        #                 db.close()
        #             except:
        #                 pass
        #     # else:
        #     #     # image has no metadata --> nothing TODO --> I can just skip things
        #     #     pass
        #     del tmp
        # except:
        #     traceback.print_exc()
        #     print('Error could not save image properties to the TA database')

        import sys
        sys.exit(0)

    # faire un fast mode
    # could offer both offers

    # TODO also store the converted height map to smthg I

    use_cpu = True
    deepTA = EZDeepLearning(use_cpu=use_cpu)
    # best on my data
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARE_model_trained_MAE_in_epyseg_29-1_50steps_per_epoch_not_outstanding_though/surface_projection_model.h5')
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARE_model_trained_MSE_error_instead_MAE_in_epyseg_29-1/surface_projection_model.h5')
    # best consensus choice I guess # but I would most likely so far need to offer more choices
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/surface_projection_model.h5') # maybe the best but really needs no addition of extra frames !!!!!# for this model extraframes are required otherwise signal is missing # very good on CARE too --> IN FACT QUITE GOOD ON BOTH --> MAYBE THE BEST BY DEFAULT STUFF # also has noise for the iamges of manue --> in that case there would be better choices
    # deepTA.load_or_build(model='/E/Sample_images/sample_images_pyta/test_merged_model.h5')  # TODO --> replace this by online model --> TODO


    # surface_proj_model = '/E/models/my_model/bckup/CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/surface_projection_model.h5'
    # denoiser_model = '/E/models/my_model/bckup/CARESEG_masked_data_and_masked_loss_to_prevent_identity_v1_210107_good/2D_denoiser_model.h5'  # not great on CARE great on others --> really need different denoisers depending on DATA --> ok --> though I doubt people will use CARE like data as input...
    # denoiser_model = '/E/models/my_model/bckup/CARESEG_another_normal_training_201216_colab_but_gave_crap/2D_denoiser_model.h5'# great for CARE not great for the others
    # denoiser_model = '/E/models/my_model/bckup/CARESEG_retrained_normally_201216/2D_denoiser_model.h5'# great for CARE not great for the others
    # denoiser_model = '/E/models/my_model/bckup/CARESEG_trained_on_CARE_data_with_dilation_good/2D_denoiser_model.h5'# great for CARE not great for the others
    surface_proj_model = 'SURFACE_PROJECTION_2'
    # surface_proj_model = 'SURFACE_PROJECTION_3'
    # surface_proj_model = 'SURFACE_PROJECTION_4'
    # denoiser_model = '2D_DENOISER'
    denoiser_model = '2D_DENOISEG'

    # pb is that
    model = create_surface_projection_denoise_and_height_map_combinatorial_model(surface_proj_model, denoiser_model)

    # shall I create a submodel ????

    print(model.summary(line_length=250))
    deepTA.model = model

    # deepTA.load_or_build(model='/E/models/my_model/bckup/FULL_CARE_MODEL_CONSENSUS_TRAINING_VERY_GOOD_140721/surface_projection_model.h5') # # nb adding frames to this one gives really crappy output --> do not do it!!! newly trained model --> really not bad in fact
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_second_correct_tested_210219_outstanding_denoise/surface_projection_model.h5')

    # best on CARE data
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_another_normal_training_201216_colab_but_gave_crap/surface_projection_model.h5')
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_masked_data_and_masked_loss_to_prevent_identity_v1_210107_good/surface_projection_model.h5') # works a bit on my data but less well than the one above
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_my_loss_mae_first_chan_dice_other_chans/surface_projection_model.h5')
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_trained_on_CARE_data_with_dilation_good/surface_projection_model.h5')
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_Zroll_models_trained_on_colab_201215/surface_projection_model.h5')

    # try find best denoiser!!!
    # not great # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_masked_data_and_masked_loss_to_prevent_identity_v1_210107_good/2D_denoiser_model.h5') # ok denoiser but not great on CARE!!!
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_another_normal_training_201216_colab_but_gave_crap/2D_denoiser_model.h5') # excellent on CARE not great on my own data
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_my_loss_mae_first_chan_dice_other_chans/2D_denoiser_model.h5') # not so bad on my data but weak signal
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_retrained_normally_201216/2D_denoiser_model.h5')
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_shuffle_n_roll_models_trained_on_colab_201215/2D_denoiser_model.h5') # a bit better on my data but not great (very weak in fact)
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_test_combined_loss_properly_made_with_reduce_mean_220105/2D_denoiser_model.h5')# second best on my data, very weak

    # second tests denoiser of my data
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARE_model_trained_MAE_in_epyseg_29-1_50steps_per_epoch_not_outstanding_though/2D_denoiser_model.h5') # good on my data but check on CARE ??? --> not great on CARE --> need really check individual ones --> TODO
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_first_correct_test/2D_denoiser_model.h5') # a bit artificial
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/2D_denoiser_model.h5') # CRAP
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_second_correct_tested_210219_outstanding_denoise/2D_denoiser_model.h5')

    # common
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_shuffle_n_roll_models_trained_on_colab_201215/2D_denoiser_model.h5')
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_test_combined_loss_properly_made_with_reduce_mean_220105/2D_denoiser_model.h5')
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_trained_different_losses_for_channels_mae_bce/2D_denoiser_model.h5') # not too bad too --> maybe the most natural one ...
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_trained_on_CARE_data_with_dilation_good/2D_denoiser_model.h5') # BEST denoiser maybe best but not great --> still maybe best
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_Zroll_models_trained_on_colab_201215/2D_denoiser_model.h5') # maybe best # second best but still not so great, in a way specific denoisers need be used

    # if I just save model weights, it's gonna really take no space... --> maybe it's a good idea !!!
    # test of a consensus denoiser
    # deepTA.load_or_build(model='/home/aigouy/mon_prog/Python/epyseg_pkg/personal/training_deep/consensus_denoise_norm_input_output_none_training_150_epochs_lr_0.001_test2/2D_denoiser_model-0.h5') # not that bad in fact, check on care # good on my data but not great on CARE --> maybe offer different stuff
    # deepTA.load_or_build(model='/home/aigouy/mon_prog/Python/epyseg_pkg/personal/training_deep/consensus_denoise_norm_input_output_none_training_150_epochs_lr_0.001/2D_denoiser_model-0.h5') # not that bad in fact, check on care # good on my data but not great on CARE --> maybe offer different stuff
    # deepTA.load_or_build(model='/home/aigouy/mon_prog/Python/epyseg_pkg/personal/training_deep/2D_denoiser_model-0.h5') # not that bad in fact, check on care # good on my data but not great on CARE --> maybe offer different stuff
    # deepTA.load_or_build(model='/home/aigouy/mon_prog/Python/epyseg_pkg/personal/training_deep/2D_denoiser_model-0-0.h5') # not that bad in fact, check on care # good on my data but not great on CARE --> maybe offer different stuff
    # deepTA.load_or_build(model='/home/aigouy/mon_prog/Python/epyseg_pkg/personal/training_deep/2D_denoiser_model-0-0-0.h5') # not that bad in fact, check on care # good on my data but not great on CARE --> maybe offer different stuff
    # deepTA.load_or_build(model='/home/aigouy/mon_prog/Python/epyseg_pkg/personal/training_deep/consensus_denoise_normalized_input_output_training_100_epochs_lr_0.004_test5_MAE/2D_denoiser_model-0-0-0.h5') # VERY GOOD DENOISER but really ugly on CARE --> I will not manage --> do the ensemble code

    # test code find bug
    # deepTA.load_or_build(model='/home/aigouy/mon_prog/Python/epyseg_pkg/personal/training_deep/consensus_denoise_normalized_input_output_training_100_epochs_lr_0.004_test5_MAE/2D_denoiser_model-0-0-0.h5')  # VERY GOOD DENOISER but really ugly on CARE --> I will not manage --> do the ensemble code
    # the denoiser sucks
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/CARESEG_model-0-0.h5')  # for this model extraframes are required otherwise signal is missing # very good on CARE too --> IN FACT QUITE GOOD ON BOTH --> MAYBE THE BEST BY DEFAULT STUFF # also has noise for the iamges of manue --> in that case there would be better choices

    # do for one image then ask which models to take --> see how much can be downloaded at the same time ???
    # doable with gitlab but will take a bit of space but ok maybe
    # maybe do a selection of the models...
    # faire une selection image par image --> pas trop bete en fait
    # faire max proj

    # TOD try my consensus denoiser just in case!!!

    # try all models on one image

    # others
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARE_masked_data_and_partial_masked_loss_to_limit_identity_but_not_remove_it_v1_210120_not_outstanding_CARESEG_is_better/surface_projection_model.h5')
    # deepTA.load_or_build(model='/E/models/my_model/bckup/CARESEG_trained_on_CARE_data_with_dilation_good/surface_projection_model.h5') # does not work well on my own data but does great on CARE

    # now really need to find a denoiser that I can append to then all done and since extra frames need do a subtraction to the file containing

    # TODO make it predict just for a single image and control everything --> in a way that would be much simpler to handle à la TA

    deepTA.get_loaded_model_params()
    deepTA.summary()

    IS_TA_OUTPUT_MODE = True  # stores as handCorrection.tif in the folder with the same name as the parent file without ext
    input_channel_of_interest = None  # 1  # assumes image is single channel or multichannel nut channel of interest is ch0, needs be changed otherwise, e.g. 1 for channel 1
    TILE_WIDTH = 512  # 256  # 128 # 64
    TILE_HEIGHT = 512  # 256  # 128 # 64
    TILE_OVERLAP = 32

    Z_FRAMES_TO_ADD = 5  # 5 #None
    # EPYSEG_PRETRAINING = 'Linknet-vgg16-sigmoid-v2'  # or 'Linknet-vgg16-sigmoid' for v1
    SIZE_FILTER = None  # 100 # set to 100 to get rid of cells having pixel area < 100 pixels

    # input_normalization = {'method': 'Rescaling (min-max normalization)', 'range': [0, 1], 'individual_channels': True}
    # input_normalization = {'method': Img.normalization_methods[7], 'range': [2, 99.8],'individual_channels': True, 'clip': False}
    input_normalization = None

    # INPUT_FOLDER = '/path/to/files_to_segment/'

    # surface proj tests

    # INPUT_FILE = '/E/Sample_images/sample_images_pyta/exposures_1_P04.tif'
    # INPUT_FILE = '/E/Sample_images/sample_images_pyta/210219.lif_t000.tif'
    # INPUT_FILE = '/E/Sample_images/sample_images_pyta/Image49.lsm'
    # INPUT_FILE = '/E/Sample_images/sample_images_pyta/Image4.lsm'

    # INPUT_FILE = ['/E/Sample_images/sample_images_pyta/exposures_1_P04.tif', '/E/Sample_images/sample_images_pyta/210219.lif_t000.tif', '/E/Sample_images/sample_images_pyta/Image49.lsm', '/E/Sample_images/sample_images_pyta/Image4.lsm']

    INPUT_FILE = ['/E/Sample_images/sample_images_denoise_manue/210922_armGFP_suz_40-54hAPF_ON.lif - Series004_t002_reveresed.tif',
                    # '/E/Sample_images/sample_images_pyta/exposures_1_P04.tif',
                  # '/E/Sample_images/sample_images_pyta/210219.lif_t000.tif',
                  # '/E/Sample_images/sample_images_pyta/Image48.lsm',
                  #   '/E/Sample_images/sample_images_pyta/Image4.lsm'
                  ]

    # denoiser tests
    # INPUT_FILE = '/E/Sample_images/sample_images_pyta/focused_Series012.png'
    # INPUT_FILE = '/E/Sample_images/sample_images_pyta/Image29.tif'
    # INPUT_FILE = '/E/Sample_images/sample_images_pyta/210219.lif_t000/handCorrection.tif'
    # INPUT_FILE = '/E/Sample_images/sample_images_pyta/Image49/handCorrection.tif'
    # INPUT_FILE = '/E/Sample_images/sample_images_pyta/exposures_1_P04/handCorrection.tif'

    # result = predict_single_image(deepTA, INPUT_FILE, TILE_WIDTH=TILE_WIDTH, TILE_HEIGHT=TILE_HEIGHT, TILE_OVERLAP=TILE_OVERLAP, input_normalization=input_normalization, SIZE_FILTER=SIZE_FILTER, Z_FRAMES_TO_ADD=Z_FRAMES_TO_ADD, input_channel_of_interest=input_channel_of_interest)
    # print(result[0].shape)

    # TODO create the library the proper way also see how I can connect directly the denoiser
    # surface_projection_pyta(deepTA, INPUT_FILE,recursion_for_denoising=4)
    surface_projection_pyta(deepTA, INPUT_FILE, recursion_for_denoising=1)

    # finalize all now then ok, transfer files to the stuff
    # sinon dire ou sauver le fichier et tt faire comme il faut
    # sinon si max proj --> sauver dans le fichier correspondant --> non en fait c'est ok!
