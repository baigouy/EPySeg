# TODO add glob to loadlist to load generic lists --> good idea I think and offer various sorting algos including natsort
# a tool to load lists that is smarter than the TA list loader

import os
from natsort import natsorted

# from epyseg.img import _transfer_voxel_size_metadata
from epyseg.tools.logger import TA_logger # logging
import glob
import shutil

logger = TA_logger()

# if ouput name is
# if save is True then save
def create_list(input_folder, save=False, output_name=None, extensions=['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tga', '*.lif'], sort_type='natsort'):
    lst = []
    if not extensions:
        logger.error('Extensions are not defined --> list cannot be created')
        return lst
    path = input_folder
    if not path.endswith('/') and not path.endswith(
            '\\') and not '*' in path and not os.path.isfile(path):
        path+='/'
    # print('corrected path', path)
    if not '*' in path:
        for ext in extensions:
            if not ext.startswith('*'):
                ext='*'+ext
            lst += glob.glob(path + ext)
    else:
        lst+=glob.glob(path)
    if sort_type=='natsort':
        lst=natsorted(lst)
    if save:
        if output_name is None:
            output_name = os.path.join(input_folder,'list.lst')
        save_list_to_file(lst,output_name)
    return lst


def loadlist(txtfile, always_prefer_local_directory_if_file_exists=True, filter_existing_files_only=False, smart_local_folder_detection=True, skip_hashtag_commented_lines=True, skip_empty_lines=True):
    if not os.path.exists(txtfile) or not os.path.isfile(txtfile):
        if not '*' in txtfile and not os.path.isdir(txtfile):
            return None
        else:
            # try creating a list
            # print("here")
            lst = create_list(txtfile, save=False)
    else:
        with open(txtfile) as f:
            if not skip_hashtag_commented_lines:
                lst = [line.rstrip() for line in f]
            else:
                # exclude commented lines anyways they will make the software crash...
                lst = [line.rstrip() for line in f if not line.rstrip().startswith('#')]
        if skip_empty_lines:
            # excludes lines containing spaces only/empty lines (this also fixes a bug that relative path cannot be found if the file contains an empty line), so altogether I think this should be on by default.
            lst = [line.rstrip() for line in lst if not line.strip() == '']

    if always_prefer_local_directory_if_file_exists:
        logger.debug('Files located in the .lst/.txt file containing folder will be preferred, if they exist.')
        # this should probably be improved for complex dir structure I would need to find the common root of all paths and apply some voodoo to keep this --> think about it for future dvpt
        list_file_directory = os.path.dirname(txtfile)
        if not smart_local_folder_detection:
            lst = [os.path.join(list_file_directory, os.path.basename(line)) if os.path.isfile(os.path.join(list_file_directory, os.path.basename(line))) else line for line in lst]
        else:
            # find common root path and apply local path to common path --> much better list opening than above (because supports sub folders too), smarter code than what I had done in TA
            common_path = os.path.commonprefix(lst)
            # print('common_path', common_path)
            if not os.path.isdir(common_path):
                common_path = os.path.dirname(common_path)
            # print(os.path.isdir(common_path))
            # print('common_path',common_path)
            if common_path != '':
                relative_paths = [os.path.relpath(path, common_path) for path in lst]
                lst = [os.path.join(list_file_directory, rel) if os.path.isfile(os.path.join(list_file_directory, rel)) else line for rel, line in zip(relative_paths, lst)]

    if filter_existing_files_only:
        logger.debug('Checking list for existing files')
        # check whether files exist otherwise skip
        lst = [line for line in lst if os.path.isfile(line)]

    return lst

# TODO maybe do
# this class will inject files in TA folder --> can be used to easily reinject deep learning generated files into TA folders

def TA_smart_pairing_list(TA_list, corresponding_list_of_files_to_inject_or_folder, stop_on_error=False):
    if TA_list.lower().endswith('.txt') or TA_list.lower().endswith('.lst'):
        TA_list = loadlist(TA_list)




    if os.path.isdir(corresponding_list_of_files_to_inject_or_folder):
        # assume same name as for input
        # corresponding_list_of_files_to_inject_or_folder
        matching_list = []
        for file in TA_list:
            filename0_without_path = os.path.basename(file)
            # filename0_without_ext = os.path.splitext(filename0_without_path)[0] # TODO maybe allow the user to change the ext some day --> can inject text files or alike then!!!
            matching_list.append(os.path.join(corresponding_list_of_files_to_inject_or_folder, filename0_without_path))
    elif corresponding_list_of_files_to_inject_or_folder.lower().endswith('.txt') or corresponding_list_of_files_to_inject_or_folder.lower().endswith('.lst'):
        matching_list =  loadlist(corresponding_list_of_files_to_inject_or_folder)

    # try zipping and print correspondance and if fails try smart matching --> but warn maybe
    if len(matching_list) != len(TA_list):
        if stop_on_error:
            print('input lists/folders don\'t match', len(TA_list), len(matching_list))
            return
        else:
            # try some rescue
            # if there is a match in name --> just use that
            # TODO implement that
            # maybe use that for max proj and alike stuff
            # use classical matching patterns and exclude extension --> TODO
            # do a matching without repick
            # can allow to macth with generic names added as prefix of suffix before extension such as proj_, max_,  _test,...

            # TODO implement that --> really useful for real TA but ok for now
            pass

    # print(matching_list)
    # print(list(zip(matching_list, TA_list)))


    zipped = list(zip(matching_list, TA_list))
    return zipped

def TA_injector(TA_list, corresponding_list_of_files_to_inject_or_folder, generic_TA_name_for_injected=None, stop_on_error=True, move_file=False, simulate_only = False):
    # if folder assume same name without the extension and inject it

    zipped = TA_smart_pairing_list(TA_list,corresponding_list_of_files_to_inject_or_folder)
    if zipped is None: #
        # nothing todo --> maybe return an error
        return
    # print(matching_things)

    for matching_things in zipped:
        # I always use this --> put it in a generic class because it can be useful
        file_to_copy = matching_things[0]
        file_to_copy_without_path = os.path.basename(file_to_copy)



        file = matching_things[-1]




        filename0_without_path = os.path.basename(file)
        filename0_without_ext = os.path.splitext(filename0_without_path)[0]
        path = os.path.dirname(file)
        destination_folder = os.path.join(path, filename0_without_ext)

        destination_name = os.path.join(destination_folder, file_to_copy_without_path if generic_TA_name_for_injected is None else generic_TA_name_for_injected)

        # make sure the folder exists for the copy
        if not simulate_only:
            os.makedirs(os.path.dirname(destination_folder), exist_ok=True)
        if not move_file:
            print('copy', file_to_copy, '-->', destination_name)
            if simulate_only:
                continue
            shutil.copy2(file_to_copy, destination_name)
        else:
            print('move', file_to_copy, '-->', destination_name)
            if simulate_only:
                continue
            shutil.move(file_to_copy, destination_name)


# creates based on the main TA list a list of files with alternative names and returns it
# NB will only change name if the file with the alternate name exists --> I think this is the behaviour I want but ok
def smart_TA_list(main_list, name_to_search_first, alternative_name_if_name_to_search_first_does_not_exist=None):
    lst = [os.path.join(os.path.splitext(line)[0], name_to_search_first) for line in main_list]
    if alternative_name_if_name_to_search_first_does_not_exist is not None:
        lst = [line if os.path.isfile(line) or not os.path.isfile(os.path.join(os.path.dirname(line), alternative_name_if_name_to_search_first_does_not_exist)) else os.path.join(os.path.dirname(line), alternative_name_if_name_to_search_first_does_not_exist) for line in lst]
    return lst

# files are separated by commas by default
def save_list_to_file(lst, filename, col_separator='\t'):
    # TODO --> maybe add mkdirs...
    if lst is not None:
        with open(filename, 'w') as f:
            for item in lst:
                if not isinstance(item,list) and not isinstance(item, tuple):
                    # TODO maybe do a smarter split if item is itself a list --> then use file sep as a parser
                    f.write("%s\n" % item)
                else:
                    # if item is a list save it as a csv or tab or any other user defined separated text
                    f.write("%s\n" % col_separator.join([str(i) for i in item]))
        return lst
    else:
        raise ValueError('Empty list --> cannot save it to a file')
        return lst

# try TODO a code that transfers metadata in files

# TODO peut etre faire un list executor --> that runs a code for all the files of a list and maybe say whether can be MTed of not
# --> maybe convert this to a class --> think about it for future dev of pyTA the future TA?

# this gives the optimal folder name for saving images without requiring other stuff, from various models --> use that for ensemble tools
def smart_output_folder_name(lst):
    # gets the smartest name for the output folder can be used for predict --> and does not require to store model name as a txt file
    # get common path then replace / or \ by _ to get a smart and meaningful minimal name even when models change!!
    common_path = os.path.commonprefix(lst)
    # print('common_path', common_path)
    if not os.path.isdir(common_path):
        common_path = os.path.dirname(common_path)

    print(common_path)

    if common_path == '':
        return None

    smart_name = [os.path.splitext(name.replace(common_path,'').replace('/','_').replace('\\','_'))[0] for name in lst]

    # then remove file extension

    # remove / if name starts with that
    smart_name = [name if not name.startswith('_') else name[1:] for name in smart_name]

    print(smart_name)



    return smart_name

# can create a list of files with the same name across several folders generated by different models for example --> can then be used for ensemble or averaging...
def get_transversal_list(root_path):

    transverse_list =[]
    # this lists all the folders --> easy to get in fact !!!

    # root_path = '/E/Sample_images/Consensus_learning/gray/CARE/'
    folders = os.listdir(root_path)
    folders = [os.path.join(root_path, folder) for folder in folders]
    folders = natsorted(folders)
    # list of all folders within the folder

    files_in_first_folder = os.listdir(folders[0])

    # DO a list of lists

    # list all files in first folder and if present in others then add them
    for file in files_in_first_folder:
        # images = {}
        multifolder_list = []
        for folder in folders:
            final_file = os.path.join(folder, file)
            if os.path.isfile(final_file):
                try:
                    multifolder_list.append(final_file)
                    # img = Img(final_file)
                    # if force_channel is not None:
                    #     if img.has_c():
                    #         img = img[..., force_channel]
                    # folder_name = os.path.basename(os.path.dirname(final_file))
                    # images[folder_name] = img
                    # # print(img.shape)
                    # # print('valid', final_file)
                except:
                    print('invalid file', final_file)
        transverse_list.append(multifolder_list)

    return transverse_list


if __name__ == '__main__':

    if True:
        transversal_list =get_transversal_list('/E/Sample_images/Consensus_learning/gray/CARE/')
        print(transversal_list)
        # for each can do an avg for example

        import sys
        sys.exit(0)


    if True:
        import sys
        models = [
            'CARE_model_trained_MAE_in_epyseg_29-1_50steps_per_epoch_not_outstanding_though/CARE_model-0.h5',  # 0 #####
            'CARE_model_trained_MSE_error_instead_MAE_in_epyseg_29-1/CARE_model-0.h5', #1
            'CARESEG3D_models_trained_on_colab_201214/CARESEG3D_model-0.h5', #2
            'CARESEG_another_normal_training_201216_colab_but_gave_crap/CARESEG_model-0.h5', # 3 # not bad in fact especially when there is an amazing amount of noise because it is an amazing denoiser...
            'CARESEG/CARESEG_model-0.h5', #4
            'CARESEG_first_correct_test/CARESEG_model-0-0-0.h5', #5
            ## 'CARESEG_first_correct_test/CARESEG_model-0-0-4.h5', #6
            'CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/CARESEG_model-0-0.h5', #7
            # 'CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/CARESEG_model-0-1.h5', #8
            # 'CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/CARESEG_model-0-2.h5', #9
            # 'CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/CARESEG_model-0-3.h5',#10
            # 'CARESEG_first_successful_retrain_on_CARE_data_very_very_good_surface_proj_but_not_for_all/CARESEG_model-0-4.h5',#11
            'CARESEG_masked_data_and_masked_loss_to_prevent_identity_v1_210107_good/CARESEG_model-0.h5',#12
            'CARESEG_masked_data_and_partial_masked_loss_to_limit_identity_but_not_remove_it_v1_210107_outstanding_on_some_just_good_on_others/CARESEG_model-0.h5',
            #####
            # 13
            'CARESEG_masked_data_and_partial_masked_loss_to_limit_identity_but_not_remove_it_v2_zRoll_stack_210108_quite_good_but_a_bit_low_signal/CARESEG_model-0.h5',
            ##### doesn't seem to require extra black frames and denoiser sucks because it has a lot of noise --> see the best of all
            ##########DO NOT USE NEO######## 'CARESEG_masked_data_and_partial_masked_loss_to_limit_identity_but_not_remove_it_v2_zRoll_stack_210108_quite_good_but_a_bit_low_signal/complete_model_with_3_outputs.h5', # NEO
            # 14
            'CARESEG_my_loss_mae_first_chan_dice_other_chans/CARESEG_model-0.h5',#15
            'CARESEG_RETRAINED_NEW_LOSS_DICE_201215/CARESEG_model-0.h5',#16
            'CARESEG_retrained_normally_201216/CARESEG_model-0.h5',  # 17 #####
            'CARESEG_second_correct_tested_210219_outstanding_denoise/CARESEG_model-0.h5',  # 18 #####
            'CARESEG_second_successful_retrain_on_CARE_data_better_than_previous_need_longer_training_and_more_diverse_one_too/CARESEG_model-0.h5',#19
            'CARESEG_shuffle_n_roll_models_trained_on_colab_201215/CARESEG_model-0.h5',#20
            'CARESEG_test_combined_loss_properly_made_with_reduce_mean_220105/CARESEG_model-0.h5',#21
            'CARESEG_test_combined_mae_jaccard_loss_properly_made_with_reduce_mean_220105_terrible_no_clue_why/CARESEG_model-0.h5',# 22
            'CARESEG_trained_different_losses_for_channels_mae_bce/CARESEG_model-0.h5', # 23
            'CARESEG_trained_on_CARE_data_with_dilation_good/CARESEG_model-0.h5',# 24
            'CARESEG_Zroll_models_trained_on_colab_201215/CARESEG_model-0.h5',# 25
            'CARE_trained_on_its_data_and_mine_not_working_on_its/CARE_model-0.h5',# 26
            'CARE_training_on_its_own_data_with_CARESEG/CARESEG_model-0-0.h5',# 27
            # 'CARE_training_on_its_own_data_with_CARESEG/CARESEG_model-0-4.h5',# 28
            'retrained_CARE_my_soft_fixed_normalization_not_outstanding_but_improving/CARE_model-0.h5',# 29
            'CARE_finally_working_retrain_with_CARE_normalization/CARE_model-0.h5',  # 30
            'CARE_models_trained_on_colab_201214/CARE_model-0.h5',  # 31 #####
            'CARE_masked_data_and_partial_masked_loss_to_limit_identity_but_not_remove_it_v1_210120_not_outstanding_CARESEG_is_better/CARE_model-0.h5',# 32
        ]

        for idx, model in enumerate(models):
            models[idx] = os.path.join('/E/models/my_model/bckup/', model)

        print(models)

        smart_output_folder_name(models)

        # get smart_output_folder_name

        sys.exit(0)


    if False:
        import sys

        # THIS CODE COPIES METADATA FROM ONE FILE TO ANOTHER
        # CAN I USE LEICA DATA TODO STUFF
        # I may still need a time file at least for a while until I change my other tools but that could be useful still
        # TODO --> make this its own function or class !!!!
        # do a generic executer within this class and say whether it can be Mted or not ... --> Could be useful
        list_file = [
            '/E/Sample_images/sample_images_denoise_manue/210402_EcadKI_mel_female_26hAPF_pupae_from_old_tube/predict/predict_model_nb_0_not_best_but_very_good_still/list.lst']  # done 3D
        list_file.append(
            '/E/Sample_images/sample_images_denoise_manue/210324_ON_suz_22h45_armGFP_line2/predict/predict_model_nb_0/mini_list.lst')
        list_file.append(
            '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/21-03/GT/predict/predict_model_nb_3/list.lst')
        list_file.append(
            '/E/Sample_images/sample_images_denoise_manue/210312_armGFP_line2_suz_39h30APF/predict/predict_model_nb_0/list.lst')

        list_file.append(
            '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/25-1/GT/predict/predict_model_nb_3/list.lst')
        list_file.append(
            '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/070219/predict/predict_model_nb_0/list.lst')
        list_file.append(
            '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/20-2/predict/predict_model_nb_0/list.lst')
        list_file.append(
            '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/210219/predict/predict_model_nb_0/list.lst')
        list_file.append(
            '/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/27-02/predict/predict_model_nb_0/list.lst')
        list_file.append(
            '/E/Sample_images/sample_images_denoise_manue/210409_EcadKI_ovipo_female_around_40hAPF/predict/predict_model_nb_4/list.lst')
        # list_file = []
        list_file.append('/E/Sample_images/sample_images_denoise_manue/210121_armGFP_suz_line2_47h30_APF/predict/before_crash/predict_model_nb_0/mini_list.lst')

        # list_file = []
        list_file.append(
            '/E/Sample_images/sample_images_denoise_manue/201104_armGFP_different_lines_tila/predict/predict_model_nb_0/mini_list.lst')  # TODO

        # HACK FOR THE LAST BECAUSE IT IS INCORRECT!!!
        # zipped = TA_smart_pairing_list('/E/Sample_images/sample_images_denoise_manue/201104_armGFP_different_lines_tila/predict/predict_model_nb_0/registered/splitted/list2_matching.lst', '/E/Sample_images/sample_images_denoise_manue/201104_armGFP_different_lines_tila/list1.lst')

        # map 2 lists
        for lst in list_file:

            parent_folder = heightmap_folder = lst.split('/predict/')[0]
            # print(os.path.exists(heightmap_folder))
            if not os.path.exists(parent_folder):
                print('error', parent_folder)

            # print(parent_folder)

            # zipped = TA_smart_pairing_list(lst, parent_folder)

            # HACK FOR THE LAST BECAUSE IT IS INCORRECT!!!
            # zipped = TA_smart_pairing_list('/E/Sample_images/sample_images_denoise_manue/201104_armGFP_different_lines_tila/predict/predict_model_nb_0/registered/splitted/list2_matching.lst', '/E/Sample_images/sample_images_denoise_manue/201104_armGFP_different_lines_tila/list1.lst')
            zipped = TA_smart_pairing_list(lst, parent_folder)
            # print(zipped)
            for files in zipped:
                print('transferring metadata from', files[0], 'to', files[1])
                _transfer_voxel_size_metadata(files[0], files[1])#.replace('.tif','.lsm')
                # pass
        sys.exit(0)

    if True:
        # TODO add glob


        import sys
        # copy and inject height maps to the file
        # TA_injector
        simulate_only = False
        generic_TA_name = 'height_map.tif'

        list_file = ['/E/Sample_images/sample_images_denoise_manue/210402_EcadKI_mel_female_26hAPF_pupae_from_old_tube/predict/predict_model_nb_0_not_best_but_very_good_still/list.lst']  # done 3D

        list_file.append('/E/Sample_images/sample_images_denoise_manue/210324_ON_suz_22h45_armGFP_line2/predict/predict_model_nb_0/mini_list.lst')
        list_file.append('/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/21-03/GT/predict/predict_model_nb_3/list.lst')
        list_file.append('/E/Sample_images/sample_images_denoise_manue/210312_armGFP_line2_suz_39h30APF/predict/predict_model_nb_0/list.lst')
        list_file.append('/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/25-1/GT/predict/predict_model_nb_3/list.lst')
        list_file.append('/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/070219/predict/predict_model_nb_0/list.lst')
        list_file.append('/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/20-2/predict/predict_model_nb_0/list.lst')
        list_file.append('/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/210219/predict/predict_model_nb_0/list.lst')
        list_file.append('/E/Sample_images/sample_images_denoise_manue/fullset_Manue/Ovipositors/27-02/predict/predict_model_nb_0/list.lst')
        list_file.append('/E/Sample_images/sample_images_denoise_manue/210409_EcadKI_ovipo_female_around_40hAPF/predict/predict_model_nb_4/list.lst')
        list_file.append(
            '/E/Sample_images/sample_images_denoise_manue/210121_armGFP_suz_line2_47h30_APF/predict/before_crash/predict_model_nb_0/mini_list.lst')
        # TODO hack the last one
        # list_file = []
        # list_file.append('/E/Sample_images/sample_images_denoise_manue/201104_armGFP_different_lines_tila/predict/predict_model_nb_0/registered/splitted/mini_list.lst')
        list_file.append('/E/Sample_images/sample_images_denoise_manue/201104_armGFP_different_lines_tila/predict/predict_model_nb_0/mini_list.lst')

        for lst in list_file:
            # TA_injector('/E/Sample_images/sample_images_denoise_manue/210402_EcadKI_mel_female_26hAPF_pupae_from_old_tube/predict/predict_model_nb_0_not_best_but_very_good_still/list.lst', '/E/Sample_images/sample_images_denoise_manue/210402_EcadKI_mel_female_26hAPF_pupae_from_old_tube/predict/height_map_test1', simulate_only=simulate_only, generic_TA_name_for_injected=generic_TA_name)
            heightmap_folder = lst.split('/predict/')[0]+'/predict/height_maps/'
            # print(os.path.exists(heightmap_folder))
            if not os.path.exists(heightmap_folder):
                print(heightmap_folder)
            # TA_injector(list, '/E/Sample_images/sample_images_denoise_manue/210402_EcadKI_mel_female_26hAPF_pupae_from_old_tube/predict/height_map_test1', simulate_only=simulate_only, generic_TA_name_for_injected=generic_TA_name)
            TA_injector(lst, heightmap_folder, simulate_only=simulate_only, generic_TA_name_for_injected=generic_TA_name)
        sys.exit(0)

    # TODO transform this into tests (just create some tmp files)
    print(loadlist('/E/Sample_images/sample_images_denoise_manue/200722_armGFP_suz_ON_47hAPF/predict/predict_model_nb_4/list.lst'))
    print(loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini/list.lst'))
    print(loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini/list.lst', always_prefer_local_directory_if_file_exists=False))
    print(loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini/list.lst', always_prefer_local_directory_if_file_exists=False, filter_existing_files_only=True))
    print(loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini/list.lst', always_prefer_local_directory_if_file_exists=True, filter_existing_files_only=True))
    print(loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini/list_smart.lst', always_prefer_local_directory_if_file_exists=True, smart_local_folder_detection=False))
    print(loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini/list_smart.lst',                  always_prefer_local_directory_if_file_exists=True))
    print(loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini/list_smart.lst')) # this is the default behavior and I do really love it
    print(loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini/list_smart.lst', always_prefer_local_directory_if_file_exists=True, smart_local_folder_detection=False, filter_existing_files_only=True))

    tmp_lst = loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini/list.lst', always_prefer_local_directory_if_file_exists=True, filter_existing_files_only=True)
    print(smart_TA_list(tmp_lst, 'tutu.tif'))
    print(smart_TA_list(tmp_lst, 'handCorrection.tif', alternative_name_if_name_to_search_first_does_not_exist=None))
    print(smart_TA_list(tmp_lst, 'handCorrection.tif', alternative_name_if_name_to_search_first_does_not_exist='handCorrection.png'))
    # I think that works as I want
    print(smart_TA_list(tmp_lst, 'handCorrection2.png', alternative_name_if_name_to_search_first_does_not_exist='handCorrection.png'))

    save_list_to_file(smart_TA_list(tmp_lst, 'handCorrection.tif', alternative_name_if_name_to_search_first_does_not_exist='handCorrection.png'), '/E/Sample_images/sample_images_PA/trash_test_mem/mini/list_masks.lst')

    print('final list',create_list('/E/Sample_images/sample_images_denoiseg/train/raw/', save=True))
