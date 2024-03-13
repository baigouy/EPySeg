# TODO add glob to loadlist to load generic lists --> good idea I think and offer various sorting algos including natsort
# a tool to load lists that is smarter than the TA list loader

import os
import traceback
from functools import partial
from epyseg.tools.early_stopper_class import early_stop
from epyseg.tools.logger import TA_logger # logging
import glob
import shutil
from natsort import natsorted

from epyseg.utils.commontools import execute_chained_functions_and_save_as_tiff

logger = TA_logger()


def list_files_in_child_dirs(parent_directory, depth=1):
    """
    Recursively list files in child directories up to a specified depth.

    Args:
        parent_directory (str): The path to the parent directory to start the search.
        depth (int): The maximum depth to search for child directories. Defaults to 1.

    Returns:
        list: A list of file paths found within the specified depth.

    Example:
        Given the directory structure:
        parent_dir/
        ├── child_dir1/
        │   ├── file1.txt
        │   └── file2.txt
        ├── child_dir2/
        │   ├── file3.txt
        └── child_dir3/
            └── file4.txt

        To list files in child directories up to a depth of 1:
        # >>> list_files_in_child_dirs("parent_dir", depth=1)
        # Output: ['parent_dir/child_dir1/file1.txt', 'parent_dir/child_dir1/file2.txt', 'parent_dir/child_dir2/file3.txt']
    """
    tif_files = []
    parent_directory, ext = parent_directory.split('*')

    def recursive_list(directory, current_depth):
        if current_depth >= depth:
            return
        for child_dir in os.listdir(directory):
            child_dir_path = os.path.join(directory, child_dir)
            if os.path.isdir(child_dir_path):
                tif_files.extend(glob.glob(os.path.join(child_dir_path, f'*{ext}')))
                recursive_list(child_dir_path, current_depth + 1)

    recursive_list(parent_directory, current_depth=0)
    return tif_files

def create_list(input_folder, save=False, output_name=None, extensions=['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tga', '*.lif'], sort_type='natsort', recursive=False):
    """
    Creates a list of file paths in the specified input folder based on the given extensions.

    Parameters:
        - input_folder: The path to the input folder where the files are located.
        - save: A flag indicating whether to save the list to a file (default is False).
        - output_name: The name of the output file (default is None, which generates a file named 'list.lst' in the input folder).
        - extensions: A list of file extensions to consider when creating the list (default is a list of common image extensions).
        - sort_type: The type of sorting to apply to the list of file paths (default is 'natsort').

    Returns:
        A list of file paths.

    Examples:
        Suppose we have an input folder named 'images' with the following files:
        - image1.jpg
        - image2.png
        - image3.tif

        >>> create_list('.',extensions=['.py'])
        ['./__init__.py', './commontools.py', './loadlist.py']
    """

    lst = []  # Initialize an empty list to store file paths

    if not extensions:
        logger.error('Extensions are not defined --> list cannot be created')
        return lst

    path = input_folder  # Assign the input_folder path to the 'path' variable

    if not path.endswith('/') and not path.endswith('\\') and not '*' in path and not os.path.isfile(path):
        path += '/'  # Append a trailing slash to the path if it doesn't have one

    if not '*' in path:
        # Iterate over the extensions and append matching file paths to the list
        for ext in extensions:
            if not ext.startswith('*'):
                ext = '*' + ext  # Prepend '*' to the extension if it doesn't have one
            lst += glob.glob(path + ext)
    else:
        if not recursive:
            lst += glob.glob(path)  # Append all file paths in the path to the list
        else:
            if isinstance(recursive, int):
                lst = list_files_in_child_dirs(path, depth=recursive)
            else:
                # Recursively search for .tif files in the folder and its subdirectories
                path, ext = path.split('*')
                for root, _,_ in os.walk(path):
                    lst.extend(glob.glob(os.path.join(root, '*'+ext)))

    if sort_type == 'natsort':
        lst = natsorted(lst)  # Apply natural sorting to the list of file paths

    if save:
        if output_name is None:
            output_name = os.path.join(input_folder, 'list.lst')  # Generate the default output file name
        save_list_to_file(lst, output_name)  # Save the list to the specified output file

    return lst  # Return the list of file paths


def get_resume_list(input_list, resume_value):
    """
    Retrieves a resumed list starting from the specified resume_value.

    Parameters:
        - input_list: The input list to retrieve the resume list from.
        - resume_value: The value in the list to resume from.

    Returns:
        A new list starting from the resume_value.

    Examples:
        >>> files = ['file1.jpg', 'file2.png', 'file3.tif', 'file4.jpg']
        >>> get_resume_list(files, 'file2.png')
        ['file2.png', 'file3.tif', 'file4.jpg']
    """

    try:
        resume_index = input_list.index(resume_value)
        resume_list = input_list[resume_index:]
        return resume_list
    except ValueError:
        return []



def loadlist(txtfile, always_prefer_local_directory_if_file_exists=True, filter_existing_files_only=False,
             smart_local_folder_detection=True, skip_hashtag_commented_lines=True, skip_empty_lines=True, recursive=False):
    """
    Loads a list from a text file with optional filtering and processing options.

    Parameters:
        - txtfile: The path to the text file to load the list from.
        - always_prefer_local_directory_if_file_exists: If True, files located in the .lst/.txt file containing folder will be preferred if they exist.
        - filter_existing_files_only: If True, filters the list to include only existing files.
        - smart_local_folder_detection: If True, applies local paths to the common root path in the list for smarter list opening.
        - skip_hashtag_commented_lines: If True, skips lines starting with '#' (commented lines) in the list.
        - skip_empty_lines: If True, skips empty lines and lines containing spaces only.

    Returns:
        The loaded list with the specified filtering and processing options.

    Examples:
        >>> list_file = 'loadlist.py'
        >>> loadlist(list_file, always_prefer_local_directory_if_file_exists=True, filter_existing_files_only=False)[:2]
        ['import os', 'import traceback']
    """

    if not os.path.exists(txtfile) or not os.path.isfile(txtfile):
        if not '*' in txtfile and not os.path.isdir(txtfile):
            return None
        else:
            lst = create_list(txtfile, save=False, recursive=recursive)  # Create a list using create_list() if txtfile is a directory or contains wildcard characters ('*')
    else:
        with open(txtfile) as f:
            if not skip_hashtag_commented_lines:
                lst = [line.rstrip() for line in f]  # Read all lines from the text file and remove trailing whitespaces
            else:
                lst = [line.rstrip() for line in f if not line.rstrip().startswith('#')]  # Exclude lines starting with '#' (commented lines)
        if skip_empty_lines:
            lst = [line.rstrip() for line in lst if not line.strip() == '']  # Exclude lines containing only spaces or empty lines

    if always_prefer_local_directory_if_file_exists:
        logger.debug('Files located in the .lst/.txt file containing folder will be preferred if they exist.')

        list_file_directory = os.path.dirname(txtfile)  # Get the directory of the txtfile

        if not smart_local_folder_detection:
            lst = [os.path.join(list_file_directory, os.path.basename(line)) if os.path.isfile(
                os.path.join(list_file_directory, os.path.basename(line))) else line for line in lst]
            # Prepend the list items with the list_file_directory if the file exists in the directory, otherwise keep the item as it is
        else:
            common_path = os.path.commonprefix(lst)  # Find the common root path of all items in the list
            if not os.path.isdir(common_path):
                common_path = os.path.dirname(common_path)

            if common_path != '':
                relative_paths = [os.path.relpath(path, common_path) for path in lst]  # Get the relative paths to the common_path
                lst = [os.path.join(list_file_directory, rel) if os.path.isfile(
                    os.path.join(list_file_directory, rel)) else line for rel, line in zip(relative_paths, lst)]
                # Prepend the list_file_directory to the relative paths if the file exists in the directory, otherwise keep the item as it is

    if filter_existing_files_only:
        logger.debug('Checking list for existing files')
        lst = [line for line in lst if os.path.isfile(line)]  # Filter the list to include only existing files

    return lst

def TA_smart_pairing_list(TA_list, corresponding_list_of_files_to_inject_or_folder, stop_on_error=False):
    """
    Performs smart pairing of two lists, TA_list and corresponding_list_of_files_to_inject_or_folder, based on certain conditions.

    Parameters:
        - TA_list: The list to be paired.
        - corresponding_list_of_files_to_inject_or_folder: The list of files or folder to be paired with TA_list.
        - stop_on_error: If True, stops the execution if the input lists/folders don't match.

    Returns:
        The paired list as a result of smart pairing.

    # Examples:
    #     >>> ta_list = 'ta_list.txt'
    #     >>> files_folder = 'files_folder'
    #     >>> TA_smart_pairing_list(ta_list, files_folder, stop_on_error=True)
    #     [('files_folder/file1.jpg', 'file1.jpg'), ('files_folder/file2.png', 'file2.png'), ('files_folder/file3.tif', 'file3.tif')]
    """

    if TA_list.lower().endswith('.txt') or TA_list.lower().endswith('.lst'):
        TA_list = loadlist(TA_list)  # Load the list from a text file if TA_list ends with '.txt' or '.lst'

    if os.path.isdir(corresponding_list_of_files_to_inject_or_folder):
        # If corresponding_list_of_files_to_inject_or_folder is a directory, assume same name as for input
        # corresponding_list_of_files_to_inject_or_folder
        matching_list = []
        for file in TA_list:
            filename0_without_path = os.path.basename(file)  # Get the filename without the path
            matching_list.append(os.path.join(corresponding_list_of_files_to_inject_or_folder, filename0_without_path))
            # Create the matching list by joining the corresponding_list_of_files_to_inject_or_folder with the filename without the path
    elif corresponding_list_of_files_to_inject_or_folder.lower().endswith('.txt') or \
            corresponding_list_of_files_to_inject_or_folder.lower().endswith('.lst'):
        matching_list = loadlist(corresponding_list_of_files_to_inject_or_folder)
        # Load the matching list from a text file if corresponding_list_of_files_to_inject_or_folder ends with '.txt' or '.lst'

    # Try zipping and print correspondence and if it fails, try smart matching
    if len(matching_list) != len(TA_list):
        if stop_on_error:
            print('Input lists/folders don\'t match', len(TA_list), len(matching_list))
            return
        else:
            # TODO: Implement rescue strategy for matching when lengths don't match
            pass

    zipped = list(zip(matching_list, TA_list))  # Zip the matching_list and TA_list together
    return zipped


def TA_injector(TA_list, corresponding_list_of_files_to_inject_or_folder, generic_TA_name_for_injected=None, stop_on_error=True, move_file=False, simulate_only=False):
    """
    Performs injection of files from the corresponding_list_of_files_to_inject_or_folder into the destination folders specified by the TA_list.

    Parameters:
        - TA_list: The list specifying the destination folders for injection.
        - corresponding_list_of_files_to_inject_or_folder: The list of files or folder to be injected.
        - generic_TA_name_for_injected: The generic name to be used for the injected files. If None, the original filename will be used.
        - stop_on_error: If True, stops the execution if the input lists/folders don't match.
        - move_file: If True, moves the files instead of copying them.
        - simulate_only: If True, simulates the injection without actually performing any file operations.

    Returns:
        None

    # Examples:
    #     >>> ta_list = ['folder1', 'folder2']
    #     >>> files_folder = 'files_folder'
    #     >>> TA_injector(ta_list, files_folder, generic_TA_name_for_injected='injected.jpg', stop_on_error=False, move_file=True, simulate_only=False)
    #     move files_folder/file1.jpg --> folder1/injected.jpg
    #     move files_folder/file2.png --> folder2/injected.jpg
    """

    zipped = TA_smart_pairing_list(TA_list, corresponding_list_of_files_to_inject_or_folder)
    if zipped is None:
        # Nothing to do, maybe return an error
        return

    for matching_things in zipped:
        file_to_copy = matching_things[0]
        file_to_copy_without_path = os.path.basename(file_to_copy)
        file = matching_things[-1]
        filename0_without_path = os.path.basename(file)
        filename0_without_ext = os.path.splitext(filename0_without_path)[0]
        path = os.path.dirname(file)
        destination_folder = os.path.join(path, filename0_without_ext)
        destination_name = os.path.join(destination_folder, file_to_copy_without_path if generic_TA_name_for_injected is None else generic_TA_name_for_injected)

        # Make sure the folder exists for the copy
        if not simulate_only:
            os.makedirs(os.path.dirname(destination_folder), exist_ok=True)

        if not move_file:
            print('copy', file_to_copy, '-->', destination_name)
            if not simulate_only:
                shutil.copy2(file_to_copy, destination_name)
        else:
            print('move', file_to_copy, '-->', destination_name)
            if not simulate_only:
                shutil.move(file_to_copy, destination_name)



# creates based on the main TA list a list of files with alternative names and returns it
# NB will only change name if the file with the alternate name exists --> I think this is the behaviour I want but ok
def smart_TA_list(main_list, name_to_search_first, alternative_name_if_name_to_search_first_does_not_exist=None):
    """
    Creates a smart list by combining the main_list with the specified name_to_search_first and an alternative name if the name_to_search_first doesn't exist.

    Parameters:
        - main_list: The main list of paths.
        - name_to_search_first: The name to search first in each path.
        - alternative_name_if_name_to_search_first_does_not_exist: The alternative name to use if the name_to_search_first doesn't exist.

    Returns:
        The smart list generated by combining the main_list with the specified names.

    # Examples:
    #     >>> main_list = ['path1/file.txt', 'path2/image.png', 'path3/document.doc']
    #     >>> name_to_search_first = 'data.txt'
    #     >>> alternative_name_if_name_to_search_first_does_not_exist = 'backup.txt'
    #     >>> smart_TA_list(main_list, name_to_search_first, alternative_name_if_name_to_search_first_does_not_exist)
    #     ['path1/data.txt', 'path2/data.txt', 'path3/data.txt']
    """

    lst = [os.path.join(os.path.splitext(line)[0], name_to_search_first) for line in main_list]
    # Generate a list by joining the path without the extension with the name_to_search_first

    if alternative_name_if_name_to_search_first_does_not_exist is not None:
        lst = [line if os.path.isfile(line) or not os.path.isfile(os.path.join(os.path.dirname(line), alternative_name_if_name_to_search_first_does_not_exist)) else os.path.join(os.path.dirname(line), alternative_name_if_name_to_search_first_does_not_exist) for line in lst]
        # Check if the line is a file or if the alternative name doesn't exist in the same directory.
        # If it's a file or the alternative name exists, keep the line as it is.
        # Otherwise, replace the line with the alternative name.

    return lst

def save_list_to_file(lst, filename, col_separator='\t'):
    """
    Saves the given list to a file.

    Parameters:
        - lst: The list to be saved.
        - filename: The name of the file to save the list.
        - col_separator: The separator character to use for items in the list.

    Returns:
        The input list if successfully saved.

    Raises:
        ValueError: If the list is empty.

    # Examples:
    #     >>> lst = ['item1', 'item2', 'item3']
    #     >>> filename = 'output.txt'
    #     >>> save_list_to_file(lst, filename, col_separator='\t')
    #     ['item1', 'item2', 'item3']
    """

    if lst is not None:
        with open(filename, 'w') as f:
            for item in lst:
                if not isinstance(item, list) and not isinstance(item, tuple):
                    f.write("%s\n" % item)
                    # Write each item in the list to a new line in the file.
                else:
                    f.write("%s\n" % col_separator.join([str(i) for i in item]))
                    # If the item is a list, join its elements with the col_separator and write to the file.

        return lst
    else:
        raise ValueError('Empty list --> cannot save it to a file')
        # Raise an error if the list is empty.


# this gives the optimal folder name for saving images without requiring other stuff, from various models --> use that for ensemble tools
def smart_output_folder_name(lst):
    """
    Generates the smartest name for the output folder based on the given list of paths.

    Parameters:
        - lst: The list of paths.

    Returns:
        The list of smart folder names.

    """

    # TODO no clue what  i wanted to do here, maybe just return the commaon path ?

    # Get the common path among all the paths in the list
    common_path = os.path.commonprefix(lst)

    if not os.path.isdir(common_path):
        common_path = os.path.dirname(common_path)

    if common_path == '':
        return None

    # Replace slashes and backslashes in the common path with underscores to create a meaningful name
    smart_name = [os.path.splitext(name.replace(common_path, '').replace('/', '_').replace('\\', '_'))[0] for name in lst]

    # Remove leading underscores from the folder names
    smart_name = [name if not name.startswith('_') else name[1:] for name in smart_name]

    return smart_name

# can create a list of files with the same name across several folders generated by different models for example --> can then be used for ensemble or averaging...
def get_transversal_list(root_path):
    """
    Generates a transversal list of files by traversing through folders.

    Parameters:
        - root_path: The root path to start the traversal.

    Returns:
        The transversal list of files.

    # Examples:
    #     >>> root_path = '/E/Sample_images/Consensus_learning/gray/CARE/'
    #     >>> get_transversal_list(root_path)
    #     [['/E/Sample_images/Consensus_learning/gray/CARE/folder1/file1.txt',
    #       '/E/Sample_images/Consensus_learning/gray/CARE/folder2/file1.txt'],
    #      ['/E/Sample_images/Consensus_learning/gray/CARE/folder1/file2.txt',
    #       '/E/Sample_images/Consensus_learning/gray/CARE/folder2/file2.txt']]
    """

    transverse_list = []

    # Get the list of folders within the root path
    folders = os.listdir(root_path)
    folders = [os.path.join(root_path, folder) for folder in folders]
    folders = natsorted(folders)

    # Get the list of files in the first folder
    files_in_first_folder = os.listdir(folders[0])

    # Traverse through the files in the first folder
    for file in files_in_first_folder:
        multifolder_list = []

        # Check if the file exists in other folders and add them to the list
        for folder in folders:
            final_file = os.path.join(folder, file)
            if os.path.isfile(final_file):
                try:
                    multifolder_list.append(final_file)
                except:
                    print('invalid file', final_file)

        transverse_list.append(multifolder_list)

    return transverse_list


def list_processor(lst, processing_fn, multithreading=True, progress_callback=None, use_save_execution_if_chained=False, name_processor_function_for_saving=None):
    """
    Processes a list using a specified processing function.

    Parameters:
        - lst: The list to process.
        - processing_fn: The processing function to apply to each element of the list.
        - multithreading: Boolean value indicating whether to use multithreading.
        - progress_callback: Callback function to track the progress of the processing.
        - use_save_execution_if_chained: Boolean value indicating whether to use save execution if the processing function is chained.
        - name_processor_function_for_saving: Name of the processor function for saving.

    Returns:
        None

    # Examples:
    #     >>>lst = loadlist('/E/Sample_images/sample_images_PA/mini_empty/list.lst')
    #     >>>def output_file_name(input_file_name):
    #     ...    return smart_name_parser(input_file_name, 'full_no_ext') + 'inverted2.tif'
    #     >>>chained_functions = [Img, invert, elastic_deform]
    #     >>>list_processor(lst=lst, processing_fn=chained_functions, multithreading=True, use_save_execution_if_chained=True,progress_callback=None, name_processor_function_for_saving=output_file_name)

    """

    from tqdm import tqdm
    import multiprocessing
    from multiprocessing import Pool
    from timeit import default_timer as timer

    start = timer()

    nb_procs = multiprocessing.cpu_count() - 1
    if nb_procs <= 0:
        nb_procs = 1

    if multithreading:
        print('using', nb_procs, 'processors')

    if isinstance(processing_fn, list):
        processing_fn = partial(execute_chained_functions_and_save_as_tiff, function_to_chain_iterable=processing_fn,
                                output_file_name=name_processor_function_for_saving, reverse=False)

    if multithreading:
        import gc
        gc.collect()
        with Pool(processes=nb_procs) as pool:
            for i, _ in enumerate(tqdm(pool.imap_unordered(processing_fn, lst), total=len(lst))):
                if early_stop.stop:
                    pool.close()
                    pool.join()
                    return
                if progress_callback is not None:
                    try:
                        from qtpy.QtWidgets import QProgressBar
                        if isinstance(progress_callback, QProgressBar):
                            progress_callback.setValue(int((i / len(lst)) * 100))
                        else:
                            progress_callback.emit(int((i / len(lst)) * 100))
                    except:
                        traceback.print_exc()
            pool.close()
            pool.join()
    else:
        for i, lst_elm in enumerate(lst):
            result = processing_fn(lst_elm)
            if early_stop.stop:
                return
            if progress_callback is not None:
                try:
                    from qtpy.QtWidgets import QProgressBar
                    if isinstance(progress_callback, QProgressBar):
                        progress_callback.setValue(int((i / len(lst)) * 100))
                    else:
                        progress_callback.emit(int((i / len(lst)) * 100))
                except:
                    traceback.print_exc()

    print('total time', timer() - start)


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
    print(loadlist('/E/Sample_images/sample_images_PA/mini/list.lst'))
    print(loadlist('/E/Sample_images/sample_images_PA/mini/list.lst', always_prefer_local_directory_if_file_exists=False))
    print(loadlist('/E/Sample_images/sample_images_PA/mini/list.lst', always_prefer_local_directory_if_file_exists=False, filter_existing_files_only=True))
    print(loadlist('/E/Sample_images/sample_images_PA/mini/list.lst', always_prefer_local_directory_if_file_exists=True, filter_existing_files_only=True))
    print(loadlist('/E/Sample_images/sample_images_PA/mini/list_smart.lst', always_prefer_local_directory_if_file_exists=True, smart_local_folder_detection=False))
    print(loadlist('/E/Sample_images/sample_images_PA/mini/list_smart.lst',                  always_prefer_local_directory_if_file_exists=True))
    print(loadlist('/E/Sample_images/sample_images_PA/mini/list_smart.lst')) # this is the default behavior and I do really love it
    print(loadlist('/E/Sample_images/sample_images_PA/mini/list_smart.lst', always_prefer_local_directory_if_file_exists=True, smart_local_folder_detection=False, filter_existing_files_only=True))

    tmp_lst = loadlist('/E/Sample_images/sample_images_PA/mini/list.lst', always_prefer_local_directory_if_file_exists=True, filter_existing_files_only=True)
    print(smart_TA_list(tmp_lst, 'tutu.tif'))
    print(smart_TA_list(tmp_lst, 'handCorrection.tif', alternative_name_if_name_to_search_first_does_not_exist=None))
    print(smart_TA_list(tmp_lst, 'handCorrection.tif', alternative_name_if_name_to_search_first_does_not_exist='handCorrection.png'))
    # I think that works as I want
    print(smart_TA_list(tmp_lst, 'handCorrection2.png', alternative_name_if_name_to_search_first_does_not_exist='handCorrection.png'))

    save_list_to_file(smart_TA_list(tmp_lst, 'handCorrection.tif', alternative_name_if_name_to_search_first_does_not_exist='handCorrection.png'), '/E/Sample_images/sample_images_PA/mini/list_masks.lst')

    print('final list',create_list('/E/Sample_images/sample_images_denoiseg/train/raw/', save=True))
