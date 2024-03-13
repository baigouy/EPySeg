import sys
import os
def add_folder_to_python_path(folder_path):
    """
    Add a folder to the Python path.

    Parameters:
    - folder_path (str): The path of the folder to be added to the Python path.
    """
    if isinstance(folder_path, str):
        sys.path.append(folder_path)
    else:
        # assume it's a list
        for path in folder_path:
            add_folder_to_python_path(path)

def add_folder_to_system_path(folder_path):
    """
    Add a folder to the system PATH.

    Parameters:
    - folder_path (str): The path of the folder to be added to the system PATH.
    """
    if isinstance(folder_path, str):
        os.environ['PATH'] += os.pathsep + folder_path
    else:
        # assume it's a list
        for path in folder_path:
            add_folder_to_system_path(path)

def find_bin_folders(root_folder):
    bin_folders = []

    # Walk through the directory tree
    for foldername, subfolders, filenames in os.walk(root_folder):
        # Check if 'bin' is in the subfolders
        if 'bin' in subfolders:
            bin_folders.append(os.path.join(foldername, 'bin'))

    return bin_folders

def add_epyseg_to_sys_path(file=None, print_only=False):
    # if print_only does not execute the command --> just gets the appropriate text --> print_only creates a portable code that will always work!!!
    # NB THIS NEEDS BE COPIED UNDER EVERY STUFF THAT NEEDS BE EXECUTED ALONE FROM COMMAND LINE --> THIS CODE NEEDS BE CUT AND PASTE AND CANNOT BE DIRECTLY COPIED THOUGH
    if file is None:
        epyseg_path = __file__
    else:
        epyseg_path = file
    epyseg_path = epyseg_path.split('epyseg_pkg')[0] + 'epyseg_pkg/'
    if not print_only:
        sys.path.append(epyseg_path)
    else:
        command = '###### add the commands below to the beginning of your python file: ######\n'
        command += 'import sys\n'
        # command +=f'epyseg_path = \'{epyseg_path}\'\n'
        command += f'epyseg_path = __file__\n'
        command += f'epyseg_path = epyseg_path.split(\'epyseg_pkg\')[0] + \'epyseg_pkg/\'\n'
        command += 'sys.path.append(epyseg_path)\n'
        command += '###### end commands ######\n'
        print(command)
        return command
    # add_folder_to_system_path(epyseg_path)

if __name__ == '__main__':
    # TODO --> make it generate also the code to be put at the top of
    print(os.cpu_count()-1)
    print(find_bin_folders('/home/aigouy/mon_prog/Python/epyseg_pkg/personal/hiC_microC_tmp/blast_executables/'))
    # add_epyseg_to_sys_path()
    add_epyseg_to_sys_path(print_only=True) # prints the code to copy to the beginning of .py files to get them to run irrespective of the environment