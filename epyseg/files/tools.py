"""
A collection of file related methods


"""

from epyseg.ta.tracking.tools import smart_name_parser
import os
import shutil
from datetime import datetime
import subprocess
import tempfile
import glob
import hashlib
import os
import platform

def backup_file_with_date(source_file, verbose=False, append_time=True):
    """
    Bcakup a file and append the current date in 'yyyymmdd' format to it.

    Args:
        source_file (str): The path to the source file to be backed up.

    Returns:
        str: The path to the copied file with the date appended.
    """
    # Check if the source file exists
    if os.path.exists(source_file):
        # Get the current date in 'yyyymmdd' format

        if append_time:
            current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            current_date = datetime.now().strftime('%Y%m%d')

        # Construct the new file name with the date appended
        new_file_name = f"{source_file}.{current_date}"

        if verbose:
            print('creating a backup', source_file, new_file_name)

        # Copy the source file to the destination with the new name
        shutil.copy(source_file, new_file_name)

        return new_file_name
    else:
        raise FileNotFoundError("Source file does not exist.")


def make_dirs_for_file_saving(save_name):
    """
    Create directories for file saving based on the given save_name.

    Parameters:
        - save_name: The name of the file to be saved.

    Returns:
        None
    """

    # Parse the parent folder name from the save_name using smart_name_parser
    parent_folder = smart_name_parser(save_name, 'parent')

    # Create the parent folder and any necessary intermediate directories
    os.makedirs(parent_folder, exist_ok=True)

def open_file_with_default_handler(file_path):
    """
    Opens a file with the default system handler.

    Args:
        file_path (str): The path to the file to be opened.

    Returns:
        bool: True if the file was opened successfully, False otherwise.
    """
    if os.path.isfile(file_path):
        try:
            subprocess.run(['open', file_path], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            return False
    else:
        print(f"The file '{file_path}' does not exist.")
        return False

def open_file_with_default_app(file_path):
    """
    Opens a file using the default system application.

    Parameters:
    - file_path (str): The path to the file to be opened.
    """

    # Determine the platform
    current_platform = platform.system()

    # Open the file using the default system application
    if current_platform == 'Windows':
        os.startfile(file_path)
    elif current_platform == 'Darwin':  # macOS
        os.system('open "{}"'.format(file_path))
    elif current_platform == 'Linux':
        os.system('xdg-open "{}"'.format(file_path))
    else:
        print("Unsupported operating system")


def create_temporary_file(extension=None, delete=True, return_file=True):
    """
    Create a temporary file with a specified file extension and automatic deletion on close.

    Args:
        extension (str): The desired file extension, including the dot (e.g., '.txt', '.csv').

    Returns:
        tempfile.NamedTemporaryFile: A NamedTemporaryFile object with the specified extension.

    Example:
        >>> with create_temporary_file() as temp_file:
        ...     temp_file.write(b"Hello, World!")
        ...     temp_file.seek(0)
        ...     data = temp_file.read()
        ...     print(data)  # Output: b'Hello, World!'
        13
        0
        b'Hello, World!'
    """
    if extension is not None and extension:
        if not extension.startswith('.'):
            extension = '.'+extension
    temp_file = tempfile.NamedTemporaryFile(delete=delete, suffix=extension)
    if return_file:
        return temp_file
    else:
        return temp_file.name

# # maybe allow glob with exceptions
# def rm(file_path, exceptions=None):
#     """
#     Delete a file if it exists.
#
#     Args:
#         file_path (str): The path to the file you want to delete.
#
#     Returns:
#         str: A message indicating the result of the deletion.
#
#     # Example:
#     # >>> rm("example.txt")
#     # >>> rm("nonexistent_file.txt")
#     """
#
#     if '*' in file_path:
#         # run glob
#
#
#     if os.path.exists(file_path):
#         try:
#             os.remove(file_path)
#         except Exception as e:
#             pass
def rm(file_path, exceptions=None):
    """
    Delete a file if it exists.

    Args:
        file_path (str): The path to the file you want to delete.
        exceptions (list, optional): A list of file names to exclude from deletion. Default is None.

    Returns:
        str: A message indicating the result of the deletion.

    # Example:
    # >>> rm("example.txt")
    # 'File 'example.txt' has been deleted.'
    # >>> rm("nonexistent_file.txt")
    # 'The file 'nonexistent_file.txt' does not exist.'
    # >>> rm("*.txt", exceptions=["example.txt"])
    # 'All matching files except ['example.txt'] have been deleted.'
    """
    if exceptions is not None:
        if isinstance(exceptions,str):
            exceptions = [exceptions]

    if '*' in file_path:
        # Use glob to create a list of files that match the pattern
        files_to_delete = glob.glob(file_path)
        if exceptions:
            files_to_delete = [f for f in files_to_delete if os.path.basename(f) not in exceptions]
        # if exceptions:
        #     files_to_delete = [f for f in files_to_delete if os.path.basename(f) not in exceptions]

        if not files_to_delete:
            # print('No files matching the pattern found or all were exceptions.')
            return

        deleted_files = []
        for file_to_delete in files_to_delete:
            try:
                os.remove(file_to_delete)
                deleted_files.append(file_to_delete)
            except Exception as e:
                pass

        if deleted_files:
            # return f'All matching files except {exceptions if exceptions else "none"} have been deleted.'
            return
        else:
            # return f'No matching files were deleted due to exceptions or errors.'
            return

    elif os.path.exists(file_path):
        try:
            os.remove(file_path)
            # return f"File '{file_path}' has been deleted."
            return
        except Exception as e:
            pass
    else:
        # return f"The file '{file_path}' does not exist."
        return


def is_file_more_recent(file1, file2):
    """
    Check if file1 was modified more recently than file2.

    Parameters:
    - file1: Path to the first file
    - file2: Path to the second file

    Returns:
    - True if file1 is more recent than file2, False otherwise
    """
    try:
        # Get the modification time of each file
        time1 = os.path.getmtime(file1)
        time2 = os.path.getmtime(file2)

        # Compare the modification times
        return time1 > time2
    except FileNotFoundError:
        print("One or both of the files do not exist.")
        return False


def get_md5_hash(filename):
    if not os.path.isfile(filename):
        return None
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_sha256(filename):
    return get_md5_hash(filename)

def get_file_size(path, return_size_in_mb=True):
    size = os.path.getsize(path)
    if return_size_in_mb:
        size/= (1024*1024)
    return size

def find_files_by_extension(folder_path, extension):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

if __name__ == '__main__':
    import sys

    if True:
        # Example usage:
        file_path1 = "/F/genomic_resources/db_100_genomes/drosophila_only_with_gunu_human_readbale_names/100genomes_with_annots/0_suz_genome_us/D_suzukii.fasta.ndb"
        file_path2 = "/F/genomic_resources/db_100_genomes/drosophila_only_with_gunu_human_readbale_names/100genomes_with_annots/0_suz_genome_us/D_suzukii.fasta"
        file_path2 = "/F/genomic_resources/db_100_genomes/drosophila_only_with_gunu_human_readbale_names/100genomes_with_annots/0_suz_genome_us/D_suzukii.fasta.nin"

        # file_path1, file_path2 = file_path2, file_path1

        if is_file_more_recent(file_path1, file_path2):
            print(f"{file_path1} is more recent than {file_path2}")
        else:
            print(f"{file_path1} is not more recent than {file_path2}")
        sys.exit(0)




    import importlib
    print(importlib.import_module("epyseg.files.tools")) #<module 'epyseg.files.tools' from '/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/files/tools.py'>

    # rm('/E/Sample_images/cells_escaping_silencing_test_count/14,5kb T2A DsRed X2 B4 Tg/14,5kb T2A DsRed X2 B4 Tg 0001/wing_mask.tif')
    rm('/E/Sample_images/cells_escaping_silencing_test_count/14,5kb T2A DsRed X2 B4 Tg/14,5kb T2A DsRed X2 B4 Tg 0001/*.tif', exceptions=['veins_deep.tif','wing_cells.tif'])
    rm('/E/Sample_images/cells_escaping_silencing_test_count/14,5kb T2A DsRed X2 B4 Tg/14,5kb T2A DsRed X2 B4 Tg 0001/*.db', exceptions='spots.db')

