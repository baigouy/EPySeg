"""
A collection of file related methods


"""

import os
from epyseg.ta.tracking.tools import smart_name_parser

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


if __name__ == '__main__':
    import importlib
    print(importlib.import_module("epyseg.files.tools")) #<module 'epyseg.files.tools' from '/home/aigouy/mon_prog/Python/epyseg_pkg/epyseg/files/tools.py'>
