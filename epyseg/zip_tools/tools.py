import gzip
import shutil
import os
import subprocess
from timeit import default_timer as timer
from epyseg.ta.tracking.tools import smart_name_parser

def compress_file_using_python(file_path, output_name=None):
    """
    Compresses the specified file using gzip compression.

    Args:
        file_path (str): The path to the file to be compressed.
        output_name (str, optional): The name of the compressed output file. If not provided, a default name is generated.
    """
    if output_name is None:
        output_name = os.path.splitext(file_path)[0] + '.gz'
    os.makedirs(smart_name_parser(output_name, 'parent'), exist_ok=True)
    with open(file_path, 'rb') as f_in:
        with gzip.open(output_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def compress_file_3rd_party(file_path, output_path=None):
    """
    Compresses the specified file using system gzip compression.

    Args:
        file_path (str): The path to the file to be compressed.
        output_path (str, optional): The path to the output file. If not provided, a default path is generated.
    """
    if output_path is None:
        output_path = file_path + '.gz'

    command = ['gzip', '-c', file_path]
    print('running: ' + ' '.join(command) + ' > ' + output_path)
    # create dirs if not exist
    os.makedirs(smart_name_parser(output_path, 'parent'), exist_ok=True)
    subprocess.run(['gzip', '-c', file_path], stdout=open(output_path, 'wb'))


if __name__ == '__main__':

    if True:
        # do a real compression of a real file
        start = timer()
        file_path = '/D/hiC_microC/hiC_data_fub1_n_ctrl/mapped.pairs'
        output_name = '/D/hiC_microC/hiC_data_fub1_n_ctrl/12-16.txt.gz'
        compress_file_3rd_party(file_path, output_name)
        print(timer() - start)
        import sys

        sys.exit(0)

    if False:
        # very slow --> only for small files
        start = timer()
        file_path = '/D/hiC_microC/hiC_data_fub1_n_ctrl/mapped.txt'
        compress_file_using_python(file_path)
        print(timer() - start)

        start = timer()
        output_name = '/D/hiC_microC/hiC_data_fub1_n_ctrl/tutu/mapped2.txt.gz'
        compress_file_using_python(file_path, output_name)
        print(timer() - start)

    if True:
        file_path = '/D/hiC_microC/hiC_data_fub1_n_ctrl/mapped.txt'
        output_path = '/D/hiC_microC/hiC_data_fub1_n_ctrl/tutu/mapped2.txt.gz'
        start = timer()
        compress_file_3rd_party(file_path, output_path)
        print(timer() - start)

        start = timer()
        compress_file_3rd_party(file_path)
        print(timer() - start)
