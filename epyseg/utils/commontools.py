import functools
import sys

def get_python_version():
    """
    Returns the Python version.

    Returns:
        str: The Python version.

    # Examples:
    #     >>> get_python_version()
    #     '3.7.6'
    """
    return sys.version

def get_version_infos():
    """
    Returns the version information of the Python interpreter.

    Returns:
        sys.version_info: Version information as a named tuple.

    # Examples:
    #     >>> get_version_infos()
    #     sys.version_info(major=3, minor=7, micro=6, releaselevel='final', serial=0)
    """
    return sys.version_info

def get_path_to_python_executable():
    """
    Returns the path to the Python executable.

    Returns:
        str: Path to the Python executable.

    # Examples:
    #     >>> get_path_to_python_executable()
    #     '/usr/local/bin/python3.7'
    """
    return sys.executable


def get_tensorflow_version():
    """
    Returns the version of TensorFlow installed.

    Returns:
        str: TensorFlow version.

    # Examples:
    #     >>> get_tensorflow_version()
    #     '2.6.5'
    """
    import tensorflow as tf
    return tf.__version__

def execute_chained_functions(function_to_chain_iterable, parameter, reverse=False):
    """
    Executes a chain of functions on a parameter.

    Args:
        function_to_chain_iterable (iterable): Iterable containing the functions to chain.
        parameter: The parameter to be processed by the chained functions.
        reverse (bool, optional): Specifies whether to reverse the order of the functions in the chain. Defaults to False.

    Returns:
        The result of applying the chained functions on the parameter.

    Examples:
        >>> def add_two(x):
        ...     return x + 2
        ...
        >>> def multiply_by_three(x):
        ...     return x * 3
        ...
        >>> def subtract_one(x):
        ...     return x - 1
        ...
        >>> functions = [add_two, multiply_by_three, subtract_one]
        >>> execute_chained_functions(functions, 5)
        20
        >>> execute_chained_functions(functions, 5, reverse=True)
        14
    """
    result = functools.reduce(lambda fn, f: f(fn), reversed(function_to_chain_iterable) if reverse else function_to_chain_iterable, parameter)
    return result

def execute_chained_functions_and_save_as_tiff(parameter, function_to_chain_iterable=None, output_file_name=None, reverse=False):
    """
    Executes a chain of functions on a parameter and saves the result as a TIFF image.

    Args:
        parameter: The parameter to be processed by the chained functions.
        function_to_chain_iterable (iterable, optional): Iterable containing the functions to chain. Defaults to None.
        output_file_name (callable, optional): A callable that returns the output file name based on the parameter. Defaults to None.
        reverse (bool, optional): Specifies whether to reverse the order of the functions in the chain. Defaults to False.

    """
    result = execute_chained_functions(function_to_chain_iterable=function_to_chain_iterable, parameter=parameter, reverse=False)
    save_as_tiff(result, output_name=output_file_name(parameter))


if __name__ == '__main__':
    if True:
        print("Python version")
        print(get_python_version())
        print("Version info")
        print(get_version_infos())
        print("Executable")
        print(get_path_to_python_executable())
        # print('tf')
        # print(get_tensorflow_version())


    # --> cool so all will be easy with this stuff
    if True:
        '''SINGLE IMAGE VERSION BELOW'''
        # quick and rapid code to chain output of fucntions --> really useful --> I really love it

        from functools import partial
        from epyseg.img import save_as_tiff, Img, elastic_deform, invert
        from epyseg.ta.tracking.tools import smart_name_parser
        from epyseg.utils.loadlist import list_processor, loadlist

        img_path = '/E/Sample_images/wings/adult_wings/N01_YwrGal4_males_wing0_projected.png'
        save = partial(save_as_tiff, output_name=smart_name_parser(
            '/E/Sample_images/wings/adult_wings/N01_YwrGal4_males_wing0_projected_inverted.tif', 'full_no_ext') + '.tif')

        chained_functions = [Img, invert, elastic_deform, save]
        print('chained functions', chained_functions)
        execute_chained_functions(chained_functions, img_path)  # load image invert it and save it

    if True:
        ''' MT VERSION BELOW '''
        from functools import partial
        from epyseg.img import save_as_tiff, Img
        from epyseg.ta.tracking.tools import smart_name_parser
        from epyseg.utils.loadlist import list_processor, loadlist


        # see if I can MT that
        lst = loadlist('/E/Sample_images/sample_images_PA/mini_empty/list.lst')
        def output_file_name(input_file_name):
            return smart_name_parser(
                input_file_name, 'full_no_ext') + 'inverted2.tif'


        chained_functions = [Img, invert, elastic_deform]

        list_processor(lst=lst, processing_fn=chained_functions, multithreading=True, use_save_execution_if_chained=True,progress_callback=None, name_processor_function_for_saving=output_file_name)

        import sys
        sys.exit(0)

        # very good for all
        # all is ok