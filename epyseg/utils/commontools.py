import functools
import sys

def get_python_version():
    return sys.version

def get_version_infos():
    return sys.version_info

def get_path_to_python_executable():
    return sys.executable

def get_tensorflow_version():
    import tensorflow as tf
    return tf.__version__

# this is for things that need be run one after the other
# TODO --> maybe make a version with independent stuff
# can pass several methods to it and the ouput of one is passed to the other
def execute_chained_functions(function_to_chain_iterable, parameter, reverse=False):
    result = functools.reduce(lambda fn, f: f(fn), reversed(function_to_chain_iterable) if reverse else function_to_chain_iterable, parameter)
    return result

def execute_chained_functions_and_save_as_tiff(parameter, function_to_chain_iterable=None, output_file_name=None,reverse=False):
    from epyseg.img import save_as_tiff
    result = execute_chained_functions(function_to_chain_iterable=function_to_chain_iterable, parameter=parameter, reverse=False)
    save_as_tiff(result, output_name=output_file_name(parameter))
    # return result/


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
        from epyseg.img import save_as_tiff, Img, elastic_deform
        from epyseg.ta.tracking.tools import smart_name_parser
        from epyseg.utils.loadlist import list_processor, loadlist

        img_path = '/E/Sample_images/adult_wings/N01_YwrGal4_males_wing0_projected.png'
        save = partial(save_as_tiff, output_name=smart_name_parser(
            '/E/Sample_images/adult_wings/N01_YwrGal4_males_wing0_projected_inverted.tif', 'full_no_ext') + '.tif')

        chained_functions = [Img, Img.invert, elastic_deform, save]
        print('chained functions', chained_functions)
        execute_chained_functions(chained_functions, img_path)  # load image invert it and save it

    if True:
        ''' MT VERSION BELOW '''
        from functools import partial
        from epyseg.img import save_as_tiff, Img
        from epyseg.ta.tracking.tools import smart_name_parser
        from epyseg.utils.loadlist import list_processor, loadlist


        # see if I can MT that
        lst = loadlist('/E/Sample_images/sample_images_PA/trash_test_mem/mini_empty/list.lst')
        def output_file_name(input_file_name):
            return smart_name_parser(
                input_file_name, 'full_no_ext') + 'inverted2.tif'


        chained_functions = [Img, Img.invert, elastic_deform]

        list_processor(lst=lst, processing_fn=chained_functions, multithreading=True, use_save_execution_if_chained=True,progress_callback=None, name_processor_function_for_saving=output_file_name)

        import sys
        sys.exit(0)

        # very good for all
        # all is ok