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



if __name__ == '__main__':
    print("Python version")
    print(get_python_version())
    print("Version info")
    print(get_version_infos())
    print("Executable")
    print(get_path_to_python_executable())
    # print('tf')
    # print(get_tensorflow_version())


    # --> cool so all will be easy with this stuff




