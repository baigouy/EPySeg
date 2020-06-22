import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='epyseg',
    version='0.1.6',
    author='Benoit Aigouy',
    author_email='baigouy@gmail.com',
    description='A deep learning based tool to segment epithelial tissues. The epyseg GUI can be uesd to build, train or run custom networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/baigouy/EPySeg',
    package_data={'': ['*.md']}, # include all .md files
    license='BSD',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    # TODO put this in requirements.txt file and read the file from here --> can have both methods work
    # below are the required files (they will be installed together with epyseg unless the '--no-deps' tag is used)
    install_requires=[
        "czifile",
        "h5py",
        "Markdown",
        "matplotlib",
        "numpy",
        "numpydoc",
        "Pillow",
        "PyQt5",
        "PyQtWebEngine",
        "read-lif",
        "scikit-image",
        "scipy",
        "segmentation-models==1.0.1",
        # "tensorflow==2.1.0",
        "tensorflow==2.1.0",
        # "tensorflow-gpu==2.1.0",
        "tensorflow-gpu==2.1.0",
        "tifffile",
        "tqdm",
        "natsort",
        "numexpr"
    ],
    python_requires='>=3.6, <3.8' # tensorflow not supported in python 3.8 yet
)

# pip3 freeze
# TODO add svg libs ???
# my versions in case that's needed some day
# should I add keras for seg models ???
# czifile==2019.7.2
# h5py==2.9.0
# Markdown==3.1.1
# matplotlib==3.1.3
# numpy==1.16.4
# numpydoc==0.9.2
# Pillow==7.0.0
# PyQt5==5.13.0
# PyQtWebEngine==5.13.0
# read-lif==0.2.1
# scikit-image==0.16.2
# scipy==1.4.1
# segmentation-models==1.0.1
# tensorflow==2.1.0
# tensorflow-gpu==2.1.0
# tifffile==2020.2.16
# tqdm==4.42.1
# natsort=7.0.1
# numexpr=2.7.1
