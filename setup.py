import setuptools
from epyseg.epygui import __MAJOR__, __MINOR__, __MICRO__, __AUTHOR__, __VERSION__, __NAME__, __EMAIL__

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='epyseg',
    version=__VERSION__,
    author=__AUTHOR__,
    author_email=__EMAIL__,
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
        # "tensorflow>=2.0.0",  # to allow for mac OS X conda support #shall I put 2.3 now
        # "tensorflow-gpu>=2.0.0;platform_system!='Darwin'",
        "tensorflow>=2.3.0",  # to allow for mac OS X conda support #shall I put 2.3 now
        "tensorflow-gpu>=2.3.0;platform_system!='Darwin'",
        "segmentation-models==1.0.1",
        # "tensorflow-gpu>=2.0.0", # not required ? # make sure it does not install on OS X to prevent crash if does not exist
        "czifile",
        # "h5py", # should be installed with tensorflow gpu so do I need it ??? # probably better to remove it to avoid intsalling erroneous versions that are incompatible with tensorflow... similarly do I really need to have numpy it will be installed with tf anyways??? --> just try
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
        "tifffile",
        "tqdm",
        "natsort",
        "numexpr",
        "urllib3" # for model download
    ],
    # extras_require = {
    #     'all':  ["tensorflow-gpu>=2.0.0"]
    # },
    python_requires='>=3.6, <=3.8' # tensorflow is now supported by python 3.8
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
