# EPySeg

**EPySeg is a package for segmenting 2D epithelial tissues.** EPySeg also ships with a graphical user interface that allows for building, training and running deep learning models. Training can be done with or without data augmentation (2D-xy and 3D-xyz data augmentation are supported). EPySeg relies on the [segmentation_models](https://github.com/qubvel/segmentation_models) library. EPySeg source code is available [here](https://github.com/baigouy/EPySeg). Cloud version available [here](https://github.com/baigouy/notebooks).

# Install

1. Install [python 3.7](https://www.python.org/downloads/) or [Anaconda 3.7](https://www.anaconda.com/distribution/) (if not already present on your system)

2. In a command prompt type: 

    ```
    pip install --user --upgrade epyseg
    ```
    or
    ```
    pip3 install --user --upgrade epyseg
    ```
    NB:
    - To open a **command prompt** on **Windows** press **'Windows'+R** then type **'cmd'**
    - To open a **command prompt** on **MacOS** press **'Command'+Space** then type in **'Terminal'**

3. To open the graphical user interface, type the following in a command:
    ```
    python -m epyseg
    ```
    or
    ```
    python3 -m epyseg
    ``` 
   
# Third party libraries

Below is a list of the 3<sup>rd</sup> party libraries used by EPySeg and/or pyTA.<br><br> <font color='red'>**IMPORTANTLY: if you disagree with any license below, <u>please uninstall EPySeg</u>**.<br></font>

| Library name            | Use                                                                                                                  | Link                                          | License            |
|-------------------------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|--------------------|
| **tensorflow**          | Deep learning library                                                                                                | https://pypi.org/project/tensorflow/          | Apache 2.0         |
| **segmentation-models** | Models                                                                                                               | https://pypi.org/project/segmentation-models/ | MIT                |
| **czifile**             | Reads Zeiss .czi files                                                                                               | https://pypi.org/project/czifile/             | BSD (BSD-3-Clause) |
| **Markdown**            | Python implementation of Markdown                                                                                    | https://pypi.org/project/Markdown/            | BSD                |
| **matplotlib**          | Plots images and graphs                                                                                              | https://pypi.org/project/matplotlib/          | PSF                |
| **numpy**               | Array/Image computing                                                                                                | https://pypi.org/project/numpy/               | BSD                |
| **numpydoc**            | Numpy documentation format                                                                                           | https://pypi.org/project/numpydoc/            | BSD                |
| **Pillow**              | Reads 'basic' images (.bmp, .png, .pnm, ...)                                                                         | https://pypi.org/project/Pillow/              | HPND               |
| **PyQt5**               | Graphical user interface (GUI)                                                                                       | https://pypi.org/project/PyQt5/               | GPL v3             |
| **PyQtWebEngine**       | Display html in GUI                                                                                                  | https://pypi.org/project/PyQtWebEngine/       | GPL v3             |
| **read-lif**            | Reads Leica .lif files                                                                                               | https://pypi.org/project/read-lif/            | GPL v3             |
| **scikit-image**        | Image processing                                                                                                     | https://pypi.org/project/scikit-image/        | BSD (Modified BSD) |
| **scipy**               | Great library to work with numpy arrays                                                                              | https://pypi.org/project/scipy/               | BSD                | 
| **tifffile**            | Reads .tiff files (also reads Zeiss .lsm files)                                                                      | https://pypi.org/project/tifffile/            | BSD                |
| **tqdm**                | Command line progress                                                                                                | https://pypi.org/project/tqdm/                | MIT, MPL 2.0       |
| **natsort**             | 'Human' like sorting of strings                                                                                      | https://pypi.org/project/natsort/             | MIT                |
| **numexpr**             | Speeds up image math                                                                                                 | https://pypi.org/project/numexpr/             | MIT                |
| **urllib3**             | Model architecture and trained models download                                                                       | https://pypi.org/project/urllib3/             | MIT                |
| **qtawesome**           | Elegant icons in pyTA                                                                                                | https://pypi.org/project/QtAwesome/           | MIT                |
| **pandas**              | Data analysis toolkit                                                                                                | https://pypi.org/project/pandas/              | BSD (BSD-3-Clause) |
| **numba**               | GPU acceleration of numpy ops                                                                                        | https://pypi.org/project/numba/               | BSD                |
| **elasticdeform**       | Image deformation (data augmentation)                                                                                | https://pypi.org/project/elasticdeform/       | BSD                |
| **CARE/csbdeep**        | pyTA uses custom trained derivatives of the CARE surface projection model to generate (denoised) surface projections | https://pypi.org/project/csbdeep/             | BSD (BSD-3-Clause) |

