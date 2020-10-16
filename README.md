# EPySeg

**EPySeg is a package for segmenting 2D epithelial tissues.** EPySeg also ships with a graphical user interface that allows for building, training and running deep learning models. Training can be done with or without data augmentation (2D-xy and 3D-xyz data augmentation arejust  supported). EPySeg relies on the [segmentation_models](https://github.com/qubvel/segmentation_models) library. EPySeg source code is available [here](https://github.com/baigouy/EPySeg). Cloud version available [here](https://github.com/baigouy/notebooks).

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