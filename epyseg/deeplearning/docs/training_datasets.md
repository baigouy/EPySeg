## Training datasets

This is the place to provide the images to be used to train the model

* '**+**': add an input/output paired set. **Input** is typically the **original/raw image**. **Output** is the **corresponding expected output/ground truth segmentation** for the given input. Importantly the number of input/output image sets should be the same. The simplest solution is to use an 'input' folder for input files and an 'output' folder for ground truth files. Fill the input folder with original images and the output folder with the corresponding ground truth segmentation. Give the same file name for each corresponding input/output pair (see a typical organisation below).
    * input (folder organisation):
        * test_0.tif (original)
        * test_1.tif (original)
        * ...
    * output (folder organisation):
        * test_0.tif (expected segmentation for the original file named 'test_0.tif')
        * test_1.tif (expected segmentation for the original file amed 'test_1.tif')
        * ...    
* '**-**': remove the selected input/output paired dataset