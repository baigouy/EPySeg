# Train tab

## Brief introduction

To train a neural network one needs to provide the network with input images and the corresponding expected model output/ground truth segmentation (i.e. the expected segmentation mask for the given input image). During training, the model will learn features from the input image and generate an output based on them. The model aims to reproduce the user provided output/segmentation (practically it does so by trying to minimize the selected loss function). The lower the loss value is the better the model output fits the user expected output (provided the loss function is wisely chosen...). Often, a human readable version of the loss is also printed in the form of a percentage. Importantly this human readable version is not used to improve the model (i.e. adjust model weights). 

## A few things to keep in mind before getting started  
 
Models will carry along every learnt bias. So it is important to pay a lot of attention to the dataset used for training a model.

# Compile Model (setting the parameters of the model optimizer)

* If the model used is loaded from a file it may or may not come up with its own settings (including loss, metrics and optimizer):
    * you may (tick 'Compile/recompile model') or may not (untick) want to change these settings
* Optimizer: model weights are optimized using gradient descent algorithms that aim to find global minima. The optimizer is the gradient descent algorithm used by the model. 'Adam' is the most commonly used one, start with it if you don't know which optimizer to use. SGD is slower than adam but supposed to be better. Other optimizers might be useful for specific problems.
* Loss: during training models try to minimize this loss function. The choice of the loss is key for model training and using the wrong loss function can impair training. If the model should output binary images a good loss function is the 'Jaccard loss' (also known as iou = intersection over union, for a detailed description see https://en.wikipedia.org/wiki/Jaccard_index). Another typical loss for binary images is 'binary cross entropy'. Both losses can be combined in a single one using 'bce_jaccard_loss'. 'Dice loss' is similar to and correlates with 'Jaccard loss'. 'Mean squared error' and 'Mean absolute error' can be used for non binary outputs. 
* Metrics: it is a human readable value (usually given in percent) that tells how well the model fits the data. It is optional and does not impact model training. Tip: several metrics can be used at the same time. Use '+' to add a metric and '-' to remove one. 

## Training datasets

This is the place to provide the images to be used to train the model

* '**+**': add an input/output paired set. **Input** is typically the **original/raw image**. **Output** is the **corresponding expected output/ground truth segmentation** for the given input. Importantly the number of input/output image sets should be the same. The simplest solution is to use an 'input' folder for input files and an 'output' folder for ground truth files. Fill the input folder with original images and the output folder with the corresponding ground truth segmentation. Give the same file name for each corresponding input/output pair (see a typical organisation below).
    * input (folder organisation):
        * test_0.tif (original)
        * test_1.tif (original)
        * ...
    * output (folder organisation):
        * test_0.tif (expected segmentation for the original file named 'test_0.tif')
        * test_1.tif (expected segmentation for the original file named 'test_1.tif')
        * ...        
* '**-**': remove the selected input/output paired dataset

**Tip: if only a part of the input/output image should be used for training, please draw a ROI over it (in 'Preview')**.

## Data augmentation

Training a model requires a high number of images. However when such a high amount of images is not readily available it is possible to use 'Data augmentation' to generate more input/output images. An example of data augmentation is 'flip', if this augmentation is used the input/output images will be flipped horizontally or vertically thereby expanding the training set. **Tip: data augmentation also limits overfit**. 

**Tip: if no augmentation is set the model will use the 'None' augmentation by default, this augmentation keeps the input/output images unchanged.** 

**Tip 2: data augmentation supports 2D and 3D images.** 
* '**+**': add an augmentation
* '**-**': remove the selected augmentation(s)

## Tiling

Models can use a high quantity of memory and memory usage scales with the size of the input image. In order to reduce the memory footprint, images can be cut into pieces, these tiles are then passed to the model (for each input tile a corresponding output tile is generated automatically). 

**Tip:** some models require a specific input size, when this is the case, it is not possible to define a tile width/height.
* Default tile width: sets the default input tile width (Tip: most models usually take multiple of 32 as input size width/height, you can use 128 or 256 for input tile parameters)
* Default tile height: sets the default input tile height

## Normalization
Most models require inputs to be normalized. Also normalization helps gradient descent algorithms. 

**Tip:** when a normalization method for training a model, it is very important to keep the same normalization for predictions (otherwise model predictions may be of poor quality). 
* remove outliers:  
    * ignore outliers: do not remove outlier pixels from the original image
    * '+' : remove a custom percentage (user defined) of pixels with high intensities
    * '-' : remove a custom percentage (user defined) of pixels with low intensities
    * '+/-': remove a custom percentage (user defined) of pixels with high and low intensities
* method: method to be used for normalization
    * rescaling (min/max normalization): scales image intensities in order for min/max to be in the user defined range
    * Max normalization (auto): scales image intensities for 0/max to be in the user defined range
    * Standardization: centers intensity values within the selected range
    * None: keep intensities unchanged (may cause trouble for gradient descent)
    
## Training parameters

Set of parameters to use during training:

* **Output models to:** path where the trained models will be saved
* **Epochs:** an epoch is when an entire dataset is passed though the network 
* **Steps per epoch:** number of images to pass through the network for each epoch. Tip: If set to -1 the full dataset will be passed to the model at each epoch.
* **Batch size:** size of the set of images used for one gradient update of model training. **Tip:** if batch size is too large, it can give rise to out of memory/oom errors. On the other hand a bigger batch size can be beneficial for training. **Tip 2:** set batch size to a large value then tick 'auto resize on oom' the training will be preformed on the user specified batch size but every time an oom error occurs the software will reduce batch size by 2 for training. If batch size reaches 0 an error is raised it is important to reduce tile width/height or change the model input parameters or buy a graphic card with more memory or use Colab TPU to prevent oom errors.
* **Keep:** number of best models (minimal loss) to keep/save.
* **Validation split:** can be used to define a validation set (optional). Validation sets are used to evaluate the model but not for training it (the model does not learn from the validation dataset, it only learns from the training dataset), this is usually used to check that the model is not overfit. It should be kept to a small percentage of the input data.
* **Upon completion of training, load the:** set whether the 'best' or the 'last' model should be loaded at the end of the training.

## Train the model

* Go (Train model): trains the model. Tip: model training can take several hours (or even days), so please be patient...
* Stop asap: stops the training as soon as possible (may take some time before it actually stops)