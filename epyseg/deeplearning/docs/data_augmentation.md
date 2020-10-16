# Data augmentation

**Data augmentation** is used to artificially increase the number of input images, it **is useful when**:
<br>
- The **amount of images available to train a model is low** 
<br>
- Also it **may improve the robustness of a model** (e.g. if a model is trained on a dataset where objects are all pointing in the same direction, then the model may be unable to detect the same object when it is not properly oriented; adding rotation augmentation enhances model detection of objects irrespective of their initial orientation) 
<br>

**List of augmentations:**
<br>
- **None** : no augmentation = keep images unchanged 
<br>
- **Shear** : shear the input and output images by the same amount of shear (up to the value specified by the user)
<br>
- **Flip** : randomly flips input and output images horizontally, vertically or both ways
<br>
- **Blur** : applies a gaussian blur to input images only (up to the value specified by the user), output images will not be blurred.
<br>
- **Rotate**: rotates input and output images by a random angle (up to the value specified by the user)
<br>
- **Translate**: applies a translation to input and output images along the x-y axes (up to the value specified by the user, this value is a percentage of image width and height)
<br>
- **Invert**: takes the negative/inverts the input image, output is kept unchanged (can be used if the model should be trained to produce the same even if input images have inverted intensities).
<br>
- **Noise**: adds noise to input images (outputs are kept unchanged, the model may indirectly learn to denoise input images)
<br>
- **Zoom**: Scales up (magnifies) or down input and output images the same way

**Important**: Data augmentation may lead to pixel interpolation/change pixels (see unsafe augmentations below). Those augmentations can severely damage output image quality and this especially when ground truth images are single px wide segmentation masks (e.g. masks produced by the watershed algorithm)

**Safe augmentations**:
<br>
-None
<br>
-Flip
<br>
-Invert
<br>
-Translate
<br>
-Noise
<br>
-Blur
<br>

**Unsafe augmentations**:
<br>
-Zoom
<br>
-Shear
<br>
-Rotate
<br>

**NB: I recommend not to use an unsafe augmentation with 1px wide masks** or to apply a dilation to the mask before (if you choose the second solution, please bear in mind that the model may learn to produce a dilated output, which may be unwanted) 
