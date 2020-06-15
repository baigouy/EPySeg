# Data augmentation

list of augmentations:
- None : no augmentation = keep images unchanged 
- Shear : shear the input and output image by the same amount of shear (up to the value specified by the user)
- Flip : randomly flips input horizontally, vertically or both ways
- Blur : applies a gaussian blur to the input image only (up to the value specified by the user), the output image will not be blurred unchanged
- Rotate: rotates images by a random angle (up to the value specified by the user)
- Translate: translates along x/y the images (up to the value specified by the user, this value is a percentage of image width and height)
- Invert: takes the negative/inverts the input image, output is kept unchanged (can be used if output should remain the same even if the intensity is inverted).
- Noise: adds noise to input images (outputs are kept unchanged) 

__Important__: some data augmentation require interpolation and therefore change pixels (see unsafe augmentations below) and can severely damage your outputs especially when these are 1px wide segmentation masks (e.g. such as the ones produced by the watershed algorithm)

Safe augmentations:
-None
-flip
-invert
-translate
-noise
-blur

Unsafe augmentations:
-zoom
-shear
-rotate

I recommend not to use unsafe augmentations with 1px wide masks or to apply a dilation to the mask before (if you choose the second solution, be aware that the model will learn to produce a dilated output, which may not be what you want)

