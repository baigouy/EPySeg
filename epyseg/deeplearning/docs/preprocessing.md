# Pre-processing

Various pre-processing can be applied to the datasets before they are passed to the model

## Inputs:

* **Invert**: Inverts input image intensity before passing it to the model

## Outputs/Ground truth masks:

* **Dilate**: applies a user specified dilation to the ground truth segmentation before passing it to the model. This may be done in order to use more aggressive data augmentation on watershed images (e.g. data augmentation that lead to pixel interpolation).
* **Remove border pixels**: removes n pixels (makes n pixels black) at the border of ground truth images. This was added because when Tissue Analyzer segments images, it typically adds a 1px wide white frame around them. When those framed files are fed to the model, the model learns that images must have a frame and it adds one around each image it predicts. That is of course a serious problem (highlighting how easy it is to introduce biases in deep learning) that one can get rid of by setting this parameter to 1. **Tip**: dilation can applied after border removal to prevent image edges from being devoid of signal and having the model learn that images must have a black frame...