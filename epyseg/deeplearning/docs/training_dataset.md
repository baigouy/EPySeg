##Pre-processing
Various pre-processing to be applied to the image before they are passed to the model
- inputs:
    - Invert: pass the the negative of the input image to the model
- outputs:
    - Dilate: applies user specified dilation to the output image before passing it to the model
    - Remove border pixels: removes pixels at the border of the output image (be careful, if the ouput image has a frame, the model may learn to reproduce it). Tip: dilation is applied after border removal.