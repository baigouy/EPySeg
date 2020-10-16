# Predict tab

A trained model can generate a segmentation output for any given input.

# Dataset

Path to a folder containing input images to pass through the network

**Tip:** the number of supported/detected input images is shown on the right. The validity of the input is indicated by a green icon. Wildcards are supported e.g. /home/user/test*.png will open all files in the user folder that start with test and end with .png.

## Tiling

See also the 'Tiling' section in 'Train'
-overlap width/height: models may be less efficient at tile edges, this can be fixed using overlapping tiles. 

**Tip:** overlap should be a multiple of 2. 

**Tip 2:** the overlap should be smaller than 0.5 times tile width/height. 

## Normalization

See 'Normalization' in 'Train' (Please do keep the same normalization for training and prediction)

## Channel number adjustment

Set of rules to apply when the network input number channels does not match expected model input number of channels
- Channel of interest (COI): defines the channel of interest (e.g. your image is RGB but only one channel should be used by the model, then select the channel of interest). Tip: A 'preview' is displayed below to help you choose the right channel. 
- Rule to reduce nb of channels (if needed): defines how channel reduction should be performed
- Rule to increase nb of channels (if needed): defines how channel augmentation should be performed

## Preview

Shows the selected channel of interest (make sure the right channel was selected)
 
## Output prediction path

Path where model predictions should be saved.
- Auto: by default model predictions are saved in a folder named 'predict' within the input folder. Output filenames are the same as the input file names.
- Tissue analyzer mode: saves model predictions in a folder that has the same name as the original image without the extension.
- Custom output directory: saves images in the user defined folder. Tip: if the output folder is valid then a green icon will be visible.

## Refine segmentation

**If not checked raw model output is saved** in the 'predict' folder. **If checked**, **the raw output** that consists of 5 watershed-like segmentations and 2 watershed-like seeds **is further post-processed to generate a single segmentation mask**. Most often the post processed data is better than raw model output (but that is npot necessarily always the case).  

## High quality predictions

If ticked prediction will be slower but segmentation should be globally better

## Predict

- Go (Predict): pass inputs to the model and recovers predictions. Tip: if an input image is corrupt or cannot be read (unknown file format), it image will be skipped and the software will move to the next image in the queue.
- Stop ASAP: stop the prediction as soon as possible

