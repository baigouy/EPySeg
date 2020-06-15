#Getting started (build/train/use a neural network)

Below is a quick tutorial to train you first convolutional neural network (CNN)

##Pre-requisites
- Have at least 20-100 images along with their segmentation masks (the more images the better...)
- Make sure the quality segmentation is good and unbiased
- Copy the original image to a folder named 'input'
- Copy the segmentation masks to a folder named 'output'
- Ensure you have the same number of files in input and output and ideally the files should have the same names in both folders (or that they are in the same alphabetical order)
- Keep a few input images the model will never see during the training in an 'unseen' folder (you will use those to see how well the model performs on unseen/unknown data, this is to make sure the model does not overfit)  

##Build a new model
- In the 'Model' tab select 'Build a new model'.
- Press 'Go' at the bottom of the 'Model' tab to build the model.
- Congratulations you have just created your first CNN. Note that the 'train' and 'predict' tabs are now active.   

##Training a model
- Select the 'Train' tab
- In 'Compile model' press '+' to add 'iou_score' to the metrics
- In 'Training dataset':
    - press '+' (a window pops up)
    - In the 'Input' tab of this window, drag n drop your input folder into the 'dataset' field (the number of detected images is now displayed along with a green sign if everything is ok)
    - In 'Channel number', select the 'Channel of interest' from your input image (check the preview if you are unsure about channel number) (NB this step is optional if input images are single channel images, mandatory otherwise)
    - Select the 'Output' tab
    - Drag n drop your 'output' folder into the 'dataset' field
    - Press 'Ok' and the window will close
- In 'Data augmentation'
    - press '+' select 'None' and press 'Ok'
    - press '+' select 'Flip' and press 'Ok'
    - press '+' select 'Translate' and press 'Ok'
    - If your segmentation is 1 px wide (e.g. watershed output, stick to these three augmentation, otherwise add 'shear', 'zoom' and 'rotate')
- In 'Tiling' set 'Default tile width' and 'Default tile height' to 128
- In 'Normalization' keep the parameters unchanged (images will be normalized to a 0-1 range)
- In 'Training parameters' check the 'Output models to' field and change it to any custom directory or use the default output directory
- Press 'Go (Train model)'
- Wait several minutes/hours/days depending on the size of your dataset
- Congratulations you have trained your first CNN

##Use the created model for prediction
- Click on the 'Predict' tab
- Drag n drop your 'unseen' folder onto the 'Dataset' field
- In 'Tiling' change width and height to 128
- In 'Channel number' select the 'Channel of interest' if your input has several channels
- Press 'Go (predict)', wait a few seconds/minutes
- Browse your 'unseen' folder, the CNN predictions can be found in a folder named 'predict'
- Congratulations you have now completed your first training

<BR>
PS: if you are not happy with the results, try to increase your training set, train the model longer and/or try different architecture/backbone combinations