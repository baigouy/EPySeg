#Getting started (load a pre-trained network and use it for segmenting epithelia) 

Below is a quick tutorial to use a pre-trained convolutional neural network (CNN) to segment epithelia

##Pre-requisites
- Place your images in a folder named 'unseen'  

##Load the pre-trained model
- In the 'Model' tab select 'Use a pre-trained model'
- Select a model in the combobox labeled with 'Models trained on 2D epithelia'
- Press 'Go' at the bottom of the 'Model' tab to load the model

##Prediction epithelia segmentation
- Click on the 'Predict' tab
- Drag n drop your 'unseen' folder onto the 'Dataset' field
- In 'Channel number' select the 'Channel of interest' if your input has several channels (a preview of the channel is available below)
- Tick (or not) 'Refine mask' (it helped in most, but not all, cases tested)
- Press 'Go (predict)', wait a few seconds/minutes
- Browse your 'unseen' folder, the CNN predictions can be found in a folder named 'predict'


__PS:__ if you are not happy with the results try re-training a/the model with your own dataset. You can also try using refine mask if you haven't tried it or try unticking it if you used it.