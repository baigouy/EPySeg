# New model parameters
 
* **Architecture**: The architecture of a model is the arrangement and connection of its layers. Unet or Linknet are good starting architectures. The architecture of a CNN is divided into an encoder (the 'eyes' of the model, that extract features of the input image) and a decoder (a network that produces the desired output from the detected features).
* **Backbone**: Set the model encoder (the 'eyes'/feature extracting part of the network).
* **Input width**: Often optional (keep it to None if the model works with any image size). Select the model input layer width (keep it low, to avoid memory errors).
* **Input height**: Often optional (keep it to None if the model works with any image size). Select the model input layer height (keep it low, to avoid memory errors).
* **Input channels**: Select the number of channels of the input layer of the model (it needs not be the number of channels of the input images, it is the number of channels the model should learn from).
* **Activation layer**: Choose an activation to be applied to the last layer of the model (sigmoid is a good choice for generating binary images, i.e. a segmentation masks).
* **Number of classes**: Select the number of outputs the model should predict.