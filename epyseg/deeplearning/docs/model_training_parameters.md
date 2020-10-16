## Training parameters

Set of parameters to use during training:

* **Output models to:** path where the trained models will be saved
* **Epochs:** an epoch is when an entire dataset is passed though the network 
* **Steps per epoch:** number of images to pass through the network for each epoch. Tip: If set to -1 the full dataset will be passed to the model at each epoch.
* **Batch size:** size of the set of images used for one gradient update of model training. **Tip:** if batch size is too large, it can give rise to out of memory/oom errors. On the other hand a bigger batch size can be beneficial for training. **Tip 2:** set batch size to a large value then tick 'auto resize on oom' the training will be preformed on the user specified batch size but every time an oom error occurs the software will reduce batch size by 2 for training. If batch size reaches 0 an error is raised it is important to reduce tile width/height or change the model input parameters or buy a graphic card with more memory or use Colab TPU to prevent oom errors.
* **Keep:** number of best models (minimal loss) to keep/save.
* **Validation split:** can be used to define a validation set (optional). Validation sets are used to evaluate the model but not for training it (the model does not learn from the validation dataset, it only learns from the training dataset), this is usually used to check that the model is not overfit. It should be kept to a small percentage of the input data.
* **Upon completion of training, load the:** set whether the 'best' or the 'last' model should be loaded at the end of the training.
