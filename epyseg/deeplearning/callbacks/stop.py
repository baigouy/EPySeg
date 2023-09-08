# from tensorflow import keras
import tensorflow as tf

class myStopCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for stopping training based on a flag.

    Attributes:
        stop_me (bool): Flag indicating whether to stop training.
    """

    def __init__(self):
        self.stop_me = False

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of each epoch.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary of logs containing the metrics of the model. Defaults to None.
        """
        if self.stop_me:
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary of logs containing the metrics of the model. Defaults to None.
        """
        if self.stop_me:
            self.model.stop_training = True

    def on_batch_begin(self, batch, logs=None):
        """
        Called at the beginning of each batch.

        Args:
            batch (int): Current batch index.
            logs (dict): Dictionary of logs containing the metrics of the model. Defaults to None.
        """
        if self.stop_me:
            self.model.stop_training = True

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of each batch.

        Args:
            batch (int): Current batch index.
            logs (dict): Dictionary of logs containing the metrics of the model. Defaults to None.
        """
        if self.stop_me:
            self.model.stop_training = True
