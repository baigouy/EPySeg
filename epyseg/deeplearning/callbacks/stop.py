import tensorflow.keras as keras

# call this to stop training
class myStopCallback(keras.callbacks.Callback):

    def __init__(self):
        self.stop_me = False

    def on_epoch_begin(self, epoch, logs={}):
        if self.stop_me:
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs={}):
        if self.stop_me:
            self.model.stop_training = True

    def on_batch_begin(self, batch, logs={}):
        if self.stop_me:
            self.model.stop_training = True

    def on_batch_end(self, batch, logs={}):
        if self.stop_me:
            self.model.stop_training = True

