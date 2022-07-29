import os
import tensorflow as tf


class CustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        checkpoint_filename = 'e{0:03d}-acc{1:.4f}-val_acc{2:.4f}-val_loss{3:.4f}.hdf5' \
            .format(epoch, logs['accuracy'],logs['val_accuracy'],logs['val_loss'])
        self.model.model.save_weights(os.path.join(self.checkpoint_dir, checkpoint_filename))


class CustomCallbackExample(tf.keras.callbacks.Callback):
    def __init__(self, variable):
        self.variable = variable

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        #print("Called when training starts; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        #print("Called when training ends; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        #print("Called when epoch {} begins; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Called when epoch {} ends; got log keys: {}".format(epoch, keys))
