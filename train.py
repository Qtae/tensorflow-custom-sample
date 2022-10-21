import os
import tensorflow as tf
from model import CustomTrainStepModel
from callbacks import CustomCheckpoint
from loss import CustomLoss
from data import load_dataset


if __name__ == '__main__':
    #set directories
    root_dir = 'D:/Work/99_OT/'
    dataset_dir = root_dir + 'Cifar10/Train'
    model_save_dir = root_dir + 'Model/SavedModel'
    checkpoint_dir = root_dir + 'Model/CheckPoint'

    #set hyperparameters
    epochs = 400
    batch_size = 16
    learning_rate = 0.0001
    valid_ratio = 0.2

    #load dataset
    print('-----------------------load dataset-----------------------')
    train_dataset, valid_dataset, train_step, valid_step, classes\
        = load_dataset(dataset_dir, batch_size, valid_ratio=valid_ratio, one_hot=True,
                       do_shuffle=True, normalize=True, resize=(32, 32))
    
    #build model
    input_layer = tf.keras.layers.Input([32, 32, 3])
    model = CustomTrainStepModel(input_layer, classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=CustomLoss(), metrics=[tf.keras.metrics.Accuracy(name='accuracy')])
    model.summary()

    #set callback list
    custom_checkpoint = CustomCheckpoint(checkpoint_dir=checkpoint_dir)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10,
                                                     cooldown=10, min_lr=0.000001, mode='auto', verbose=True)
    callbacks_list = [custom_checkpoint, reduce_lr]

    #train
    model.fit(train_dataset, epochs=epochs, steps_per_epoch=train_step,
              validation_data=valid_dataset, validation_steps=valid_step, callbacks=callbacks_list)

    #trained model save
    model.model.save(model_save_dir)
