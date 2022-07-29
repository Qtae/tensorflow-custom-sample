import tensorflow as tf


class CustomLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true, y_pred)
        return loss
