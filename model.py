import tensorflow as tf


class CustomTrainStepModel(tf.keras.Model):
    def __init__(self, input_layer, classes):
        super(CustomTrainStepModel, self).__init__()
        self.classes = classes
        self.model = self.simple_model(input_layer, classes)

    def compile(self, optimizer='rmsprop', loss=None, metrics=None):
        super(CustomTrainStepModel, self).compile(optimizer=optimizer, loss=loss,
                                                  metrics=metrics)

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            #calculate loss
            loss = self.compiled_loss(y_true, y_pred)
        #get trainable parameters in model
        trainable_vars = self.model.trainable_variables
        #calculate gradient
        gradients = tape.gradient(loss, trainable_vars)
        #update weight
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        #calculate accuracy
        self.compiled_metrics[0].update_state(y_true, y_pred)
        #update metrics
        results = {"loss": loss, "accuracy": self.custom_metrics[0].result()}
        return results

    def test_step(self, data):
        x, y_true = data
        y_pred = self.model(x, training=False)
        # calculate loss
        val_loss = self.compiled_loss(y_true, y_pred)
        #calculate accuracy
        self.compiled_metrics[0].update_state(y_true, y_pred)
        # update metrics
        results = {"loss": val_loss, "accuracy": self.compiled_metrics[0].result()}
        return results

    def summary(self):
        self.model.summary()

    def simple_model(self, input, classes):
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        output = tf.keras.layers.Dense(classes, activation='softmax', name='predictions')(x)

        model = tf.keras.models.Model(inputs=input, outputs=output)
        return model
