import tensorflow as tf

class PlateModel:
    @staticmethod
    def build(width, height, depth, classes):

        inputShape = (height, width, depth)
        chanDim = -1

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.BatchNormalization(axis=chanDim),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.BatchNormalization(axis=chanDim),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.BatchNormalization(axis=chanDim),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.BatchNormalization(axis=chanDim),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.BatchNormalization(axis=chanDim),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.BatchNormalization(axis=chanDim),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(classes, activation="softmax")

        ])


        return model

    '''
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.BatchNormalization(axis=chanDim),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(classes, activation="softmax")
    '''