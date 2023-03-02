import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self, n_input=7, n_past_points=40, n_labels=3, data_noise_dim=50):

        super(Generator, self).__init__()
        # DATA
        self.d1 = tf.keras.layers.Dense(n_past_points * n_input, use_bias=False, input_shape=(data_noise_dim,))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.lRe1 = tf.keras.layers.LeakyReLU()

        self.conv1 = tf.keras.layers.Conv1DTranspose(32, 5, strides=1, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.lRe2 = tf.keras.layers.LeakyReLU()

        # LABEL
        self.d12 = tf.keras.layers.Dense(n_past_points * n_labels, use_bias=False, input_shape=(3,))
        self.bn12 = tf.keras.layers.BatchNormalization()
        self.lRe12 = tf.keras.layers.LeakyReLU()

        self.conv12 = tf.keras.layers.Conv1DTranspose(32, 5, strides=1, padding='same', use_bias=False)
        self.bn22 = tf.keras.layers.BatchNormalization()
        self.lRe22 = tf.keras.layers.LeakyReLU()

        # DATA + LABEL
        self.conv2 = tf.keras.layers.Conv1DTranspose(32, 5, strides=2, padding='same', use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.lRe3 = tf.keras.layers.LeakyReLU()

        self.conv3 = tf.keras.layers.Conv1D(n_input, 5, strides=2, padding='same', use_bias=False, activation='tanh')
        self.bn32 = tf.keras.layers.BatchNormalization()
        self.lRe32 = tf.keras.layers.LeakyReLU()

    def call(self, inputs, training=None):
        # data
        inputs_1 = tf.slice(inputs, [0, 0], [-1, 50])
        # label
        inputs_2 = tf.slice(inputs, [0, 50], [-1, 3])

        # INPUT
        x = self.d1(inputs_1)
        x = self.bn1(x, training=training)
        x = self.lRe1(x, training=training)
        x = tf.reshape(x, [-1, 40, 7])

        # LABELS
        y = self.d12(inputs_2)
        y = self.bn2(y, training=training)
        y = self.lRe2(y, training=training)
        y = tf.reshape(y, [-1, 40, 3])

        # LABELS + INPUT
        x = tf.concat([x, y], axis=2)
        x = self.conv12(x)
        x = self.bn22(x, training=training)
        x = self.lRe22(x, training=training)

        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = self.lRe3(x, training=training)

        x = self.conv3(x)
        x = self.bn32(x, training=training)
        x = self.lRe32(x, training=training)

        return x

    def loadModel(self, model_path):
        self.load_weights(model_path)

class Model(tf.keras.Model):

    def __init__(self, n_classes, n_input):
        super(Model, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=6, padding='VALID', activation='sigmoid')
        self.p1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='VALID')
        self.avg1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='VALID')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.f1 = tf.keras.layers.Flatten()
        self.drop1 = tf.keras.layers.Dropout(rate=0.4)

        self.conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=6, padding="VALID")
        self.p2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='VALID')
        self.avg2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='VALID')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.f2 = tf.keras.layers.Flatten()
        self.drop2 = tf.keras.layers.Dropout(rate=0.4)

        self.d1 = tf.keras.layers.Dense(64, kernel_regularizer='l1', activation='relu')
        self.drop3 = tf.keras.layers.Dropout(rate=0.4)
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.d2 = tf.keras.layers.Dense(32, kernel_regularizer='l1', activation='relu')
        self.drop4 = tf.keras.layers.Dropout(rate=0.4)
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.d3 = tf.keras.layers.Dense(16, kernel_regularizer='l1', activation='relu')
        self.drop5 = tf.keras.layers.Dropout(rate=0.4)
        self.bn5 = tf.keras.layers.BatchNormalization()

        self.d4 = tf.keras.layers.Dense(n_classes, activation='sigmoid')

    def call(self, inputs, training=None):

        x = self.conv1(inputs)
        x = self.avg1(x)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)

        x = self.f2(x, training=training)

        x = self.d3(x)
        x = self.drop5(x, training=training)
        x = self.bn5(x, training=training)

        return self.d4(x)

    def loadModel(self, model_path):
        self.load_weights(model_path)
