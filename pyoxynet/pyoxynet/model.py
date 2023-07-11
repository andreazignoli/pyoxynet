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

        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding='causal')
        self.p1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.avg1 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.f1 = tf.keras.layers.Flatten()
        self.drop1 = tf.keras.layers.Dropout(rate=0.2)

        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="causal")
        self.p2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.avg2 = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.f2 = tf.keras.layers.Flatten()
        self.drop2 = tf.keras.layers.Dropout(rate=0.2)

        self.d1 = tf.keras.layers.Dense(64, kernel_regularizer='l2', activation='relu')
        self.drop3 = tf.keras.layers.Dropout(rate=0.2)
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.d2 = tf.keras.layers.Dense(32, kernel_regularizer='l2', activation='relu')
        self.drop4 = tf.keras.layers.Dropout(rate=0.2)
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.d3 = tf.keras.layers.Dense(16, kernel_regularizer='l2', activation='relu')
        self.drop5 = tf.keras.layers.Dropout(rate=0.2)
        self.bn5 = tf.keras.layers.BatchNormalization()

        self.d4 = tf.keras.layers.Dense(n_classes, activation='sigmoid')

    def call(self, inputs, training=None):

        x = self.conv1(inputs)
        # x = self.avg1(x)
        x = self.p1(x)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)

        x = self.conv2(x)
        # x = self.avg2(x)
        x = self.p2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)
        x = self.f2(x, training=training)

        x = self.d1(x)
        x = self.drop3(x, training=training)
        x = self.bn3(x, training=training)

        x = self.d2(x)
        x = self.drop4(x, training=training)
        x = self.bn4(x, training=training)

        x = self.d3(x)
        x = self.drop5(x, training=training)
        x = self.bn5(x, training=training)

        return self.d4(x)

    def loadModel(self, model_path):
        # self.model = Model(n_classes, n_input)
        self.load_weights(model_path)

class TCN(tf.keras.Model):
    def __init__(self, num_classes, num_filters, kernel_size, dilation_rates):
        super(TCN, self).__init__()
        self.conv_layers = []

        for dilation_rate in dilation_rates:
            self.conv_layers.append(tf.keras.layers.Conv1D(filters=num_filters,
                                                           kernel_size=kernel_size,
                                                           padding='causal',
                                                           dilation_rate=dilation_rate,
                                                           activation='sigmoid'))
        self.f = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(rate=0.4)
        self.dense = tf.keras.layers.Dense(16, kernel_regularizer='l1', activation='sigmoid')
        self.bn = tf.keras.layers.BatchNormalization()

        self.output_layer = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs, training=None):
        x = inputs

        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.f(x)
        x = self.drop(x)
        x = self.dense(x)
        x = self.bn(x, training=training)
        output = self.output_layer(x)
        return output

    def loadModel(self, model_path):
        # self.model = Model(n_classes, n_input)
        self.load_weights(model_path)