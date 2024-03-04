import tensorflow as tf

class generator(tf.keras.Model):
    def __init__(self, n_input=7, n_past_points=40, n_labels=3, data_noise_dim=50):

        super(generator, self).__init__()
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

        self.conv3 = tf.keras.layers.Conv1D(n_input, 5, strides=2, padding='same', use_bias=False, activation='linear')
        self.bn32 = tf.keras.layers.BatchNormalization()
        self.lRe32 = tf.keras.layers.LeakyReLU()

    def call(self, inputs, training=None):
        # data
        inputs_1 = tf.slice(inputs, [0, 0], [-1, 20])
        # label
        inputs_2 = tf.slice(inputs, [0, 20], [-1, 3])

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

class Model(tf.keras.Model):

    def __init__(self, n_classes, n_input):
        super(Model, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding='causal')
        self.p1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.drop1 = tf.keras.layers.Dropout(rate=0.2)

        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="causal")
        self.p2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.drop2 = tf.keras.layers.Dropout(rate=0.2)
        self.f2 = tf.keras.layers.Flatten()

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

        x = self.conv1(inputs, training=training)
        x = self.p1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)

        x = self.conv2(x, training=training)
        x = self.p2(x, training=training)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)
        x = self.f2(x, training=training)

        x = self.d1(x, training=training)
        x = self.drop3(x, training=training)
        x = self.bn3(x, training=training)

        x = self.d2(x, training=training)
        x = self.drop4(x, training=training)
        x = self.bn4(x, training=training)

        x = self.d3(x, training=training)
        x = self.drop5(x, training=training)
        x = self.bn5(x, training=training)

        return self.d4(x)

    def loadModel(self, model_path):
        # self.model = Model(n_classes, n_input)
        self.load_weights(model_path)

class TCN(tf.keras.Model):

    def __init__(self, n_output, n_input, num_layers=6, num_filters=64, kernel_size=3, dropout_rate=0.2):
        super(TCN, self).__init__()

        # Initial Input Layer
        self.input_layer = tf.keras.layers.Input(shape=(None, n_input))

        # TCN Blocks
        for i in range(num_layers):
            dilation_rate = 2 ** i
            tcn_block = self.build_tcn_block(num_filters, kernel_size, dilation_rate, dropout_rate)
            setattr(self, f'tcn_block_{i}', tcn_block)

        # Fully Connected Layers
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')

        # Output Layer for Regression
        self.output_layer = tf.keras.layers.Dense(n_output, activation='softmax')

    def build_tcn_block(self, num_filters, kernel_size, dilation_rate, dropout_rate):
        tcn_block = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate,
                          activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(dropout_rate)
        ])
        return tcn_block

    def call(self, inputs, training=None):
        x = inputs  # No need to call the input layer explicitly

        # TCN Blocks
        for i in range(8):
            x = getattr(self, f'tcn_block_{i}')(x)

        # Global Average Pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.fc2(x)

        # Output Layer for Regression
        return self.output_layer(x)

    def loadModel(self, model_path):
        self.load_weights(model_path)

class LSTMGRUModel(tf.keras.Model):
    def __init__(self, n_input, num_units=32):
        super(LSTMGRUModel, self).__init__()

        # Input Layer
        self.input_layer = tf.keras.layers.Input(shape=(None, n_input))

        # Flatten
        self.f1 = tf.keras.layers.Flatten()

        # Avg pooling
        self.avg1 = tf.keras.layers.GlobalAveragePooling1D()

        # LSTM and GRU Layers
        self.lstm_layer = tf.keras.layers.LSTM(num_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        # self.lstm_bn = tf.keras.layers.BatchNormalization()
        self.gru_layer = tf.keras.layers.GRU(num_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        # self.gru_bn = tf.keras.layers.BatchNormalization()
        self.gru_layer = tf.keras.layers.GRU(num_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout = tf.keras.layers.Dropout(0.2)

        # Dense Layer for Binary Classification
        self.output_layer = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs, training=None):
        x = inputs
        x = self.lstm_layer(x)
        #x = self.lstm_bn(x)
        x = self.gru_layer(x)
        #x = self.gru_bn(x)
        x = self.dropout(x, training=training)
        x = self.avg1(x)
        return self.output_layer(x)

    def load_model(self, model_path):
        self.load_weights(model_path)
