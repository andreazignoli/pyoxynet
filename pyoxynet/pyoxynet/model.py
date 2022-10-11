import tensorflow as tf

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

        # x = self.conv2(x)
        # x = self.avg2(x)
        # x = self.bn2(x, training=training)
        x = self.f2(x, training=training)
        # x = self.drop2(x, training=training)

        # x = self.d1(x)
        # x = self.drop3(x, training=training)
        # x = self.bn3(x, training=training)
        #
        # x = self.d2(x)
        # x = self.drop4(x, training=training)
        # x = self.bn4(x, training=training)

        x = self.d3(x)
        x = self.drop5(x, training=training)
        x = self.bn5(x, training=training)

        return self.d4(x)

    def loadModel(self, model_path):
        # self.model = Model(n_classes, n_input)
        self.load_weights(model_path)
