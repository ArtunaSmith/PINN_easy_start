import tensorflow as tf


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def call(self, x, *args, **kwargs):
        # shape of x: (batch, 2)
        x = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(x)


class MyNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = MyLayer(1, 4)
        self.layer2 = MyLayer(4, 1)

    def call(self, x, *args, **kwargs):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
