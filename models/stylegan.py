import tensorflow as tf

generated_boxes = tf.random.normal(mean=0.0, stddev=1, shape=(4, 64, 64, 64, 1))

class G(tf.keras.Model):
    def __init__(self, **kwargs):
        super(G, self).__init__(**kwargs)
    def build(self, input_shape):
        pass

class HBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        pass

    