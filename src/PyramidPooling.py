import tensorflow as tf


class PyramidPooling(tf.keras.layers.Layer):
    def __init__(self, pool_list, **kwargs):
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i for i in pool_list])
        super(PyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[2]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2] * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(PyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        input_shape = tf.shape(x)
        num_cols = input_shape[1]

        col_length = [tf.dtypes.cast(num_cols, tf.float32) / i for i in self.pool_list]

        outputs = []
        for pool_num, num_pool_regions in enumerate(self.pool_list):
            for ix in range(num_pool_regions):
                x1 = ix * col_length[pool_num]
                x2 = ix * col_length[pool_num] + col_length[pool_num]

                x1 = tf.dtypes.cast(tf.math.round(x1), tf.int32)
                x2 = tf.dtypes.cast(tf.math.round(x2), tf.int32)

                new_shape = [input_shape[0], x2 - x1, input_shape[2]]

                x_crop = x[:, x1:x2, :]
                #xm = K.reshape(x_crop, new_shape)
                pooled_val = tf.math.reduce_max(x_crop, axis=1)
                outputs.append(pooled_val)

        outputs = tf.concat(outputs, axis=-1)

        return outputs
