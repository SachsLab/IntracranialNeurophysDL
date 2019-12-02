import numpy as np
import tensorflow as tf


class KernelLengthRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, kernel_size, window_scale=1e-2, window_func='poly', poly_exp=2,
                 threshold=tf.keras.backend.epsilon()):
        self.kernel_size = kernel_size
        self.window_scale = window_scale
        self.window_func = window_func
        self.poly_exp = poly_exp
        self.threshold = threshold
        self.rebuild_window()

    def rebuild_window(self):
        windows = []
        for win_dim, win_len in enumerate(self.kernel_size):
            if win_len == 1:
                window = np.ones((1,), dtype=np.float32)
            else:
                if self.window_func == 'hann':
                    window = 1 - tf.signal.hann_window(win_len, periodic=False)
                elif self.window_func == 'hamming':
                    window = 1 - tf.signal.hamming_window(win_len, periodic=False)
                else:  # if window_func == 'linear':
                    hl = win_len // 2
                    window = np.zeros((win_len,), dtype=np.float32)
                    window[:hl] = np.arange(1, hl + 1)[::-1]  # Negative slope line to 0 for first half.
                    window[-hl:] = np.arange(1, hl + 1)  # Positive slope line from 0 for second half.
                    window = window / hl  # Scale so it's -1 -- 0 -- 1
                    window = window ** self.poly_exp  # Exponent

            win_shape = [1] * (len(self.kernel_size) + 2)
            win_shape[win_dim] = win_len
            window = tf.reshape(window, win_shape)
            windows.append(window)

        self.window = tf.linalg.matmul(*windows)
        self.window = self.window / tf.reduce_max(self.window)

    def get_config(self):
        return {'kernel_size': self.kernel_size,
                'window_scale': self.window_scale,
                'window_func': self.window_func,
                'poly_exp': self.poly_exp,
                'threshold': self.threshold}

    def __call__(self, weights):
        weights = tf.sqrt(tf.square(weights))  # Differentiable abs
        # non_zero = tf.cast(weights > self.threshold, tf.float32)
        non_zero = tf.nn.sigmoid(weights - self.threshold)

        regularization = self.window_scale * self.window * non_zero
        # regularization = tf.reduce_max(regularization, axis=[0, 1], keepdims=True)
        regularization = tf.reduce_sum(regularization)
        return regularization
