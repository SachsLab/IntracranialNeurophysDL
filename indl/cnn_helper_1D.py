"""
#Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes.

Results example: ![Visualization](http://i.imgur.com/4nj4KjN.jpg)
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import scipy.interpolate


def normalize(x):
    """utility function to normalize a tensor.

    # Arguments
        x: An input tensor.

    # Returns
        The normalized input tensor.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def upsample_timeseries(test_dat, n_resamples, axis=1):
    y = test_dat.numpy()
    x = np.linspace(0, 1, y.shape[axis])
    f_interp = scipy.interpolate.interp1d(x, y, axis=axis)
    xnew = np.linspace(0, 1, n_resamples)
    y = f_interp(xnew)
    return tf.convert_to_tensor(y.astype(np.float32))


def _stitch_filters(max_acts, n=None):
    """Draw the best filters in a nxn grid.

    # Arguments
        filters: A List of generated images and their corresponding losses
                 for each processed filter.
        n: dimension of the grid.
           If none, the largest possible square will be used
    """
    if n is None:
        n = int(np.floor(np.sqrt(len(max_acts))))

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top n*n filters.
    max_acts.sort(key=lambda x: x[1], reverse=True)
    max_acts = max_acts[:n * n]

    output_dim = max_acts[0][0].shape

    act_dat = np.stack([_[0] for _ in max_acts], axis=1)
    for act_ix in range(act_dat.shape[1]):
        temp = act_dat[:, act_ix, :]
        f_min, f_max = temp.min(), temp.max()
        act_dat[:, act_ix, :] = (temp - f_min) / (f_max - f_min)

    MARGIN = 5
    n_x = n * output_dim[0] + (n - 1) * MARGIN
    stitched_filters = np.nan * np.ones((n_x, n, output_dim[1]))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            dat = act_dat[:, i * n + j, :]
            width_margin = (output_dim[0] + MARGIN) * i
            stitched_filters[width_margin: width_margin + output_dim[0], j, :] = dat - j

    return stitched_filters


def visualize_layer(model, layer_idx,
                    output_dim=(801, 58), filter_range=(0, None),
                    step=1., epochs=200,
                    upsampling_steps=9, upsampling_factor=1.2
                    ):
    """Visualizes the most relevant filters of one conv-layer in a certain model.
    https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py
    # Arguments
        model: The model containing layer_name.
        layer_idx: The index of the layer to be visualized.
        step: step size for gradient ascent.
        epochs: Number of iterations for gradient ascent.
        upscaling_steps: Number of upscaling steps.
        upscaling_factor: Factor to which to slowly upgrade the timeseries towards output_dim.
        output_dim: [n_timesteps, n_channels] The output image dimensions.
        filter_range: [lower, upper]
                      Determines the to be computed filter numbers.
                      If the second value is `None`, the last filter will be inferred as the upper boundary.
    """

    output_layer = model.layers[layer_idx]

    max_filts = len(output_layer.get_weights()[1])
    max_filts = max_filts if filter_range[1] is None else min(max_filts, filter_range[1])

    # iterate through each filter in this layer and generate its corresponding time series
    maximizing_activations = []
    for f_ix in range(filter_range[0], max_filts):
        s_time = time.time()
        if isinstance(output_layer, tf.keras.layers.Conv1D):
            model_output = output_layer.output[:, :, f_ix]
        else:
            model_output = output_layer.output[f_ix]
        max_model = tf.keras.Model(model.input, model_output)

        # we start with some random noise that is smaller than the expected output.
        n_samples_out = output_dim[0]
        n_samples_intermediate = int(n_samples_out / (upsampling_factor ** upsampling_steps))
        test_dat = tf.convert_to_tensor(
            np.random.random((1, n_samples_intermediate, output_dim[-1])).astype(np.float32))

        for up in reversed(range(upsampling_steps)):
            # Run gradient ascent
            for _ in range(epochs):
                with tf.GradientTape() as tape:
                    tape.watch(test_dat)
                    layer_act = max_model(test_dat)
                    loss_value = K.mean(layer_act)
                gradients = tape.gradient(loss_value, test_dat)
                # normalization trick: we normalize the gradient
                gradients = normalize(gradients)
                test_dat += gradients * step

                # some filters get stuck to 0, we can skip them
                if loss_value <= K.epsilon():
                    test_dat = None
                    break

            if test_dat is None:
                break

            # Now upsample the timeseries
            n_samples_intermediate = int(n_samples_intermediate / (upsampling_factor ** up))
            test_dat = upsample_timeseries(test_dat, n_samples_intermediate, axis=1)

        if test_dat is not None:
            print('Costs of filter: {:5.0f} ( {:4.2f}s )'.format(loss_value, time.time() - s_time))
            test_dat = upsample_timeseries(test_dat, n_samples_out, axis=1)
            maximizing_activations.append((test_dat[0].numpy(), loss_value.numpy()))

    print('{} filters processed.'.format(len(maximizing_activations)))
    # Stitch timeseries together into one mega timeseries with NaN gaps.
    return _stitch_filters(maximizing_activations)


if __name__ == '__main__':
    from pathlib import Path
    import os
    from tensorflow.keras.models import load_model

    if Path.cwd().stem == 'indl':
        os.chdir(Path.cwd().parent)

    model_file = Path.cwd() / 'data' / 'kjm_ecog' / 'converted' / 'faces_basic' / 'mv_model_full.h5'
    model = load_model(str(model_file))

    if False:
        import tempfile
        # TODO: When processing softmax layer, second last dense layer should be converted from relu to linear.
        model.layers[-1].activation = tf.keras.activations.linear
        # Save and load the model to actually apply the change.
        tmp_path = Path(tempfile.gettempdir()) / (next(tempfile._get_candidate_names()) + '.h5')
        try:
            model.save(str(tmp_path))
            model = load_model(str(tmp_path))
        finally:
            tmp_path.unlink()
    model.summary()

    layer_idx = 14  # [2, 6, 10, 14]
    stitched_data = visualize_layer(model, layer_idx,  upsampling_steps=1, upsampling_factor=1,
                                    filter_range=(0, 12),
                                    output_dim=(801, model.get_input_shape_at(0)[-1]))

    import matplotlib.pyplot as plt

    # Create a colour code cycler e.g. 'C0', 'C1', etc.
    from itertools import cycle
    colour_codes = map('C{}'.format, cycle(range(10)))

    plt.figure()
    for chan_ix in range(3):
        plt.plot(stitched_data[:, :, chan_ix], color=next(colour_codes))
    plt.show()

