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


def normalize(x):
    """utility function to normalize a tensor.

    # Arguments
        x: An input tensor.

    # Returns
        The normalized input tensor.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def deprocess_timeseries(x):
    """utility function to convert a float array into a valid multi-channel timeseries.

    # Arguments
        x: A numpy-array representing the generated image.

    # Returns
        A processed numpy-array, which could be used in e.g. imshow.
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def process_timeseries(x, former):
    """utility function to convert a valid uint8 image back into a float array.
       Reverses `deprocess_image`.

    # Arguments
        x: A numpy-array, which could be used in e.g. imshow.
        former: The former numpy-array.
                Need to determine the former mean and variance.

    # Returns
        A processed numpy-array representing the generated image.
    """
    if K.image_data_format() == 'channels_first':
        x = x.transpose((2, 0, 1))
    return (x / 255 - 0.5) * 4 * former.std() + former.mean()


def visualize_layer(model,
                    layer_idx,
                    step=1.,
                    epochs=200,
                    upscaling_steps=9,
                    upscaling_factor=1.2,
                    output_dim=(801, 58),
                    filter_range=(0, None)):
    """Visualizes the most relevant filters of one conv-layer in a certain model.
    https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py
    # Arguments
        model: The model containing layer_name.
        layer_name: The name of the layer to be visualized.
                    Has to be a part of model.
        step: step size for gradient ascent.
        epochs: Number of iterations for gradient ascent.
        upscaling_steps: Number of upscaling steps.
                         Starting image is in this case (80, 80).
        upscaling_factor: Factor to which to slowly upgrade
                          the image towards output_dim.
        output_dim: [img_width, img_height] The output image dimensions.
        filter_range: Tupel[lower, upper]
                      Determines the to be computed filter numbers.
                      If the second value is `None`,
                      the last filter will be inferred as the upper boundary.
    """

    def _generate_filter_dat(output_shape,
                             layer_output,
                             filter_index):
        """Generates image for one particular filter.

        # Arguments
            input_img: The input-image Tensor.
            layer_output: The output-image Tensor.
            filter_index: The to be processed filter number.
                          Assumed to be valid.

        #Returns
            Either None if no image could be generated.
            or a tuple of the image (array) itself and the last loss.
        """
        s_time = time.time()

        loss_object = K.mean(layer_output[:, :, filter_index])

        @tf.function
        def iterate(train_dat):
            with tf.GradientTape() as tape:
                tape.watch(train_dat)
                predictions = model(train_dat)
                loss = loss_object(predictions)
            gradients = tape.gradient(loss, train_dat)
            # normalization trick: we normalize the gradient
            gradients = normalize(gradients)
            return loss, gradients

        # we start with some random noise
        # intermediate_dim = tuple(
        #     int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim)
        #
        # input_dat = np.random.random((1, intermediate_dim[0], intermediate_dim[1])).astype(np.float32)

        # Slowly upscaling towards the original size prevents
        # a dominating high-frequency of the to visualized structure
        # as it would occur if we directly compute the 412d-image.
        # Behaves as a better starting point for each following dimension
        # and therefore avoids poor local minima

        input_data = tf.convert_to_tensor(np.random.randn(*output_shape).astype(np.float32)[None, :, :])

        for up in reversed(range(upscaling_steps)):
            # we run gradient ascent for e.g. 20 steps
            for _ in range(epochs):
                loss_value, grads_value = iterate([input_data])
                input_data += grads_value * step

                # some filters get stuck to 0, we can skip them
                if loss_value <= K.epsilon():
                    return None

            if False:
                # Calulate upscaled dimension
                intermediate_dim = tuple(
                    int(x / (upscaling_factor ** up)) for x in output_dim)
                # TODO: Upscale
                dat = deprocess_timeseries(input_dat[0])
                # TODO: Reshape
                input_dat = [process_timeseries(dat, input_dat[0])]

        if False:
            # decode the resulting input image
            dat = deprocess_timeseries(input_dat[0])
            e_time = time.time()
            print('Costs of filter {:3}: {:5.0f} ( {:4.2f}s )'.format(filter_index,
                                                                      loss_value,
                                                                      e_time - s_time))
        return input_data[0], loss_value

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

        MARGIN = 5
        width = n * output_dim[0] + (n - 1) * MARGIN
        height = n * output_dim[1] + (n - 1) * MARGIN
        stitched_filters = np.zeros((width, height, 3), dtype='uint8')

        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                img, _ = max_acts[i * n + j]
                width_margin = (output_dim[0] + MARGIN) * i
                height_margin = (output_dim[1] + MARGIN) * j
                stitched_filters[
                    width_margin: width_margin + output_dim[0],
                    height_margin: height_margin + output_dim[1], :] = img

    output_layer = model.layers[layer_idx]
    max_filts = len(output_layer.get_weights()[1])
    max_filts = max_filts if filter_range[1] is None else min(max_filts, filter_range[1])
    # iterate through each filter and generate its corresponding time series
    maximizing_activations = []
    for f_ix in range(filter_range[0], max_filts):
        if isinstance(output_layer, tf.keras.layers.Conv1D):
            model_output = output_layer.output[:, :, f_ix]
        else:
            model_output = output_layer.output[f_ix]
        max_model = tf.keras.Model(model.input, model_output)
        max_data, loss_vals = _generate_filter_dat(output_dim, max_model)
        if max_data is not None:
            maximizing_activations.append(max_data)

    print('{} filter processed.'.format(len(maximizing_activations)))
    # Stitch timeseries together into one mega timeseries with NaN gaps.
    _stitch_filters(maximizing_activations)


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
    visualize_layer(model, layer_idx,
                    upscaling_steps=1, upscaling_factor=1.0,
                    output_dim=model.get_input_shape_at(0)[1:])
