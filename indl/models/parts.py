import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, constraints
from ..regularizers import KernelLengthRegularizer


def Conv2DBlockRegu(input_shape,
                    n_kerns=4, kern_space=1, kern_length=64,
                    kern_regu_scale=0.0, l1=0.000, l2=0.000,
                    n_pool=4,
                    dropout_rate=0.25, dropout_type='Dropout',
                    activation='elu'):

    if dropout_type == 'SpatialDropout2D':
        dropout_type = layers.SpatialDropout2D
    elif dropout_type == 'Dropout':
        dropout_type = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    num_chans = input_shape[0]
    _input = layers.Input(shape=(num_chans, None, 1))

    if kern_regu_scale:
        kern_regu = KernelLengthRegularizer((1, kern_length),
                                            window_func='poly',
                                            window_scale=kern_regu_scale,
                                            poly_exp=2, threshold=0.0015)
    elif l1 > 0 or l2 > 0:
        kern_regu = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
    else:
        kern_regu = None

    # Temporal-filter-like
    _y = layers.Conv2D(n_kerns, (kern_space, kern_length),
                       padding='same',
                       kernel_regularizer=kern_regu,
                       use_bias=False)(_input)
    _y = layers.BatchNormalization()(_y)
    _y = layers.Activation(activation)(_y)
    if n_pool > 1:
        _y = layers.AveragePooling2D((1, n_pool))(_y)
    _y = dropout_type(dropout_rate)(_y)

    return models.Model(inputs=_input, outputs=_y)


def DepthwiseConv2DBlock(input_shape,
                         depth_multiplier=4,
                         depth_pooling=1,
                         dropout_rate=0.25, dropout_type='Dropout',
                         activation='elu'):
    if dropout_type == 'SpatialDropout2D':
        dropout_type = layers.SpatialDropout2D
    elif dropout_type == 'Dropout':
        dropout_type = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    _input = layers.Input(shape=input_shape)
    _y = layers.DepthwiseConv2D((input_shape[0], 1), use_bias=False,
                                depth_multiplier=depth_multiplier,
                                depthwise_constraint=constraints.max_norm(1.))(_input)
    _y = layers.BatchNormalization()(_y)
    _y = layers.Activation(activation)(_y)
    if depth_pooling > 1:
        _y = layers.AveragePooling2D((1, depth_pooling))(_y)
    _y = dropout_type(dropout_rate)(_y)
    return models.Model(inputs=_input, outputs=_y)


def SeparableConv2DBlock(input_shape,
                    n_kerns=4, kern_space=1, kern_length=64,
                    kern_regu_scale=0.0, l1=0.000, l2=0.000,
                    n_pool=4,
                    dropout_rate=0.25, dropout_type='Dropout',
                    activation='elu'):

    if dropout_type == 'SpatialDropout2D':
        dropout_type = layers.SpatialDropout2D
    elif dropout_type == 'Dropout':
        dropout_type = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    num_chans = input_shape[0]
    _input = layers.Input(shape=(num_chans, None, 1))

    _y = layers.SeparableConv2D(n_kerns, (kern_space, kern_length),
                                padding='same', use_bias=False)(_input)
    _y = layers.BatchNormalization()(_y)
    _y = layers.Activation(activation)(_y)
    if n_pool > 1:
        _y = layers.AveragePooling2D((1, n_pool))(_y)
    _y = dropout_type(dropout_rate)(_y)

    return models.Model(inputs=_input, outputs=_y)


def Bottleneck(input_shape, latent_dim=32, activation='elu', dropoutRate=0.25):

    _input = layers.Input(input_shape)
    flatten = layers.Flatten()(_input)

    # Force through bottleneck
    bottleneck = layers.Dense(latent_dim, activation=activation)(flatten)
    bottleneck = layers.Dropout(dropoutRate)(bottleneck)
    bottleneck = layers.Activation(activation, name='latent')(bottleneck)

    return models.Model(inputs=_input, outputs=bottleneck)


def CombineInputs(input_shapes):
    inputs, subblocks = [], []
    for shape in input_shapes:
        inputs.append(layers.Input(shape=shape))
        subblocks.append(inputs)

    concat = layers.Concatenate(axis=-1, name='concat')(subblocks)
    return models.Model(inputs=inputs, outputs=concat)
