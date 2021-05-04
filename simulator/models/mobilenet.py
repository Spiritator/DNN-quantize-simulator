"""Quantized MobileNet v1 model

Rebuild MobileNet v1 model on quantized keras layer.
The following code is base on the keras-application mobilenet.py


MobileNet v1 models for Keras.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNets support any input size greater than 32 x 32, with larger image sizes
offering better performance.
The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 16 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.75, 0.5 and 0.25.
For each of these `alpha` values, weights for 4 different input image sizes
are provided (224, 192, 160, 128).

The following table describes the size and accuracy of the 100% MobileNet
on size 224 x 224:
----------------------------------------------------------------------------
Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
----------------------------------------------------------------------------
|   1.0 MobileNet-224    |    70.6 %     |        529        |     4.2     |
|   0.75 MobileNet-224   |    68.4 %     |        325        |     2.6     |
|   0.50 MobileNet-224   |    63.7 %     |        149        |     1.3     |
|   0.25 MobileNet-224   |    50.6 %     |        41         |     0.5     |
----------------------------------------------------------------------------

The following table describes the performance of
the 100 % MobileNet on various input sizes:
------------------------------------------------------------------------
      Resolution      | ImageNet Acc | Multiply-Adds (M) | Params (M)
------------------------------------------------------------------------
|  1.0 MobileNet-224  |    70.6 %    |        529        |     4.2     |
|  1.0 MobileNet-192  |    69.1 %    |        529        |     4.2     |
|  1.0 MobileNet-160  |    67.2 %    |        529        |     4.2     |
|  1.0 MobileNet-128  |    64.4 %    |        529        |     4.2     |
------------------------------------------------------------------------

The weights for all 16 models are obtained and translated
from TensorFlow checkpoints found at
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md

# Reference

- [MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf))
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings

from keras_applications import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape

BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
                    'releases/download/v0.6/')

import tensorflow.keras.backend as backend
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.utils as keras_utils

from tqdm import tqdm
from ..layers.quantized_layers import QuantizedConv2D, QuantizedDepthwiseConv2D, QuantizedBatchNormalization
from ..layers.quantized_ops import quantizer,build_layer_quantizer


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, data_format='channels_last', mode='tf', **kwargs)


def QuantizedMobileNetV1(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000,
              batch_size=None,
              nbits=16,
              fbits=8, 
              BN_nbits=None, 
              BN_fbits=None,
              rounding_method='nearest',
              quant_mode='hybrid',
              overflow_mode=False,
              stop_gradient=False,
              verbose=True,
              **kwargs):
    """Instantiates the MobileNet architecture.

    To load a MobileNet model via `load_model`, import the custom
    objects `relu6` and pass them to the `custom_objects` parameter.
    E.g.
    model = load_model('mobilenet.h5', custom_objects={
                       'relu6': mobilenet.relu6})

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or (3, 224, 224) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution
            (also called the resolution multiplier)
        dropout: dropout rate
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """

    if verbose:
        print('\nBuilding model : Quantized MobileNet V1')
        pbar=tqdm(total=16)

    if BN_nbits is None:
        BN_nbits=nbits

    if BN_fbits is None:
        BN_fbits=fbits
        
    layer_quantizer=build_layer_quantizer(nbits,fbits,rounding_method,overflow_mode,stop_gradient)
            
    layer_BN_quantizer=build_layer_quantizer(BN_nbits,BN_fbits,rounding_method,overflow_mode,stop_gradient)
    
    if verbose:
        pbar.set_postfix_str('Preparation')
        pbar.update()


    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

        if rows != cols or rows not in [128, 160, 192, 224]:
            if rows is None:
                rows = 224
                warnings.warn('MobileNet shape is undefined.'
                              ' Weights for input shape '
                              '(224, 224) will be loaded.')
            else:
                raise ValueError('If imagenet weights are being loaded, '
                                 'input must have a static square shape '
                                 '(one of (128, 128), (160, 160), '
                                 '(192, 192), or (224, 224)). '
                                 'Input shape provided = %s' % (input_shape,))

    if backend.image_data_format() != 'channels_last':
        warnings.warn('The MobileNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height).'
                      ' You should set `image_data_format="channels_last"` '
                      'in your Keras config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        backend.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = layers.Input(batch_shape=(batch_size,)+input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, batch_shape=(batch_size,)+input_shape)
        else:
            img_input = input_tensor

    if verbose:
        pbar.set_postfix_str('building standard conv block')
    x = _conv_block(img_input, 32, alpha, strides=(2, 2), 
                    layer_quantizer=layer_quantizer, 
                    layer_BN_quantizer=layer_BN_quantizer, 
                    quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 1')
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1, 
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('building depthwise conv block 2')
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2,
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 3')
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3,
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('building depthwise conv block 4')
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4,
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 5')
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5,
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('building depthwise conv block 6')
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6,
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 7')
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7, 
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 8')
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8, 
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 9')
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9, 
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 10')
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10, 
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 11')
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11, 
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('building depthwise conv block 12')
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12,
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 13')
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13,
                              layer_quantizer=layer_quantizer, 
                              layer_BN_quantizer=layer_BN_quantizer, 
                              quant_mode=quant_mode)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('building output block')
    if include_top:
        if backend.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape(shape, name='reshape_1')(x)
        x = layers.Dropout(dropout, name='dropout')(x)
        x = QuantizedConv2D(classes, 
                            kernel_size=(1, 1),
                            quantizers=layer_quantizer,
                            padding='same',
                            name='conv_preds',
                            quant_mode=quant_mode,
                            last_layer=True)(x)
        x = layers.Activation('softmax', name='act_softmax')(x)
        x = layers.Reshape((classes,), name='reshape_2')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    if verbose:
        pbar.update()
    

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='quantized_mobilenet_%0.2f_%s' % (alpha, rows))
    
    if verbose:
        pbar.set_postfix_str('Model Built')
        pbar.close()

    # load weights
    if weights == 'imagenet':
        if backend.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_first" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        if include_top:
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = keras_utils.get_file(model_name,
                                                weight_path,
                                                cache_subdir='models')
        else:
            model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = keras_utils.get_file(model_name,
                                                weight_path,
                                                cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if old_data_format:
        backend.set_image_data_format(old_data_format)
    return model


def _conv_block(inputs, 
                filters, 
                alpha, 
                kernel=(3, 3), 
                strides=(1, 1), 
                layer_quantizer=quantizer(16,8), 
                layer_BN_quantizer=quantizer(16,8), 
                quant_mode='hybrid'):
    """Adds an initial convolution layer (with batch normalization and relu6).

    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
        
    x = layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = QuantizedConv2D(filters, 
                        kernel_size=kernel,
                        quantizers=layer_quantizer, 
                        padding='valid',
                        use_bias=False,
                        strides=strides,
                        name='conv1', 
                        quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=channel_axis, 
                                    name='conv1_bn', 
                                    quant_mode=quant_mode)(x)
    return layers.ReLU(6., name='conv1_relu')(x)


def _depthwise_conv_block(inputs, 
                          pointwise_conv_filters, 
                          alpha,
                          depth_multiplier=1, 
                          strides=(1, 1), 
                          block_id=1,
                          layer_quantizer=quantizer(16,8), 
                          layer_BN_quantizer=quantizer(16,8), 
                          quant_mode='hybrid'):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    

    x = layers.ZeroPadding2D((1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = QuantizedDepthwiseConv2D(kernel_size=(3, 3),
                                 quantizers=layer_quantizer,
                                 padding='valid',
                                 depth_multiplier=depth_multiplier,
                                 strides=strides,
                                 use_bias=False,
                                 name='conv_dw_%d' % block_id, 
                                 quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=channel_axis, 
                                    name='conv_dw_%d_bn' % block_id, 
                                    quant_mode=quant_mode)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = QuantizedConv2D(pointwise_conv_filters, 
                        kernel_size=(1, 1),
                        quantizers=layer_quantizer,
                        padding='same',
                        use_bias=False,
                        strides=(1, 1),
                        name='conv_pw_%d' % block_id, 
                        quant_mode=quant_mode)(x)
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=channel_axis,
                                    name='conv_pw_%d_bn' % block_id, 
                                    quant_mode=quant_mode)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)




#==============================================================================
#    FUSED BATCHNORMALIZATION MODEL
#==============================================================================


def QuantizedMobileNetV1FusedBN(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000,
              batch_size=None,
              nbits=16,
              fbits=8, 
              rounding_method='nearest',
              quant_mode='hybrid',
              overflow_mode=False,
              stop_gradient=False,
              verbose=True,
              **kwargs):
    """Instantiates the MobileNet architecture.

    To load a MobileNet model via `load_model`, import the custom
    objects `relu6` and pass them to the `custom_objects` parameter.
    E.g.
    model = load_model('mobilenet.h5', custom_objects={
                       'relu6': mobilenet.relu6})

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or (3, 224, 224) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution
            (also called the resolution multiplier)
        dropout: dropout rate
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    if verbose:
        print('\nBuilding model : Quantized MobileNet V1 Fused BatchNornalization')
        pbar=tqdm(total=16)
    
    layer_quantizer=build_layer_quantizer(nbits,fbits,rounding_method,overflow_mode,stop_gradient)
    if verbose:
        pbar.set_postfix_str('Preparation')
        pbar.update()

    if not os.path.exists(weights):
        raise ValueError('The `weights` argument must be the path to the weights file to be loaded. File not found!')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

        if rows != cols or rows not in [128, 160, 192, 224]:
            if rows is None:
                rows = 224
                warnings.warn('MobileNet shape is undefined.'
                              ' Weights for input shape '
                              '(224, 224) will be loaded.')
            else:
                raise ValueError('If imagenet weights are being loaded, '
                                 'input must have a static square shape '
                                 '(one of (128, 128), (160, 160), '
                                 '(192, 192), or (224, 224)). '
                                 'Input shape provided = %s' % (input_shape,))

    if backend.image_data_format() != 'channels_last':
        warnings.warn('The MobileNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height).'
                      ' You should set `image_data_format="channels_last"` '
                      'in your Keras config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        backend.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = layers.Input(batch_shape=(batch_size,)+input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, batch_shape=(batch_size,)+input_shape)
        else:
            img_input = input_tensor

    if verbose:
        pbar.set_postfix_str('building standard conv block')
    x = _conv_block_fused_BN(img_input, 32, alpha, strides=(2, 2), 
                             layer_quantizer=layer_quantizer, 
                             quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 1')
    x = _depthwise_conv_block_fused_BN(x, 64, alpha, depth_multiplier, block_id=1, 
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('building depthwise conv block 2')
    x = _depthwise_conv_block_fused_BN(x, 128, alpha, depth_multiplier,
                                       strides=(2, 2), block_id=2,
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 3')
    x = _depthwise_conv_block_fused_BN(x, 128, alpha, depth_multiplier, block_id=3,
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('building depthwise conv block 4')
    x = _depthwise_conv_block_fused_BN(x, 256, alpha, depth_multiplier,
                                       strides=(2, 2), block_id=4,
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 5')
    x = _depthwise_conv_block_fused_BN(x, 256, alpha, depth_multiplier, block_id=5,
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('building depthwise conv block 6')
    x = _depthwise_conv_block_fused_BN(x, 512, alpha, depth_multiplier,
                                       strides=(2, 2), block_id=6,
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 7')
    x = _depthwise_conv_block_fused_BN(x, 512, alpha, depth_multiplier, block_id=7, 
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 8')
    x = _depthwise_conv_block_fused_BN(x, 512, alpha, depth_multiplier, block_id=8, 
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 9')
    x = _depthwise_conv_block_fused_BN(x, 512, alpha, depth_multiplier, block_id=9, 
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 10')
    x = _depthwise_conv_block_fused_BN(x, 512, alpha, depth_multiplier, block_id=10, 
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 11')
    x = _depthwise_conv_block_fused_BN(x, 512, alpha, depth_multiplier, block_id=11, 
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('building depthwise conv block 12')
    x = _depthwise_conv_block_fused_BN(x, 1024, alpha, depth_multiplier,
                                       strides=(2, 2), block_id=12,
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('building depthwise conv block 13')
    x = _depthwise_conv_block_fused_BN(x, 1024, alpha, depth_multiplier, block_id=13,
                                       layer_quantizer=layer_quantizer, 
                                       quant_mode=quant_mode)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('building output block')
    if include_top:
        if backend.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape(shape, name='reshape_1')(x)
        x = layers.Dropout(dropout, name='dropout')(x)
        x = QuantizedConv2D(classes, 
                            kernel_size=(1, 1),
                            quantizers=layer_quantizer,
                            padding='same',
                            name='conv_preds',
                            quant_mode=quant_mode,
                            last_layer=True)(x)
        x = layers.Activation('softmax', name='act_softmax')(x)
        x = layers.Reshape((classes,), name='reshape_2')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    if verbose:
        pbar.update()

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='quantized_mobilenet_fusedBN_%0.2f_%s' % (alpha, rows))
    
    if verbose:
        pbar.set_postfix_str('Model Built')
        pbar.close()

    # load weights
    if weights is not None:
        model.load_weights(weights)

    if old_data_format:
        backend.set_image_data_format(old_data_format)
    return model


def _conv_block_fused_BN(inputs, 
                         filters, 
                         alpha, 
                         kernel=(3, 3), 
                         strides=(1, 1),
                         layer_quantizer=quantizer(16,8), 
                         quant_mode='hybrid'):
    """Adds an initial convolution layer (with batch normalization and relu6).

    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    #channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
        
    x = layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = QuantizedConv2D(filters, 
                        kernel_size=kernel,
                        quantizers=layer_quantizer,
                        padding='valid',
                        strides=strides,
                        name='conv1', 
                        quant_mode=quant_mode)(x)
    return layers.ReLU(6., name='conv1_relu')(x)


def _depthwise_conv_block_fused_BN(inputs, 
                                   pointwise_conv_filters, 
                                   alpha,
                                   depth_multiplier=1, 
                                   strides=(1, 1), 
                                   block_id=1,
                                   layer_quantizer=quantizer(16,8), 
                                   quant_mode='hybrid'):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        Output tensor of block.
    """
    #channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    

    x = layers.ZeroPadding2D((1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = QuantizedDepthwiseConv2D(kernel_size=(3, 3),
                                 quantizers=layer_quantizer,
                                 padding='valid',
                                 depth_multiplier=depth_multiplier,
                                 strides=strides,
                                 name='conv_dw_%d' % block_id, 
                                 quant_mode=quant_mode)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = QuantizedConv2D(pointwise_conv_filters, 
                        kernel_size=(1, 1),
                        quantizers=layer_quantizer,
                        padding='same',
                        strides=(1, 1),
                        name='conv_pw_%d' % block_id, 
                        quant_mode=quant_mode)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

