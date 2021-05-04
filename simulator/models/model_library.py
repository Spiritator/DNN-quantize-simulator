# -*- coding: utf-8 -*-
"""
Model library for custom quantized model LeNet-5, 4C2F-CNN, DroneNet

@author: Yung-Yu Tsai

"""
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
import numpy as np
from tqdm import tqdm

from ..layers.quantized_layers import QuantizedConv2D, QuantizedDense, QuantizedBatchNormalization, QuantizedFlatten
from ..layers.quantized_ops import quantizer,build_layer_quantizer


def quantized_lenet5(nbits=8, fbits=4, rounding_method='nearest', 
                     input_shape=(28,28,1), num_classes=10, batch_size=None, 
                     quant_mode='hybrid', 
                     overflow_mode=False, stop_gradient=False,
                     verbose=True):
    
    if verbose:
        print('\nBuilding model : Quantized Lenet 5')
        pbar=tqdm(total=9)
    
    layer_quantizer=build_layer_quantizer(nbits,fbits,rounding_method,overflow_mode,stop_gradient)
    
    if verbose:
        pbar.set_postfix_str('Preparation')
        pbar.update()
        
        pbar.set_postfix_str('Building Layer 0')
    input_shape = Input(batch_shape=(batch_size,)+input_shape)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 1')
    x = QuantizedConv2D(filters=16,
                        quantizers=layer_quantizer,
                        kernel_size=(5,5),
                        padding='same',
                        strides=(1, 1),                              
                        activation='relu',
                        quant_mode=quant_mode)(input_shape)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 2')
    x = MaxPooling2D(pool_size=(2,2))(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 3')
    x = QuantizedConv2D(filters=36,
                        quantizers=layer_quantizer,
                        kernel_size=(5,5),
                        padding='same',
                        strides=(1, 1),
                        activation='relu',
                        quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 4')
    x = MaxPooling2D(pool_size=(2,2))(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 5')
    x = QuantizedFlatten()(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 6')
    x = QuantizedDense(128,
                       quantizers=layer_quantizer,
                       activation='relu',
                       quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 7')
    x = QuantizedDense(num_classes,
                       quantizers=layer_quantizer,
                       activation='softmax',
                       quant_mode=quant_mode,
                       last_layer=True)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Model Built')
        pbar.close()

    model=Model(inputs=input_shape, outputs=x, name='quantized_lenet5')
    
    return model

def quantized_4C2F(nbits=8, fbits=4, rounding_method='nearest', 
                   input_shape=(32,32,3), num_classes=10, batch_size=None, 
                   quant_mode='hybrid', 
                   overflow_mode=False, stop_gradient=False,
                   verbose=True):
    
    if verbose:
        print('\nBuilding model : Quantized 4C2F CNN')
        pbar=tqdm(total=14)
    
    layer_quantizer=build_layer_quantizer(nbits,fbits,rounding_method,overflow_mode,stop_gradient)
    
    if verbose:
        pbar.set_postfix_str('Preparation')
        pbar.update()
    
        pbar.set_postfix_str('Building Layer 0')
    input_shape = Input(batch_shape=(batch_size,)+input_shape)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 1')
    x = QuantizedConv2D(filters=32,
                        quantizers=layer_quantizer,
                        kernel_size=(3, 3),
                        padding='same',
                        strides=(1, 1),
                        activation='relu',
                        quant_mode=quant_mode)(input_shape)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 2')
    x = QuantizedConv2D(filters=32,
                        quantizers=layer_quantizer,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        activation='relu',
                        quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 3')
    x = MaxPooling2D(pool_size=(2, 2))(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 4')
    x = Dropout(0.25)(x)
    if verbose:
        pbar.update()
    
        pbar.set_postfix_str('Building Layer 5')
    x = QuantizedConv2D(filters=64,
                        quantizers=layer_quantizer,
                        kernel_size=(3, 3),
                        padding='same',
                        strides=(1, 1),
                        activation='relu',
                        quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 6')
    x = QuantizedConv2D(filters=64,
                        quantizers=layer_quantizer,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        activation='relu',
                        quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 7')
    x = MaxPooling2D(pool_size=(2, 2))(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 8')
    x = Dropout(0.25)(x)
    if verbose:
        pbar.update()
    
        pbar.set_postfix_str('Building Layer 9')
    x = QuantizedFlatten()(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 10')
    x = QuantizedDense(512,
                       quantizers=layer_quantizer,
                       activation='relu',
                       quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 11')
    x = Dropout(0.5)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 12')
    x = QuantizedDense(num_classes,
                       quantizers=layer_quantizer,
                       activation='softmax',
                       quant_mode=quant_mode,
                       last_layer=True)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Model Built')
        pbar.close()
    
    model=Model(inputs=input_shape, outputs=x, name='quantized_4C2F')
    
    return model


def quantized_4C2FBN(nbits=8, fbits=4, BN_nbits=None, BN_fbits=None, rounding_method='nearest', 
                     input_shape=(32,32,3), num_classes=10, batch_size=None, 
                     quant_mode='hybrid', 
                     overflow_mode=False, stop_gradient=False,
                     verbose=True):
    
    if verbose:
        print('\nBuilding model : Quantized 4C2FBN CNN')
        pbar=tqdm(total=24)
    
    if BN_nbits is None:
        BN_nbits=nbits

    if BN_fbits is None:
        BN_fbits=fbits
        
    layer_quantizer=build_layer_quantizer(nbits,fbits,rounding_method,overflow_mode,stop_gradient)
        
    layer_BN_quantizer=build_layer_quantizer(BN_nbits,BN_fbits,rounding_method,overflow_mode,stop_gradient)
    
    if verbose:
        pbar.set_postfix_str('Preparation')
        pbar.update()
        
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    if verbose:
        pbar.set_postfix_str('Building Layer 0')
    input_shape = Input(batch_shape=(batch_size,)+input_shape)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 1')
    x = QuantizedConv2D(filters=32,
                        quantizers=layer_quantizer,
                        kernel_size=(3, 3),
                        padding='same',
                        strides=(1, 1),
                        quant_mode=quant_mode)(input_shape)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 2')
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=channel_axis, 
                                    quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 3')
    x = Activation('relu')(x)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('Building Layer 4')
    x = QuantizedConv2D(filters=32,
                        quantizers=layer_quantizer,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 5')
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=channel_axis, 
                                    quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 6')
    x = Activation('relu')(x)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('Building Layer 7')
    x = MaxPooling2D(pool_size=(2, 2))(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 8')
    x = Dropout(0.25)(x)
    if verbose:
        pbar.update()
    
        pbar.set_postfix_str('Building Layer 9')
    x = QuantizedConv2D(filters=64,
                        quantizers=layer_quantizer,
                        kernel_size=(3, 3),
                        padding='same',
                        strides=(1, 1),
                        quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 10')
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=channel_axis, 
                                    quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 11')
    x = Activation('relu')(x)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('Building Layer 12')
    x = QuantizedConv2D(filters=64,
                        quantizers=layer_quantizer,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 13')
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=channel_axis, 
                                    quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 14')
    x = Activation('relu')(x)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('Building Layer 15')
    x = MaxPooling2D(pool_size=(2, 2))(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 16')
    x = Dropout(0.25)(x)
    if verbose:
        pbar.update()
    
        pbar.set_postfix_str('Building Layer 17')
    x = QuantizedFlatten()(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 18')
    x = QuantizedDense(512,
                       quantizers=layer_quantizer,
                       quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 19')
    x = QuantizedBatchNormalization(quantizers=layer_BN_quantizer,
                                    axis=channel_axis, 
                                    quant_mode=quant_mode)(x)
    if verbose:
        pbar.update()

        pbar.set_postfix_str('Building Layer 20')
    x = Activation('relu')(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 21')
    x = Dropout(0.5)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Building Layer 22')
    x = QuantizedDense(num_classes,
                       quantizers=layer_quantizer,
                       activation='softmax',
                       quant_mode=quant_mode,
                       last_layer=True)(x)
    if verbose:
        pbar.update()
        pbar.set_postfix_str('Model Built')
        pbar.close()
    
    model=Model(inputs=input_shape, outputs=x, name='quantized_4C2FBN')
    
    return model


    
# model with activation function as a independent layer for examine the feature maps distribution
def quantized_lenet5_splt_act(nbits=8, fbits=4, rounding_method='nearest', 
                              input_shape=(28,28,1), num_classes=10, batch_size=None, 
                              quant_mode='hybrid', 
                              overflow_mode=False, stop_gradient=False,):
    
    print('\nBuilding model : Quantized Lenet 5')
    pbar=tqdm(total=11)
    
    layer_quantizer=build_layer_quantizer(nbits,fbits,rounding_method,overflow_mode,stop_gradient)
        
    pbar.set_postfix_str('Building Layer 0')
    input_shape = Input(batch_shape=(batch_size,)+input_shape)
    pbar.update()
    pbar.set_postfix_str('Building Layer 1')
    x = QuantizedConv2D(filters=16,
                        quantizers=layer_quantizer,
                        kernel_size=(5,5),
                        padding='same',
                        strides=(1, 1),                              
                        activation=None,
                        quant_mode=quant_mode)(input_shape)
    pbar.update()
    pbar.set_postfix_str('Building Layer 2')
    x = Activation('relu')(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 3')
    x = MaxPooling2D(pool_size=(2,2))(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 4')
    x = QuantizedConv2D(filters=36,
                        quantizers=layer_quantizer,
                        kernel_size=(5,5),
                        padding='same',
                        strides=(1, 1),
                        activation=None,
                        quant_mode=quant_mode)(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 5')
    x = Activation('relu')(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 6')
    x = MaxPooling2D(pool_size=(2,2))(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 7')
    x = QuantizedFlatten()(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 8')
    x = QuantizedDense(128,
                       quantizers=layer_quantizer,
                       activation=None,
                       quant_mode=quant_mode)(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 9')
    x = Activation('relu')(x)    
    pbar.update()
    pbar.set_postfix_str('Building Layer 10')
    x = QuantizedDense(num_classes,
                       quantizers=layer_quantizer,
                       activation='softmax',
                       quant_mode=quant_mode,
                       last_layer=True)(x)
    pbar.update()
    pbar.set_postfix_str('Model Built')
    pbar.close()

    model=Model(inputs=input_shape, outputs=x, name='quantized_lenet5')
    
    return model


# model with activation function as a independent layer for examine the feature maps distribution
def quantized_4C2F_splt_act(nbits=8, fbits=4, rounding_method='nearest', 
                            input_shape=(32,32,3), num_classes=10, batch_size=None, 
                            quant_mode='hybrid', 
                            overflow_mode=False, stop_gradient=False):
    
    print('\nBuilding model : Quantized 4C2F CNN')
    pbar=tqdm(total=18)
    
    layer_quantizer=build_layer_quantizer(nbits,fbits,rounding_method,overflow_mode,stop_gradient)
    
    pbar.set_postfix_str('Building Layer 0')
    input_shape = Input(batch_shape=(batch_size,)+input_shape)
    pbar.update()
    pbar.set_postfix_str('Building Layer 1')
    x = QuantizedConv2D(filters=32,
                        quantizers=layer_quantizer,
                        kernel_size=(3, 3),
                        padding='same',
                        strides=(1, 1),
                        activation=None,
                        quant_mode=quant_mode)(input_shape)
    pbar.update()
    pbar.set_postfix_str('Building Layer 2')
    x = Activation('relu')(x)    
    
    pbar.update()
    pbar.set_postfix_str('Building Layer 3')
    x = QuantizedConv2D(filters=32,
                        quantizers=layer_quantizer,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        activation=None,
                        quant_mode=quant_mode)(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 4')
    x = Activation('relu')(x)    
    
    pbar.update()
    pbar.set_postfix_str('Building Layer 5')
    x = MaxPooling2D(pool_size=(2, 2))(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 6')
    x = Dropout(0.25)(x)
    pbar.update()
    
    pbar.set_postfix_str('Building Layer 7')
    x = QuantizedConv2D(filters=64,
                        quantizers=layer_quantizer,
                        kernel_size=(3, 3),
                        padding='same',
                        strides=(1, 1),
                        activation=None,
                        quant_mode=quant_mode)(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 8')
    x = Activation('relu')(x)    

    pbar.update()
    pbar.set_postfix_str('Building Layer 9')
    x = QuantizedConv2D(filters=64,
                        quantizers=layer_quantizer,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        activation=None,
                        quant_mode=quant_mode)(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 10')
    x = Activation('relu')(x)    
    
    pbar.update()
    pbar.set_postfix_str('Building Layer 11')
    x = MaxPooling2D(pool_size=(2, 2))(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 12')
    x = Dropout(0.25)(x)
    pbar.update()
    
    pbar.set_postfix_str('Building Layer 13')
    x = QuantizedFlatten()(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 14')
    x = QuantizedDense(512,
                       quantizers=layer_quantizer,
                       activation=None,
                       quant_mode=quant_mode)(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 15')
    x = Activation('relu')(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 16')
    x = Dropout(0.5)(x)
    pbar.update()
    pbar.set_postfix_str('Building Layer 17')
    x = QuantizedDense(num_classes,
                       quantizers=layer_quantizer,
                       activation='softmax',
                       quant_mode=quant_mode,
                       last_layer=True)(x)
    pbar.update()
    pbar.set_postfix_str('Model Built')
    pbar.close()
    
    model=Model(inputs=input_shape, outputs=x, name='quantized_4C2F')
    
    return model

