# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:20:35 2021

@author: Yung-Yu Tsai

Example for view intermediate of each DNN layer
"""

# import this
from simulator.inference.verification import view_intermediate

import numpy as np
import tensorflow.keras.backend as K

from simulator.models.model_library import quantized_lenet5
from tensorflow.keras.preprocessing import image


#%% build model

weight_name='../mnist_lenet5_weight.h5'

model=quantized_lenet5(nbits=8,
                       fbits=3,
                       rounding_method=['down','nearest','down'],
                       batch_size=None,
                       quant_mode='hybrid')

model.load_weights(weight_name)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print('model build')

#%% load test image

# load image path
img_path = '../test_images and intermediate/number4.jpg'
img = image.load_img(img_path, target_size=(28, 28))
x = image.img_to_array(img)
# gray scale
x = np.mean(x,axis=2,keepdims=True)
# add batch dimension
x = np.expand_dims(x, axis=0)
# preprocess normalize
x = np.subtract(1,np.divide(x,255))


lenet_intermediate=view_intermediate(model,x,eager_mode=True) 
# eager mode set to true
# graph mode can't represent the same value as RTL verilog

print('Prediction: %d | Probability: %f'%(np.argmax(lenet_intermediate[-1]),np.max(lenet_intermediate[-1])))

      
