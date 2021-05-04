# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:33:23 2018

@author: Yung-Yu Tsai

evaluate quantized testing result with custom Keras quantize layer of LeNet-5
"""

# setup

import numpy as np
import tensorflow.keras.backend as K
import time

from simulator.models.model_library import quantized_lenet5
from simulator.utils_tool.weight_conversion import convert_original_weight_layer_name
from simulator.utils_tool.dataset_setup import dataset_setup
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
from tensorflow.keras.losses import categorical_crossentropy
from simulator.metrics.topk_metrics import top2_acc

#%% model setup

weight_name='../mnist_lenet5_weight.h5'
batch_size=25

# model setup
# all arguments use the same quantize precision
#model=quantized_lenet5(nbits=8,fbits=4,rounding_method='nearest')
# each argument uses different quantize precision. information list [input, weight, output]
#model=quantized_lenet5(nbits=[10,4,10],fbits=[5,2,5],rounding_method='nearest')
# intrinsic quantization
#model=quantized_lenet5(nbits=8,fbits=2,rounding_method='nearest',batch_size=batch_size,quant_mode='hybrid',overflow_mode=[True,False,True])
#model=quantized_lenet5(nbits=8,fbits=3,rounding_method='nearest',batch_size=batch_size,quant_mode='intrinsic')
model=quantized_lenet5(nbits=8,fbits=3,rounding_method=['down','nearest','down'],batch_size=batch_size,quant_mode='hybrid')
#model=quantized_lenet5(nbits=8,fbits=[6,6,6],rounding_method=['down','nearest','down'],batch_size=batch_size,quant_mode='intrinsic')

weight_name=convert_original_weight_layer_name(weight_name)
model.load_weights(weight_name)
print('orginal weight loaded')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,pred_vary_10,pred_vary_20])

# multi GPU
#parallel_model = multi_gpu_model(model, gpus=2)
#parallel_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])


#%% dataset setup

x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('mnist')

#%% view test result

t = time.time()
test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)

t = time.time()-t
print('\nruntime: %f s'%t)

print('\nLoss: %f'%test_result[0])
print('Top1 Accuracy: %f'%test_result[1])
print('Top2 Accuracy: %f'%test_result[2])

#%% draw confusion matrix

prediction = model.predict(x_test, verbose=1,batch_size=batch_size)

print('\n')
#prediction = model.predict(x_test, verbose=1,batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',normalize=False)


