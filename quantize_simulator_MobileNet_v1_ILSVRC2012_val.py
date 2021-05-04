# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:32:50 2018

@author: Yung-Yu Tsai

evaluate quantized testing result with custom Keras quantize layer of MobileNetV1
"""

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from simulator.models.mobilenet import QuantizedMobileNetV1,QuantizedMobileNetV1FusedBN
from simulator.utils_tool.dataset_setup import dataset_setup
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
from tensorflow.keras.losses import categorical_crossentropy
from simulator.metrics.topk_metrics import top5_acc

import time


# dimensions of our images.
img_width, img_height = 224, 224

class_number=1000
batch_size=40

set_size=2
validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator_setsize_2'
nb_validation_samples = 50000

#%% model setup

#print('Building model...')
# t = time.time()
# model = QuantizedMobileNetV1(weights='../mobilenet_1_0_224_tf.h5', 
#                              nbits=16,
#                              fbits=8, 
#                              BN_nbits=16, 
#                              BN_fbits=8,
#                              rounding_method='nearest',
#                              batch_size=batch_size,
#                              quant_mode='hybrid')

# model = QuantizedMobileNetV1FusedBN(weights='../mobilenet_1_0_224_tf_fused_BN.h5', 
#                                     nbits=16,
#                                     fbits=8, 
#                                     rounding_method='nearest',
#                                     batch_size=batch_size,
#                                     quant_mode='hybrid')
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top5_acc])
# model.summary()
# t = time.time()-t
# print('model build time: %f s'%t)

# multi GPU model
print('Building multi GPU model...')
t = time.time()
strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])
with strategy.scope():
    parallel_model = QuantizedMobileNetV1FusedBN(weights='../mobilenet_1_0_224_tf_fused_BN.h5', 
                                        nbits=16,
                                        fbits=8, 
                                        rounding_method='nearest',
                                        batch_size=batch_size,
                                        quant_mode='hybrid')

    parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top5_acc])
    parallel_model.summary()

t = time.time()-t
print('multi GPU model build time: %f s'%t)

#%% dataset setup

print('preparing dataset...')
x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('ImageDataGenerator', img_rows = img_width, img_cols = img_height, batch_size = batch_size*strategy.num_replicas_in_sync, data_augmentation = False, data_dir = validation_data_dir)
print('dataset ready')


#%% test

t = time.time()
print('evaluating...')

test_result = parallel_model.predict(datagen, verbose=1, steps=len(datagen))

t = time.time()-t
print('\nruntime: %f s'%t)

print('\nLoss: %f'%test_result[0])
print('Top1 Accuracy: %f'%test_result[1])
print('Top5 Accuracy: %f'%test_result[2])

#%% draw confusion matrix

#prediction = parallel_model.predict(datagen, verbose=1, steps=len(datagen))
#prediction = model.predict(datagen, verbose=1, steps=len(datagen))

#print('\n')
#prediction = np.argmax(prediction, axis=1)
#
#show_confusion_matrix(datagen.classes,prediction,datagen.class_indices.keys(),'Confusion Matrix',figsize=(10,8),normalize=False,big_matrix=True)

