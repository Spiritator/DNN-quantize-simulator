# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 13:19:59 2018

@author: Yung-Yu Tsai 

train cifar10 on droneNet V2
"""

from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
import os

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#network setup

input_shape=x_train.shape[1:]

def droneNet(inputs=None, include_top=True, classes=10, *args, **kwargs):
    if inputs is None :
        if K.image_data_format() == 'channels_first':
            input_shape = Input(shape=(3, 224, 224))
        else:
            input_shape = Input(shape=(224, 224, 3))
    else:
        input_shape=inputs

    outputs = []

    x = Conv2D(32, (3, 3), strides=(1, 1),use_bias=False)(input_shape)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    outputs.append(x)

    for i in range(3):
        x = Conv2D(64*(2**i), (3, 3), strides=(1, 1),use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        outputs.append(x)

    x = Conv2D(256, (3, 3), strides=(1, 1),use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    outputs.append(x)
    

    if include_top:
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(classes, activation='sigmoid')(x)
        return Model(inputs=input_shape, outputs=x, *args, **kwargs)
    else:
        return Model(inputs=input_shape, outputs=outputs, *args, **kwargs)
    
model=droneNet(inputs=input_shape,classes=num_classes)
model.summary()

def top2_acc(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=2)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', top2_acc])


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
model.save_weights('cifar10_droneNet_v2_weight.h5')
model.save('cifar10_droneNet_v2_model.h5')

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test top1 accuracy:', scores[1])
print('Test top2 accuracy:', scores[2])

