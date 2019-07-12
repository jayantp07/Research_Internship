import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

training_data1=np.load("./training_data1/training_data.npy", allow_pickle=True)
X = np.array([i[0] for i in training_data1]).reshape(-1,256,256,1)
Y = np.array([i[1] for i in training_data1]).reshape(-1,256,256,1)
x_train=X[0:185, :, :, :]
y_train=Y[0:185, :, :, :]
x_test=X[185:265, :, :, :]
y_test=Y[185:265, :, :, :]

data_gen_args = dict(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1
image_datagen.fit(x_train, augment=True, seed=seed)
mask_datagen.fit(y_train, augment=True, seed=seed)

image_generator = image_datagen.flow(x_train, seed=seed, batch_size=10)
mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=10)

train_generator = zip(image_generator, mask_generator)
 
import os
import skimage.io as io
import tensorflow as tf
import skimage.transform as trans
from keras.layers import Input, BatchNormalization, Dropout, Lambda
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from time import time


smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x



def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    s = BatchNormalization()(input_img)
    c1 = conv2d_block(s, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy', dice_coef])
    
    model.summary()

    return model

#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
input_img = Input((256, 256, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.load_weights("unet_nodes_new_model.hdf5")
#model_checkpoint = ModelCheckpoint('unet_nodes_new_model.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(train_generator, steps_per_epoch=1295, epochs=150, verbose=1, callbacks=[model_checkpoint], validation_data=(x_test, y_test))

results = model.predict(x_test,verbose=1)

for i, item in enumerate(results):
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(np.squeeze(x_test[i]), cmap="gray")

    plt.subplot(2,2,2)
    plt.imshow(np.squeeze(y_test[i]), cmap="gray")

    plt.subplot(2,2,3)
    plt.imshow(np.squeeze(results[i]), cmap="gray")

    plt.savefig("./results_new_model/Test_"+str(i)+".png")
    plt.close()