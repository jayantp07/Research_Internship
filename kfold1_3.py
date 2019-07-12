from __future__ import print_function
import numpy as np
import os
import sys
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils import shuffle
import keras
from keras.models import Model
import tensorflow as tf
from time import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPool3D
from tensorflow.keras.callbacks import TensorBoard
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

def tta_prediction(datagen, model, image, n_examples):
    samples=np.expand_dims(image, 0)
    it=datagen.flow(samples, batch_size=n_examples)
    yhats=model.predict_generator(it, steps=n_examples, verbose=0)
    summed=np.sum(yhats, axis=0)
    probs=np.mean(yhats, axis=0)
    return probs[1], np.argmax(summed)

plt.figure(0).clf()

batch_size=32
epochs=20
np.set_printoptions(threshold=sys.maxsize)
seed=7
np.random.seed(seed)

save_dir = os.path.join(os.getcwd(), 'my_models1')
model_name = 'lymph_nodes_model_3_6x_end.h5'

train_path="./DATA_FINAL/3D/train/"
test_path="./DATA_FINAL/3D/test/"

ids=os.listdir("./MED")
ids.sort()
ids=np.array(ids)

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []
cohens_kappa_list = []
ROC_AUC_list = []
true_negative = []
false_negative = []
false_positive = []
true_positive = []
cnt=0

for train, test in kfold.split(ids):
    cnt=cnt+1
    train_ids=ids[train]
    test_ids=ids[test]
    training_data=np.load(train_path+train_ids[0]+".npy", allow_pickle=True)
    X=np.array([i[0] for i in training_data]).reshape(-1,32,32,32,1)
    Y=np.array([i[1] for i in training_data])

    training_data=np.load(test_path+test_ids[0]+".npy", allow_pickle=True)
    x_test=np.array([i[0] for i in training_data]).reshape(-1,32,32,32,1)
    y_test=np.array([i[1] for i in training_data])


    for i in range(1, len(train_ids)):
        training_data=np.load(train_path+train_ids[i]+".npy", allow_pickle=True)
        X_temp=np.array([i[0] for i in training_data]).reshape(-1,32,32,32,1)
        Y_temp=np.array([i[1] for i in training_data])
        X=np.concatenate((X, X_temp))
        Y=np.concatenate((Y, Y_temp))

    for i in range(1, len(test_ids)):
        training_data=np.load(test_path+test_ids[i]+".npy", allow_pickle=True)
        x_test_temp=np.array([i[0] for i in training_data]).reshape(-1,32,32,32,1)
        y_test_temp=np.array([i[1] for i in training_data])
        x_test=np.concatenate((x_test, x_test_temp))
        y_test=np.concatenate((y_test, y_test_temp))

    X, Y=shuffle(X, Y, random_state=seed)
    x_train, x_val, y_train, y_val=train_test_split(X, Y, test_size=0.1, random_state=seed)

    print()
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)
    print()

    ## input layer
    input_layer = Input((32, 32, 32, 1))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=32, kernel_size=(3, 3, 3))(input_layer)
    conv_layer1 = BatchNormalization()(conv_layer1)
    conv_layer1 = Activation('relu')(conv_layer1)
    conv_layer2 = Conv3D(filters=32, kernel_size=(3, 3, 3))(conv_layer1)
    conv_layer2 = BatchNormalization()(conv_layer2)
    conv_layer2 = Activation('relu')(conv_layer2)

    ## add max pooling to obtain the most imformatic features
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)
    pooling_layer1 = Dropout(0.3)(pooling_layer1)

    conv_layer3 = Conv3D(filters=64, kernel_size=(3, 3, 3))(pooling_layer1)
    conv_layer3 = BatchNormalization()(conv_layer3)
    conv_layer3 = Activation('relu')(conv_layer3)
    conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3))(conv_layer3)
    conv_layer4 = BatchNormalization()(conv_layer4)
    conv_layer4 = Activation('relu')(conv_layer4)


    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)
    pooling_layer2 = Dropout(0.3)(pooling_layer2)

    ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
    flatten_layer = Flatten()(pooling_layer2)

    ## add dropouts to avoid overfitting / perform regularization
    dense_layer2 = Dense(units=512, activation='relu', name="SVM")(flatten_layer)
    dense_layer2 = Dropout(0.5)(dense_layer2)
    output_layer = Dense(units=2, activation='softmax')(dense_layer2)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    '''data_gen_args=dict(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode='constant', cval=-1)
    image_datagen=ImageDataGenerator(**data_gen_args)
    image_datagen.fit(x_train, seed=seed)
    image_generator=image_datagen.flow(x_train, y_train, seed=seed, batch_size=32)
    model.fit_generator(image_generator, epochs=130, validation_data=(x_val, y_val), workers=4, steps_per_epoch=(10*(x_train.shape[0]))//32)'''

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), shuffle=False)

    '''if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)'''

    print()
    print("#######################################################")
    print()

    '''datagen=ImageDataGenerator(**data_gen_args)
    n_examples_per_image=10
    yhats=[]
    probs=[]
    for i in range(x_test.shape[0]):
        prob, yhat=tta_prediction(datagen, model, x_test[i], n_examples_per_image)
        yhats.append(yhat)
        probs.append(prob)'''
    
    yhat_probs=model.predict(x_test, verbose=1)
    yhats=yhat_probs.argmax(axis=-1)
    probs=yhat_probs[:, 1]
    testy=y_test[:, 1]

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(testy, yhats)
    print('Accuracy: %f' % accuracy)

    # precision tp / (tp + fp)
    precision = precision_score(testy, yhats)
    print('Precision: %f' % precision)

    # recall: tp / (tp + fn)
    recall = recall_score(testy, yhats)
    print('Recall: %f' % recall)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(testy, yhats)
    print('F1 score: %f' % f1)
 
    # kappa
    kappa = cohen_kappa_score(testy, yhats)
    print('Cohens kappa: %f' % kappa)

    # ROC AUC
    auc = roc_auc_score(testy, probs)
    print('ROC AUC: %f' % auc)

    # confusion matrix
    matrix = confusion_matrix(testy, yhats)
    print(matrix)

    fpr, tpr, thresh = roc_curve(testy, probs)
    plt.plot(fpr, tpr, label="fold "+str(cnt)+", auc="+str(auc), linestyle='dashed')

    true_negative.append(matrix[0][0])
    false_negative.append(matrix[1][0])
    false_positive.append(matrix[0][1])
    true_positive.append(matrix[1][1])
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1)
    cohens_kappa_list.append(kappa)
    ROC_AUC_list.append(auc)
    accuracy_list.append(accuracy * 100)

    print()
    print("#######################################################")
    print()

    '''svm_training=[]
    svm_testing=[]
    intermediate_layer_model=Model(inputs=model.input, outputs=model.get_layer("SVM").output)

    intermediate_output1=intermediate_layer_model.predict(x_train)
    for i in range(intermediate_output1.shape[0]):
        svm_training.append([np.array(intermediate_output1[i]), np.array(y_train[i])])

    intermediate_output1=intermediate_layer_model.predict(x_test)
    for i in range(intermediate_output1.shape[0]):
        svm_testing.append([np.array(intermediate_output1[i]), np.array(y_test[i])])

    np.save("./SVM1/3D/training_"+str(cnt)+".npy", svm_training)
    np.save("./SVM1/3D/testing_"+str(cnt)+".npy", svm_testing)'''

plt.legend(loc=0)
plt.savefig("./ROC_CURVES/3_small_noTTA")
plt.close()

print("%.2f%% (+/- %.2f%%)" % (np.mean(accuracy_list), np.std(accuracy_list)))
print("%.2f (+/- %.2f)" % (np.mean(precision_list), np.std(precision_list)))
print("%.2f (+/- %.2f)" % (np.mean(recall_list), np.std(recall_list)))
print("%.2f (+/- %.2f)" % (np.mean(f1_score_list), np.std(f1_score_list)))
print("%.2f (+/- %.2f)" % (np.mean(cohens_kappa_list), np.std(cohens_kappa_list)))
print("%.2f (+/- %.2f)" % (np.mean(ROC_AUC_list), np.std(ROC_AUC_list)))
print("%.2f (+/- %.2f)" % (np.mean(true_negative), np.std(true_negative)))
print("%.2f (+/- %.2f)" % (np.mean(false_negative), np.std(false_negative)))
print("%.2f (+/- %.2f)" % (np.mean(false_positive), np.std(false_positive)))
print("%.2f (+/- %.2f)" % (np.mean(true_positive), np.std(true_positive)))
