from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
from keras.losses import categorical_crossentropy, binary_crossentropy
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import confusion_matrix
import itertools


from cus_load import data
dataset = data("output")
x_train, y_leng, y_widt, y_colo, y_angl = dataset.load()

npy_path_images = x_train
# npy_path_labels = '../data/labels.npy'
npy_path_line = y_leng
npy_path_width = y_widt
npy_path_color = y_colo
npy_path_angle = y_angl

# path1 = '/home/baba/GPU_Keras/Assignment/Input_Images'
# path2 = '/home/baba/GPU_Keras/Assignment/Input_Images_Pre'

# os.chdir(path1)

# listing = sorted(os.listdir(path1))

# image_mat = np.array([np.array(mpimg.imread(image)).reshape(-1) for image in listing])
# np.save(npy_path, image_mat)
image_matrix = x_train
# label = np.load(npy_path_labels)
Y_line = y_leng
Y_width = y_widt
Y_color = y_colo
Y_angle = y_angl

# label = np.load(npy_path_labels)
# print(image_matrix)

def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.get_cmap('Blues')):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    ticks_marks = np.arange(len(classes))
    plt.xticks(ticks_marks, classes, rotation=45)
    plt.yticks(ticks_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion_matrix, without normalization')
    print(cm)
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted label")

X, Y_line, Y_width, Y_color, Y_angle = shuffle(image_matrix, Y_line, Y_width, Y_color, Y_angle, random_state=2)
# train_data = [Data, Label]

batch_size = 100
# nb_classes = 96
nb_epochs = 10
img_rows, img_cols = 28, 28
img_channels = 3
nb_filters = 3
nb_conv = 7
nb_pool = 2

# (X, Y) = (train_data[0], train_data[1])
X_train, X_test, Y_train_line, Y_test_line, Y_train_width, Y_test_width, Y_train_color, Y_test_color, Y_train_angle, Y_test_angle= train_test_split(X, Y_line, Y_width, Y_color, Y_angle, test_size=0.2, random_state=4)
# print(Y_train_angle.shape)
# print('***')
# print(Y_test_angle.shape)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

X_train = np.true_divide(X_train, 255)
X_test = np.true_divide(X_test, 255)

Y_train_angle = np_utils.to_categorical(Y_train_angle, 12)
Y_test_angle = np_utils.to_categorical(Y_test_angle, 12)
# print(Y_train_angle)
# print(Y_test_angle.shape)

# Y_test_labels = []
# for y in Y_test:
    # index = list(y).index(1.)
    # Y_test_labels.append(index)
# '''
# BUILDING THE NON-SEQUENTIAL NETWORK

image_input = Input(shape=(img_rows, img_cols, 3), dtype='float64', name='Main_Image_Input')
# image_input = Input(np.expand_dims(X_train[0], axis=0))
first_conv = Convolution2D(nb_filters, nb_conv, nb_conv)(image_input)
first_activation = Activation('relu')(first_conv)
second_conv = Convolution2D(nb_filters, nb_conv, nb_conv)(first_activation)
second_activation = Activation('relu')(second_conv)
first_max_pool = MaxPooling2D(pool_size=(nb_pool, nb_pool))(second_activation)
# first_dropout = Dropout(0.5)(first_max_pool)
feature = Flatten()(first_max_pool)

line_dense_1 = Dense(1024, activation='relu')(feature)
# line_dropout_1 = Dropout(0.5)(line_dense_1)(line_dense_1)
line_dense_2 = Dense(512, activation='relu')(line_dense_1)
# line_dropout_2 = Dropout(0.5)(line_dense_2)(line_dense_2)
line_output = Dense(1, activation='sigmoid', name='line_output')(line_dense_2)

width_dense_1 = Dense(1024, activation='relu')(feature)
# width_dropout_1 = Dropout(0.5)(line_dense_1)(width_dense_1)
width_dense_2 = Dense(512, activation='relu')(width_dense_1)
# width_dropout_2 = Dropout(0.5)(line_dense_2)(width_dense_2)
width_output = Dense(1, activation='sigmoid', name='width_output')(width_dense_2)

color_dense_1 = Dense(1024, activation='relu')(feature)
# color_dropout_1 = Dropout(0.5)(line_dense_1)(color_dense_1)
color_dense_2 = Dense(512, activation='relu')(color_dense_1)
# color_dropout_2 = Dropout(0.5)(line_dense_2)(color_dense_2)
color_output = Dense(1, activation='sigmoid', name='color_output')(color_dense_2)

angle_dense_1 = Dense(1024, activation='relu')(feature)
# angle_dropout_1 = Dropout(0.5)(line_dense_1)(angle_dense_1)
angle_dense_2 = Dense(512, activation='relu')(angle_dense_1)
# angle_dropout_2 = Dropout(0.5)(line_dense_2)(angle_dense_2)
angle_output = Dense(12, activation='sigmoid', name='angle_output')(angle_dense_2)

model = Model(inputs=[image_input], outputs=[line_output, width_output, color_output, angle_output])
model.compile(optimizer='Adam', metrics=['accuracy'], loss={'line_output': 'binary_crossentropy',
            'width_output': 'binary_crossentropy', 'color_output': 'binary_crossentropy',
            'angle_output': 'categorical_crossentropy'})

model.fit([X_train], [Y_train_line, Y_train_width, Y_train_color, Y_train_angle],
        epochs=2, batch_size=32, verbose=1)
# score = model.evaluate(X_test, Y_test, verbose=1)
####################################

'''
'''
# Testing conv layers
for i in range(5):
    test_image_batch = np.expand_dims(X_train[i], axis=0)
    test_image_conv = model.predict(test_image_batch)
    test_img = np.squeeze(test_image_conv, axis=0)
    print(test_img.shape)
    plt.imshow(test_img)
    plt.show()

#####################################'''