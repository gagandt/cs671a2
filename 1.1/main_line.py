import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from load_data import data
dataset = data("output")
x_train, y_train = dataset.load()


print(len(x_train))        
print(len(y_train))  


X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4, random_state=42)
 
print(len(X_train))
print(X_train.shape)


#reshaping
X_train = X_train.reshape(len(X_train),3,28,28)
X_test = X_test.reshape(len(X_test),3,28,28)


#Converting to binary class matrix
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(X_train.shape)

#Model
model = Sequential()
#Architecture
model.add(Conv2D(32, (7, 7), padding='same',
                     input_shape=(3,28,28)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(96, activation='softmax'))


#Compiling
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#training
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

print(history.history.keys())
# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#predicting
#predicting
model.evaluate(X_test, y_test)


