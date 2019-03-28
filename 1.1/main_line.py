import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from load_data import data
x_train, y_train = data("output")

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
                     input_shape=(3, 28,28)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(96, activation='softmax'))


#Compiling
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#predicting
model.predict(X_test[:])

y_test[:10]


