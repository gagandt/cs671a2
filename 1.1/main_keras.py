from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#reshaping
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#Converting to binary class matrix
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#Model
model = Sequential()
#Architecture
model.add(Conv2D(32, kernel_size=7, input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))


#Compiling
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#predicting
model.predict(X_test[:])

y_test[:10]