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

print(X_test.shape)


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
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2)

print(history.history.keys())
# summarize history for accuracy
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
model.evaluate(X_test, y_test)


