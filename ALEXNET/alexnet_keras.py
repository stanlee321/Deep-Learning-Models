import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import numpy as np

np.random.seed(1000)


class AlexNet:
    @staticmethod
    def build(input_shape, classes):
        #Instantiate an empty model
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding="valid"))
        model.add(Activation("relu"))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding="valid"))
        model.add(Activation("relu"))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
        model.add(Activation("relu"))

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
        model.add(Activation("relu"))

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid"))
        model.add(Activation("relu"))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation("relu"))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation("relu"))
        # Add Dropout
        model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        model.add(Dense(1000))
        model.add(Activation("relu"))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
"""

# network and training
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT=0.2
IMG_ROWS, IMG_COLS = 28, 28 # input image dimensions
NB_CLASSES = 10 # number of outputs = number of digits
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)
# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
k.set_image_dim_ordering("th")
# consider them as float and normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# we need a 60K x [1 x 28 x 28] shape as input to the CONVNET
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)
"""
# initialize the optimizer and model
INPUT_SHAPE = (227,227,3)
NB_CLASSES = 1000
model = AlexNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
print(model.summary())
"""
model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
metrics=["accuracy"])
history = model.fit(X_train, y_train,
batch_size=BATCH_SIZE, epochs=NB_EPOCH,
verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=["accuracy"]) 

"""

