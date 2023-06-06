from path import Path
data_dir = 'train'
# Importing TensorFlow Module and Functions Within It
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout, MaxPooling2D, Conv2D
from tensorflow.keras.models import Sequential
# Importing PLT For Graphing
import matplotlib.pyplot as plt
# Importing Module That Will Conduct Mathematical Operations On Arrays
import numpy as np
# Importing Pillow - Image Processor
from PIL import Image
import PIL
# Importing OS - Used To Navigate Through File Structures
import os
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2




# Storing All The Data Into A Variable - Resized Images To 256x256
data = tf.keras.utils.image_dataset_from_directory('train')
tf.data.AUTOTUNE
# The 5 Different Classes In The Car Class
carModel = ['Happiness', 'Neutral', 'Sadness', 'Anger', 'Surprise', 'Disgust', 'Fear']

# Normalzing The Data
data = data.map(lambda x, y: (x / 255, y))  # Performing Pixel Transformation To Range Of [0, 1] To Pipeline
scaled_iterator = data.as_numpy_iterator()  # Grabbing The Next Batch
batch = scaled_iterator.next()
batch[0].max()  # Testing If Max Value Is 1 - Which It Should Be If Images Are Transformed Correctly

fig, ax = plt.subplots(ncols=4, figsize=(8, 8))  # 4 Creating A Figure With 4 Figures With 8x8 Size
for idx, img in enumerate(batch[0][:4]):  # Getting Last 4 Images From Batch 0
    ax[idx].imshow(img)  # Showing The Images
    ax[idx].title.set_text(batch[1][idx])  # Stating Which Class The Image Is From

# Finding The Size For Each Dataset 
data_size = len(data)
train_size = int(len(data) * 0.7)  # Around 70% Of Data For Train
val_size = int(len(data) * 0.2) + 1  # Around 20% Of Data For Validation
test_size = int(len(data) * 0.1) + 1  # Around 10% Of Data For Test
train_size = 7
val_size = 2
print(data_size, train_size, val_size, test_size)

# Allocating Certain Amount Of Batches For Each Dataset Based On Above Calc.
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# model = Sequential([
#     Conv2D(128, (3, 3), activation='relu', input_shape=(256, 256, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(32, (3, 3), activation='relu'),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(7, activation='softmax')
# ])
def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)


input_shape = (256, 256, 3)
num_classes = 7
model = mini_XCEPTION(input_shape, num_classes)
# Choosing An Optimizer And Loss Function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

# Compiling The Model While Making The Metrics To Be Accuracy
model.compile(optimizer, loss_object, metrics='accuracy')

# Creating A File To Save Training Logs
logdir = 'logs'
# Logging information about the training process
tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# Trains the model for 20 epochs using the training set, and validates it using the validation set
hist = model.fit(train, epochs=1, validation_data=val, callbacks=tensorboard_callbacks)

model.save(os.path.join('models', 'emotionModel.h5'))

