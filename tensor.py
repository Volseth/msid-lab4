from keras_preprocessing.image import ImageDataGenerator
from tensorflow import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

img_width = 28
img_height = 28
channels = 1
nb_of_generations = 1
epochs = 10


def imageGeneration(prepared_image, number_of_gen):
    # Parameters of newly generated images based on the original ones
    image_gen = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.01,
                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   data_format='channels_last',
                                   )
    images = []
    prepared_image = prepared_image.reshape(1, img_width, img_height, channels)
    iteration = 0
    for generated_image in image_gen.flow(prepared_image, batch_size=1):
        images.append(generated_image)
        iteration += 1
        if iteration >= number_of_gen:
            break
    return images


def generateData(X_train, y_train):
    train_images = []
    train_labels = []
    print("Data generating")
    for x_, y_ in zip(X_train, y_train):
        # Pixel value scaling to 0-1
        x_ = x_ / 255.0
        new_x = imageGeneration(x_, nb_of_generations)
        for new_image in new_x:
            train_images.append(new_image.reshape(img_width, img_height, channels))
            train_labels.append(y_)
    return np.array(train_images), np.array(train_labels)


def predictFashion(X_train, y_train, X_test, y_test):
    (train_labels, test_labels) = y_train, y_test
    print(keras.backend.image_data_format())

    # Prepare data for 3D conv layer
    train_images = np.expand_dims(X_train, axis=3).astype('float32')
    test_images = np.expand_dims(X_test, axis=3).astype('float32')

    # Generate more training images for model
    gen_img, gen_labels = generateData(X_train, y_train)

    # Append new data to original training set
    train_images = np.append(train_images, gen_img, axis=0)
    train_labels = np.append(train_labels, gen_labels, axis=0)

    # Preparing validation set for model selection
    val_images = train_images[0:10000]
    val_labels = y_train[0:10000]

    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=4, activation='relu', input_shape=(img_width, img_height, channels)),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(img_width, img_height, channels)),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10, activation='softmax')

    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    fitting = model.fit(train_images[10000:train_images.shape[0]], train_labels[10000:train_labels.shape[0]],
                        validation_data=(val_images, val_labels),
                        epochs=epochs, verbose=2)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(fitting.history['accuracy'], label='train')
    plt.plot(fitting.history['val_accuracy'], label='valid')
    plt.legend()
    plt.show()
    model.summary()
    print('\nTest accuracy:', test_acc)
