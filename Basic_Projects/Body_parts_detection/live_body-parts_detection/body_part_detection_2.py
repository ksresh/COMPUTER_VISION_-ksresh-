import cv2
import os
import random
import numpy as np
import pandas as pd
import tensorflow

####################################################################

DIRECTORY = 'Body_parts_detection'
CATEGORIES = [cat for cat in os.listdir(DIRECTORY) if cat != '.DS_Store']

images = []
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)  # to the folder
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os .path.join(folder, img) # to the image
        image = cv2.imread(img_path)  # reading image
        if image is not None:
            image = cv2.resize(image, (256, 256))
            images.append([image, label])

random.shuffle(images)

x,y = zip(*images)
x = np.array(x)/256
y = np.array(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=45, test_size=0.2)

####################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
model1 =  Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(256,256,3), padding='same'),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2), strides=(2,2)),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    Conv2D(256, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2), strides=(2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2), strides=(3,3)),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(16, activation='relu'),

    Dense(len(CATEGORIES), activation='softmax')])

model1.compile(loss='sparse_categorical_crossentropy',
               optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
               metrics=['accuracy'])

early_stop = tensorflow.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='auto',
    min_delta=0.0001,
    verbose=1,
    patience=5,
    baseline=None)

'''result1 = model1.fit(x_train, y_train,
           batch_size=45,
           validation_data=(x_test,y_test),
           epochs=5,
           callbacks=[early_stop])'''

####################################################################

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.4,
    zoom_range=0.4,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest')

data_generator = datagen.flow(x_train, y_train, shuffle=True)

model2 = tensorflow.keras.models.clone_model(model1)

model2.compile(loss='sparse_categorical_crossentropy',
               optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
               metrics=['accuracy'])

'''result2 = model2.fit(data_generator,
                     epochs=5,
                     validation_data=(x_test,y_test),
                     callbacks=early_stop,
                     batch_size=45)'''

####################################################################

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

vgg16 = VGG16(input_shape=(256,256,3), weights='imagenet', include_top=False)

for layer in vgg16.layers:
    layer.trainable = False

X = Flatten()(vgg16.output)
X1 = Dense(16, activation='relu')(X)
predictions = Dense(len(CATEGORIES), activation='softmax')(X1)

vgg16_model1 = Model(inputs=vgg16.input, outputs=predictions)

vgg16_model1.compile(loss='sparse_categorical_crossentropy',
                    optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=['accuracy'])

'''vgg16_result1 = vgg16_model1.fit(x_train,
                                y_train,
                                epochs=30,
                                validation_data=(x_test,y_test),
                                batch_size=45,
                                callbacks=early_stop)'''

####################################################################

vgg16_model2 = tensorflow.keras.models.clone_model(vgg16_model1)

vgg16_model2.compile(loss='sparse_categorical_crossentropy',
                     optimizer = tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.001),
                     metrics=['accuracy'])

'''vgg16_result2 = vgg16_model2.fit(data_generator,
                                 epochs=30,
                                 validation_data=(x_test,y_test),
                                 batch_size=45,
                                 callbacks=early_stop)'''

####################################################################

from tensorflow.keras.applications import ResNet152

resnet152 = ResNet152(input_shape=(256,256,3), weights='imagenet', include_top=False)

for layer in resnet152.layers:
    layer.trainable=False

X = Flatten()(resnet152.output)
X1 = Dense(16, activation='relu')(X)
predictions = Dense(len(CATEGORIES), activation='softmax')(X1)

resnet152_model1 = Model(inputs=resnet152.input, outputs=predictions)

resnet152_model1.compile(loss='sparse_categorical_crossentropy',
                         optimizer=tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.001),
                         metrics=['accuracy'])

'''resnet152_result1 = resnet152_model1.fit(x_train,
                                         y_train,
                                         epochs=30,
                                         validation_data=(x_test,y_test),
                                         batch_size=45,
                                         callbacks=early_stop)'''

####################################################################

resnet152_model2 = tensorflow.keras.models.clone_model(resnet152_model1)

resnet152_model2.compile(loss='sparse_categorical_crossentropy',
                         optimizer=tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.001),
                         metrics=['accuracy'])

'''resnet152_result2 = resnet152_model2.fit(
    data_generator,
    epochs=30,
    validation_data=(x_test,y_test),
    batch_size=45,
    callbacks=early_stop)'''

####################################################################



####################################################################

print(f'model1           : {model1.evaluate(x_test, y_test, verbose=0)}')
print(f'model2           : {model2.evaluate(x_test, y_test, verbose=0)}')
print(f'vgg16_model1     : {vgg16_model1.evaluate(x_test, y_test, verbose=0)}')
print(f'vgg16_model2     : {vgg16_model2.evaluate(x_test, y_test, verbose=0)}')
print(f'resnet152_model1 : {resnet152_model1.evaluate(x_test, y_test, verbose=0)}')
print(f'resnet152_model2 : {resnet152_model2.evaluate(x_test, y_test, verbose=0)}')

####################################################################

vgg16_model1.save('vgg16_model1.h5')