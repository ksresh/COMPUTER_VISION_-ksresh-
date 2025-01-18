import cv2
import numpy as np
import os
import random
import tensorflow

##########################################################
DIRECTORIES = r'digits'
CATEGORIES = os.listdir(DIRECTORIES)
CATEGORIES = [cat for cat in CATEGORIES if cat != '.DS_Store']

data = []
for category in CATEGORIES:
    folder = os.path.join(DIRECTORIES, category)

    label = CATEGORIES.index(category)

    for image in os.listdir(folder):
        img_path = os.path.join(folder, image)
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (256,256))

        data.append([img_array, label])

random.shuffle(data)

##########################################################
x, y = zip(*data)

x = np.array(x)
x = x/256
y = np.array(y)

print(f'x : {x.shape}')
print(f'y : {y.shape}')
print(f'________________________________________________')

##########################################################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f'x_train : {x_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'x_test : {x_test.shape}')
print(f'y_test : {y_test.shape}')
print(f'________________________________________________')


##########################################################
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

data_generator = datagen.flow(x_train, y_train, batch_size=42)

img_aug, lab_aug = [],[]
for i in range(len(x_train)):
    img = x_train[i]
    img = np.expand_dims(img, axis=0)
    lab = y_train[i]

    image_generator1 = datagen.flow(img, batch_size=1)
    for j in range(2):
        image_d = next(image_generator1)[0]

        img_aug.append(image_d)
        lab_aug.append(lab)

img_aug = np.array(img_aug)
lab_aug = np.array(lab_aug)

Images = np.concatenate([x_train, img_aug])
Labels = np.concatenate([y_train, lab_aug])

print(f'img_aug : {img_aug.shape}')
print(f'lab_aug : {lab_aug.shape}')
print(f'Images : {Images.shape}')
print(f'Labels : {Labels.shape}')
print(f'________________________________________________')

##########################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model1 = Sequential([
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(256,256,3)),
    MaxPooling2D((2,2), strides=(2,2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    Flatten(),

    Dense(16, activation='relu'),
    Dense(len(CATEGORIES), activation='softmax')])


model1.compile(loss='sparse_categorical_crossentropy',
               optimizer=tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.001),
               metrics=['accuracy'])
print(f'________________________________________________')

##########################################################
early_stop = tensorflow.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    mode='auto',
    verbose=1,
    patience=20,
    baseline=None)

'''result1 = model1.fit(Images,
                     Labels,
                     epochs=30, 
                     batch_size=16,
                     validation_data=(x_test,y_test),
                     callbacks=[early_stop])'''
print(f'________________________________________________')

##########################################################

from tensorflow.keras.models import Model

from tensorflow.keras.applications import VGG16
vgg16 = VGG16(input_shape=(256,256,3), weights='imagenet', include_top=False)

for layer in vgg16.layers:
    layer.trainable=False

x = Flatten()(vgg16.output)
x1 = Dense(16, activation='relu')(x)
predictions = Dense(len(CATEGORIES), activation='softmax')(x1)

model_vgg16 = Model(inputs=vgg16.input, outputs=predictions)

model_vgg16.compile(loss='sparse_categorical_crossentropy',
               optimizer=tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.001),
               metrics=['accuracy'])

result_vgg16 = model_vgg16.fit(
    x_train,
    y_train,
    epochs=30,
    validation_data=(x_test,y_test),
    batch_size=42,
    callbacks=[early_stop])
print(f'________________________________________________')


##########################################################

print(f'model1_da_ge : {model_vgg16.evaluate(x_test,y_test, verbose=0)}')

###########################################################

model_vgg16.save('model_vgg16.h5')



