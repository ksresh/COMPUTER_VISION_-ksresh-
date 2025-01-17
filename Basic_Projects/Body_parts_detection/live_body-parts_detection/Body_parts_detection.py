import cv2
import numpy as np
import os
import random

DIRECTORY = 'Body_parts_detection'

CATEGORIES = os.listdir('Body_parts_detection')
CATEGORIES = [cat for cat in CATEGORIES if cat != '.DS_Store']

data = []
for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)

    label = CATEGORIES.index(category)

    for image in os.listdir(folder):
        image_path = os.path.join(folder, image)
        image_array = cv2.imread(image_path)
        if image_array is not None:
            image_array = cv2.resize(image_array, (256, 256))

            data.append([image_array, label])


random.shuffle(data)

x,y = zip(*data)
x = np.array(x)/255
y = np.array(y)

####################################################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=45, test_size=0.2,)

'''print(f'x_train : {x_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'x_test : {x_test.shape}')
print(f'y_test : {y_test.shape}')'''

####################################################
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=40,
                              zoom_range = 0.4,
                              shear_range=0.4,
                              width_shift_range=0.4,
                              height_shift_range=0.4,
                              horizontal_flip=True,
                              fill_mode='nearest')

data_generator = datagen.flow(x_train, y_train, shuffle=True)

##############################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model1 = Sequential([
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(256,256,3)),
    Conv2D(64, (3,3),activation='relu', padding='same'),
    MaxPooling2D((2,2), strides=(2,2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    Flatten(),

    Dense(64, activation='relu'),
    Dense(len(CATEGORIES), activation='softmax')])


model1.compile(loss = 'sparse_categorical_crossentropy',
               optimizer  = tensorflow.keras.optimizers.Adam(learning_rate=0.001),
               metrics = ['accuracy'])

early_stop = tensorflow.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    mode='auto',
    verbose=1,
    patience=20,
    baseline=None)

# result1 = model1.fit(x_train, y_train, epochs=10, batch_size=45, validation_data=(x_test,y_test), callbacks=[early_stop])

##################################################################

model2 = tensorflow.keras.models.clone_model((model1))

model2.compile(loss='sparse_categorical_crossentropy',
               optimizer=tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.001),
               metrics=['accuracy'])

# result2 = model2.fit(data_generator, epochs=100, validation_data=(x_test, y_test), batch_size=45, callbacks=[early_stop])


##################################################################
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

vgg16 = VGG16(input_shape=(256, 256, 3), weights='imagenet', include_top=False)

for layer in vgg16.layers:
    layer.trainable = False


x = Flatten()(vgg16.output)
predictions = Dense(len(CATEGORIES), activation='softmax')(x)

model_vgg16 = Model(inputs=vgg16.input , outputs= predictions)

model_vgg16.compile(loss='sparse_categorical_crossentropy',
                    optimizer = tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                    metrics = ['accuracy'])

'''result_vgg16 = model_vgg16.fit(x_train, y_train,
                               epochs=100, 
                               validation_data=(x_test,y_test), 
                               batch_size=42, 
                               callbacks=[early_stop])'''

#################################################################



model_vgg16_aug = tensorflow.keras.models.clone_model((model_vgg16))

model_vgg16_aug.compile(loss='sparse_categorical_crossentropy',
                        optimizer=tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.001),
                        metrics=['accuracy'])

'''result_vgg16_aug = model_vgg16_aug.fit(data_generator, 
                                       epochs=100, 
                                       validation_data=(x_test, y_test), 
                                       batch_size=42, 
                                       callbacks=[early_stop])'''

################################################################

from tensorflow.keras.applications import ResNet152

resnet152 = ResNet152(input_shape=(256,256,3), weights='imagenet', include_top=False)

for layer in resnet152.layers:
    layer.trainable=False

x = Flatten()(resnet152.output)
x1 = Dense(16, activation='relu')(x)
predictions = Dense(len(CATEGORIES), activation='softmax')(x1)

model_resnet152 = Model(inputs=resnet152.input, outputs=predictions)

model_resnet152.compile(loss='sparse_categorical_crossentropy',
                        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
                        metrics=['accuracy'])

'''result_resnet152 = model_resnet152.fit(x_train, y_train,
                                       epochs=100,
                                       validation_data=(x_test,y_test),
                                       batch_size=42,
                                       callbacks=[early_stop])'''

##################################################################

model_resnet152_aug = tensorflow.keras.models.clone_model((model_resnet152))

model_resnet152_aug.compile(loss='sparse_categorical_crossentropy',
                            optimizer=tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.001),
                            metrics=['accuracy'])

'''result_resnet152_aug = model_resnet152_aug.fit(data_generator, epochs=100,
                                               validation_data=(x_test,y_test),
                                               batch_size=42,
                                               callbacks=[early_stop])'''

#######################################################################

print(f'model1              : {model1.evaluate(x_test,y_test, verbose=0)}')
print(f'model1_aug          : {model2.evaluate(x_test,y_test, verbose=0)}')
print(f'model_vgg16         : {model_vgg16.evaluate(x_test,y_test, verbose=0)}')
print(f'model_vgg16_aug     : {model_vgg16_aug.evaluate(x_test,y_test, verbose=0)}')
print(f'model_resnet152     : {model_resnet152.evaluate(x_test,y_test, verbose=0)}')
print(f'model_resnet152_aug : {model_resnet152_aug.evaluate(x_test,y_test, verbose=0)}')



#######################################################################

import pickle
pickle.dump(model_vgg16, open('model_vgg16.pkl', 'wb'))

#######################################################################

model_vgg16.save('model_vgg16.h5')
