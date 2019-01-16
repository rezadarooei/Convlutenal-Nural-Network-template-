# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:20:30 2019

@author: Reza Darooei

"""
#bofore start dowmnload thano tensorflow and keras
#part  1-

#import important packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#intializing CNN
classifier=Sequential()

#Step 1-Convulution
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu"))


#step 2-Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))



#step3 - flattening
classifier.add(Flatten())

#step 4 - full conection
classifier.add(Dense(output_dim=128,activation='relu'))
#output layer
classifier.add(Dense(output_dim=1,activation="sigmoid"))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Image augmantion to Enrich dataset
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set= train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
