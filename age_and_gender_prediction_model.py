import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, model_from_json, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import vgg16
import os
import cv2
from tensorflow.keras import callbacks

#import data
from os import listdir
path = "./data/UTKFace/"
files = os.listdir(path)

#load images
import random
images = []
ages = []
genders = []
for file in random.sample(files, 5000):
    if len(file.split('_'))==4: 
      # encode age
        age = int(file.split('_')[0])
        if age > 80:
            age = '10'
        elif age > 60:
            age = '9'
        elif age > 47:
            age = '8'
        elif age > 37:
            age = '7'
        elif age > 31:
            age = '6'
        elif age > 22:
            age = '5'
        elif age > 16:
            age = '4'
        elif age > 10:
            age = '3'
        elif age > 5:
            age = '2'
        elif age > 2:
            age = '1'
        else:
            age = '0'
        ages.append(age)
        genders.append(int(file.split('_')[1]))
        image = cv2.imread(path+file)
        image = cv2.resize(image,dsize=(64,64))
        images.append(image)

# GENDER MODEL

# train test split
X=np.array(images)
y_gender=np.array(genders)
X_train, X_test, y_train, y_test=train_test_split(X,y_gender,test_size=0.2,random_state=42)
X_train=X_train/255.
X_test=X_test/255.

# build gender model
gender_model=Sequential()
gender_model.add(Conv2D(32,(3, 3),activation='relu', input_shape=(64,64,3)))
gender_model.add(Conv2D(32,(3, 3),activation='relu'))
gender_model.add(MaxPooling2D(2, 2))
gender_model.add(Conv2D(32,(3, 3),activation='relu'))
gender_model.add(Conv2D(32,(3, 3),activation='relu'))
gender_model.add(MaxPooling2D(2, 2))
gender_model.add(Conv2D(64,(3, 3),activation='relu'))
gender_model.add(Conv2D(64,(3, 3),activation='relu'))
gender_model.add(MaxPooling2D(2, 2))
gender_model.add(Dropout(0.25))
gender_model.add(Flatten())

gender_model.add(Dense(128, activation='relu'))
gender_model.add(Dense(64, activation='relu'))
gender_model.add(Dense(32, activation='relu'))
gender_model.add(Dropout(0.5))
gender_model.add(Dense(1, activation='sigmoid'))
gender_model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])

# test on model
early_stop=callbacks.EarlyStopping(monitor='val_loss', patience=3)
hist=gender_model.fit(X_train, y_train, epochs=20, batch_size=20, validation_data=(X_test, y_test),callbacks=[early_stop])

# gender model summary
gender_model.summary()


# AGE MODEL

# train test split
X=np.array(images)
y_age=np.array(ages)
X_train, X_test, y_train, y_test=train_test_split(X,y_age,test_size=0.2,random_state=42)
X_train=X_train/255.
X_test=X_test/255.

# categorical 10 classe
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

# build age model
age_model=Sequential()
age_model.add(Conv2D(32,(3, 3),activation='relu', input_shape=(64,64,3)))
age_model.add(MaxPooling2D(2, 2))
age_model.add(Conv2D(64,(3, 3),activation='relu'))
age_model.add(MaxPooling2D(2, 2))
age_model.add(Conv2D(128,(3, 3),activation='relu'))
age_model.add(MaxPooling2D(2, 2))
age_model.add(Conv2D(256,(3, 3),activation='relu'))
age_model.add(MaxPooling2D(2, 2))
age_model.add(Dropout(0.25))
age_model.add(Flatten())

age_model.add(Dense(128, activation='relu'))
age_model.add(Dense(64, activation='relu'))
age_model.add(Dense(32, activation='relu'))
age_model.add(Dropout(0.5))
age_model.add(Dense(10, activation='softmax'))
age_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss ='categorical_crossentropy', metrics=['accuracy'])

early_stop=callbacks.EarlyStopping(monitor='val_loss', patience=3)
hist=age_model.fit(X_train, y_train, epochs=50, batch_size=20, validation_data=(X_test, y_test),callbacks=[early_stop])

# age model summary
age_model.summary()
