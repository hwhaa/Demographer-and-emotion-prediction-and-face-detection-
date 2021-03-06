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
            age = 9 #81-116
        elif age > 64:
            age = 8 #65-80
        elif age > 54:
            age = 7 #55-64
        elif age > 44:
            age = 6 #45-54
        elif age > 34:
            age = 5 #35-44
        elif age > 24:
            age = 4 #25-34
        elif age > 17:
            age = 3 #18-24
        elif age > 11:
            age = 2 #12-17
        elif age > 2:
            age = 1 #3-11
        else:
            age = 0 #0-2
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

# Save JSON and h5
model_structure = gender_model.to_json()
f = Path("cnn_gender_rgb.json")
f.write_text(model_structure)
gender_model.save_weights("cnn_gender_rgb.h5")

model_structure = age_model.to_json()
f = Path("cnn_age10_rgb.json")
f.write_text(model_structure)
age_model.save_weights("cnn_age10_rgb.h5")
