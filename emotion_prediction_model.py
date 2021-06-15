import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Flatten,Dense,Dropout
from pathlib import Path
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam


train_dir = "train" #passing the path with training images
test_dir = "test"   #passing the path with testing images
img_size = 48 #original size of the image

# Apply ImageData Generator
train_datagen = ImageDataGenerator(rotation_range = 22,
                                         width_shift_range = 0.1,
                                         height_shift_range = 0.1,
                                         horizontal_flip = True,
                                         rescale = 1./255,
                                         #zoom_range = 0.2,
                                         validation_split = 0.2
                                        )
validation_datagen = ImageDataGenerator(rescale = 1./255,
                                         validation_split = 0.2)

train_generator = train_datagen.flow_from_directory(directory = train_dir,
                                                    target_size = (img_size,img_size),
                                                    batch_size = 64,
                                                    color_mode = "grayscale",
                                                    class_mode = "categorical",
                                                    subset = "training"
                                                   )
validation_generator = validation_datagen.flow_from_directory( directory = test_dir,
                                                              target_size = (img_size,img_size),
                                                              batch_size = 64,
                                                              color_mode = "grayscale",
                                                              class_mode = "categorical",
                                                              subset = "validation"
                                                             )

# build emotion model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer = Adam(lr=0.0001),loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 15
batch_size = 64
history = model.fit(x = train_generator,epochs = epochs,validation_data = validation_generator)

model.summary()

# Save JSON and h5
model_structure = model.to_json()
f = Path("EMO_MODEL_STRUCTURE.json")
f.write_text(model_structure)
model.save_weights("EMO_MODEL_WEIGHTS.h5")
