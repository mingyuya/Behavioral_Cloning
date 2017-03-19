import os
import csv
import cv2
import numpy as np

samples = []
with open('./record/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #name = './IMG/'+batch_sample[0].split('/')[-1]
                name = batch_sample[0] # [1] : Lest, [2] : Right
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image - 70 rows from the top, 20 rows from the bottom
            X_train = np.array(images)
            #X_train = np.array(images[20:89,:,:])
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

model = Sequential()

# Nvidia's Model with Batch Normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20),(0,0)), input_shape=(160, 320, 3)))
model.add(BatchNormalization())
model.add(Conv2D(24, 5, 5, subsample=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(36, 5, 5, subsample=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(48, 5, 5, subsample=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3))
model.add(BatchNormalization())
model.add(Conv2D(64, 3, 3))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Dense(1))

adam = Adam(lr=0.01)
model.compile(loss='mse', optimizer=adam)

history_object = model.fit_generator(train_generator, 
                                     validation_data = validation_generator,
                                     samples_per_epoch = len(train_samples),
                                     nb_val_samples = len(validation_samples), 
                                     nb_epoch=8,
                                     verbose=1)

model.save('model.h5')

exit()
