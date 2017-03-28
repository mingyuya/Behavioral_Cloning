'''
  Author      : Matt, Min-gyu, Kim
  Contact     : mingyuya_at_gmail_dot_com
  Filename    : model.py
  Decription  : Building Training the CNN for Behavioral Cloning
'''

import os
import csv
import cv2
import numpy as np
from math import atan, radians

##############################
# Load Training Data
##############################
samples = []
with open('./record_new/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
samples = shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

###############################
# Parameters
###############################
b_size  = 64    # Batch Size
lr_rate = 0.01  # Learning Rate
EPOCH   = 5     # Epochs


###############################
# Generator
###############################
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0] # [1] : Lest, [2] : Right
                center_img = cv2.imread(name)
                center_ang = float(batch_sample[3])
                images.append(center_img)
                angles.append(center_ang)
                
                # Augmentation : Lateral Flipping
                flipped_center_img = np.fliplr(center_img)
                flipped_center_ang = -center_ang
                images.append(flipped_center_img)
                angles.append(flipped_center_ang)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator      = generator(train_samples,      batch_size=b_size)
validation_generator = generator(validation_samples, batch_size=b_size)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

####################################################################################
# Define the model
#-----------------------------------------------------------------------------------
# 1) Based on Nvidia's Model
# 2) Batch Normalization and Exponenetial Linear Unit for fast and accurate training
# 3) Dropout to avoid that the model is overfitted to driving straight
# 4) No activation at the last layer in this (regression) case
# 4) Optimizer : Adam
# 5) Loss function : MSE
####################################################################################
model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20),(0,0)), input_shape=(160, 320, 3)))
model.add(Conv2D(24, 5, 5, subsample=(2,2)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(36, 5, 5, subsample=(2,2)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(48, 5, 5, subsample=(2,2)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(64, 3, 3))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Conv2D(64, 3, 3))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Dense(1))

adam = Adam(lr = lr_rate)
model.compile(loss='mse', optimizer=adam)

history_object = model.fit_generator(train_generator, 
                                     validation_data   = validation_generator,
                                     samples_per_epoch = len(train_samples)*2,
                                     nb_val_samples    = len(validation_samples)*2, 
                                     nb_epoch          = EPOCH,
                                     verbose           = 1)

model.save('model.h5')

####################################################################################
# The code for plotting the training and validation loss history can be added here
####################################################################################
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])

###############################
# Draw the architecture
###############################
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')

exit()
