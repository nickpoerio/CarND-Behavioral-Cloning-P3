import os
import csv

samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
				# Flipping
                images.append(cv2.flip(center_image,1))
				angles.append(-1.*center_angle)
			
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, Convolution2D, Dropout, Flatten, Dense
from keras.regularizers import l2

def Preprocessing():
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.)-.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((60,20),(0,0))))
    return model

def NVIDIAmodel():
	drop_rate1 = 0.1
	drop_rate2 = 0.33
	reg_rate = .01
    model = Preprocessing()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
	model.add(Dropout(drop_rate1))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
	model.add(Dropout(drop_rate1))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
	model.add(Dropout(drop_rate1))
    model.add(Convolution2D(64,3,3, activation='relu'))
	model.add(Dropout(drop_rate1))
    model.add(Convolution2D(64,3,3, activation='relu'))
	model.add(Dropout(drop_rate1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
	model.add(Dropout(drop_rate2))
    model.add(Dense(50, activation='relu'))
	model.add(Dropout(drop_rate2))
    model.add(Dense(10, activation='relu'),W_regularizer=l2(reg_rate),b_regularizer=l2(reg_rate))
    model.add(Dense(1),W_regularizer=l2(reg_rate),b_regularizer=l2(reg_rate))
return model

model = NVIDIAmodel()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=3)

import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save("model.h5")

"""
If the above code throw exceptions, try 
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
"""
