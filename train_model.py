from math import ceil
import cv2
import csv
import sklearn
import numpy as np

lines = []
with open('driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples , validation_samples = train_test_split(lines,test_size=0.2)

def generator(samples,batch_size=32):
    num_samples = len(samples)
    while 1:
        #shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            previous = 0
            alpha = .5
            beta = 1 - alpha
            filtered = 0
            for line in batch_samples:
                for source_path in line[0:3]:
                    filtered = float(line[3])*alpha + previous*beta 
                    file_name = source_path.split('/')[-1]
                    current_path = 'IMG/' + file_name
                    image = cv2.imread(current_path)
                    image_flipped = cv2.flip(image,1)
                    images.append(image)
                    if file_name.split('_')[0] == 'center':
                        measurements.append(filtered)
                        images.append(image_flipped)
                        measurements.append(-1*filtered)
                    elif file_name.split('_')[0] == 'left':
                        measurements.append(filtered+.3)
                        images.append(image_flipped)
                        measurements.append(-1*filtered+.3)
                    else:
                        measurements.append(filtered-.3)
                        images.append(image_flipped)
                        measurements.append(-1*(filtered)-.3)
                    previous = filtered

            X_train = np.array(images)
            Y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, Y_train)

#print("the size of the images are: ")
#print(len(images))
#X_train = np.array(images)
#Y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, AveragePooling2D,Cropping2D,Dropout
import tensorflow as tf

batch_size =32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
tf.random.set_seed(0)


model = Sequential()
model.add(Lambda(lambda x: x/255,input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(units=84, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator,steps_per_epoch=ceil(len(train_samples)/batch_size),validation_data=validation_generator,validation_steps=ceil(len(validation_samples)/batch_size),epochs=2)
model.save('model_1.h5')
model.save('model_1',save_format=('tf'))


