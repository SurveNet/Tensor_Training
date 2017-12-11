
'''
# Convolutional Neural Network

This model trains a CNN using to conv2D layers, 2 MaxPool layers
and 2 fully connected layers.
Uses image folder names as labels
and trains for 25 epochs

Model is saved to a h5 file and can be loaded for later use in test.py

@Author: Aaron Ward
'''
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 
from keras.models import load_model

#Initialise the CNN
classifier = Sequential() 

# 1 - Build the convolution

# 32 filters or a 3x3 grid
classifier.add(Conv2D(32, 3, 3, input_shape = (64,64,3), activation = 'relu'))

# 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Second layer
classifier.add(Conv2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# 3 - Flattening 
classifier.add(Flatten())

# 4 - Full Connection, making an ANN

classifier.add(Dense(output_dim = 128, activation = 'relu'))

#Binary outcome so sigmoid is being used
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

## Compiling the NN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the neural network for the images
from keras.preprocessing.image import ImageDataGenerator

#Augment the images to improve accuracy
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                  '/input/training_set',
                  target_size=(64, 64),
                  batch_size=32,
                  class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '/input/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


###### EXECUTE THIS TO TRAIN MODEL ##############
classifier.fit_generator( training_set,
                   steps_per_epoch=8000,
                   epochs=25,
                   validation_data=test_set,
                   validation_steps=2000)

classifier.save_weights("/output/out.h5")
classifier.save("/output/saved_model.h5")
print("-----SAVED OUTPUT-----------")

