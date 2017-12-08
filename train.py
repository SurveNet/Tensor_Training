# Convolutional Neural Network

# Written by: Aaron Ward

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
print("-----SAVED OUTPUT-----------")


print('-------LOADED MODEL----------')
from keras.models import load_model

import h5py

classifier.load("trained_data/out.h5")
classifier.load_weights("trained_data/out.h5", by_name=True)
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("Loaded model from disk")
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#--------- New Prediction -------------

# ==============================================================================
import numpy as np
from keras.preprocessing import image

# #Load the image
test_image = image.load_img('input/single_prediction/smile1.jpg',target_size=(64, 64))
#Change to a 3 Dimensional array because it is a colour image
test_image = image.img_to_array(test_image)
# #add a forth dimension
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

# #treshold of 50% to classify the image
if result[0][0] > 0.5:
    prediction = 'Happy'
else:
    prediction = 'Sad'
        
print(result[0][0], "% certainty of being a", prediction)
#==============================================================================
