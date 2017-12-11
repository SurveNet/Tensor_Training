# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:06:03 2017

@author: aaron

This file loads the trained weights to a CNN and 
tests an input image

"""

from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import Sequential
import numpy as np
import cv2

def VGG_16(weights_path=None):
  classifier = Sequential() 
  classifier.add(Conv2D(32, 3, 3, input_shape = (64,64,3), activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2,2)))
  classifier.add(Conv2D(32, 3, 3, activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2,2)))
  classifier.add(Flatten())
  classifier.add(Dense(output_dim = 128, activation = 'relu'))
  classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
  classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
  
  if weights_path:
      classifier.load_weights(weights_path)
  return classifier

if __name__ == "__main__":
    from keras.preprocessing import image  
    
    #Load trained weights to neural net
    model = VGG_16('trained_data/saved_weights.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #Import image and convert to array
    test_image = image.load_img('test_images/sad.jpg',target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    
    #Make a prediction
    out = model.predict(test_image)
          
    #Predict the result
    result = model.predict(test_image)
    if result[0][0] > 0.5:
        prediction = 'Happy'
    else:
        prediction = 'Sad'
    print("Emotion detected: ", prediction)

    #Print a summary of the model
    print(model.summary())

#==================Write model to JSON===============================================
#     model_json = model.to_json()
#     with open('./trained_data/model.json', 'w') as json_file:
#         json_file.write(model_json) 
#==============================================================================
    
    

