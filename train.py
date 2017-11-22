import numpy as np 
import os
import cv2
import tensorflow as tf 
import tflearn 
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tensorflow.python.framework import ops

# supress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

TRAINING_DIR = 'data/training_circles'
TESTING_DIR = 'data/testing_circles'
IMAGE_SIZE = 50
LR = 0.001
MODEL_NAME = 'Image-Classifier'

def create_label(image_name):
    word_label = image_name.split('.')[-2] #One hot encoder
    if word_label == 'circle':
        return np.array([1, 0])
    elif word_label == 'line':
        return np.array([0, 1])

# getting images and resizing
def create_training_data():
    training_data = []
    for img in tqdm(os.listdir(TRAINING_DIR)):
        path = os.path.join(TRAINING_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMAGE_SIZE, IMAGE_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('training_data.npy', training_data)
    return training_data

def create_testing_data():
    testing_data = []
    for img in tqdm(os.listdir(TESTING_DIR)):
        path = os.path.join(TESTING_DIR, img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMAGE_SIZE, IMAGE_SIZE))
        testing_data.append([np.array(img_data), img_num])
    shuffle(training_data)
    np.save('testing_data.npy', testing_data)
    return training_data 

training_data = create_training_data()
testing_data = create_testing_data()

# training_data = np.load('training_data.npy')
# testing_data = np.load('training_data.npy')


train = training_data[:-200]
test = training_data[-100:]

X_train = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE,IMAGE_SIZE, 1 )
Y_train = [i[1] for i in train ]

X_test = np.array([i[0] for i in test]).reshape(-1, IMAGE_SIZE,IMAGE_SIZE, 1 )
Y_test = [i[1] for i in test ]


ops.reset_default_graph()

# BUILD MODEL
convnet = input_data(shape=[None, 32, 32, 3], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# convnet = conv_2d(convnet, 128, 2, activation='relu')
# convnet = max_pool_2d(convnet, 2)

# convnet = conv_2d(convnet, 64, 2, activation='relu')
# convnet = max_pool_2d(convnet, 2)

# convnet = conv_2d(convnet, 32, 2, activation='relu')
# convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer = 'adam', learning_rate = LR,
                 loss = 'categorical_crossentropy', name = 'target')

model = tflearn.DNN(convnet, tensorboard_dir="log", tensorboard_verbose=0)
model.fit({'input' : X_train}, {'target' : Y_train},
          n_epoch=10,
          validation_set=({'input' : X_test}, {'target' : Y_test}),
          snapshot_step=200, show_metric=True, run_id = MODEL_NAME)

fig = plg.figure(figsize=(16, 12))

for num, data in enumerate(testing_data[:16]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(4, 4, num+1)
    orig = img_data
    data = img_data.reshape=(IMAGE_SIZE, IMAGE_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) > .5:
        str_label = 'circle'
    else:
        str_label = 'line'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_variable(False)
    y.axes.get_yaxis().set_variable(False)

plt.show()