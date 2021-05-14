from sys import path
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import datetime
import math
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
from keras.metrics import mean_absolute_error
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense,Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import Sequential
from tensorflow.keras.applications.xception import Xception
import pickle
import fastai
from fastai.vision import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

#Directorios

train = './train/' #Aqui va el directorio para las imagenes de entrenamiento
#validation = './test/' #Aqui van las imagenes para el test

csv_train = pd.read_csv('train.csv')
csv_test = pd.read_csv('test.csv')

csv_train['grupo'] = csv_train['group'].map(str)

print(csv_train.head())
print(csv_train.describe())

#Splitting train to train and validation
df_train, df_val = train_test_split(csv_train, train_size=0.5, test_size = 0.25, random_state = 0)


imgSize = 255
batchSize = 50

trainGen = ImageDataGenerator(rescale=1./255, 
        shear_range=0.2, 
        horizontal_flip=True)

validGen = ImageDataGenerator(rescale=1./255, 
        shear_range=0.2, 
        horizontal_flip=True)

imageTrain = trainGen.flow_from_dataframe(dataframe=df_train, directory=train, target_size=(imgSize, imgSize), x_col= 'name', y_col= 'grupo', batch_size = batchSize, seed = 42, shuffle = True, class_mode= 'categorical', flip_vertical = True, color_mode = 'rgb')
#imageVal = validGen.flow_from_dataframe(dataframe=df_val, directory=train, target_size=(imgSize, imgSize), x_col= 'name', y_col= 'grupo', batch_size =batchSize, seed = 42, shuffle = True, class_mode= 'categorical', flip_vertical = True, color_mode = 'rgb')


#testDataGenerator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
#testGen = testDataGenerator.flow_from_directory(directory = validation, shuffle = True, class_mode = None, color_mode = 'rgb', target_size = (imgSize, imgSize))

testX, testY = next(validGen.flow_from_dataframe(dataframe=df_val, directory=train, target_size=(imgSize, imgSize), x_col= 'name', y_col= 'grupo', batch_size = batchSize, seed = 42, shuffle = True, class_mode= 'categorical', flip_vertical = True, color_mode = 'rgb'))

print("Creando el modelo")

classifier = NearestNeighbors(n_neighbors=10)
classifier.fit(imageTrain)

print("modelo creado")

knn_file = open('modelo.pkl', 'wb')

pickle.dump(classifier, knn_file)

print("Accuracy: ", accuracy_score(testY, classifier.predict(testX)))