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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Directorios

train = './train/' #Aqui va el directorio para las imagenes de entrenamiento
validation = './test/' #Aqui van las imagenes para el test

csv_train = pd.read_csv('train.csv')
csv_test = pd.read_csv('test.csv')



print(csv_train.head())
print(csv_train['group'].head())

grupos = csv_train['group']
g = list(grupos.drop_duplicates())

#Splitting train to train and validation
df_train, df_val = train_test_split(csv_train, test_size = 0.25, random_state = 0)


imgSize = 256
batchSize = 50

trainGen = ImageDataGenerator(rescale=1. /255)

validGen = ImageDataGenerator(rescale=1. /255)

'''imageTrain = trainGen.flow_from_dataframe(dataframe = df_train, directory = train, x_col= 'filename', y_col= "class", batch_size=batchSize, seed = 42, shuffle = True, class_mode= 'categorical', flip_vertical = True, color_mode = 'rgb', target_size = (imgSize, imgSize))
imageVal = testGen.flow_from_dataframe(dataframe = df_val, directory = validation, x_col = 'filename', y_col = "class", batch_size=1, seed = 42, shuffle = True, class_mode = 'categorical', flip_vertical = True, color_mode = 'rgb', target_size = (imgSize, imgSize))

print("Imagenes convertidas y cargadas")

testX, testY = next(testGen.flow_from_dataframe(dataframe = df_val, directory = train, x_col= 'filename', y_col= "class", target_size = (imgSize, imgSize), batch_size = 50, class_mode = 'categorical'))

x_col= 'name', y_col= g, batch_size = 50, seed = 42, shuffle = True, class_mode= 'categorical', flip_vertical = True, color_mode = 'rgb',

'''

imageTrain = trainGen.flow_from_dataframe(dataframe = df_train, directory = train,  batch_size=50, target_size = (imgSize, imgSize), class_mode='input')
imageVal = validGen.flow_from_dataframe(dataframe = df_val, directory = train, batch_size=50, target_size = (imgSize, imgSize), class_mode='input')


testDataGenerator = ImageDataGenerator(rescale=1. /255)
testGen = testDataGenerator.flow_from_directory(directory = validation, shuffle = True, class_mode = None, color_mode = 'rgb', target_size = (imgSize, imgSize))

testX, testY = next(testDataGenerator.flow_from_dataframe(dataframe = df_val, directory = train, target_size = (imgSize, imgSize), batch_size = 2523), class_mode='input')

print("Tests")

classifier = KNeighborsClassifier(n_neighbors=3).fit(imageTrain, imageVal)

print("modelo creado")

knn_file = open('modelo.pkl', 'wb')

pickle.dump(classifier, knn_file)

print("Accuracy: ", accuracy_score(testY, classifier.predict(testX)))