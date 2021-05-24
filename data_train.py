
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display

import pandas as pd
import numpy as np

from PIL import Image
import joblib
from skimage.feature import hog
from skimage.color import rgb2gray
from numba import jit, cuda
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier


train = './train' #Aqui va el directorio para las imagenes de entrenamiento


csv_train = pd.read_csv('./train.csv')

csv_train['grupo'] = csv_train['group'].map(str)

print(csv_train.head())
print(csv_train.describe())
print(len(csv_train))

#Funciones de preprocesamiento 
#RGB a escala de grises

def get_image(row_id, root=train):
    name = csv_train.iloc[row_id]['name']
    filename = "{}".format(name)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    img = img.resize((255, 255))
    
    return np.array(img)


labels = csv_train.iloc[:, 0]

row = labels.index[0]
print(row)

plt.imshow(get_image(row))
plt.show()

#Grayscale

imagen = get_image(1)
gray_img = rgb2gray(imagen)

plt.imshow(gray_img, cmap=mpl.cm.gray)

print(gray_img.shape)

#Histograma de grises

hog_features, hog_image = hog(gray_img,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=(16, 16))

plt.imshow(hog_image, cmap=mpl.cm.gray)

def create_features(img):
    img = rgb2gray(img)
    color_features = img.flatten()
    flat_features = np.hstack(color_features)

    return flat_features

img_features = create_features(imagen)

print(img_features)

#Iterando labels de imagenes para su futuro procesamiento

def create_feature_matrix(label_indexes):
    features_list = []
    
    for img_id in tqdm(range(len(label_indexes))):
        img = get_image(label_indexes[img_id])
        image_features = create_features(img)
        features_list.append(image_features)

    feature_matrix = np.array(features_list)

    return feature_matrix


#Agarrando las imagenes random
from random import randint
int_list = list()
y_l = list()
for i in range(0, 4000):
    rnd = randint(0, len(csv_train))
    if rnd in int_list:
        rnd = randint(0, len(csv_train))
    else:
        int_list.append(rnd)
        y_l.append(csv_train.iloc[rnd]['grupo'])
print(int_list[0])
comp = len(int_list)
y = pd.Series(y_l)
feature_matrix = create_feature_matrix(int_list)
        
    
print(feature_matrix.shape)


ss = StandardScaler()
img_stand = ss.fit_transform(feature_matrix)

pca = PCA(n_components=comp)
img_pca = ss.fit_transform(img_stand)
print('PCA matrix shape is: ', img_pca.shape)


#Separando test/train
X = pd.DataFrame(img_pca)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234123)

print(pd.Series(y_train).value_counts())

print(type(X_test))
print(X_test.shape)
input()

#KNN model
knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10))
knn.fit(X_train,y_train)
print('KNN Accuracy: %.3f' % accuracy_score(y_test, knn.predict(X_test)))


#SVM model
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('SVM Accuracy is: ', accuracy)

r = randint(0, len(int_list))

img = create_feature_matrix([int_list[r]])
#print(int_list[r])
#print(img.shape)


#print(svm.predict(img)[0])
#print(knn.predict(img)[0])



print("Guardando modelo")
filename = 'modeloKNN.sav'

joblib.dump(knn, filename)
joblib.dump(svm, filename) 