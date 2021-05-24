
from unicodedata import name
import numpy as np
import joblib
from random import randint
import os
from PIL import Image
from numpy.core.fromnumeric import size
from skimage.color import rgb2gray
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




def create_features(img):
    img = rgb2gray(img)
    color_features = img.flatten()
    flat_features = np.hstack(color_features)

    return flat_features

def get_image(root=''):
    file_path = os.path.join(root)
    img = Image.open(file_path)
    img = img.resize((255, 255))
    
    return np.array(img)

def prediction(path):
    features_list = []
    print(str(path))
    img = get_image(path)
    image_features = create_features(img)
    features_list.append(image_features)

    feature_matrix = np.array(features_list)

    return feature_matrix


def app(path):

    root = "./train/"

    img = prediction(path)

    ss = StandardScaler()
    img_stand = ss.fit_transform(img)

    pca = PCA(n_components=1)
    img_pca = ss.fit_transform(img_stand)

    imd = pd.DataFrame(img_pca)

    modelo = joblib.load('./modeloKNN.sav')


    grupo = modelo.predict(img)

    csv_train = pd.read_csv('./train.csv')

    csv_train['grupo'] = csv_train['group'].map(str)

    img_list = []

    images = csv_train.loc[csv_train['grupo'] == grupo[0], 'name'].values.tolist()

    for i in range(10):
        img_list.append(get_image(root + images[i]))

    fig = plt.figure(figsize=(8, 8))

    rows = 5
    cols = 2

    

    for i in range(len(img_list)):
        try:
            fig.add_subplot(rows, cols, i+1)
            plt.imshow(img_list[i])
            plt.axis('off')
        except:
            pass
    
    plt.show()
