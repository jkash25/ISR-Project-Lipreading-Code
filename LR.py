import numpy as np
import warnings
from pathlib import Path
import pandas as pd
import os
import imutils
import dlib
import cv2
import imageio
from PIL import Image
from imutils import face_utils
import time
from keras.utils import np_utils, generic_utils
import shutil
from skimage.transform import resize
from sklearn.utils import shuffle
from skimage.io import imread, imsave, imshow
import tensorflow
import keras
from keras import layers
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.layers import Activation, ZeroPadding3D, TimeDistributed, LSTM, GRU, Reshape,BatchNormalization, ConvLSTM2D
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
# from tensorflow.keras.applications.resnet50 import ResNet50
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    words = ['Begin','Choose','Connection','Navigation','Next', 'Previous', 'Start','Stop','Hello','Web']
    words_di = {i:words[i] for i in range(len(words))}
    unseen_test = ['F04']
    unseen_validation = ['F07','M02']
    starting_path = 'C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\dataset\\dataset'
    people = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
    data_types = ['words','phrases']
    folder_nums = ['01','02','03','04','05','06','07','08','09','10']
    instances = ['01','02','03','04','05','06','07','08','09','10']
    image_nums = ['color_001','color_002','color_003','color_004','color_005','color_006','color_007','color008','color_009','color_010']
    def crop_and_save():
        MAX_HEIGHT = 100
        MAX_WIDTH = 100
        max_seq_length = 10
        t1 = time.time()
        hog_face_detector = dlib.get_frontal_face_detector()
        dlib_facelandmark = dlib.shape_predictor('C:\\Users\\Jai K\\CS Stuff\\Python\\ISR Project\\shape_predictor_68_face_landmarks.dat')

        for person in people:
            tx1 = time.time()
            for data_type in data_types:
                for word_index,folder in enumerate(folder_nums):
                    for instance in instances:
                        sequence = []
                        #sequence = np.array(sequence)
                        for image in image_nums:
                            path = starting_path + '\\' + person + '\\' + data_type + '\\' + folder + '\\'+instance + '\\' + image + '.jpg'
                            #print('Hello path is here: ',path)
                            #print('Type is here: ',type(path))

                            if(os.path.exists(path) and path.__contains__('words')):
                                #print('Path here again:',path)
                                #print('Type here again:',type(path))
                                frame = cv2.imread(path)
                                cropped = frame[270:400,200:300]
                                x_arr = []
                                y_arr = []
                                cropped = imutils.resize(cropped, width=500)
                                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                                copy = gray.copy()
                                faces = hog_face_detector(gray)
                                for face in faces:

                                    face_landmarks = dlib_facelandmark(gray, face)

                                    for n in range(48, 68):
                                        x = face_landmarks.part(n).x
                                        y = face_landmarks.part(n).y
                                        x_arr.append(x)
                                        y_arr.append(y)
                                        #print(face_landmarks.part(n))

                                        xmax = max(x_arr)
                                        xmin = min(x_arr)
                                        ymax = max(y_arr)
                                        ymin = min(y_arr)

                                        cv2.circle(gray, (x, y), 1, (0, 255, 255), 1)

                                copy = copy[ymin-5:ymax+20, xmin-25:xmax+25]
                                #print("Gray mouth shape: ", copy.shape)

                                scale_percent = 200
                                width = int(copy.shape[1] * scale_percent / 100)
                                height = int(copy.shape[0] * scale_percent / 75)
                                width2 = 100#int(100*(scale_percent/100))

                                height2 = 100#int(75*(scale_percent/100))
                                dim = (width2, height2)
                                resized_cropped = cv2.resize(copy, dim, interpolation = cv2.INTER_AREA)
                                MAX_WIDTH, MAX_HEIGHT = resized_cropped.shape
                                max_seq_length = 10
                                resized_cropped = resized_cropped*255
                                resized_cropped = resized_cropped.astype(np.uint8)
                                #print("Shape: ",resized_cropped.shape)
                                sequence.append(resized_cropped)
                            else:
                                continue
                        pad_array = [np.zeros((MAX_WIDTH, MAX_HEIGHT))]           
                        sequence.extend(pad_array * (max_seq_length - len(sequence)))
                        sequence = np.array(sequence)
                        if person in unseen_test:
                            x_test.append(sequence)
                            y_test.append(word_index)
                        elif person in unseen_validation:
                            x_val.append(sequence)
                            y_val.append(word_index)
                        else:
                            x_train.append(sequence)
                            y_train.append(word_index)
            tx2 = time.time()
            print(f'Finished reading images for person {person}. Time taken : {tx2 - tx1} secs.')
        t2 = time.time()
        print(f"Time taken to create 3D Tensor from cropped lip images: {t2 - t1} secs")



    crop_and_save()


    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_val = np.array(x_val)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)



    print("X Training shape: ",x_train.shape)
    print("X Test shape: ",x_test.shape)
    print("X Validation shape: ",x_val.shape)
    print("Y Training shape: ",y_train.shape)
    print("Y Test shape: ",y_test.shape)
    print("Y Validation shape: ",y_val.shape)


    def normalize_it(X):
        v_min = X.min(axis=(2, 3), keepdims=True)
        v_max = X.max(axis=(2, 3), keepdims=True)
        #print(v_min)
        #print('Max: ',v_max)
        X = (X - v_min)/(v_max - v_min)
        X = np.nan_to_num(X)
        return X



    def clean(x_train,x_val,x_test,y_train,y_val,y_test):
        print()
        print("Normalizing data...")
        t1 = time.time()
        x_train = normalize_it(x_train)
        #print(X_train)
        x_val = normalize_it(x_val)
        x_test = normalize_it(x_test)
        t2 =time.time()
        print()
        print(f"Time taken to normalize images: {t2 - t1} secs")

        print()
        print("One hot encoding labels...")
        t3 = time.time()
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)
        y_val = np_utils.to_categorical(y_val, 10)
        t4 = time.time()
        print()
        print(f"Time taken to convert labels to categorical: {t4 - t3} secs")

        print()
        print("Shuffling data...")
        t5 = time.time()
        X_train, y_train = shuffle(x_train, y_train, random_state=0)
        X_test, y_test = shuffle(x_test, y_test, random_state=0)
        X_val, y_val = shuffle(x_val, y_val, random_state=0)
        t6 = time.time()
        print()
        print(f"Time taken to shuffle data: {t6 - t5} secs")
        
        print()
        print("Reshaping data...")
        t7 = time.time()
        x_train = np.expand_dims(x_train, axis=4)
        x_val = np.expand_dims(x_val, axis=4)
        x_test = np.expand_dims(x_test, axis=4)
        t8 = time.time()
        print()
        print(f"Time taken to reshape data: {t8 - t7} secs")
        print()

        print(x_train.shape)
        print(x_val.shape)
        print(x_test.shape)
        return x_train, x_val, x_test, y_train, y_val, y_test
        
        # ADD HERE









