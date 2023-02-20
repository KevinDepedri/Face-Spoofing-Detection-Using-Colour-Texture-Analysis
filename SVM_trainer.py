import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import full_histogram



def compute_histogram_on_dataset():
    """
    Take all the images we have in the dataset, compute the full histogram on all of them and save the
    result in a file called data1.pickle. Once we've done it, we don't need to run this part of the program again because
    the result can be directly recovered by the saved pickle file.
    :return: pickle file containing the dataset
    """

    dir = 'C:\\Users\\matte\\OneDrive\\Desktop\\SIV_Project\\images dataset'

    categories = ['True', 'Fake']
    dataset = []

    for category in categories:
        path = os.path.join(dir, category)
        label = categories.index(category) # I'll get the indexes of the categories -> 0 for true and 1 for fake

        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            print(imgpath)
            face_image = cv2.imread(imgpath, 0)
            try:
                face_image = cv2.resize(face_image, (30, 30))
                print('Computing histogram on {}'.format(imgpath))
                image = full_histogram.final_function(imgpath,,
                dataset.append([image, label])
                print([image, label])
            except Exception as e:
                pass

    pick_in = open('data1.pickle', 'wb')
    pickle.dump(dataset, pick_in)
    pick_in.close()

#open the dataset from the pickle file
dataset = []
pick_in = open('data1.pickle', 'rb')
dataset = pickle.load(pick_in)
pick_in.close()

#create an array composed by all the input samples and one with all the corresponding labels
input_sample = []
labels = []

for sample, label in dataset:
    input_sample.append(sample)
    labels.append(label)

#train the model with a linear SVC

#If I want to test my dataset I can split it into two different sets: a training part and a testing part
#X_train, X_test, y_train, y_test = train_test_split(input_sample, labels, train_size=0.8)

model = LinearSVC(C=100.0, random_state=42, dual=False)
#model.fit(X_train, y_train) -> I use this whenever I split my dataset in two parts
model.fit(input_sample, labels)

#save the trained model in order to use it without training it every single time
pick = open('model0,8.sav', 'wb')
pickle.dump(model, pick)
pick.close()

"""
#If I want to test the accuracy of the model I can use the following functions
prediction = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print('Accuracy:{}'.format(accuracy))
"""