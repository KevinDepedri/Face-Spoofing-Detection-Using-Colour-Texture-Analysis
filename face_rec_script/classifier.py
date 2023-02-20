import numpy as np
from PIL import Image
import os
import cv2

"""
In this code we will train the classifier so the system is able to recognize the person
"""


def train_classifier(dataset):
    """
    This method trains the classifier to recognize the face
    :param dataset: directory where all the images are saved
    :return: creates a separate file -> classifier.yml
    """
    # save the path of all the images contained in the dataset directory
    images_path = [os.path.join(dataset, f) for f in os.listdir(dataset)]
    faces = []
    ids = []

    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists
    for image in images_path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Create the classifier using LBPH, train it and save the result in a yml file
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.yml")

train_classifier("dataset")