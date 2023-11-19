
#import yalefaces
from LBPH2 import LBPH2
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
import matplotlib.pyplot as plt

def train_test(images, labels, num=0):
    mp = {}
    for i in range(len(images)):
        image, label = images[i], labels[i]
        mp.setdefault(label, [])
        mp[label].append(image)
    train_x, train_y, test_x, test_y = [], [], [], []
    for person in mp.keys():
        for i in range(len(mp[person])):
            if i < num:
                train_x.append(mp[person][i])
                train_y.append(person)
            else:
                test_x.append(mp[person][i])
                test_y.append(person)
    return train_x, train_y, test_x, test_y
""" 
def yalefaces_test():
    images, labels = yalefaces.yale_data(100)
    # lbph = LBPH(1, 100, images, labels)

    train_images, train_labels, test_images, test_labels = train_test(images, labels)
    lbph = LBPH2(1, 100, train_images, train_labels)

    count = 0
    for i in range(len(test_images)):
        img, lab = test_images[i], test_labels[i]
        heap = lbph.knn(lbph.get_Histogram(img), "EuclideanDistance")
        print(heap)
        print(lab)
        print(heap[0][1] == lab)
        if heap[0][1] == lab: count += 1

    print(count)
"""
def lfw_test(NUM_FEATURES):
    face_dataset = datasets.fetch_lfw_people(min_faces_per_person=50)

    face_dataset.images = [cv2.resize(image, (NUM_FEATURES, NUM_FEATURES)) for image in face_dataset.images]
    kept_images = []
    kept_labels = []
    frequencies = {}

    for image, label in zip(face_dataset.images, face_dataset.target):
        if label not in frequencies:
            frequencies[label] = 0
        if frequencies[label] > 150:
            continue
        kept_images.append(image)
        kept_labels.append(label)
        frequencies[label] += 1
    
    images = np.array([vector.reshape((NUM_FEATURES, NUM_FEATURES)) for vector in kept_images])
    labels = np.asarray(kept_labels)

    labeldict = {i: name for i, name in enumerate(face_dataset.target_names)}
    reversedict = {name:i for i,name in enumerate(face_dataset.target_names)}
    labelnames = []
    for i in range(len(labels)):
        labelnames.append(labeldict[labels[i]])

    labelnames = np.asarray(labelnames)

    indices = np.arange(len(images))
    np.random.seed(18)
    np.random.shuffle(indices)

    prop = .25
    cut = int((1 - prop) * len(images))//1

    X_train =  images[indices[:cut]]
    X_test = images[indices[cut:]]

    y_train = labelnames[indices[:cut]]
    y_actual = labelnames[indices[cut:]]
    lbph = LBPH2(5, 100, X_train, y_train)

    count = 0
    for i in range(len(X_test)):
        img, lab = X_test[i], y_actual[i]
        heap = lbph.knn(lbph.get_Histogram(img), "ChiSquare")
        pred = [n for d, n in heap]
        pred_val = max(set(pred), key=pred.count)
        


        print(heap)
        print(lab, pred_val)
        print(pred_val == lab)
        if pred_val == lab: count += 1


    print(count)
    print(count/len(X_test))

#lfw_test(50)
