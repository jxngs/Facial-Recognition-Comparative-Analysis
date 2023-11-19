
import numpy as np
from sklearn import datasets
import cv2
from fisherfaces import Fisherfaces
def fisher_runner(NUM_FEATURES):
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
    f = Fisherfaces(X_train, y_train, NUM_FEATURES)

    accuracy = 0
    for x,y in zip(X_test, y_actual):
        
        pred = f.predict(x)
        print('pred', y, pred)
        if y == pred[0]: accuracy += 1
    print('accuracy', accuracy/len(X_test))

fisher_runner(40)
