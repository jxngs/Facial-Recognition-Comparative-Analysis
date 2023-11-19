
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
import matplotlib.pyplot as plt

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
    f = Fisherfaces(X_train, y_train, NUM_FEATURES)

    conf_matrix = np.zeros((len(reversedict), len(reversedict)))

    accuracy = 0
    for x,y in zip(X_test, y_actual):
        
        pred = f.predict(x)
        print('pred', y, pred)
        conf_matrix[reversedict[y], reversedict[pred[0]]] += 1

        if y == pred[0]: accuracy += 1
    print('accuracy', accuracy/len(X_test))
    display_labels = [labeldict[i] for i in range(len(reversedict))]
    display = ConfusionMatrixDisplay(conf_matrix, display_labels=display_labels)
    display.plot(xticks_rotation="vertical")
    plt.tight_layout()
    plt.show()


fisher_runner(40)
