# this file will run implementations from other files
#from eigenfaces import Eigenfaces
from fisherfaces import Fisherfaces
import matplotlib.pyplot as plt
import numpy as np
import yalefaces
from filereader import FileReader
NUM_FEATURES = 110
#e = Eigenfaces()
import json
from PIL import Image
bestacc, bestfeat = -1,-1
accuracies=[0]

for NUM_FEATURES in range(130,133,5):
    labels, images = FileReader.readFilesToVectors('/Users/noahloewy/Documents/Facial-Recognition-Comparative-Analysis/lfw-deepfunneled', NUM_FEATURES)
    
    """
    json_filename = "names.json"
    with open(json_filename, 'r') as json_file:
        labels = json.load(json_file)
    images = np.load("imagevectors.npy")
    """l

    images = np.array([vector.reshape((NUM_FEATURES, NUM_FEATURES)) for vector in images])
    f = Fisherfaces(images, labels, NUM_FEATURES)

    accuracy = 0
    prevlabel=None
    for label in set(labels):
        if label == "Venus_Williams":
            pred = f.predict('/Users/noahloewy/Documents/Facial-Recognition-Comparative-Analysis/lfw-deepfunneled/' + label + '/' + label + '_0001.jpg')
            print('pred', label, pred)
            if label == pred[0]: accuracy += 1

    print('accuracy for ' + str(NUM_FEATURES) + " ", accuracy/len(set(labels)))
    accuracies.append(accuracy/len(labels))
x_values = np.arange(len(accuracies)) * 5
"""plt.plot(x_values,accuracies)
plt.xlabel('NUM FEATURES')
plt.ylabel('Accuracy')
plt.title('Features vs Accuracy')
plt.show()
"""
#f.save_face(f.mean_face, 'mean.jpeg')






