from LBPH import LBPH
import yalefaces
from LBPH2 import LBPH2
import random
from filereader import FileReader
import numpy as np


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

def lfw_test(features, num):
    labels, images = FileReader.readFilesToVectors('./lfw-deepfunneled', features, num, num)
    images = [np.resize(img, (features, features)) for img in images]
    # cross_val = FileReader.getCrossValidationGroups(labels, images)
    # print(cross_val[0])
    
    # train_labels, train_images  = cross_val[0]
    # test_labels, test_images =  cross_val[1]
    train_images, train_labels, test_images, test_labels = train_test(images, labels, num)
    print(train_labels, test_labels)
    lbph = LBPH2(10, 100, train_images, train_labels)

    count = 0
    for i in range(len(test_images)):
        img, lab = test_images[i], test_labels[i]
        # heap = lbph.mean_knn(lbph.get_Histogram(img), "EuclideanDistance")
        # pred_val = heap[0][1]
        heap = lbph.knn(lbph.get_Histogram(img), "EuclideanDistance")
        pred = [n for d, n in heap]
        pred_val = max(set(pred), key=pred.count)
        


        print(heap)
        print(lab, pred_val)
        print(pred_val == lab)
        if pred_val == lab: count += 1


    print(count)
    print(count/len(test_images))

lfw_test(50, 20)

#yalefaces_test()