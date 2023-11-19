from LBPH import LBPH
import yalefaces
from LBPH2 import LBPH2
import random

def train_test(images, labels):
    mp = {}
    for i in range(len(images)):
        image, label = images[i], labels[i]
        mp.setdefault(label, [])
        mp[label].append(image)
    train_x, train_y, test_x, test_y = [], [], [], []
    for person in mp.keys():
        for i in range(len(mp[person])):
            if i < 10:
                train_x.append(mp[person][i])
                train_y.append(person)
            else:
                test_x.append(mp[person][i])
                test_y.append(person)
    return train_x, train_y, test_x, test_y


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


