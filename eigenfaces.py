import numpy as np
import keras
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import cv2

face_dataset = datasets.fetch_lfw_people(min_faces_per_person=50)
IMAGE_SIZE = face_dataset.images[0].shape
# Show the first image in the image dataset
face_dataset.images = [cv2.resize(image, (100, 100)) for image in face_dataset.images]
first_image = face_dataset.images[0]
img = keras.preprocessing.image.array_to_img(np.expand_dims(first_image * 255, axis=-1))

# filter out images if they're the > 150th image for a class -- don't want too many of some classes
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

images = np.array([vector.reshape((100, 100)) for vector in kept_images])
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
y_test = labelnames[indices[cut:]]

n_components = 150

import numpy as np

class CustomPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variances = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        centered_data = X - self.mean
        covariance_matrix = np.cov(centered_data, rowvar=False)
        eigendata = np.linalg.eigh(covariance_matrix)
        eigenvalues = eigendata[0]
        eigenvectors = eigendata[1]
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        self.components = eigenvectors[:, :self.n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variances = eigenvalues[sorted_indices][:self.n_components] / total_variance

    def transform(self, X):
        centered_data = X - self.mean
        projected_data = np.dot(centered_data, self.components)
        whitened_data = projected_data / np.sqrt(self.explained_variances)
        return whitened_data

X_train_reshaped = X_train.reshape((len(X_train), -1))
X_test_reshaped = X_test.reshape((len(X_test), -1))

n_components = 150
custom_pca = CustomPCA(n_components=n_components)
custom_pca.fit(X_train_reshaped)

X_train_pca = custom_pca.transform(X_train_reshaped)
X_test_pca = custom_pca.transform(X_test_reshaped)

classifier = SVC(kernel='linear', C=1)
classifier.fit(X_train_pca, y_train)
# Predict on the test set
y_pred = classifier.predict(X_test_pca)
# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

