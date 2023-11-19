import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import io

# https://github.com/pavlin-policar/facial-recognition/blob/master/facial_recognition/model.py
class Fisherfaces:
    def __init__(self, images, labels, num_features):
        self.images, self.labels = np.array([img.reshape(num_features, num_features) for img in images]), np.array(labels)
        self.classes = list(set(labels))
        self.num_features = num_features
        self.fisherfaces, self.mean_face, self.class_means, self.basis = self.compute_fisherfaces(self.images, self.labels)

    def normalize(self, matrix, mean_all):
        min_val = np.min(mean_all)
        max_val = np.max(mean_all)
        normalized_matrix = 255 * (matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    def compute_fisherfaces(self, images, labels):
        n = len(images)
        print(images.shape)
        unique_labels = list(set(labels))
        class_means = []
        labels = np.array(labels)

        for i in range(len(unique_labels)):
            label = unique_labels[i]
            matching_indexes = np.where(labels == label)[0]
            class_samples = images[matching_indexes]  # each c
            class_mean = np.mean(class_samples, axis=0)
            class_means.append(class_mean)
            
        class_means = np.array(class_means)
        mean_all = np.mean(class_means, axis=0)

        X = np.array([image for image in images])

        Sw = np.zeros((self.num_features, self.num_features))  # within class
        Sb = np.zeros((self.num_features, self.num_features))  # between-class

        for i in range(len(unique_labels)):
            label = unique_labels[i]
            class_samples = X[labels == label]  # each c

            for class_sample in class_samples:
                Sw += np.dot((class_sample - class_means[i]).T, (class_sample - class_means[i]))

            Sb += len(class_samples) * np.dot((class_means[i] - mean_all), (class_means[i] - mean_all).T)

        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(Sw), (Sb)))
        sort = np.argsort(eigenvalues)[:-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

        class_means = np.dot(class_means, eigenvectors)
        
        return eigenvectors, mean_all, class_means, eigenvalues

    def convert_bin(self, filename):
        with open(filename, 'rb') as file:
            binary_data = file.read()
            image = Image.open(io.BytesIO(binary_data))
            resized_image = image.resize((self.num_features, self.num_features))
            return np.array(resized_image)

    def convert_jpeg(self, filename):
        image = Image.open(filename)
        resized_image = image.resize((self.num_features, self.num_features))

        pixel_data = list(resized_image.getdata())
        greyscale_data = [0 for x in range(self.num_features * self.num_features)]

        for i in range(len(pixel_data)):
            rgb = pixel_data[i]
            greyscale_data[i] = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

        greyscale_data = np.asarray(greyscale_data)
        greyscale_mat = greyscale_data.reshape(self.num_features, self.num_features)

        return greyscale_mat

    def predict_with_fisher(self, ff, image):
        projection = np.dot(image, ff)

        base_filepath = './altered_images'
        class_means = []
        names = []
        for class_name in os.listdir(base_filepath):
            if class_name == 'proj' or class_name == 'class0.jpeg':
                continue
            filepath = base_filepath + '/' + class_name
            names.append(class_name)
            image = ImageOps.grayscale(Image.open(filepath))
            class_means.append(np.resize(np.array(image), (self.num_features, self.num_features)))

        distances = []
        for c in class_means:
            distances.append(np.linalg.norm(projection - np.resize(c, (self.num_features, self.num_features))))
        distances = np.array(distances)

        predicted_label = np.argmin(distances)
        distance = distances[predicted_label]
        smallest_indexes = sorted(range(len(distances)), key=lambda i: distances[i])[:100]

        return names[predicted_label], distance, np.array(names)[np.array(smallest_indexes)], np.array(distances)[
            np.array(smallest_indexes)]

    def predict(self, image):

        projection = np.dot(image, self.fisherfaces)

        distances = []
        for c in self.class_means:
            distances.append(np.linalg.norm(projection - c))
        distances = np.array(distances)

        predicted_label = np.argmin(distances)

        distance = distances[predicted_label]
        return self.classes[predicted_label], distance

    def get_fisherface(self):
        return self.fisherfaces

    
