from nis import match
import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import io

#https://github.com/pavlin-policar/facial-recognition/blob/master/facial_recognition/model.py

class Fisherfaces:

    def __init__(self, images, labels, num_features):
            self.images, self.labels = images, labels
            self.classes = list(set(labels))
            self.num_features = num_features
            self.fisherfaces, self.mean_face, self.class_means, self.basis = self.compute_fisherfaces(self.images, self.labels)
            

    def normalize(self, matrix, mean_all):
        
    
        # return normalized image
        min_val = np.min(mean_all)
        max_val = np.max(mean_all)
    
        normalized_matrix = 255 * (matrix - min_val) / (max_val - min_val)

        return normalized_matrix
        
    def compute_fisherfaces(self, images, labels):
        # class means
        unique_labels = list(set(labels))
        class_means = []
        labels = np.array(labels)
        for i in range(len(unique_labels)):
            label = unique_labels[i]
            matching_indexes = np.where(labels == label)[0]
            class_samples = images[matching_indexes] # each c
            class_mean = np.mean(class_samples, axis=0)
            class_means.append(class_mean)
            self.save_face(class_mean, 'class' + label + '.jpeg')

        class_means = np.array(class_means)
        mean_all = np.mean(class_means, axis=0)
        # class means and mean_all are correct

        # first, normalize all images
        X = np.array([self.normalize(image, mean_all) for image in images])
        
        Sw = np.zeros((self.num_features, self.num_features)) # within class
        Sb = np.zeros((self.num_features, self.num_features)) # between-class

        # Compute class means and scatter matrices
        # c is each label
        for i in range(len(unique_labels)):
            label = unique_labels[i]
            class_samples = X[labels == label] # each c
            
            # Within-class scatter matrix
            for class_sample in class_samples:
                Sw += np.dot((class_sample - class_means[i]).T, (class_sample - class_means[i]))

            # Between-class scatter matrix
            Sb += len(X) * np.dot((class_means[i] - mean_all), (class_means[i] - mean_all).T)

        # create projection, W, that maximizes class separability criterion
        
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(Sw),(Sb)))
        sort = np.argsort(eigenvalues)[:-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

        return eigenvectors, mean_all, np.dot(class_means, eigenvectors), eigenvalues

    def convert_bin(self, filename):
        with open(filename, 'rb') as file:
        # Read the content of the binary file
            binary_data = file.read()
            image = Image.open(io.BytesIO(binary_data))
            resized_image =  image.resize((self.num_features,self.num_features))
            return np.array(resized_image)
    
    def convert_jpeg(self, filename):
            image = Image.open(filename)
            resized_image = image.resize((self.num_features,self.num_features))

            # extract the pixel data from the image and convert to greyscale
            pixel_data = list(resized_image.getdata())
            greyscale_data = [0 for x in range(self.num_features*self.num_features)]
            for i in range(len(pixel_data)):
                # using the weighted grayscale method, which weights colors according to their wavelengths
                rgb = pixel_data[i]
                greyscale_data[i] = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]

            # save the images for our viewing pleasure
            greyscale_data = np.asarray(greyscale_data)
            greyscale_mat = greyscale_data.reshape(self.num_features, self.num_features)

            return greyscale_mat
        
    def predict(self, filename):
   
        image = self.convert_jpeg(filename)
    
        projection = np.dot(image, self.fisherfaces)
        
        self.save_face(self.fisherfaces, 'proj/' + 'fisher' + '.jpeg')
        self.save_face(image, 'proj/' + 'img' + '.jpeg')
        self.save_face(projection, 'proj/' + 'proj' + '.jpeg')

        distances = []
        for c in self.class_means:
            distances.append(np.linalg.norm(projection - c))
        distances = np.array(distances)
       
        predicted_label = np.argmin(distances)

        distance = distances[predicted_label]
        return self.classes[predicted_label], distance        

    def get_fisherface(self):
        return self.fisherfaces

    def save_face(self, matrix, filename):
         plt.imsave('altered_images/'+filename, matrix)
    

   














