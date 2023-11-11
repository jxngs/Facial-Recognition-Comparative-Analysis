import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_WIDTH = 10
IMAGE_HEIGHT = 10

class Fisherfaces:

    # Loading Images from the Images Folder
    def loadimages(self):
        # for each file in the images directory...
        for filename in os.listdir('images'):
            # open the image, resize it, and convert it to a one-dimensional array
            image = Image.open('images/'+filename)
            resized_image = image.resize((IMAGE_HEIGHT,IMAGE_WIDTH))

            # extract the pixel data from the image and convert to greyscale
            pixel_data = list(resized_image.getdata())
            greyscale_data = [0 for x in range(IMAGE_WIDTH*IMAGE_HEIGHT)]
            for i in range(len(pixel_data)):
                # using the weighted grayscale method, which weights colors according to their wavelengths
                rgb = pixel_data[i]
                greyscale_data[i] = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
                # update average image pixel by pixel as we go, used later for computations
                self.average_image[i] = (self.average_image[i]*self.num_images + greyscale_data[i])/(self.num_images+1)

            # save the images for our viewing pleasure
            greyscale_data = np.asarray(greyscale_data)
            greyscale_mat = greyscale_data.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
            plt.gray()
            plt.imsave('altered_images/'+filename, greyscale_mat)

            # update class instance variables for later
            self.num_images = self.num_images + 1
            self.images.append(greyscale_mat)
            self.labels.append(filename)

    def normalize(self, matrix):

        # return normalized image
        min_val = np.min(matrix)
        max_val = np.max(matrix)
    
        normalized_matrix = 255 * (matrix - min_val) / (max_val - min_val)

        return normalized_matrix
        
    def compute_fisherfaces(self, images, labels):
        # first, normalize all images
        # X = normalize()
        X = np.array([self.normalize(image) for image in images])

        # reduce dimensionality potentially
        # num samples is how many images
        # num features should be 250 x 250
        num_samples, num_features, width = X.shape

        # X is a random sample from c classes
            # how to determine classes???
        
        mean_all = np.mean(X, axis=0) # take mean

        # Initialize within-class scatter matrix and between-class scatter matrix
        Sw = np.zeros((num_features, num_features)) # within class
        Sb = np.zeros((num_features, num_features)) # between-class

        unique_labels = set(labels)

        class_means = [0 for i in labels]
        
        # Compute class means and scatter matrices
        # c is each label
        for label in unique_labels:
            class_samples = X[np.array(labels) == label] # each c
            class_mean = np.mean(class_samples, axis=0) # calculate ui
            class_means[np.array(labels) == label] = class_mean
            # Within-class scatter matrix

            for class_sample in class_samples:
                Sw += np.dot((class_sample - class_mean).T, (class_sample - class_mean))

                # Between-class scatter matrix
            Sb += len(class_samples) * np.dot((class_mean - mean_all), (class_mean - mean_all).T)

        # create projection, W, that maximizes class separability criterion
        # Solve the generalized eigenvalue problem
        # print(Sw, Sb, Sw.shape, Sb.shape)
        # print(np.linalg.det(Sw), np.linalg.det(Sb))

        # temp fix, test on more values
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues.real)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top eigenfaces
        # ?? potential experiments here
        # if num_components is not None:
        #     eigenvectors = eigenvectors[:, :num_components]

        # how to compute class means???
        return eigenvectors.real.T, mean_all, class_means  # Return the top eigenfaces as rows of the matrix

    def predict(self, image):
        diff = image - self.mean_face
        projection = np.dot(diff, self.fisherfaces)

        distances = np.linalg.norm(projection - self.class_means, axis=1)
        predicted_label = np.argmin(distances)
        confidence = 1 / (1 + distances[predicted_label])

        return predicted_label, confidence        

    

    def get_fisherface(self):
        return self.fisherfaces

    def __init__(self):
        self.num_images = 0
        self.average_image = [0 for x in range(IMAGE_WIDTH*IMAGE_HEIGHT)]
        self.images = []
        self.labels = []
        self.loadimages()
        self.fisherfaces, self.meanface, self.class_means = self.compute_fisherfaces(self.images, self.labels)

   
