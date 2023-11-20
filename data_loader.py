import io
import random
import cv2
import numpy as np
from PIL import Image
import os
import keras
from sklearn.model_selection import train_test_split
from sklearn import datasets

"""
VARIABLES INVOLVED WITH DATA LOADING

Eigenfaces, Fisherfaces, and LBPH:
- self.images
- self.labelnames
- self.indices

CNN:
- self.images
- self.labelnames
- self.labels_to_numbers
- self.numbers_to_labels
"""

class DatasetLoader:       
    def split_data(self, prop):
        np.random.seed(18)
        np.random.shuffle(self.indices)
        cut = int((1 - prop) * len(self.images))//1
        X_train =  self.images[self.indices[:cut]]
        X_test = self.images[self.indices[cut:]]
        y_train = self.labelnames[self.indices[:cut]]
        y_actual = self.labelnames[self.indices[cut:]]
        
        return X_train, X_test, y_train, y_actual
    
class TA_Data(DatasetLoader):

    def load_data(self, modeltype, num_features, prop, minimages = None, maximages = None):
        self.images = []
        self.labelnames = []
        for filename in os.listdir('./ta_data'):
            # read in file from its filepath
            image = Image.open('./ta_data/' + filename).resize((num_features,num_features))
            # store name of person in file
            name = filename[0:filename.index('_')]
            # convert RGB pixel data into a list
            pixels = list(image.getdata())
            greyscale = np.zeros(num_features*num_features)
            for i in range(len(pixels)):
                rgb = pixels[i]
                # use weighted grayscale function to optimize lighting differences among color hues
                greyscale[i] = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2] 
            # reshape greyscale vector back into matrix form       
            greyscale = greyscale.reshape((num_features,num_features)) 

            # add greyscale image and label name for image to the 2 respective lists          
            self.images.append(greyscale)
            self.labelnames.append(name)
        
        # turn them into numpy arrays for proper indexing
        self.images = np.array(self.images)
        self.labelnames = np.array(self.labelnames)
        
        # CNN Data Handling
        if modeltype=="CNN":
            self.labels_to_numbers = {}
            i = 0
            for label in self.labelnames:
                if label in self.labels_to_numbers.keys():
                    continue
                self.labels_to_numbers[label] = i
                i += 1

            self.numbers_to_labels = {value:key for key, value in self.labels_to_numbers.items()}

            images = np.expand_dims(np.asarray(self.images), axis=-1) # necessary to show there is 1 channel (grayscale)?
            labels = keras.utils.to_categorical(np.asarray([self.labels_to_numbers[label] for label in self.labelnames]))
            self.X_train, self.X_test, self.y_train, self.y_actual = train_test_split(images, labels, test_size=prop, random_state=18)
        # Eigenfaces, Fisherfaces, LBPH Data Handling
        else:
            self.indices = np.arange(len(self.images))
            self.X_train, self.X_test, self.y_train, self.y_actual = self.split_data(prop)

    def split_data(self, prop=.25):
        return super().split_data(prop)
    
class LFW(DatasetLoader):

    def load_data(self, modeltype, num_features, prop, minimages = 50, maximages = 150):
        face_dataset = datasets.fetch_lfw_people(min_faces_per_person=minimages)
        self.target_names = face_dataset.target_names
        face_dataset.images = [cv2.resize(image, (num_features, num_features)) for image in face_dataset.images]
        
        self.images = []
        self.labelnames = []
        frequencies = {}

        for image, label in zip(face_dataset.images, face_dataset.target):
            if label not in frequencies:
                frequencies[label] = 0
            if frequencies[label] > maximages:
                continue
            self.images.append(image)
            self.labelnames.append(label)
            frequencies[label] += 1

        self.images = np.array(self.images)
        self.labelnames = np.asarray(self.labelnames)
        
        if modeltype=="CNN":
            self.labels_to_numbers = {}
            i = 0
            for label in self.labelnames:
                if label in self.labels_to_numbers.keys():
                    continue
                self.labels_to_numbers[label] = i
                i += 1

            self.numbers_to_labels = {value:key for key, value in self.labels_to_numbers.items()}

            images = np.expand_dims(np.asarray(self.images), axis=-1) # necessary to show there is 1 channel (grayscale)?
            labels = keras.utils.to_categorical(np.asarray([self.labels_to_numbers[label] for label in self.labelnames]))
            self.X_train, self.X_test, self.y_train, self.y_actual = train_test_split(images, labels, test_size=prop, random_state=18)
       
        else:
            self.indices = np.arange(len(self.images))
            self.X_train, self.X_test, self.y_train, self.y_actual = self.split_data(prop)


    def split_data(self, prop=.2):
        return super().split_data(prop)
    
class Yale_Data(DatasetLoader):
    def load_data(self, modeltype, num_features, prop, minimages = None, maximages = None):
        images = []
        labels = []
        for filename in os.listdir('yalefaces_binary'):
            if filename == 'data' or filename == 'Readme.txt': continue
            with open('yalefaces_binary/' + filename, 'rb') as file:
            # Read the content of the binary file
                binary_data = file.read()
                image = Image.open(io.BytesIO(binary_data))
                resized_image = image.resize((num_features,num_features))
                images.append(np.array(resized_image))
                labels.append(filename[:filename.find('.')])
        
        self.images = np.array(images)
        self.labelnames = np.array(labels)

        if modeltype=="CNN":
            self.labels_to_numbers = {}
            i = 0
            for label in self.labelnames:
                if label in self.labels_to_numbers.keys():
                    continue
                self.labels_to_numbers[label] = i
                i += 1

            self.numbers_to_labels = {value:key for key, value in self.labels_to_numbers.items()}
            images = np.expand_dims(np.asarray(self.images), axis=-1) # necessary to show there is 1 channel (grayscale)?
            labels = keras.utils.to_categorical(np.asarray([self.labels_to_numbers[label] for label in self.labelnames]))
            self.X_train, self.X_test, self.y_train, self.y_actual = train_test_split(images, labels, test_size=prop, random_state=18)
        else:
            self.indices = np.arange(len(self.images))
            self.X_train, self.X_test, self.y_train, self.y_actual = self.split_data(prop)
    
    def split_data(self, prop=.2):
        return super().split_data(prop)
    
class BFW_Probabilistic(DatasetLoader):
    def read_image(self, path, width=80, height=80):
        with open(path, 'rb') as file:
            binary_data = file.read()
            image = Image.open(io.BytesIO(binary_data))
            resized_image = image.resize((width, height))
            grayscaled_image = resized_image.convert('L')
            return np.array(grayscaled_image)

    def load_data(self, modeltype, num_features, prop, minimages=None, maximages=None, probabilities=None):
        names = []
        images = []
        demographics = []
        random.seed(10)
        categories = [folder for folder in os.scandir('bfw') if folder.name != '.DS_Store']
        if probabilities is None:
            probabilities = {}

        for category in categories:
            people = [folder for folder in os.scandir(category.path) if folder.name != '.DS_Store']
            demographic = category.name
            demographic_probability = probabilities[demographic] if demographic in probabilities else 1
            for person in people[:1]:
                name = person.name
                person_images = [file for file in os.scandir(person.path) if file.name != '.DS_Store']
                if random.random() > demographic_probability:
                    continue
                for image in person_images:
                    images.append(self.read_image(image, num_features, num_features))
                    names.append(name)
                    demographics.append(demographic)
        
        self.images = np.array(images)
        self.labelnames = np.array(names)
        self.indices = np.arange(len(self.images))
        self.labels_to_numbers = {name: i for i, name in enumerate(set(names))}
        self.numbers_to_labels = {i: name for name, i in self.labels_to_numbers.items()}
        
        if modeltype=="CNN":
            images = np.expand_dims(np.asarray(self.images), axis=-1) # necessary to show there is 1 channel (grayscale)?
            labels = keras.utils.to_categorical(np.asarray([self.labels_to_numbers[label] for label in self.labelnames]))
            self.X_train, self.X_test, self.y_train, self.y_actual = train_test_split(images, labels, test_size=prop, random_state=18)
       
        else:
            self.X_train, self.X_test, self.y_train, self.y_actual = self.split_data(prop)

    def split_data(self, prop=.2):
        return super().split_data(prop)
    
b = BFW_Probabilistic()
b.load_data("CNN",40,0.1)
print(b.labelnames)
print()
print(b.y_train)
print()
print(b.y_actual)