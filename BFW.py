from os import listdir
import os
import numpy as np
from PIL import Image
import io
import random
import keras
from sklearn.model_selection import train_test_split

def read_image(path, width=80, height=80):
    with open(path, 'rb') as file:
        binary_data = file.read()
        image = Image.open(io.BytesIO(binary_data))
        resized_image = image.resize((width, height))
        grayscaled_image = resized_image.convert('L')
        return np.array(grayscaled_image)
    
class DatasetLoader:

    #CREATED IN LOAD DATA : self.images, self.indices, self.labelnames, self.labeldict,self.reversedict, self.labels, self.target_names
    #CREATED IN SPLIT DATA: self.X_train, self.X_test, self.y_train, self.y_actual

    def load_data(self):
        # Implement the logic to load data from a specific dataset
        pass
        
    def split_data(self, prop):
        np.random.seed(18)
        np.random.shuffle(self.indices)

        cut = int((1 - prop) * len(self.images))//1

        X_train =  self.images[self.indices[:cut]]
        X_test = self.images[self.indices[cut:]]

        y_train = self.labelnames[self.indices[:cut]]
        y_actual = self.labelnames[self.indices[cut:]]
        
        return X_train, X_test, y_train, y_actual

class BFW_Probabilistic(DatasetLoader):
    def load_data(self, modeltype, num_features=100, prop=0.2, minimages=None, maximages=None, probabilities=None):
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
                    images.append(read_image(image, num_features, num_features))
                    names.append(name)
                    demographics.append(demographic)
        
        unique_labels = list(set(names))
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
        
        return self
        
        
class BFW_Balanced(BFW_Probabilistic):
    def load_data(self, modeltype, num_features=100, prop=0.2, minimages=None, maximages=None):
        super().load_data(modeltype, num_features, prop, maximages, minimages, {})
        
        # self.images = s.images
        # self.labelnames = s.labelnames
        # self.indices = s.indices
        # self.labels_to_numbers = s.labels_to_numbers
        # self.numbers_to_labels = s.numbers_to_labels
        # self.X_train, self.X_test, self.y_train, self.y_actual = s.X_train, s.X_test, s.y_train, s.y_actual
    def split_data(self, prop=0.25):
        return super().split_data(prop)