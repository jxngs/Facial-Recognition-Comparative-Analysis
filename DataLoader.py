
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
import cv2
import keras
from tensorflow.keras import layers
import random
import os
import io
from PIL import Image
from sklearn.model_selection import train_test_split

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
    def load_data(self, modeltype, num_features=100, prop=0.2, probabilities=None):
        names = []
        images = []
        demographics = []
        random.seed(18)
        categories = [folder for folder in os.scandir('bfw') if folder.name != '.DS_Store']
        if probabilities is None:
            probabilities = {}

        max_proportion = max(probabilities.values()) if probabilities else 100
        max_proportion_number = 20
        if max_proportion_number != 100:
            print(f"Warning: Leaving some people out of dataset (only using {max_proportion_number / 100 * 100}% for highest group)")

        for category in categories:
            people = [folder for folder in os.scandir(category.path) if folder.name != '.DS_Store']
            demographic = category.name
            demographic_probability = (probabilities[demographic] if demographic in probabilities else max_proportion)/max_proportion
            demographic_number = round(max_proportion_number * demographic_probability)
            print(f"Demographic {demographic} has {demographic_number} people selected")
            selected_people = random.sample(people, demographic_number)
            for person in selected_people:
                name = person.name
                person_images = [file for file in os.scandir(person.path) if file.name != '.DS_Store']
                for image in person_images:
                    images.append(self.read_image(image, num_features, num_features))
                    names.append(name)
                    demographics.append(demographic)
        
        unique_labels = list(set(names))
        self.images = np.array(images)
        self.labelnames = np.array(names)
        self.indices = np.arange(len(self.images))
        self.labels_to_numbers = {name: i for i, name in enumerate(set(names))}
        self.numbers_to_labels = {i: name for name, i in self.labels_to_numbers.items()}

        self.labels_to_demographics = {names[i]: demo for i, demo in enumerate(demographics)}
        if modeltype=="CNN":
            images = np.expand_dims(np.asarray(self.images), axis=-1) # necessary to show there is 1 channel (grayscale)?
            labels = keras.utils.to_categorical(np.asarray([self.labels_to_numbers[label] for label in self.labelnames]))
            self.X_train, self.X_test, self.y_train, self.y_actual = train_test_split(images, labels, test_size=prop, random_state=18)
       
        else:
            self.X_train, self.X_test, self.y_train, self.y_actual = self.split_data(prop)
        
        return self
    
    def read_image(self, path, width=80, height=80):
        with open(path, 'rb') as file:
            binary_data = file.read()
            image = Image.open(io.BytesIO(binary_data))
            resized_image = image.resize((width, height))
            grayscaled_image = resized_image.convert('L')
            return np.array(grayscaled_image)
        
class BFW_Balanced(BFW_Probabilistic):
    def load_data(self, modeltype, num_features=100, prop=0.2, probabilities=None):
        super().load_data(modeltype, num_features, prop, {})
        
        # self.images = s.images
        # self.labelnames = s.labelnames
        # self.indices = s.indices
        # self.labels_to_numbers = s.labels_to_numbers
        # self.numbers_to_labels = s.numbers_to_labels
        # self.X_train, self.X_test, self.y_train, self.y_actual = s.X_train, s.X_test, s.y_train, s.y_actual
    def split_data(self, prop=0.25):
        return super().split_data(prop)
class LFW(DatasetLoader):

        def load_data(self, modeltype, num_features=100, prop = .25, probabilities=None):
        
            face_dataset = datasets.fetch_lfw_people(min_faces_per_person=50)
            self.target_names = face_dataset.target_names
            if modeltype=='Fisher':
                num_features = min(num_features, min(face_dataset.images[0].shape))
            if modeltype!='CNN':
                face_dataset.images = [cv2.resize(image, (num_features, num_features)) for image in face_dataset.images]
            
            self.num_features = num_features
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

            if modeltype!='CNN':
                kept_images = np.array([vector.reshape((num_features, num_features)) for vector in kept_images])
            self.labels = np.asarray(kept_labels)

            self.numbers_to_labels = {i: name for i, name in enumerate(face_dataset.target_names)}
            self.labels_to_numbers = {name: i for i, name in enumerate(face_dataset.target_names)}
            
            labelnames = []
            for i in range(len(self.labels)):
                labelnames.append(self.numbers_to_labels[self.labels[i]])
            self.labelnames = np.asarray(labelnames)

            self.indices = np.arange(len(kept_images))
            self.images=np.asarray(kept_images)
            
            if modeltype=="CNN":
                
                self.X_train, self.X_test, self.y_train, self.y_actual = self.split_data()
                self.X_train = np.expand_dims(np.asarray(self.X_train), axis=-1) # necessary to show there is 1 channel (grayscale)?
                self.X_test = np.expand_dims(np.asarray(self.X_test), axis=-1) # necessary to show there is 1 channel (grayscale)?
                self.y_train = [self.labels_to_numbers[item] for item in self.y_train]
                self.y_actual = [self.labels_to_numbers[item] for item in self.y_actual]
                
                self.y_train = keras.utils.to_categorical(np.asarray(self.y_train))
                self.y_actual = keras.utils.to_categorical(np.asarray(self.y_actual))

            else:
                self.X_train, self.X_test, self.y_train, self.y_actual = self.split_data()


        def split_data(self, prop=.2):
            return super().split_data(prop)
class TA_Data(DatasetLoader):

    def load_data(self, modeltype, num_features, prop, probabilities = None):
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
        self.labels_to_numbers = {}
        i = 0
        for label in self.labelnames:
            if label in self.labels_to_numbers.keys():
                continue
            self.labels_to_numbers[label] = i
            i += 1

        self.numbers_to_labels = {value:key for key, value in self.labels_to_numbers.items()}
        
        # CNN Data Handling
        if modeltype=="CNN":
            images = np.expand_dims(np.asarray(self.images), axis=-1) # necessary to show there is 1 channel (grayscale)?
            labels = keras.utils.to_categorical(np.asarray([self.labels_to_numbers[label] for label in self.labelnames]))
            self.X_train, self.X_test, self.y_train, self.y_actual = train_test_split(images, labels, test_size=prop, random_state=18)
        # Eigenfaces, Fisherfaces, LBPH Data Handling
        else:
            self.indices = np.arange(len(self.images))
            self.X_train, self.X_test, self.y_train, self.y_actual = self.split_data(prop)

    def split_data(self, prop=.25):
        return super().split_data(prop)


class Yale_Data(DatasetLoader):
    def load_data(self, modeltype, num_features, prop, probabilities = None):
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

        self.labels_to_numbers = {}
        i = 0
        for label in self.labelnames:
            if label in self.labels_to_numbers.keys():
                continue
            self.labels_to_numbers[label] = i
            i += 1

        self.numbers_to_labels = {value:key for key, value in self.labels_to_numbers.items()}
        if modeltype=="CNN":

            images = np.expand_dims(np.asarray(self.images), axis=-1) # necessary to show there is 1 channel (grayscale)?
            labels = keras.utils.to_categorical(np.asarray([self.labels_to_numbers[label] for label in self.labelnames]))
            self.X_train, self.X_test, self.y_train, self.y_actual = train_test_split(images, labels, test_size=prop, random_state=18)
        else:
            self.indices = np.arange(len(self.images))
            self.X_train, self.X_test, self.y_train, self.y_actual = self.split_data(prop)
    
    def split_data(self, prop=.2):
        return super().split_data(prop)
