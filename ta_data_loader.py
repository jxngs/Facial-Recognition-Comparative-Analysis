import os
from PIL import Image
import numpy as np
import keras

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

class TA_Data(DatasetLoader):
    def loaddata(self, modeltype, IMAGE_SIZE = 100):
        self.images = []
        self.labelnames = []
        for filename in os.listdir('./ta_data'):
            image = Image.open('./ta_data/' + filename).resize((IMAGE_SIZE,IMAGE_SIZE))
            name = filename[0:filename.index('_')]
            pixels = list(image.getdata())
            greyscale = np.zeros(IMAGE_SIZE*IMAGE_SIZE)

            for i in range(len(pixels)):
                rgb = pixels[i]
                # use weighted grayscale function to optimize lighting differences among color hues
                greyscale[i] = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
            
            greyscale = greyscale.reshape((IMAGE_SIZE,IMAGE_SIZE))
            
            self.images.append(greyscale)
            self.labelnames.append(name)

        if modeltype=="CNN":
            images = np.expand_dims(np.asarray(self.images), axis=-1) # necessary to show there is 1 channel (grayscale)?
            labels = keras.utils.to_categorical(np.asarray(self.labelnames))
            self.X_train, self.X_test, self.y_train, self.y_actual = train_test_split(images, labels, test_size=0.2, random_state=18)
        else:
            self.X_train, self.X_test, self.y_train, self.y_actual = self.split_data()

        print(self.labelnames)

    def split_data(self, prop=.25):
        return super().split_data(prop)

loader = TA_Data()
print(loader.labelnames)
print(loader.y_train)
print(loader.y_actual)