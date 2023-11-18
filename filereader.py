import io
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100

class FileReader:

    # should pass in './lfw-deepfunneled' into this method to get the correct images
    @staticmethod 
    def readFilesToVectors(base_filepath, num=IMAGE_HEIGHT, threshold=None):
        names = []
        image_vectors = []

        # loop through folders in the image dataset
        for person_name in os.listdir(base_filepath):
            count = 0

            if str(person_name).startswith('.'):
                continue
            filepath = base_filepath + '/' + person_name
            # get the image files from each folder
            for filename in os.listdir(filepath):
                if threshold and count > threshold: break
                # add person's name to the list of names for each image
                names.append(person_name)

                # open and process image
                image = Image.open(filepath + '/' + filename).resize((num,num))
                pixels = list(image.getdata())
                greyscale = np.zeros(num * num)

                for i in range(len(pixels)):
                    rgb = pixels[i]
                    # use weighted grayscale function to optimize lighting differences among color hues
                    greyscale[i] = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
                
                image_vectors.append(greyscale)
                if threshold: count += 1

        return names, image_vectors
    
    def getCrossValidationGroups(names, image_vectors, num_splits = 10, random_seed = 1):
        shuffled_names, shuffled_images = shuffle(names, image_vectors, random_state = random_seed)

        names_split = np.array_split(shuffled_names, num_splits)
        image_vectors_split = np.array_split(shuffled_images, num_splits)

        return [(names_split[i], image_vectors_split[i]) for i in range(num_splits)]


        
