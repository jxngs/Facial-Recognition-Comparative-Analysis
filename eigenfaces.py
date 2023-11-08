import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_WIDTH = 250
IMAGE_HEIGHT = 250

class Eigenfaces:

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
            self.images.append(greyscale_data)

    
    def __init__(self):
        self.num_images = 0
        self.average_image = [0 for x in range(IMAGE_WIDTH*IMAGE_HEIGHT)]
        self.images = []
        self.loadimages()
        