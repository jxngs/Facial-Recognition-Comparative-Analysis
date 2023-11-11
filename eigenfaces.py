import enum
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_WIDTH = 250
IMAGE_HEIGHT = 250

class Eigenfaces:
    
    def __init__(self):
        self.num_images = 0
        self.average_image = [0 for x in range(IMAGE_WIDTH*IMAGE_HEIGHT)]
        self.images = []
        self.loadimages()

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

    def getNeighbors(self, mat, r, c):
        if r<=0 or c<=0 or r>=len(mat)-1 or c>=len(mat[0])-1:
            print("Error. Invalid Input")
            return
        topleft = mat[r-1][c-1]
        topcenter = mat[r-1][c]
        topright = mat[r-1][c+1]
        right = mat[r][c+1]
        botright = mat[r+1][c+1]
        botcenter = mat[r+1][c]
        botleft = mat[r+1][c-1]
        left = mat[r][c-1]
        return np.array([topleft,topcenter,topright,right,botright,botcenter,botleft,left])
    
    def get_LBP_Val(self,thresh,arr):
        ret=0
        above_thresh = (arr >= thresh).astype(np.uint8)
        for ind,val in enumerate(above_thresh):
            if val:
                ret+=2**ind
        return ret

    def get_LBP_Mat(self, grayscaled_mat):
        M,N=grayscaled_mat.shape
        lbp_matrix = np.zeros((M, N), dtype=np.uint8)
        for r in range(1,M-1):
            for c in range(1,N-1):
                pixel_threshold = grayscaled_mat[r][c]
                neighbors = self.getNeighbors(grayscaled_mat,r,c)
                lbp_rc = self.get_LBP_Val(pixel_threshold,neighbors)
                lbp_matrix[r][c]=lbp_rc
        return lbp_matrix

    def show_histogram(self, histogram):
        plt.bar(range(len(histogram[0])), histogram[0])
        plt.title("LBP Histogram")
        plt.xlabel("Bin")
        plt.ylabel("Frequency")
        plt.show()

    def get_Histogram(self, grayscaled_mat, show_hist=False): 
        lbp_mat = self.get_LBP_Mat(grayscaled_mat)
        lbp_values = lbp_mat.flatten()
        histogram = np.histogram(lbp_values, bins=256, range=(0, 256))
        if show_hist:
            self.show_histogram(histogram)
        return histogram
    
x=Eigenfaces()
x.get_Histogram(x.images[0],True)
