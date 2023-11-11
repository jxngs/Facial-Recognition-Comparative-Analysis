import os
import numpy as np
from PIL import Image
import io


IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50

def yale_data():
    # for each file in the images directory...
    images = []
    labels = []
    for filename in os.listdir('yalefaces_binary'):
        if filename == 'data' or filename == 'Readme.txt': continue
        with open('yalefaces_binary/' + filename, 'rb') as file:
        # Read the content of the binary file
            binary_data = file.read()
            image = Image.open(io.BytesIO(binary_data))
            resized_image = image.resize((IMAGE_HEIGHT,IMAGE_WIDTH))

            #pixel_data = list(resized_image.getdata())

            # greyscale_data = [0 for x in range(IMAGE_WIDTH*IMAGE_HEIGHT)]

            # print(pixel_data)
            # for i in range(len(pixel_data)):
            #     # using the weighted grayscale method, which weights colors according to their wavelengths
            #     rgb = pixel_data[i]
            #     print(rgb)
            #     greyscale_data[i] = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]

            # # save the images for our viewing pleasure
            # greyscale_data = np.asarray(greyscale_data)
            # greyscale_mat = greyscale_data.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

           # pixel_data = np.frombuffer(binary_data, dtype=np.uint8).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))
            images.append(np.array(resized_image))
            labels.append(filename[:filename.find('.')])
    return np.array(images), np.array(labels)


def greyscale(images):
        for img in images:
            # open the image, resize it, and convert it to a one-dimensional array
            image = Image.fromarray(img)

            resized_image = image.resize((IMAGE_HEIGHT,IMAGE_WIDTH))

            # extract the pixel data from the image and convert to greyscale
            pixel_data = list(resized_image.getdata())
            greyscale_data = [0 for x in range(IMAGE_WIDTH*IMAGE_HEIGHT)]
            for i in range(len(pixel_data)):
                # using the weighted grayscale method, which weights colors according to their wavelengths
                rgb = pixel_data[i]
                greyscale_data[i] = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
                # update average image pixel by pixel as we go, used later for computations
                # self.average_image[i] = (self.average_image[i]*self.num_images + greyscale_data[i])/(self.num_images+1)

            # save the images for our viewing pleasure
            pixel_data = np.asarray(pixel_data).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
            # plt.gray()
            # plt.imsave('altered_images/'+filename, greyscale_mat)

            # update class instance variables for later
            # self.num_images = self.num_images + 1
            # self.images.append(greyscale_mat)
            # self.labels.append(filename)
            return greyscale_data