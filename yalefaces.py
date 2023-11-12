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