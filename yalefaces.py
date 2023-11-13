import os
import numpy as np
from PIL import Image
import io

def yale_data(num_features):
    # for each file in the images directory...
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
    return np.array(images), np.array(labels)