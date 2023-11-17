# this file will run implementations from other files
#from eigenfaces import Eigenfaces
from fisherfaces import Fisherfaces
import yalefaces
NUM_FEATURES = 100
#e = Eigenfaces()
from PIL import Image, ImageOps
from filereader import FileReader
import numpy as np
import os

images, labels = yalefaces.yale_data(NUM_FEATURES)

if False:
    labels, images = FileReader.readFilesToVectors('./lfw-deepfunneled')
    f = Fisherfaces(images, labels, NUM_FEATURES)
    print(f.get_fisherface())
    f.save_face(f.get_fisherface(), 'proj/fisher.jpeg')
elif True:
    base_filepath = './lfw-deepfunneled'
    count = 0

    for person_name in os.listdir(base_filepath):

            if str(person_name).startswith('.'):
                continue
            # get the image files from each folder
            count += 1
            if person_name == 'George_W_Bush': print(count)
            if count > 1900: print(count, person_name)
            if count == 1910: break

else:
    image = np.array(ImageOps.grayscale(Image.open('lfw-deepfunneled/George_W_Bush/George_W_Bush_0001.jpg').resize((NUM_FEATURES, NUM_FEATURES))))
    fisherface = np.array(ImageOps.grayscale(Image.open('altered_images/proj/fisher.jpeg').resize((NUM_FEATURES, NUM_FEATURES))))



    f = Fisherfaces(images, labels, NUM_FEATURES)
    pred = f.predict_with_fisher(fisherface, image)
    print(pred)



# accuracy = 0
# for i in range(1, 16):
#     num = '0' + str(i) if i < 10 else str(i)
#     pred = f.predict('yalefaces_binary/subject' + num + '.centerlight', 'bin')
#     print('pred', i, pred)
#     if 'subject' + num == pred[0]: accuracy += 1
# print('accuracy', accuracy/15)

# f.save_face(f.mean_face, 'mean.jpeg')
