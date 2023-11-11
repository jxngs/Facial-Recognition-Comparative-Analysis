# this file will run implementations from other files
from eigenfaces import Eigenfaces
from fisherfaces import Fisherfaces
import yalefaces

#e = Eigenfaces()
images, labels = yalefaces.yale_data()
f = Fisherfaces(images, labels)
print(f.get_fisherface())

# labels = ['a', 'b']
# img = ['aaaa', 'bbb']
# print(img[labels == 'a'])