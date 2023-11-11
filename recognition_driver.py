# this file will run implementations from other files
from eigenfaces import Eigenfaces
from fisherfaces import Fisherfaces

#e = Eigenfaces()
f = Fisherfaces()
print(f.get_fisherface())

# labels = ['a', 'b']
# img = ['aaaa', 'bbb']
# print(img[labels == 'a'])