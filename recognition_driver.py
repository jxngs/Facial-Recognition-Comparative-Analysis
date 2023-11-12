# this file will run implementations from other files
from eigenfaces import Eigenfaces
from fisherfaces import Fisherfaces
import yalefaces

#e = Eigenfaces()
images, labels = yalefaces.yale_data()
f = Fisherfaces(images, labels)
#print(f.get_fisherface())
print(f.predict('yalefaces_binary/subject01.centerlight', 'bin'))
print(f.predict('yalefaces_binary/subject02.centerlight', 'bin'))
print(f.predict('yalefaces_binary/subject03.centerlight', 'bin'))
print(f.predict('yalefaces_binary/subject04.centerlight', 'bin'))

# labels = ['a', 'b']
# img = ['aaaa', 'bbb']
# print(img[labels == 'a'])