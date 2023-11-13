# this file will run implementations from other files
#from eigenfaces import Eigenfaces
from fisherfaces import Fisherfaces
import yalefaces
NUM_FEATURES = 300
#e = Eigenfaces()
images, labels = yalefaces.yale_data(NUM_FEATURES)

f = Fisherfaces(images, labels, NUM_FEATURES)
#print(f.get_fisherface())
accuracy = 0
for i in range(1, 16):
    num = '0' + str(i) if i < 10 else str(i)
    pred = f.predict('yalefaces_binary/subject' + num + '.centerlight', 'bin')
    print('pred', i, pred)
    if 'subject' + num == pred[0]: accuracy += 1
print('accuracy', accuracy/15)
# labels = ['a', 'b']
# img = ['aaaa', 'bbb']
# print(img[labels == 'a'])