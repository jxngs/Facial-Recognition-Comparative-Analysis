from filereader import FileReader

names, images = FileReader.readFilesToVectors('./lfw-deepfunneled')
fold0, fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9 = FileReader.getCrossValidationGroups(names, images)

print(fold0)

'''
HERE IS WHAT A FOLD LOOKS LIKE:
(array(['Tommy_Haas', 'Sarah_Hughes', 'Jamir_Miller', ...,
       'Britney_Spears', 'Prince_Charles', 'Donald_Rumsfeld'],
      dtype='<U35'), array([[ 76.23 ,  76.23 ,  75.23 , ..., 145.295, 134.686, 131.974],
       [  0.   ,   0.   ,   0.   , ...,   0.   ,   0.   ,   0.   ],
       [  0.   ,   0.   ,   0.   , ...,   0.   ,   0.   ,   0.   ],
       ...,
       [  0.   ,   0.   ,   0.   , ...,   0.456,   0.456,   0.228],
       [  2.239,   1.761,  79.614, ...,  29.015,   6.043,   0.587],
       [  0.299,   0.   ,   0.299, ...,   3.783,   0.598,   0.598]]))

IT IS A TUPLE WITH TWO ARRAYS, ONE FOR THE NAMES AND THE OTHER FOR THE IMAGES. THE ARRAYS ARE EQUALLY SIZED
'''