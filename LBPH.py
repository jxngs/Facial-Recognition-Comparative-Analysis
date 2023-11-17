import enum
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from heapq import *
from filereader import *


class LBPH:
    
    def __init__(self,k):
        self.k=k
        if os.path.exists("imagevectors.npy") and os.path.exists("names.json"):
            json_filename = "names.json"
            with open(json_filename, 'r') as json_file:
                names = json.load(json_file)
            images = np.load("imagevectors.npy")
        else:
            names, images = FileReader.readFilesToVectors('/Users/noahloewy/Documents/Facial-Recognition-Comparative-Analysis/__pycache__/Try')
        images = [vector.reshape((100, 100)) for vector in images]
        self.cv(names,images,"EuclideanDistance")
        
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

    def get_LBP_Mat(self, grayscaled_mat):
        M,N=grayscaled_mat.shape
        lbp_matrix = np.zeros((M, N), dtype=np.uint8)
        for r in range(1,M-1):
            for c in range(1,N-1):
                pixel_threshold = grayscaled_mat[r][c]
                neighbors = self.getNeighbors(grayscaled_mat,r,c)
                above_thresh = neighbors >= pixel_threshold
                lbp_rc = np.sum(above_thresh * (2 ** np.arange(len(neighbors))))
                lbp_matrix[r, c] = lbp_rc
        return lbp_matrix

    def show_histogram(self, histogram, name):
        plt.bar(range(len(histogram[0])), histogram[0])
        plt.title("LBP Histogram" + name)
        plt.xlabel("Bin")
        plt.ylabel("Frequency")
        plt.show()

    def get_Histogram(self, grayscaled_mat, name, show_hist=False): 
        lbp_mat = self.get_LBP_Mat(grayscaled_mat)
        lbp_values = lbp_mat.flatten()
        histogram = np.histogram(lbp_values, bins=256, range=(0, 256))
        if show_hist:
            self.show_histogram(histogram, name)
        return histogram[0]
    
    def distance(self,vector1, vector2, metric):
        if metric == "ChiSquare":
            distance = np.sum((vector1 - vector2)**2 / (vector1 + vector2 + 1e-10))

        elif metric == "EuclideanDistance":
            distance = np.linalg.norm(vector1 - vector2)

        elif metric == "NormalizedEuclideanDistance":
            distance = np.linalg.norm(vector1 - vector2) / np.linalg.norm(vector1)

        elif metric == "AbsoluteValue":
            distance = np.sum(np.abs(vector1 - vector2))

        else:
            raise ValueError("Invalid metric. Supported metrics: 'ChiSquare', 'EuclideanDistance', 'NormalizedEuclideanDistance', 'AbsoluteValue'.")

        return distance

    def knn(self, testpoint, training, metric):
        heap = [] 
        for person,point in training:  
            dist = self.distance(testpoint[1], point[1], metric)
            print(testpoint[0], person, dist)
            if len(heap)<self.k:
                heappush(heap,(-dist,person))
            else:
                heappushpop(heap, (-dist,person))
        heap.sort()
        return heap

    def test(self, training, testing, metric):
        correct=0
        for ind,testingpoint in enumerate(testing):
            nearest_neighbors = self.knn(testingpoint, training, metric)
            freqs={}
            for neighbor in nearest_neighbors:
                if neighbor[1] not in freqs:
                    freqs[neighbor[1]]=0
                freqs[neighbor[1]]=freqs[neighbor[1]]+1
            maxkey,maxval="",0
            for k,v in freqs.items():
                if v>maxval:
                    maxval=v
                    maxkey=k
            conf_measure = maxval/v
            if maxkey==testing[ind][0]:
                correct+=1
            print(testingpoint[0] + " Classified As " + maxkey + " with conf measure " + str(conf_measure)) 
        return correct/len(testing)

    def trainAll(self,images,names):
        for ind,trainingpoint in enumerate(images):
            images[ind]=self.get_Histogram(trainingpoint,names[ind])
            if ind%10==0:
                print(str(100*ind/len(images)) + "%")
        return images
    
    def cv(self, names, images, metric):
        images, names  = images[:200], names[:200]
        images = self.trainAll(images,names)
        partitions = FileReader.getCrossValidationGroups(names, images)
        for i in range(len(partitions)):
            partitions[i] = list(zip(partitions[i][0], partitions[i][1]))
        for i in range(len(partitions)): 
            training=[]
            for j in range(len(partitions)):
                if i!=j:
                    training=training+partitions[j]
            testing = partitions[i]
            print("Accuracy : " + str(self.test(training,testing,metric)))
x=LBPH(k = 1)