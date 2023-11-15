import enum
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from heapq import *
IMAGE_WIDTH = 250
IMAGE_HEIGHT = 250

class Eigenfaces:
    
    def __init__(self):
        self.num_images = 0
        self.trainingimages = []
        self.gettrainingdata()
     
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
    
    def get_LBP_Val(self,thresh,arr):
        ret=0
        above_thresh = (arr >= thresh).astype(np.uint8)
        for ind,val in enumerate(above_thresh):
            if val:
                ret+=2**ind
        return ret

    def get_LBP_Mat(self, grayscaled_mat):
        M,N=grayscaled_mat.shape
        lbp_matrix = np.zeros((M, N), dtype=np.uint8)
        for r in range(1,M-1):
            for c in range(1,N-1):
                pixel_threshold = grayscaled_mat[r][c]
                neighbors = self.getNeighbors(grayscaled_mat,r,c)
                lbp_rc = self.get_LBP_Val(pixel_threshold,neighbors)
                lbp_matrix[r][c]=lbp_rc
        return lbp_matrix

    def show_histogram(self, histogram):
        plt.bar(range(len(histogram[0])), histogram[0])
        plt.title("LBP Histogram")
        plt.xlabel("Bin")
        plt.ylabel("Frequency")
        plt.show()

    def get_Histogram(self, grayscaled_mat, show_hist=False): 
        lbp_mat = self.get_LBP_Mat(grayscaled_mat)
        lbp_values = lbp_mat.flatten()
        histogram = np.histogram(lbp_values, bins=256, range=(0, 256))
        if show_hist:
            self.show_histogram(histogram)
        return histogram
    
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

    def knn(self, testhist, k, metric):
        heap = []
        for ind,traininghist in enumerate(self.traininghists):
            dist = self.distance(testhist, traininghist[0], metric)
            if len(heap)<k:
                heappush(heap,(-dist,ind))
            else:
                heappushpop(heap, (-dist,ind))
        heap.sort()
        return heap

    def train(self, k, metric):
        training, testing = getData()
        
        for ind,trainingpoint in enumerate(training):
            training[ind][0]=self.get_Histogram(trainingpoint[0])
        correct=0
        for ind,testingpoint in enumerate(testing):
            testing[ind][0] = self.get_Histogram(testingpoint)
            nearest_neighbors = self.knn(testing[ind][0], k, metric)
            freqs={}
            for neighbor in nearest_neighbors:
                if neighbor not in freqs:
                    freqs[neighbor]=0
                freqs[neighbor]=freqs[neighbor]+1
            maxkey,maxval="",0
            for k,v in freqs.items():
                if v>maxval:
                    maxval=v
                    maxkey=k
            conf_measure = maxval/k
            testing[ind] = [self.get_Histogram(testingpoint), maxval, maxkey, conf_measure]
            if maxkey==testing[ind][1]:
                correct+=1
            print("Classified As " + maxkey)
        return correct/len(testing)


x=Eigenfaces()

