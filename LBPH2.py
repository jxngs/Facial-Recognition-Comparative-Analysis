import enum
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from heapq import *
from filereader import *
import json

class LBPH2:
    
    def __init__(self,k, size, images=[], names=[], radius=1):
        self.k=k
        self.size = size
        self.images = images
        self.names = names
        self.radius = radius
        self.histograms = []

        unique_names = list(set(names))
        self.names_map = {}
        for i, name in enumerate(self.names):
            self.names_map.setdefault(name, [])
            self.names_map[name].append(i)
        
        self.cv()
        
    def cv(self):
        histograms = {}
        for i, image in enumerate(self.images):
            histograms.setdefault(self.names[i], [])
            histograms[self.names[i]].append(self.get_Histogram(image))
        self.histograms = histograms
    
    def get_Histogram(self, grayscaled_mat): 
        lbp_mat = self.get_LBP_Mat(grayscaled_mat)
        #plt.imsave('altered_images/lbp'+name+".jpeg", lbp_mat)
        lbp_values = lbp_mat.flatten()
        histogram = np.histogram(lbp_values, bins=256, range=(0, 256))
        # if show_hist:
        #     self.show_histogram(histogram, name)
        return histogram[0]
    
    def get_LBP_Mat(self, grayscaled_mat):
        M,N=grayscaled_mat.shape
        lbp_matrix = np.zeros((M, N), dtype=np.uint8)
        for r in range(1,M-1):
            for c in range(1,N-1):
                pixel_threshold = grayscaled_mat[r][c]
               # neighbors = self.getNeighbors(grayscaled_mat,r,c)
                neighbors = self.getRNeighbors(grayscaled_mat, r, c, self.radius)
                above_thresh = neighbors >= pixel_threshold
                lbp_rc = np.sum(above_thresh * (2 ** np.arange(len(neighbors))))
                lbp_matrix[r, c] = lbp_rc
        return lbp_matrix

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

    def knn(self, testpoint, metric):
        heap = [] 
        for person in self.histograms:
            for histogram in self.histograms[person]:
                dist = self.distance(testpoint, histogram, metric)
                if len(heap)<self.k:
                    heappush(heap,(-dist,person))
                else:
                    heappushpop(heap, (-dist,person))
        heap.sort()
        return heap
    
    def getRNeighbors(self, mat, r, c, radius):
        arr = []
        for i in range(r - radius, r + radius + 1):
            if i not in range(self.size): continue
            for j in range(c - radius, c + radius + 1):
                if j not in range(self.size): continue
                arr.append(mat[i][j])
        return np.array(arr)