# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:21:39 2019

@author: Adam Morse

Trace Folder: Morse1

Program Description: K-Means implementation

"""

import random
import numpy as np

items = []      # holds initial dataset values for file reading

def open_file(file):
    f = open(file, "r") #input.txt
    for line in f:
        lyst = line.split() # tokenize input line, it also removes EOL marker
        lyst = list(map(float, lyst))
        items.append(lyst)
    return items

""" Initialize clusters randomly """
def get_clusters(D, k):     # D = set of elements, k = # of clusters
    random.seed(4); 
    centroids = ()
    m1 = random.uniform(0, 5)   # first centroid 
    m2 = random.uniform(0, 5)
    m3 = random.uniform(0, 5)   # second centroid
    m4 = random.uniform(0, 5)
    m5 = random.uniform(0, 5)   # third centroid
    m6 = random.uniform(0, 5)
    
    centroids = ( [[m1, m2]], [[m3, m4]], [[m5, m6]] )
    return centroids

""" Compute distance between a given object and the mean of a given cluster
    using euclidean distance. """ 
def euclidean_distance(v1, v2, mean):
    return np.sqrt(((mean[0] - v1) **2) + (mean[1] - v2)**2)
   
""" Determine which cluster is the least distance between a given object and the
    mean of a given cluster. """      
def get_cluster_class(distance, data_point, centroids):
    index = min(distance, key=distance.get)
    return (index, data_point, centroids[index])
  
""" Assign a given object to a certain cluster based on previously computed
    min distance. """
def assign_cluster(label, clusters):
    
    if label[1] in clusters[label[0]]:  # if an object is already in a cluster, just return that cluster 
        return clusters
    clusters[label[0]].append(label[1]) # assign the object to the cluster at class label
    for i in range(len(clusters)):
        if label[1] in clusters[i]:     # if the object is in the set of clusters
            if i != label[0]:           
                clusters[i].remove(label[1])    # remove the object from this cluster (this object has switched 
                                                # due to recalculation of means, and needs to be removed from that cluster)
                                                # to avoid repeating objects within multiple clusters.
          
    return clusters
  
""" Calculate the mean of each given cluster. This is calculated after 
    all objects have been assigned to a cluster after each iteration. This
    function holds the constantly updating mean. """
def get_mean(clusters):
    mean =  [ [], [], [] ]
    xVals = [ [], [], [] ]
    yVals = [ [], [], [] ]
    for i in range(len(clusters)):
        for k in range(len(clusters[i])):
            xVals[i].append(clusters[i][k][0])      # Grab the x coordinates
            yVals[i].append(clusters[i][k][1])      # Grab the y coordinates
        xVals[i] = sum(xVals[i])
        yVals[i] = sum(yVals[i])
        mean[i].append(xVals[i] / len(clusters[i])) # Find the mean x value (x val / length of objects in cluster)
        mean[i].append(yVals[i] / len(clusters[i])) # Find the mean y value (y val / length of objects in cluster)
    return mean

""" K-Means algorithm. Termination criteria is a max number of iterations, set to 100. """
def k_means(D, k, clusters, termination):
    label = []
    distance = {}
   
    for x in range(termination):            # terminate after 50 attempts (runs)
        mean = get_mean(clusters)  # calculate the new mean for each iteration 
        for i in range(len(D)):
            v1 = D[i][0]            # first point in coordinate (x val)
            v2 = D[i][1]            # second point in coordinate (y val)
            for j in range(k):
                distance[j] = euclidean_distance(v1, v2, mean[j])   # calculate distance between given coordinate and
                                                                    # the mean for each cluster
            data_point = D[i][0:2]      # cutoff class label 
            label = get_cluster_class(distance, data_point, clusters)  # determine which cluster this given coordinate belongs to
            new_centroids = assign_cluster(label, clusters)            # assign coordinate to cluster
            # update clusters to new_centroids (new_centroids = the new list of objects in the clusters)
            clusters = new_centroids
    del clusters[0][0]         # remove the initial random mean points from the clusters (otherwise total cluster size is 503)
    del clusters[1][0]
    del clusters[2][0]
    # return the calculated clusters
    return clusters

""" Determine the accuracy of the K-Means implementation. """  
def get_accuracy(D, label, classified):
    count = []      # holds total objects in each cluster
    total = 0       # holds value of misclassified objects
    for i in range(len(label)):
        count.append(len(label[i]))
        total += len(classified[i])
    c1 = (count[0]) # count (size) for cluster 0
    c2 = (count[1]) # count (size) for cluster 1
    c3 = (count[2]) # count (size) for cluster 2
    total = (len(D) - total) / len(D)
    return (c1, c2, c3, total * 100) 

""" Determine which objects that were assigned to a given cluster, were ultimately 
    misclassified based on the original class label given in the original dataset. """
def get_misclassified_objects(clusters, D):
    dataset = []    # holds dataset without class labels for the values in label 0.0
    dataset1 = []   # holds dataset without class labels for the values in label 1.0 
    dataset2 = []   # holds dataset without class labels for the values in label 2.0
    classed = []    # holds the misclassified objects (coordinates) in each labeled class (cluster)
    for i in range(len(D)):
        if D[i][2] == 0.0:
            x1 = D[i][0:2]
            dataset.append(x1)
        if D[i][2] == 1.0:
            x1 = D[i][0:2]
            dataset1.append(x1)
        if D[i][2] == 2.0:
            x1 = D[i][0:2]
            dataset2.append(x1)
    # if coordinate not in the original dataset class (0.0, 1.0, or 2.0), determine it to be misclassified into the wrong class
    classed = [ x for x in clusters[0] if x not in dataset ]    
    classed1 = [ x for x in clusters[1] if x not in dataset1 ]
    classed2 = [ x for x in clusters[2] if x not in dataset2 ]
    return (classed, classed1, classed2)

""" Determine which class label a misclassified object actually belongs to. """
def get_class_label(D, centroids, misclassed):
    dataset = []    # holds dataset without class labels for the values in label 0.0
    dataset1 = []   # holds dataset without class labels for the values in label 1.0 
    dataset2 = []   # holds dataset without class labels for the values in label 2.0
    for i in range(len(D)):
        if D[i][2] == 0.0:
            x1 = D[i][0:2]
            dataset.append(x1)
        if D[i][2] == 1.0:
            x1 = D[i][0:2]
            dataset1.append(x1)
        if D[i][2] == 2.0:
            x1 = D[i][0:2]
            dataset2.append(x1)
    
    # find which dataset(0,1,2) the misclassified object actually belongs to. This then
    # gives what class label the object should have been classified to. Search within
    # the datasets that it is not clustered with. Meaning, misclassed[0] represents 
    # all objects that were clustered into cluster 0, yet should not have been. Check
    # the rest of the dataset to determine what cluster it should be in. 
    label1 = [x for x in misclassed[0] if x in dataset1]    
    label2 = [x for x in misclassed[0] if x in dataset2]
    
    label0 = [x for x in misclassed[1] if x in dataset]
    label2 += ([x for x in misclassed[1] if x in dataset2])
    
    label0 += ([x for x in misclassed[2] if x in dataset])
    label1 += ([x for x in misclassed[2] if x in dataset1])

      
    return label0, label1, label2


""" Main """
def main():
    if(__name__ == "__main__"):
        cluster_file = open_file("synthetic_2D.txt")
        k = 3
        termination = 50       # stoppping critera value (50 iterations)
        centroids = get_clusters(cluster_file, k)   # grab file data
        
        print("Initial k means are:")
        for i in range(len(centroids)):
            print("mean[",i,"]", centroids[i][0])   
        
        kMeans = k_means(cluster_file, len(centroids), centroids, termination)       # k means
        print("kMeans:", kMeans)
        
        misclassed = get_misclassified_objects(kMeans, cluster_file)    # misclassed objects
        print("missed", misclassed)
        
        accuracy = get_accuracy(cluster_file, kMeans, misclassed)       # accuracy rate (also contains size of clusters)
        
        get_original_label = get_class_label(cluster_file, kMeans, misclassed)  # get original class label

        count = 0   # counter for positions of clusters
        
        # miscounter = bool variable to determine if while printing k means objects and a misclassified object
        # is to be printed to screen, call get_class_label fucntion to determine its original class label
        miscounter = False
        for i in range(k):
            print()
            print("Cluster", count)
            print("Size of cluster", count, "is", accuracy[i])
            print("Cluster label: ", count)
            print("Number of objects misclustered in this cluster is", len(misclassed[i]))
            for j in range(len(kMeans[i])):
                for x in range(len(get_original_label)):
                    if kMeans[i][j] in get_original_label[x]:
                        # switch to True because misclassified object was found
                        miscounter = True
                        # x will determine the index in get_original_label list, which
                        # also correlates to which class label it truly is
                        orginal_label = x
                if (miscounter):
                    print((kMeans[i][j], orginal_label))
                else:
                    print((kMeans[i][j], count))
                    # reset back to false for potential upcoming misclassified objects
                miscounter = False
            count += 1
        print()
        print("Accuracy rate is", repr(accuracy[3]) + "%")      # print accuracy rate
        print()
        print("Initial k means are:")
        for i in range(len(centroids)):
            print("mean[",i,"]", centroids[i][0]) 
        
        print("End of Program")
main()          # call main

""" End of Program """











