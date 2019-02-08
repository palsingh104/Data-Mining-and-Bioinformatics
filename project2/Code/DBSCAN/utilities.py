import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Utilities:
    def __init__(self):
        pass

    """
    @params: data = 2-D matrix, eps = threshold value,
            min_pts = minimum number of points in a cluster

    @returns: labels = cluster labels
    """
    def compute_dbscan(self, data, eps, min_pts):
        num_of_genes = len(data)
        cluster_labels = [0]*num_of_genes
        print(num_of_genes)
        current_cluster_id = 0

        for row_idx in range(0, num_of_genes):
            # Only points that have not already been claimed can be picked as new 
            # seed points.    
            # If the point's label is not 0, continue to the next point.
            if not (cluster_labels[row_idx] == 0):
               continue
            
            # Find all of P's neighboring points.
            neighboring_points = check_neighbors(data, row_idx, eps)

            if len(neighboring_points) < min_pts:
                cluster_labels[row_idx] = -1
            # Otherwise, if there are at least MinPts nearby, use this point as the 
            # seed for a new cluster.    
            else: 
                current_cluster_id += 1
                expand_cluster(data, cluster_labels, row_idx, neighboring_points, current_cluster_id, eps, min_pts)
        
        return cluster_labels


    def expand_cluster(self, data, cluster_labels, row_idx, neighboring_points, current_cluster_id, eps, min_pts):
        # Assign the cluster label to the seed point.
        cluster_labels[row_idx] = current_cluster_id
        
        idx = 0
        while idx < len(neighboring_points):    
            
            # Get the next point from the queue.        
            next_point = neighboring_points[idx]
           
            if cluster_labels[next_point] == -1:
                cluster_labels[next_point] = current_cluster_id
            
            elif cluster_labels[next_point] == 0:
                cluster_labels[next_point] = current_cluster_id
                next_point_neighbors = check_neighbors(data, next_point, eps)
                if len(next_point_neighbors) >= min_pts:
                    neighboring_points = neighboring_points + next_point_neighbors           
            
            # Advance to the next point in the queue.
            idx += 1

    """
    @params: data = 2-D matrix, eps = threshold value,
             point_idx = row in the data
    @description:  calculates the distance between a point_idx and other 
                    points in the dataset, and then returns only those points which are within the 
                    threshold distance.
    @returns: neighbors
    """
    def check_neighbors(self, data, point_idx, eps):
        neighbors = []
        num_of_genes = len(data)
        
        for point in range(0, num_of_genes):
            if np.linalg.norm(data[point_idx] - data[point]) < eps:
                neighbors.append(point)
                
        return neighbors

    def plot_PCA(self, data,labels,plot_title):
        data=data[:,2:]
        pca = PCA(n_components=2)
        data = np.matrix(data).T
        pca.fit(data)
        data_pca = pca.components_
        title =  plot_title
        plt.figure(figsize=(8,6))
        px=data_pca[0,]
        py=data_pca[1,]
        unique = list(set(labels))
        colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
        for i, u in enumerate(unique):
            xi = [px[j] for j  in range(len(px)) if labels[j] == u]
            yi = [py[j] for j  in range(len(px)) if labels[j] == u]
            plt.scatter(xi, yi, c=colors[i], label=str(u))
        
        plt.legend()
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.title(title)
        plt.show()