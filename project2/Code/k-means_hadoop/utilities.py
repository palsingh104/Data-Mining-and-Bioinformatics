import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess

class Utilities:
	def __init__(self, file_name):
		# pass
		self.file_name = file_name

	def execute_command(self, command):
		std_out = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
		return std_out

	def get_data(self, file_path):
		file = open(file_path,'r')
		lines = file.readlines()
		file.close()
		data = []
		ground_truth = []
		for line in lines:
			values = line.strip().split("\t")
			ground_truth.append(int(values[1]))
			data.append(values[2:])

		return ground_truth, np.array(data).astype(float)

	def initilialize_centroids(self, data, file_path, num_of_centroids):
		initial_centroids = np.random.randint(0,data.shape[0],num_of_centroids)
		print "initial centroids: ",initial_centroids
		output = open(file_path, 'w')

		for i in initial_centroids:
			x = "\t".join([str(data[i][j]) for j in xrange(len(data[i]))])
			output.write(x + "\n")

		output.close()
		return data[initial_centroids]

	def check_convergence(self, old_clusters, new_clusters):
		print "new_clusters: " + str(len(new_clusters))
		print "old_clusters: " + str(len(old_clusters))

		for i in xrange(len(old_clusters)):
			if np.linalg.norm(old_clusters[i] - new_clusters[i]) > 0:
				return False
		return True

	def compute_centroids(self, data, file_path):
		data = data.split('\n')
		output = []
		output_file = open(file_path, 'w')
		for i in data:
			temp = i.split("\t")
			if len(temp)>1:
				temp = temp[1:]
				output.append(np.array(temp, dtype=float))
				writeData = "\t".join([str(temp[j]) for j in xrange(len(temp))])
				output_file.write(writeData + "\n")

		output_file.close()
		return output

	def generate_matrices(self, ground_truth, result):
		N = len(ground_truth)
		P = [[0 for j in xrange(N)] for i in xrange(N)]
		C = [[0 for j in xrange(N)] for i in xrange(N)]
		for i in xrange(N):
			for j in xrange(N):
				if ground_truth[i] == ground_truth[j]:
					P[i][j] = 1
					P[j][i] = 1
		for i in xrange(N):
			for j in xrange(N):
				if result[i] == result[j]:
					C[i][j] = 1
					C[j][i] = 1
		return P,C


	def compute_jaccard_coefficient(self, ground_truth, result):
		N = len(ground_truth)
		P,C = self.generate_matrices(ground_truth,result)
		m11 = 0
		m10 = 0
		for i in xrange(N):
			for j in xrange(N):
				if C[i][j] == P[i][j] == 1:
					m11 +=1
				elif C[i][j] != P[i][j]:
					m10 +=1
		return float(m11)/(m11+m10)

	def compute_rand_idx(self, ground_truth, result):
		N = len(ground_truth)
		P,C = self.generate_matrices(ground_truth,result)
		m11 = 0
		m10 = 0
		for i in xrange(N):
			for j in xrange(N):
				if C[i][j] == P[i][j]:
					m11 +=1
				else:
					m10 +=1
		return float(m11)/(m11+m10)

	def compute_PCA_and_plot(self, data, clusters, centroids = None):
		pca = PCA(n_components = 2)
		pca.fit(data)
		data = pca.transform(data)
		fig = plt.figure()
		clusters = np.array(clusters)
		cmap = plt.get_cmap('viridis')
		colors = cmap(np.linspace(0, 1, len(clusters)))
		for i in xrange(1, len(set(clusters))+1):
			indices = np.where(clusters == i)
			plt.scatter(data[indices,0],data[indices,1], c="C"+str(i-1), label=i)
		if centroids:
			centroids = pca.transform(centroids)
			plt.scatter(centroids[:,0], centroids[:,1], c="black", marker = "x", label = "Centroids")

		plt.xlabel('PC 1', fontsize=14)
		plt.ylabel('PC 2', fontsize=14)
		plt.title(self.file_name)
		plt.suptitle( "K-Means Clustering - Hadoop", fontsize=14)
		plt.legend()
		plt.grid(axis='both')
		fig.savefig(self.file_name + '_k-means-hadoop.png')

	def map_centroid_data(self, data, centroids):
		resultClusters = []
		finalClusters = {}
		geneID = 1
		for point in data:
			cluster = min([(a[0],np.linalg.norm(point-a[1])) for a in enumerate(centroids)], key = lambda x:x[1])[0]
			resultClusters.append(cluster)

			if cluster not in finalClusters:
				finalClusters[cluster] = []
			finalClusters[cluster].append(geneID)
			geneID+=1
		return resultClusters, finalClusters