import numpy as np
import pandas as pd
import plotly
import plotly.plotly as py
# plotly offline 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

from sklearn.manifold import TSNE

class Utilities:
	def __init__(self):
		self.features = None
		self.unique_disease = None

	def load_data(self, file_name):
		# Load the data from file
		data = pd.read_csv(file_name, sep="\t", header=None)
		num_cols = len(data.columns)

		# split the data in features and diseases 
		self.features = data.iloc[:,0:num_cols-1].values
		# print('self.features.shape')
		# print(self.features.shape)
		
		self.disease = data.iloc[:,num_cols-1].values
		# print('self.disease.shape')
		# print(self.disease.shape)
		self.unique_disease = np.unique(self.disease)

	def compute_covar_matrix(self, print_covar = False):
		# # compute the mean of each column
		# Mean = np.mean(self.features, axis=0)
		# print(Mean)
		
		# covariance = (self.features - Mean).T.dot((self.features - Mean)) / (self.features.shape[0]-1)
		# print("from mean:" covariance)
		
		# since numpy's covariance function and above custom implementation generate the same
		# covariance matrix, I am using numpy's function
		# calculate the covariance matrix
		self.covariance = np.cov(self.features.T)
		# print('self.covariance')
		# print(self.covariance)
		if(print_covar):
			print("numpy:", covariance)

	def find_pc(self):
		self.compute_covar_matrix()

		# calculate the eigen values from covariance matrix
		eigen_values, eigen_vector = np.linalg.eig(self.covariance)
		# print(eigen_vector)
		# print(eigen_values)
		
		# assign index to each eigen value making a tuple of (index, eigen value)
		eigen_pairs = [0]*len(eigen_values)
		for i, value in enumerate(eigen_values):
		    eigen_pairs[i] = [i, value]
		
		# print(eigen_pairs)
		
		# sort the eigen values in decreasing order
		sorted_eigen_pairs = sorted(eigen_pairs, key=lambda x:x[1], reverse=True)
		# print(sorted_eigen_pairs)
		
		# select the top 2 eigen values
		eigen_pairs = sorted_eigen_pairs[0:2]
		# print(eigen_pairs)
		
		# high variance eigen vectors
		high_var_vectors = eigen_vector[:,(eigen_pairs[0][0], eigen_pairs[1][0])]
		# print('high_var_vectors')
		# print(high_var_vectors)

		# calculate principal components based on top 2 eigen vectors
		Y = self.features.dot(high_var_vectors)
		# print("y")
		# print(Y)
		return Y


	def implement_SVD(self):
		u,s,v = np.linalg.svd(self.features, full_matrices=True)
		# print("u")
		# print(u.shape)

		# calculate top 2 dimensions
		Y = u[:,:2]
		# print("Y")
		# print(Y.shape)
		# print(Y)
		return Y

	def implement_tSNE(self):
		tsne = TSNE(n_components=2, init='pca', n_iter=1000, learning_rate=100)
		tsne_results = tsne.fit_transform(self.features)
		# print('self.features')
		# print(self.features.shape)
		# print(self.features)
		# print('tsne_results')
		# print(tsne_results.shape)
		# print(tsne_results)
		return tsne_results

	def generate_scatter_plot(self, Y, data_file, method):
		traces = []
		Y = np.real(Y)
		file_name = data_file + "_" + method
		title = '{0} Scatter plot for {1}'.format(method, data_file)

		# generate scatter plot. the following code is adapted from plotly documentation
		for name in self.unique_disease:
		    trace = go.Scatter(
		        x=Y[self.disease==name,0],
		        y=Y[self.disease==name,1],
		        mode='markers',
		        name=name,
		        marker=go.scatter.Marker(
		            size=12,
		            line=go.scatter.marker.Line(
		                color='rgba(217, 217, 217, 0.14)',
		                width=0.5),
		            opacity=0.8))
		    traces.append(trace)
		
		
		layout = go.Layout(showlegend=True,
			title=title,
			titlefont=dict(
            family='Courier New, monospace',
            size=24,
            color='#7f7f7f'
            ),
            xaxis=dict(
            	title='component 1'
            	),
            yaxis=dict(
            	title='component 2',
            	)
            )
		
		fig = go.Figure(data=traces, layout=layout)
		plot(fig, filename=file_name)