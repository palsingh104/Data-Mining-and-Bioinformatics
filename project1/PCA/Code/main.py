from utilities import Utilities
import sys


def pca(data_file):
	#  file: pca_a
	util = Utilities()
	util.load_data(data_file)
	# PCA
	Y = util.find_pc()
	util.generate_scatter_plot(Y, data_file, 'PCA')

	# svd
	Y = util.implement_SVD()
	util.generate_scatter_plot(Y, data_file, 'SVD')

	# # t-SNE
	Y = util.implement_tSNE()
	util.generate_scatter_plot(Y, data_file, 't-SNE')


"""
:params file path
"""
def main():
	#  file: name is given as input
	pca(sys.argv[1])


if __name__ == '__main__':
	main()