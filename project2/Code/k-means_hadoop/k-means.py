import sys

from utilities import Utilities

def main():
	file_path = 'data/{}'.format(sys.argv[1])
	num_of_centroids = 5
	max_iter = 100
	if len(sys.argv) > 2:
		num_of_centroids = int(sys.argv[2])

	util = Utilities(sys.argv[1])
	# export hadoop path for quick access
	util.execute_command("export PATH=$PATH:/usr/local/hadoop/bin/")

	# clear input data of previous runs
	remove_input_data = "hdfs dfs -rm -r ~/input"
	create_input_folder = "hdfs dfs -mkdir ~/input"
	hdfs_upload_data = "hdfs dfs -put {} ~/input/".format(file_path)
	remove_output_data = "hdfs dfs -rm -r output"
	run_hadoop = "hadoop jar hadoop-streaming-2.7.7.jar -files mapper.py,reducer.py,centroids.txt -mapper mapper.py -reducer reducer.py -input ~/input/ -output output"
	download_output = "hdfs dfs -cat output/*"

	is_converged = False
	num_iterations = 0

	ground_truth, data = util.get_data(file_path)
	# initial centroids
	curr_centroids = util.initilialize_centroids(data,'centroids.txt',num_of_centroids)
	# remove input data of previous runs
	util.execute_command(remove_input_data)

	# create input folder
	util.execute_command(create_input_folder)

	# upload input data on hadoop fs
	util.execute_command(hdfs_upload_data)
	
	while (not is_converged) and (num_iterations != max_iter):
		util.execute_command(remove_output_data)
		# execute map reduce on centroids
		util.execute_command(run_hadoop)
		reducer_output = util.execute_command(download_output)
		# calculate new centroids
		new_centroids = util.compute_centroids(reducer_output, 'centroids.txt')
		num_iterations += 1
		if util.check_convergence(curr_centroids,new_centroids):
		 	is_converged = True
		curr_centroids = new_centroids

	print "Total num of iterations to converge: ", num_iterations
	
	result, final_clusters = util.map_centroid_data(data, curr_centroids)

	print "Jaccard:", util.compute_jaccard_coefficient(ground_truth, result)
	print "Rand:", util.compute_rand_idx(ground_truth, result)

	util.compute_PCA_and_plot(data, result, curr_centroids)


if __name__ == "__main__":
	main()