#!/usr/bin/env python

import sys
import numpy as np

curr_cluster = None
clusters = []
cluster = None

for line in sys.stdin:
	line = line.strip()

	cluster, data = line.split('\t', 1)
	try:
		data = np.array(data.strip().split(), dtype=float)
	except:
		continue
	if curr_cluster == cluster:
		clusters.append(data)
	else:
		if curr_cluster:
			new_centroids = np.mean(clusters, axis=0)
			new_centroids = "\t".join([str(a) for a in new_centroids])
			print '%s\t%s' % (curr_cluster, new_centroids)
		clusters = [data]
		curr_cluster = cluster

if curr_cluster == cluster:
	new_centroids = np.mean(clusters, axis=0)
	new_centroids = "\t".join([str(a) for a in new_centroids])
	print '%s\t%s' % (curr_cluster, str(new_centroids))