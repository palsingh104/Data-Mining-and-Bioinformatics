#!/usr/bin/env python

import sys
import numpy as np

f = open("centroids.txt", 'r')
lines = f.readlines()
f.close()
centroids = []

for centroid in lines:
	points = centroid.strip().split("\t")
	centroids.append(np.array(points, dtype=float))

for line in sys.stdin:
	line = line.strip()
	point = line.split("\t")
	point = np.array(point[2:], dtype=float)
	cluster_id = min([(a[0],np.linalg.norm(point-a[1])) for a in enumerate(centroids)], key = lambda y:y[1])[0]
	point = "\t".join([str(a) for a in point])
	print '%s\t%s' % (cluster_id, point)