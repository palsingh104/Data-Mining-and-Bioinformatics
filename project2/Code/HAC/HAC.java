import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Scanner;

public class HAC {
	static ArrayList<String> input = null;
	static float[][] records = null;
	static float[][] centroids = null;
	static int[] centres = null;
	static int nclusters;

	static int[][] ground;

	static ArrayList<HashSet<Integer>> clusters = new ArrayList<HashSet<Integer>>();
	static ArrayList<HashSet<Integer>> output = new ArrayList<HashSet<Integer>>();
	static ArrayList<ArrayList<HashSet<Integer>>> result = new ArrayList<ArrayList<HashSet<Integer>>>();

	public static void fillInput() {
		Scanner sc = new Scanner(System.in);
		String data = null;

		System.out.println("Enter file path");
		String file_path = sc.next();
		try {
			// file_path
			data = new String(Files.readAllBytes(Paths.get(file_path)));

		} catch (Exception e) {
			System.out.println("file was not found");
		}
		String[] genes = data.split("\n");
		input = new ArrayList(Arrays.asList(genes));

		records = new float[input.size()][input.get(0).split("\t").length];
		// centres = new int[input.size()];
		int i = 0, j = 0;
		for (String gene : input) {
			String[] values = gene.split("\t");
			for (String value : values) {
				records[i][j] = Float.parseFloat(value);
				j++;
			}
			i++;
			j = 0;
		}

		for (int k = 0; k < records.length; k++) {
			ArrayList<Integer> al = new ArrayList<>();
			al.add((int) records[k][0]);
			clusters.add(new HashSet<Integer>(al));

		}

	}

	public static double[][] fillCache() {
		double dsqrt;
		double[][] cache = new double[records.length][records.length];
		for (int i = 0; i < records.length; i++) {
			for (int j = 0; j < i; j++) {
				float distance = 0f;
				for (int k = 0; k < input.get(0).split("\t").length - 2; k++)
					distance += Math.pow(records[i][k + 2] - records[j][k + 2],
							2);

				dsqrt = Math.sqrt(distance);
				cache[i][j] = dsqrt;

			}
		}
		/*
		 * double [][] cache = new double[][]{{0,0,0,0,0,0}, {0.23,0,0,0,0,0},
		 * {0.22,0.15,0,0,0,0}, {0.37,0.2,0.15,0,0,0},
		 * {0.34,0.14,0.28,0.29,0,0}, {0.23,0.25,0.11,0.22,0.39,0}};
		 */
		return cache;
	}

	public static double[][] formClusters(double[][] cache) {

		int mini = 0, minj = 0;
		double[][] temp_cache = new double[cache.length - 1][cache.length];
		double min_distance = Double.MAX_VALUE;

		for (int i = 0; i < cache.length; i++) {
			for (int j = 0; j < i; j++) {
				if (cache[i][j] == -1)
					continue;
				if (min_distance > cache[i][j]) {
					mini = i;
					minj = j;
					min_distance = cache[i][j];
				}
			}
		}

		for (int i = minj + 1; i < cache.length; i++) {

			cache[i][minj] = Math.min(cache[i][minj], cache[mini][i]);
			// cache[mini][j]=-1;
		}

		for (int j = 0; j < cache[0].length; j++) {

			cache[minj][j] = Math.min(cache[minj][j], cache[mini][j]);
			cache[mini][j] = -1;
		}

		boolean increment = false;
		int ti = 0, tj = 0;
		for (int i = 0; i < cache.length; i++) {
			increment = false;
			for (int j = 0; j <= i; j++) {
				if (cache[i][j] != -1) {
					temp_cache[ti][tj] = cache[i][j];

					increment = true;
					tj++;
				}

			}
			tj = 0;
			if (increment)
				ti++;
		}

		clusters.get(minj).addAll(clusters.get(mini));
		clusters.remove(mini);

		for (HashSet<Integer> al : clusters)
			output.add((HashSet<Integer>) al.clone());
		result.add((ArrayList<HashSet<Integer>>) output.clone());
		System.out.println("mini=" + mini + " minj=" + minj);
		output.clear();
		return temp_cache;
	}

	public static void findJaccard() {
		int n = 0, d = 0;
		float jaccardCoef;
		ground = new int[input.size()][input.size()];
		// cluster = new int[input.size()][input.size()];
		centres = new int[input.size()];
		int index = 0;
		for (int i = 0; i < input.size(); i++) {
			index = 0;

			for (HashSet<Integer> hs : clusters) {
				if (hs.contains((int) records[i][0]))
					centres[i] = index;

				index++;
			}

		}
		for (int i = 0; i < input.size(); i++) {
			for (int j = 0; j <= i; j++) {
				if (centres[i] == centres[j] && records[i][1] == records[j][1]) {
					n += 1;
				} else if (centres[i] == centres[j]
						|| records[i][1] == records[j][1])
					d += 1;

			}
		}

		jaccardCoef = (float) n / (n + d);
		System.out.println("Jaccard coefficient is = " + jaccardCoef);
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		fillInput();
		Scanner sc = new Scanner(System.in);
		double[][] cache = fillCache();
		// boolean start = true;
		double[][] temp_cache = cache;
		System.out.println("How many clusters you want to form?");
		int clnum = sc.nextInt();
		while (clusters.size() != clnum) {
			double[][] cache_ip = temp_cache;
			temp_cache = formClusters(cache_ip);
		}
		findJaccard();
	}

}
