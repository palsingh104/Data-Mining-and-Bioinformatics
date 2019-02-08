import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class k_means {
	static ArrayList<String> input = null;
	static float[][] records = null;
	static float[][] centroids = null;
	static int[] centres = null;
	static int clusters;
	static int[][] ground;//
	static int[][] cluster; //

	public static void fillInput() {
		Scanner sc = new Scanner(System.in);
		String data = null;

		System.out.println("Enter file path");
		String file_path = sc.next();
		try {
			data = new String(Files.readAllBytes(Paths.get(file_path)));
		} catch (Exception e) {
			System.out.println("file was not found");
		}
		String[] genes = data.split("\n");
		input = new ArrayList(Arrays.asList(genes));
		records = new float[input.size()][input.get(0).split("\t").length];
		centres = new int[input.size()];
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

	}

	public static void fillCentroids() {
		System.out.println("How many clusters do you want to form?");

		Scanner sc = new Scanner(System.in);
		clusters = Integer.parseInt(sc.next());

		centroids = new float[clusters][input.get(0).split("\t").length - 2];
		for (int i = 0; i < clusters; i++) {
			System.out.println("Enter mean ID " + (i + 1));
			int geneID = Integer.parseInt(sc.next());
			for (int j = 0; j < input.get(0).split("\t").length - 2; j++)
				centroids[i][j] = records[geneID - 1][j + 2];
		}
	}

	public static void iterate() {
		boolean matched = false;
		System.out.println("How many iterations do you want?");
		Scanner sc = new Scanner(System.in);
		int iter = sc.nextInt();
		int counter = 0;
		while (counter < iter || matched) {

			for (int i = 0; i < input.size(); i++) {
				float prev_distance = 0.0f;
				int centroid = centres[i];
				for (int j = 0; j < input.get(0).split("\t").length - 2; j++) {
					prev_distance += Math.pow(centroids[centroid][j]
							- records[i][j + 2], 2);
				}
				for (int k = 0; k < clusters; k++) {
					float current_distance = 0;
					for (int j = 0; j < input.get(0).split("\t").length - 2; j++)
						current_distance += Math.pow(centroids[k][j]
								- records[i][j + 2], 2);
					if (current_distance < prev_distance) {
						prev_distance = current_distance;
						centres[i] = k;
						matched = false;
					}
				}

			}

			adjustCentroids();
			counter++;

		}
	}

	public static float[] getRatio(int cluster_index, int attribute) {
		float total = (float) 0.0;
		float freq = 0;
		float[] op = new float[2];
		for (int i = 0; i < input.size(); i++) {
			if (centres[i] == cluster_index) {
				freq++;
				total += records[i][attribute + 2];
			}
		}
		op[0] = total;
		op[1] = freq;
		return op;
	}

	public static void adjustCentroids() {
		for (int i = 0; i < clusters; i++) {
			for (int j = 0; j < input.get(0).split("\t").length - 2; j++) {
				float[] freqTotal = getRatio(i, j);
				if (freqTotal[1] != 0)
					centroids[i][j] = freqTotal[0] / freqTotal[1];
			}
		}
	}

	public static void findJaccard() {

		int n = 0, d = 0;
		float jaccardCoef;

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
		fillCentroids();
		iterate();
		findJaccard();

		// fillDistances();
	}

}
