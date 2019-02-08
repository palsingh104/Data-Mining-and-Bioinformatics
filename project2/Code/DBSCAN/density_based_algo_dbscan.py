import sys
from utilities import utilities

def main():
    file_path = "data/" + sys.argv[1]
    util = Utilities()
    # Load the data from file
    data = pd.read_csv(file_path, sep="\t", header=None)
    num_cols = len(data.columns)
    print("num of columns: ", num_cols)

    # split the data in features and diseases 
    attributes = data.iloc[:,2:num_cols].values
    # print(attributes)
    print(attributes.shape)

    result = util.compute_dbscan(attributes, 1, 5)

    util.plot_PCA(data.values, result, "DBSCAN for " + sys.argv[1])


if __name__ == '__main__':
    main()