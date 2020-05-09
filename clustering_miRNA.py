import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
filepath = "/Users/macbook/Downloads/miRNA_byBMI_forR.txt"


data = pd.read_csv(filepath, sep='\t')
data=data.set_index('Geneid')
print(data)
transposed_data = data.T

print(transposed_data)
print(transposed_data.columns.values)

kmeans = KMeans(n_clusters=2).fit(transposed_data)
centroids = kmeans.cluster_centers_

# Nice Pythonic way to get the indices of the points for each corresponding cluster
mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

print("in cluster 0:")
print(mydict)
#print("Centroids")
#print(centroids)