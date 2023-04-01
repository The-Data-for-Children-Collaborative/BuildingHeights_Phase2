import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os


def find_elbow_point_k(X, max_k=10):
    "INPUT  :  X matrix with data , number of clusters "
    "OUTPUT : the elbow point (=optimal number of clusters) after K-means algorithm "

    wss = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        wss.append(kmeans.inertia_)

    # Calculate the difference between successive WSS values
    wss_diff = np.diff(wss)

    # Calculate the second difference
    wss_diff2 = np.diff(wss_diff)

    # Find the index corresponding to the knee point on the second difference curve
    knee = np.argmax(wss_diff2) + 2
    return knee


def save_clusters_to_csv(filenames, cluster_labels, optimal_k, output_dir):
    "    INPUT: "
    "    filenames : the list of names of the initial dataset data "
    "    cluster_labels : generated after the k_means algorithm  "
    "    optimal_k : generated after the k_means algorithm (with elbow method) "
    "                     "
    "    OUTPUT: files cluster_{i}.csv containing the list of files in each cluster, one per each final cluster"

    # Create a directory for saving the clusters
#    if not os.path.exists(output_dir + '/cluster_dir_{optimal_k}'):  # not sure about the syntax!!!!
#        os.makedirs(f"output_dir/cluster_dir_{optimal_k}", exist_ok=True)
    # Save each cluster to a separate CSV file
    for i in range(optimal_k):
        cluster_indices = (cluster_labels == i)
        cluster_filenames = filenames[cluster_indices]
        cluster_data = pd.DataFrame({"filename": cluster_filenames})
        output_path = os.path.join(output_dir, f"cluster_{i}.csv") #new
        #cluster_data.to_csv(f"~/Desktop/photos_valentina/cluster_dir_{optimal_k}/cluster_{i}.csv", index=False)
        cluster_data.to_csv(output_path, index=False)

def generate_summary_file(clustered_data, cluster_labels, output_dir):
    # Get the number of files in each cluster
    cluster_counts = clustered_data.groupby("cluster_label").size().reset_index(name="count")
    #print(cluster_counts)

    # Calculate useful statistics about the files in each cluster
    cluster_stats = clustered_data.groupby("cluster_label").describe()
    #print(cluster_stats)

    # Write the summary to a file
    filename="summary.txt"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        for i in range(len(cluster_counts)):
            cluster_num = cluster_counts.loc[i, "cluster_label"]
            count = cluster_counts.loc[i, "count"]
            stats = cluster_stats.loc[cluster_num]

            f.write(f"Cluster {cluster_num}\n")
            f.write(f"Number of files: {count}\n")
            f.write(f"Statistics:\n{stats}\n\n")
