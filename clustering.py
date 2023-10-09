import math
from collections import defaultdict
from turtle import distance

# import dtw
import joblib
import plotly.express as px
from minisom import MiniSom
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import seaborn as sns
# from dtaidistance import dtw
from tslearn.metrics import dtw
# from dtw import dtw

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tslearn.barycenters import dtw_barycenter_averaging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans, silhouette_score


class Normalizer:
    def __int__(self, scaler, mySeries):
        self.scaler = scaler
        self.mySeries = mySeries

    def normalize_data(self):
        """
          This functions normalises the data based on the input scaler method
          and save the results both into  mySeries list,
          and into another list named Series_arrats as arrays
          Arguments:
            scaler: normalization method
            mySeries: list of df data
            field: Column to apply the normalisation
          Returns:
            mySeries: list of series
            Series_arrays: list of arrays
        """
        Series_arrays = {}
        namesofMySeries = joblib.load("joblib_objects/namesofMySeries.joblib")
        for i in range(len(self.mySeries)):
            if self.scaler == 'minmax':
                norm_scaler = MinMaxScaler()
            elif self.scaler == 'standard':
                norm_scaler = StandardScaler()
            elif self.scaler == 'robust':
                norm_scaler = RobustScaler()
            else:
                norm_scaler = MinMaxScaler()
            key = namesofMySeries[i]
            Series_arrays[key] = self.mySeries[i]
            self.mySeries[i] = norm_scaler.fit_transform(self.mySeries[i][['Temperature']])
            Series_arrays[key]['Normalised'] = self.mySeries[i]
            self.mySeries[i] = self.mySeries[i].reshape(len(self.mySeries[i]))

        return self.mySeries, Series_arrays


class SomClustering:
    def __init__(self, type):
        self.type = type

    def cluster_distribution(self, win_map, som, som_x, som_y, mySeries, namesofMySeries, scaler, method):
        cluster_c = []
        cluster_n = []
        for x in range(som_x):
            for y in range(som_y):
                cluster = (x, y)
                if cluster in win_map.keys():
                    cluster_c.append(len(win_map[cluster]))
                else:
                    cluster_c.append(0)
                cluster_number = x * som_y + y + 1
                cluster_n.append(f"Cluster {cluster_number}")

        plt.figure(figsize=(25, 5))
        plt.title("Cluster Distribution for SOM")
        plt.bar(cluster_n, cluster_c)
        plt.show()
        plt.savefig(f'som/som_plots/{scaler}_{method}_cl_db_{self.type}_6_clusters.jpg')
        # Cluster mapping
        cluster_map = []
        for idx in range(len(mySeries)):
            winner_node = som.winner(mySeries[idx])
            cluster_map.append((namesofMySeries[idx], f"Cluster {winner_node[0] * som_y + winner_node[1] + 1}"))

        clusters_map = pd.DataFrame(cluster_map, columns=["Series", "Cluster"]).sort_values(by="Cluster").set_index(
            "Series")
        print(clusters_map)
        return clusters_map

    def plot_som_series_averaged_center(self, som_x, som_y, win_map, method_shape, scaler):
        """
          Args:
            som_x:
        """
        fig, axs = plt.subplots(som_x, som_y, figsize=(25, 25))
        fig.suptitle('Clusters')
        for x in range(som_x):
            for y in range(som_y):
                cluster = (x, y)
                if cluster in win_map.keys():
                    for series in win_map[cluster]:
                        axs[cluster].plot(series, c="gray", alpha=0.5)
                    if method_shape == 'avg':
                        axs[cluster].plot(np.average(np.vstack(win_map[cluster]), axis=0), c="red")
                    elif method_shape == 'dba':
                        axs[cluster].plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])), c="red")
                cluster_number = x * som_y + y + 1
                axs[cluster].set_title(f"Cluster {cluster_number}")

        plt.show()
        plt.savefig(f'som/som_plots/{scaler}_{method_shape}_{self.type}_{som_x}{som_y}_combo_plot.jpg')

    def find_best_cluster_number(self):
        if self.type == "monthly":
            mySeries = joblib.load("joblib_objects/monthlySeries.joblib")
        if self.type == "yearly":
            mySeries = joblib.load("joblib_objects/yearlySeries.joblib")
        else:
            mySeries = joblib.load("joblib_objects/mySeries.joblib")

        norm = Normalizer()
        norm.__int__(scaler="minmax", mySeries=mySeries)
        norm_ma_Series, Series_arrays = norm.normalize_data()
        X = np.vstack(norm_ma_Series)

        # Run SOM with different cluster numbers
        silhouette_scores = []
        for i in range(2, 11):
            som = MiniSom(10, 10, 3653, sigma=1.0, learning_rate=0.5)
            som.pca_weights_init(X)
            som.train_random(X, 100)
            labels = som.labels_map(data=X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)

        # Print silhouette scores for each cluster number
        for i in range(len(silhouette_scores)):
            print(f"Number of clusters: {i + 2}, silhouette score: {silhouette_scores[i]}")

    def som_tuning(self):
        namesofMySeries = joblib.load("joblib_objects/namesofMySeries.joblib")
        som_tuning_results = pd.DataFrame(
            columns=["shape_method", "epochs", "quantization_error", "cluster_distribution"])
        shape_methods = ['cosine', 'euclidean', 'manhattan']
        epochs = [10, 50, 100, 200]

        counter = 0
        for method in shape_methods:
            for epoch in epochs:
                somxy = joblib.load(f"som/somxy_best_{self.type}.joblib")
                print(f'SOMXY FOR {self.type} data are: {somxy}')
                som_x, som_y = somxy[0], somxy[1]
                print(f"Calculating per: {counter}/30")
                counter += 1
                print(f'METHOD {method} EPOCHS {epoch} OF TYPE {self.type}')
                if self.type == "monthly":
                    mySeries = joblib.load("joblib_objects/monthlySeries.joblib")
                if self.type == "yearly":
                    mySeries = joblib.load("joblib_objects/yearlySeries.joblib")
                else:
                    mySeries = joblib.load("joblib_objects/mySeries.joblib")
                norm = Normalizer()
                norm.__int__(scaler="minmax", mySeries=mySeries)
                norm_ma_Series, Series_arrays = norm.normalize_data()

                som = MiniSom(som_x, som_y, len(norm_ma_Series[0]), sigma=0.3,
                              learning_rate=0.1, activation_distance=method)

                som.random_weights_init(norm_ma_Series)
                som.train(norm_ma_Series, epoch)
                win_map = som.win_map(norm_ma_Series)
                labels = [som.winner(x) for x in norm_ma_Series]
                labels = [t[1] for t in labels]
                silhouette_avg = silhouette_score(norm_ma_Series, labels)

                q_error = som.quantization_error(norm_ma_Series)
                # #
                # # Assign cluster labels to data points
                # n_rows, n_cols = som.get_weights().shape[:2]
                # labels = np.zeros(len(norm_ma_Series))
                # for i, x in enumerate(norm_ma_Series):
                #     winner = som.winner(x)
                #     # Convert winner indices to a single label
                #     label = winner[0] * n_cols + winner[1]
                #     labels[i] = label
                #
                # # Compute pairwise DTW distances within each cluster
                # cluster_dtw_distances = [[] for _ in range(n_rows * n_cols)]
                # for i in range(len(norm_ma_Series)):
                #     for j in range(i + 1, len(norm_ma_Series)):
                #         if labels[i] == labels[j]:
                #             dist = dtw(norm_ma_Series[i], norm_ma_Series[j])
                #             cluster_dtw_distances[int(labels[i])].append(dist)
                #
                # # Compute average DTW distance within each cluster
                # avg_cluster_distances = []
                # for i in range(len(cluster_dtw_distances)):
                #     avg_cluster_distances.append(np.mean(cluster_dtw_distances[i]))
                #
                # print("Average DTW distance within clusters:", avg_cluster_distances)
                #

                # self.plot_som_series_averaged_center(som_x, som_y, win_map, method, scaler=scaler)
                distance_map = som.distance_map()
                plt.imshow(distance_map, cmap='gray_r')
                plt.show()
                clusters_map = self.cluster_distribution(win_map=win_map, som=som, som_x=som_x, som_y=som_y,
                                                         mySeries=norm_ma_Series,
                                                         namesofMySeries=namesofMySeries, scaler="minmax",
                                                         method=method)
                new_row = {
                    "shape_method": method,
                    "epochs": epoch,
                    "quantization_error": q_error,
                    "silhouete_score": silhouette_avg,
                    "cluster_distribution": clusters_map,
                }
                som_tuning_results = som_tuning_results.append(new_row, ignore_index=True)
        som_tuning_results.to_csv(f'som/som_tuning_results_{self.type}_som_xy_combos.csv', index=False)

    def pca_for_som(self):
        if self.type == "monthly":
            mySeries = joblib.load("joblib_objects/monthlySeries.joblib")
        if self.type == "yearly":
            mySeries = joblib.load("joblib_objects/yearlySeries.joblib")
        else:
            mySeries = joblib.load("joblib_objects/mySeries.joblib")
        norm = Normalizer()
        norm.__int__(scaler="minmax", mySeries=mySeries)
        norm_ma_Series, Series_arrays = norm.normalize_data()

        time_series_array = np.array(norm_ma_Series)

        # Reshape the array to have shape (num_time_series, num_time_points)
        num_time_series, num_time_points = time_series_array.shape
        time_series_array = time_series_array.reshape(num_time_series, num_time_points)

        # Perform PCA on the time series
        pca = PCA()
        pca.fit(time_series_array)

        # Calculate the cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Estimate the effective dimensionality
        effective_dimensionality = np.argmax(cumulative_variance > 0.95) + 1

        # Select an appropriate SOM grid size
        som_x = int(np.sqrt(effective_dimensionality))
        som_y = int(np.ceil(effective_dimensionality / som_x))

        print("Effective dimensionality:", effective_dimensionality)
        print("SOM grid size:", som_x, "x", som_y)
        return (som_x, som_y)


class Kmeans():
    def __init__(self, timebased):
        self.timebased = timebased

    def find_best_silhouete(self):
        if self.timebased == "monthly":
            mySeries = joblib.load("joblib_objects/monthlySeries.joblib")
        if self.timebased == "yearly":
            mySeries = joblib.load("joblib_objects/yearlySeries.joblib")
        else:
            mySeries = joblib.load("joblib_objects/mySeries.joblib")
        norm = Normalizer()
        norm.__int__(scaler="minmax", mySeries=mySeries)
        norm_ma_Series, Series_arrays = norm.normalize_data()

        # find best k with elbow method
        k_range = range(3, 11)
        silhouette_scores = []
        X = np.vstack(norm_ma_Series)

        # find best k number with shilouete score
        for k in k_range:
            print(f"FOR K {k}")
            model = TimeSeriesKMeans(n_clusters=k, metric="dtw", n_jobs=-1, max_iter=20, n_init=10)
            model.fit(X)
            labels = model.labels_
            score = silhouette_score(X, labels, metric="euclidean")
            silhouette_scores.append(score)

        # Find the index of the maximum Silhouette score
        best_index = silhouette_scores.index(max(silhouette_scores))
        print(silhouette_scores)
        joblib.dump(silhouette_scores, f"kmeans/silhouette_scores_{self.timebased}.joblib")
        # The optimal number of clusters is k_values[best_index]
        optimal_k = k_range[best_index]
        print(f"Optimal number of clusters for {self.timebased}: ", optimal_k)
        return {"timebased": self.timebased, "silhouete_scores": silhouette_scores, "best_k_index": best_index}

    def kmeans_clustering(self):
        timebased_list = []
        sil_daily = joblib.load("kmeans/silhouette_scores_daily.joblib")
        sil_scores_df = joblib.load("kmeans/silhouette_scores_total_list.joblib")
        if self.timebased == "monthly":
            mySeries = joblib.load("joblib_objects/monthlySeries.joblib")
        if self.timebased == "yearly":
            mySeries = joblib.load("joblib_objects/yearlySeries.joblib")
        else:
            mySeries = joblib.load("joblib_objects/mySeries.joblib")

        mask = sil_scores_df['timebased'] == self.timebased
        subset = sil_scores_df[mask]
        print(f'Best k index based on silhouette score: {subset}')
        cluster_count = subset['best_k_index'].tolist()[0]

        norm = Normalizer()
        norm.__int__(scaler="minmax", mySeries=mySeries)
        norm_ma_Series, Series_arrays = norm.normalize_data()
        namesofMySeries = joblib.load("joblib_objects/namesofMySeries.joblib")

        # try different distance metrics
        distance_metrics = ["euclidean", "dtw","softdtw"]
        for metric in distance_metrics:
            print(metric)
            km = TimeSeriesKMeans(n_clusters=cluster_count, metric=metric, n_init=cluster_count)
            labels = km.fit_predict(norm_ma_Series)
            fancy_names_for_labels = [f"Cluster {label}" for label in labels]
            labels_found = pd.DataFrame(zip(namesofMySeries, fancy_names_for_labels),
                                        columns=["Series", "Cluster"]).sort_values(by="Cluster").set_index("Series")
            plot_count = math.ceil(math.sqrt(cluster_count))

            fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
            fig.suptitle(f'Clusters for {metric} and {self.timebased}')
            row_i = 0
            column_j = 0
            # For each label there is,
            # ploting every series with that label
            for label in set(labels):
                cluster = []
                for i in range(len(labels)):
                    if (labels[i] == label):
                        axs[row_i, column_j].plot(norm_ma_Series[i], c="gray", alpha=0.4)
                        cluster.append(norm_ma_Series[i])
                if len(cluster) > 0:
                    axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
                axs[row_i, column_j].set_title("Cluster " + str(row_i * 2 + column_j))
                column_j += 1
                if column_j % plot_count == 0:
                    row_i += 1
                    column_j = 0
            plt.show()
            labels = km.labels_
            centroids = km.cluster_centers_
            X = np.vstack(norm_ma_Series)
            distances = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                distances[i] = np.linalg.norm(X[i] - centroids[labels[i]])

            quantization_error = np.sum(distances ** 2)
            silhouette_avg = silhouette_score(X, labels)
            new_row = {"timebased": self.timebased, "labels_found": labels_found, "metric": metric,
                       "quantization_error": quantization_error, "silhouette_score": silhouette_avg}
            timebased_list.append(new_row)
        print(f" Timebased list for {self.timebased} data: {timebased_list}")
        return timebased_list

    def save_shilouette_score_results(self):
        silhouette_results = {
            "timebased": ["daily", "monthly", "yearly"],
            "silhouete_scores": [
                [0.1372229131884166, 0.12095421872713526, 0.18165509041650274, 0.12372353773875969, 0.09705864311828181,
                 0.06154873223242554, 0.025539522260859992, 0.015853744207920985],
                [0.1372229131884166, 0.13027462498619066, 0.10935714022540272, 0.16729299248537483, 0.0979940962666733,
                 0.06154873223242554, 0.025539522260859992, 0.015853744207920985],
                [0.3740055497106633, 0.3357265155175149, 0.10349905364797415, 0.03497453797901185, 0.14948157558002348,
                 0.016656009531640984, -0.007218012372264082, 0.011468168795233918]
            ],
            "best_k_index": [5, 6, 3]
        }
        # save results into csv
        joblib.dump(silhouette_results, f"kmeans/silhouete_scores_results.joblib")

    def save_kmeans_results(self):
        types = ["daily", "monthly", "yearly"]
        timebased_kmeans_results = []
        for type in types:
            print(f"Tuning Kmeans on {type} data")
            kmeans = Kmeans(timebased=type)
            timebased_list = kmeans.kmeans_clustering()
            timebased_kmeans_results.extend(timebased_list)
            print(f'Right now timebased kmeans results is: {timebased_kmeans_results}')
        print(f'KMEANS RESULTS DF: {timebased_kmeans_results}')
        joblib.dump(timebased_kmeans_results, "kmeans/kmeans_tuning_results.joblib")


class ClusteringEvaluation():
    def __init__(self, timebased):
        self.timebased = timebased

    def get_dtw_heatmap(self):
        namesofMySeries = joblib.load("joblib_objects/namesofMySeries.joblib")
        mySeries = joblib.load("joblib_objects/mySeries.joblib")
        if self.timebased == 'monthly':
            mySeries = joblib.load("joblib_objects/monthlySeries.joblib")
        if self.timebased == 'yearly':
            mySeries = joblib.load("joblib_objects/yearlySeries.joblib")
        if self.timebased == 'daily':
            mySeries = joblib.load("joblib_objects/mySeries.joblib")

        norm = Normalizer()
        norm.__int__(scaler="minmax", mySeries=mySeries)
        norm_ma_Series, Series_arrays = norm.normalize_data()
        ds = dtw.distance_matrix_fast(norm_ma_Series)
        df = pd.DataFrame(ds, index=namesofMySeries,
                          columns=namesofMySeries)

        ax = sns.heatmap(df, linewidth=0.5)
        plt.savefig(f"clustering_evaluation/heatmap_{self.timebased}.png")

    def get_dendrogram(self):
        if self.timebased == 'daily':
            mySeries = joblib.load("joblib_objects/mySeries.joblib")
        if self.timebased == 'monthly':
            mySeries = joblib.load("joblib_objects/monthlySeries.joblib")
        if self.timebased == 'yearly':
            mySeries = joblib.load("joblib_objects/yearlySeries.joblib")
        norm = Normalizer()
        norm.__int__(scaler="minmax", mySeries=mySeries)
        mySeries, _ = norm.normalize_data()

        ts_array = np.array(mySeries)
        nameofMySeries = joblib.load("joblib_objects/namesofMySeries.joblib")
        dist_matrix = np.zeros((len(mySeries), len(mySeries)))
        for i in range(len(mySeries)):
            for j in range(i + 1, len(mySeries)):
                dist = dtw(ts_array[i], ts_array[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # Convert the distance matrix to a condensed distance matrix
        condensed_dist = squareform(dist_matrix)

        # Compute the linkage matrix using complete linkage
        linkage_matrix = linkage(condensed_dist, method='complete')

        # Plot the dendrogram to visualize the clusters
        plt.figure(figsize=(13, 6))
        dendrogram(linkage_matrix, labels=nameofMySeries, color_threshold=0)
        plt.xticks(rotation=90)
        plt.savefig(f"clustering_evaluation/dendrogram_{self.timebased}.png")

    def try_diff_clusters(self):
        mySeries = joblib.load("joblib_objects/monthlySeries.joblib")
        num_clusters = 6
        norm = Normalizer()
        norm.__int__(scaler="minmax", mySeries=mySeries)
        ts_list, _ = norm.normalize_data()
        ts_array = np.array(ts_list)
        nameofMySeries = joblib.load("joblib_objects/namesofMySeries.joblib")

        # compute the distance matrix using DTW
        dist_matrix = np.zeros((len(mySeries), len(mySeries)))
        for i in range(len(mySeries)):
            for j in range(i + 1, len(mySeries)):
                dist = dtw(ts_array[i], ts_array[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # create a dataframe from the distance matrix
        df_dist = pd.DataFrame(dist_matrix, index=nameofMySeries, columns=nameofMySeries)

        cluster_labels = pd.cut(df_dist.mean(), bins=num_clusters, labels=list(range(num_clusters)))
        cluster_labels_df = cluster_labels.to_frame()
        cluster_labels_df.to_csv(f"clustering_evaluation/clustermap_monthly6.csv", index=True)

    def get_cluster(self):
        mySeries = joblib.load("joblib_objects/mySeries.joblib")
        if self.timebased == 'monthly':
            num_clusters = 6  # replace with the number of desired clusters
            mySeries = joblib.load("joblib_objects/monthlySeries.joblib")
        if self.timebased == 'yearly':
            num_clusters = 3  # replace with the number of desired clusters
            mySeries = joblib.load("joblib_objects/yearlySeries.joblib")
        if self.timebased == 'daily':
            num_clusters = 6  # replace with the number of desired clusters
            mySeries = joblib.load("joblib_objects/mySeries.joblib")

        norm = Normalizer()
        norm.__int__(scaler="minmax", mySeries=mySeries)
        ts_list, _ = norm.normalize_data()
        ts_array = np.array(ts_list)
        nameofMySeries = joblib.load("joblib_objects/namesofMySeries.joblib")

        # compute the distance matrix using DTW
        dist_matrix = np.zeros((len(mySeries), len(mySeries)))
        for i in range(len(mySeries)):
            for j in range(i + 1, len(mySeries)):
                dist = dtw(ts_array[i], ts_array[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # create a dataframe from the distance matrix
        df_dist = pd.DataFrame(dist_matrix, index=nameofMySeries, columns=nameofMySeries)

        # create a clustermap
        sns.clustermap(df_dist, cmap='viridis', vmin=0, vmax=1)
        # plt.savefig(f"clustering_evaluation/clustermap_{self.timebased}.png")
        # extract the cluster labels for each time series

        cluster_labels = pd.cut(df_dist.mean(), bins=num_clusters, labels=list(range(num_clusters)))
        cluster_labels_df = cluster_labels.to_frame()
        print(cluster_labels_df)
        cluster_labels_df.to_csv(f"clustering_evaluation/clustermap_{self.timebased}_6.csv", index=True)


if __name__ == '__main__':
    types = ['daily','monthly','yearly']
    for type in types:
        sil_scores = joblib.load(f'kmeans/silhouette_scores_{type}.joblib')


    times = ['daily', 'monthly', 'yearly']
    # evaluation = ClusteringEvaluation(timebased='monthly')
    # evaluation.get_cluster()
    # print(f'PCA Analysis for som clustering...')
    # for type in times:
    #     som = SomClustering(type=type)
    #     somxy = som.pca_for_som()
    #     joblib.dump(somxy, f"som/somxy_best_{type}.joblib")
    # print(f'som clustering just started..')

    for timebased in times:
        print(f'Som clustering for {timebased} data...')
        som = SomClustering(type=timebased)
        som.som_tuning()

    # sil_scores_total = pd.DataFrame(columns=['timebased', 'silhouete_scores', 'best_k_index'])
    # for timebased in times:
    #     print(f'Finding best silhouette score for {timebased} data...')
    #     kmeans = Kmeans(timebased=timebased)
    #     sil_score = kmeans.find_best_silhouete()
    #     print('new row is: ',sil_score)
    #     new_row = pd.Series(sil_score.values(), index=sil_scores_total.columns)
    #     sil_scores_total = sil_scores_total.append(new_row, ignore_index=True)
    #     print('sil_scores_total after appending row: ',sil_scores_total)
    # joblib.dump(sil_scores_total, f"kmeans/silhouette_scores_total_list.joblib")

    # kmeans_timebased_total = pd.DataFrame(columns=["timebased", "labels_found", "metric"])

    # load list and convert it into csv
    # list_kmeans = joblib.load("kmeans/kmeans_timebased_total.joblib")
    # kmeans_result_df = pd.DataFrame(columns=["timebased","labels_found", "metric"])
    # for item in list_kmeans:
    #     new_row = pd.Series(item.values(), index=kmeans_result_df.columns)
    #     kmeans_result_df = kmeans_result_df.append(new_row, ignore_index=True)
    # kmeans_result_df.to_csv("kmeans/kmeans_timebased_total.csv", index=False)
    #
    kmeans_timebased_total = []
    for timebased in times:
        print(f'Kmeans clustering for {timebased} data just started...')
        kmeans = Kmeans(timebased=timebased)
        timebased_list = kmeans.kmeans_clustering()
        print(timebased_list)
        kmeans_timebased_total.extend(timebased_list)
        # kmeans_timebased_total.append(timebased_list, ignore_index=True)
    joblib.dump(kmeans_timebased_total, "kmeans/kmeans_timebased_total_.joblib")
    kmeans_dict = pd.DataFrame(columns=['timebased', 'labels_found', 'metric', 'quantization_error','silhouette_score'])
    for row in kmeans_timebased_total:
        kmeans_dict = kmeans_dict.append(row, ignore_index=True)
    kmeans_dict.to_csv("kmeans/kmeans_timebased_total_df.csv")

    ###
