import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score, silhouette_samples  # type: ignore
from sklearn.cluster import KMeans, DBSCAN # type: ignore

def sil_dbscan(data, cols, opt_eps, opt_samp) -> plt.plot:
    x = data[[cols]].iloc[:, :].values
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(13, 5) 

    # The 1st subplot is the Silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]

    ax1.set_xlim([-0.1, 1])

    # Initialize the clusterer with DBSCAN clustering
    clusterer = DBSCAN(eps=opt_eps, min_samples=opt_samp, metric="euclidean")
    clusterer_labels = clusterer.fit_predict(x)
    n_clusters = len(np.unique(clusterer_labels))

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])  

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = silhouette_score(x, clusterer_labels)
    print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(x, clusterer_labels)

    y_lower = 10
    for i in range(n_clusters):

        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[clusterer_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(clusterer_labels.astype(float) / n_clusters)
        ax2.scatter(
                x= x[:, 0],
                y= x[:, 1],
                marker=".",
                s=30,
                lw=0,
                alpha=0.7,
                c=colors,
                edgecolor="k",
            )

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

def kmeans_plot(data, n, cols) -> plt.plot:
    """
        cluster_plot returns a 2D plot of clusters based on a KMeans model.
        args:
            data: dataset
            n: number of clusters
        returns:
            plt.imshow()
    """
    # Adjusting model.
    kmeans = KMeans(
            n_clusters=n,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
            tol=0.0001,
            algorithm="elkan",
        )
    # Training model.
    x1 = data[cols].iloc[:, :].values
    kmeans = kmeans.fit(x1)
    labels1 = kmeans.labels_
    centroids1 = kmeans.cluster_centers_

    # Plotting clusters
    h = 0.02
    x_min, x_max = x1[:, 0].min() - 1, x1[:, 0].max() + 1
    y_min, y_max = x1[:, 1].min() - 1, x1[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(1, figsize=(10, 6))
    plt.clf()
    Z = Z.reshape(xx.shape)
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Pastel2,
        aspect="auto",
        origin="lower",
    )
    plt.scatter(data=data, x=cols[0], y=cols[1], c=labels1, s=70)
    plt.scatter(x=centroids1[:, 0], y=centroids1[:, 1], marker="D", s=100, color="r")

def elbow_plot(data, n, cols):
    x1 = data[cols].iloc[:, :].values
    inertia = []
    for i in range(1,n):
        algorithm = KMeans(
            n_clusters = i, 
            init="k-means++", 
            n_init=10, 
            max_iter=300, 
            tol=0.0001, 
            random_state=111, 
            algorithm="elkan" # Search for this in documentation.
        )
        algorithm.fit(x1)
        wcss = algorithm.inertia_
        inertia.append(wcss)
    
    plt.figure(1 , figsize = (15 ,6))
    plt.plot(np.arange(1 , n) , inertia , 'o')
    plt.plot(np.arange(1 , n) , inertia , '-' , alpha = 0.5)
    plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
    plt.show()

def silhouette_method(data, cols, n=15):
    # Silhouette method
    X = data[cols].iloc[:, :].values
    silhouette_scores = []
    K = range(2,n)
    for _ in K:
        km = KMeans(n_clusters=_)
        km = km.fit(X)
        y_pred = km.predict(X)
        silhouette_scores.append(silhouette_score(X, y_pred))

    plt.plot(K,silhouette_scores, "rx-")
    plt.ylabel(r"Silhouette score")
    plt.xlabel("Number of clusters")
    plt.title("Silhouette method for finding optimal clusters")
    plt.show()
