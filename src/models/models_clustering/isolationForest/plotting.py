import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def plot_clusters_3d(stations_df):
    """
    Plots meteorological station clusters in a 3D scatter plot using Plotly.

    Parameters:
    stations_df (pd.DataFrame): The dataset containing station coordinates and cluster labels.

    The function creates a 3D scatter plot with:
    - Latitude (Y-axis), Longitude (X-axis), and Altitude (Z-axis).
    - Color-coded clusters.
    - Hover info displaying station name and altitude.
    """
    fig = px.scatter_3d(
        stations_df,
        x="Longitude",
        y="Latitude",
        z="Altitude",
        color="cluster",
        title="3D Visualization of Station Clusters",
        labels={"cluster": "Cluster"},
        hover_data=["Nom"],
    )

    fig.update_traces(
        marker=dict(size=3, opacity=0.8)
    )  # Adjust marker size & transparency
    fig.update_layout(legend_title_text="Cluster")

    fig.show()


def plot_elbow_and_silhouette(data, max_k=10):
    """
    Plots the Elbow Method and Silhouette Score to determine the optimal number of clusters for K-Means.

    Parameters:
    data (pd.DataFrame): The dataset containing features for clustering.
    max_k (int): The maximum number of clusters to evaluate. Default is 10.

    The function generates a plot with:
    - The elbow method (inertia) to assess within-cluster variance.
    - The silhouette score to measure how well-separated the clusters are.
    """
    inertia = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(k_values), y=inertia, mode="lines+markers", name="Elbow Method"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(k_values),
            y=silhouette_scores,
            mode="lines+markers",
            name="Silhouette Score",
        )
    )
    fig.update_layout(
        title="Elbow Method and Silhouette Score",
        xaxis_title="Number of Clusters",
        yaxis_title="Score",
    )
    fig.show()
