# Name: Joshua Jerry Selorm Yegbe
# Index Number: 10211100403

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io

def clustering_page():
    st.title("Clustering Problem")
    st.write("Upload a CSV file for clustering.")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="clustering_upload")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", data.head())

        # Select number of clusters
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)

        # Preprocessing: Drop NaN and select numeric columns
        data = data.dropna()
        X = data.select_dtypes(include=[np.number])

        # K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        data["Cluster"] = clusters

        # Visualization (assuming 2D for simplicity)
        if X.shape[1] >= 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap="viridis")
            centers = kmeans.cluster_centers_
            ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=200, label="Centroids")
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            ax.legend()
            st.pyplot(fig)
        else:
            st.write("Visualization requires at least 2 numeric features.")

        # Download clustered data
        csv = data.to_csv(index=False)
        st.download_button("Download Clustered Data", csv, "clustered_data.csv", "text/csv")

if __name__ == "__main__":
    clustering_page()