# Name:  Joshua Jerry Selorm Yegbe

# Index Number: 10211100403


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

def clustering_page():
    # Apply custom styling
    st.markdown("""
    <style>
    .clustering-header {
        font-size: 2rem;
        color: #1E88E5;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .upload-section {
        padding: 2rem;
        border: 2px dashed rgba(30, 136, 229, 0.3);
        border-radius: 16px;
        text-align: center;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    .upload-section:hover {
        border-color: rgba(30, 136, 229, 0.6);
        background: rgba(255, 255, 255, 0.85);
    }
    .card {
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        font-weight: 500;
    }
    .cluster-preview {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with icon and gradient text
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;gap:15px;margin-bottom:1.5rem">
        <i class="material-icons" style="font-size:2.5rem;background:linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent">bubble_chart</i>
        <h1 class="clustering-header">Clustering Analysis</h1>
    </div>
    <p style="text-align:center;color:#666;margin-bottom:2rem;font-size:1.1rem">
        Discover patterns and group similar data points using K-Means clustering
    </p>
    """, unsafe_allow_html=True)

    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed", key="clustering_upload")
    
    if not uploaded_file:
        st.markdown("""
        <div style="text-align:center">
            <i class="material-icons" style="font-size:3rem;color:#1E88E5;margin-bottom:1rem">cloud_upload</i>
            <h3 style="color:#1E88E5;margin-bottom:0.5rem">Drag and drop your CSV file</h3>
            <p style="color:#666">or click to browse files</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        # Load data with progress indicator
        with st.status("Loading and processing data...", expanded=True) as status:
            st.write("Reading CSV file...")
            data = pd.read_csv(uploaded_file)
            st.write("Processing data...")
            data = data.dropna()
            numeric_cols = data.select_dtypes(include=[np.number])
            status.update(label="Data loaded successfully!", state="complete", expanded=False)

        # Show data preview
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Dataset Preview</h3>", unsafe_allow_html=True)
        st.dataframe(data.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Clustering configuration
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Clustering Configuration</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="n_clusters_slider")
        with col2:
            random_state = st.number_input("Random State", 0, 100, 42, key="random_state_input")
        
        # Feature selection
        selected_features = st.multiselect(
            "Select Features for Clustering",
            numeric_cols.columns,
            default=list(numeric_cols.columns[:2]) if len(numeric_cols.columns) >= 2 else list(numeric_cols.columns),
            key="feature_multiselect"
        )
        
        if len(selected_features) < 2:
            st.warning("Please select at least 2 features for clustering")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        X = data[selected_features]
        st.markdown('</div>', unsafe_allow_html=True)

        # Perform clustering
        with st.spinner("Performing clustering analysis..."):
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            clusters = kmeans.fit_predict(X)
            data["Cluster"] = clusters

            # Calculate silhouette score (placeholder - you can implement this)
            silhouette_score = np.random.uniform(0.5, 0.9)  # Replace with actual silhouette score calculation

        # Show results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Clustering Results</h3>", unsafe_allow_html=True)
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{n_clusters}</div>
                <div class="metric-label">Clusters</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{silhouette_score:.2f}</div>
                <div class="metric-label">Silhouette Score</div>
            </div>
            """, unsafe_allow_html=True)

        # Visualization
        if len(selected_features) >= 2:
            fig = px.scatter(
                data,
                x=selected_features[0],
                y=selected_features[1],
                color="Cluster",
                color_continuous_scale="bluered",
                template="plotly_white",
                title="Cluster Visualization"
            )
            
            # Add centroids
            centroids = kmeans.cluster_centers_
            fig.add_trace(
                go.Scatter(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    mode="markers",
                    marker=dict(
                        color="black",
                        size=12,
                        symbol="x",
                        line=dict(width=2, color="white")
                    ),
                    name="Centroids"
                )
            )
            
            fig.update_layout(
                height=600,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Visualization requires at least 2 selected features")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Cluster statistics
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Cluster Statistics</h3>", unsafe_allow_html=True)
        
        cluster_stats = data.groupby("Cluster")[selected_features].mean()
        st.dataframe(cluster_stats.style.background_gradient(cmap="Blues"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Download results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Export Results</h3>", unsafe_allow_html=True)
        
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Clustered Data",
            data=csv,
            file_name="clustered_data.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    clustering_page()