# Name:  Joshua Jerry Selorm Yegbe

# Index Number: 10211100403


import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

def neural_network_page():
    # Apply custom styling
    st.markdown("""
    <style>
    .nn-header {
        font-size: 2rem;
        color: #1E88E5;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-align: center;
    }
    .upload-section {
        padding: 2rem;
        border: 2px dashed #1E88E5;
        border-radius: 16px;
        text-align: center;
        background-color: rgba(255, 255, 255, 0.9);
        margin-bottom: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: rgba(255, 255, 255, 0.95);
        margin-bottom: 1.5rem;
        border: 1px solid #e1e4e8;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
        border: 1px solid #e1e4e8;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        font-weight: 500;
    }
    .model-architecture {
        background-color: rgba(240, 248, 255, 0.9);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with improved contrast
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;gap:15px;margin-bottom:1.5rem">
        <i class="material-icons" style="font-size:2.5rem;color:#1E88E5">psychology</i>
        <h1 class="nn-header">Neural Network Explorer</h1>
    </div>
    <p style="text-align:center;color:#333;margin-bottom:2rem;font-size:1.1rem">
        Build and train neural networks for classification tasks
    </p>
    """, unsafe_allow_html=True)

    # Dataset selection
    st.markdown('<div class="card">', unsafe_allow_html=True)
    dataset_option = st.radio("Choose Dataset", ["MNIST (Default)", "Upload CSV"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    X, y, target_col, is_mnist = None, None, None, False
    X_test = None  # Initialize X_test for MNIST case

    if dataset_option == "MNIST (Default)":
        is_mnist = True
        with st.spinner("Loading MNIST dataset..."):
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            X = X_train.reshape(-1, 28*28) / 255.0
            y = tf.keras.utils.to_categorical(y_train, 10)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>MNIST Dataset Preview</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(X_train[0], caption=f"Label: {y_train[0]}", width=100)
            with col2:
                st.image(X_train[1], caption=f"Label: {y_train[1]}", width=100)
            with col3:
                st.image(X_train[2], caption=f"Label: {y_train[2]}", width=100)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{X.shape[0]}</div>
                <div class="metric-label">Training Samples</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")
        
        if not uploaded_file:
            st.markdown("""
            <div style="text-align:center">
                <i class="material-icons" style="font-size:3rem;color:#1E88E5;margin-bottom:1rem">cloud_upload</i>
                <h3 style="color:#1E88E5;margin-bottom:0.5rem">Drag and drop your CSV file</h3>
                <p style="color:#333">or click to browse files</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Processing dataset..."):
                data = pd.read_csv(uploaded_file)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Dataset Preview</h3>", unsafe_allow_html=True)
                st.dataframe(data.head(), use_container_width=True)
                
                target_col = st.selectbox("Select Target Column", data.columns)
                X = data.drop(columns=[target_col]).values
                y = pd.get_dummies(data[target_col]).values
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{X.shape[0]}</div>
                    <div class="metric-label">Samples</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{X.shape[1]}</div>
                    <div class="metric-label">Features</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{y.shape[1]}</div>
                    <div class="metric-label">Classes</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if X is not None:
        # Model configuration
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Model Configuration</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", 1, 20, 10)
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, step=0.0001, format="%.4f")
        with col2:
            batch_size = st.slider("Batch Size", 16, 256, 32)
            validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2, step=0.05)
        
        st.markdown("<h4 style='color:#1E88E5;margin-top:1rem;margin-bottom:1rem'>Model Architecture</h4>", unsafe_allow_html=True)
        st.markdown('<div class="model-architecture">', unsafe_allow_html=True)
        st.code("""
        Sequential([
            Dense(128, activation='relu', input_shape=({input_dim},)),
            Dense(64, activation='relu'),
            Dense({output_dim}, activation='softmax')
        ])
        """.format(input_dim=X.shape[1], output_dim=y.shape[1]), language="python")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Initialize model in session state
        if 'model' not in st.session_state:
            st.session_state.model = None
        
        if st.button("Train Model", use_container_width=True, type="primary"):
            with st.spinner("Building and training model..."):
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(y.shape[1], activation='softmax')
                ])
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                history = model.fit(
                    X, y, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    validation_split=validation_split,
                    verbose=0
                )
                
                # Store model and history in session state
                st.session_state.model = model
                st.session_state.history = history
                
                # Training metrics
                final_train_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                final_train_loss = history.history['loss'][-1]
                final_val_loss = history.history['val_loss'][-1]
                
                # Display metrics
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Training Results</h3>", unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
创新
                        <div class="metric-value">{final_train_acc:.2f}</div>
                        <div class="metric-label">Train Accuracy</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{final_val_acc:.2f}</div>
                        <div class="metric-label">Val Accuracy</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{final_train_loss:.4f}</div>
                        <div class="metric-label">Train Loss</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{final_val_loss:.4f}</div>
                        <div class="metric-label">Val Loss</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Training history plots
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, epochs+1)),
                    y=history.history['accuracy'],
                    name='Training Accuracy',
                    line=dict(color='#1E88E5', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(1, epochs+1)),
                    y=history.history['val_accuracy'],
                    name='Validation Accuracy',
                    line=dict(color='#0D47A1', width=3, dash='dash')
                ))
                fig.update_layout(
                    title='Training History - Accuracy',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy',
                    template='plotly_white',
                    height=400,
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
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, epochs+1)),
                    y=history.history['loss'],
                    name='Training Loss',
                    line=dict(color='#FF5722', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(1, epochs+1)),
                    y=history.history['val_loss'],
                    name='Validation Loss',
                    line=dict(color='#E64A19', width=3, dash='dash')
                ))
                fig.update_layout(
                    title='Training History - Loss',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    template='plotly_white',
                    height=400,
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
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Prediction section (always visible if model exists)
        if st.session_state.model is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Model Prediction</h3>", unsafe_allow_html=True)
            
            if is_mnist:
                sample_idx = st.slider("Select Test Sample", 0, len(X_test)-1, 0)
                sample = X_test[sample_idx].reshape(1, -1) / 255.0
                
                if st.button("Predict Selected Sample", use_container_width=True):
                    pred = st.session_state.model.predict(sample, verbose=0)
                    predicted_class = np.argmax(pred)
                    confidence = np.max(pred)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<h4 style='color:#1E88E5;margin-bottom:1rem'>Input Image</h4>", unsafe_allow_html=True)
                        st.image(X_test[sample_idx], width=150)
                    
                    with col2:
                        st.markdown("<h4 style='color:#1E88E5;margin-bottom:1rem'>Prediction</h4>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="font-size:2rem;font-weight:bold;color:#1E88E5;margin-bottom:0.5rem">
                            {predicted_class}
                        </div>
                        <div style="color:#333">Confidence: {confidence:.2%}</div>
                        """, unsafe_allow_html=True)
                        
                        # Class probabilities
                        fig = px.bar(
                            x=list(range(10)),
                            y=pred[0],
                            labels={'x': 'Digit', 'y': 'Probability'},
                            color=pred[0],
                            color_continuous_scale='blues',
                            template='plotly_white'
                        )
                        fig.update_layout(
                            height=300,
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False,
                            xaxis=dict(tickvals=list(range(10))),  # Fixed: Added comma
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("<h4 style='color:#1E88E5;margin-bottom:1rem'>Custom Input</h4>", unsafe_allow_html=True)
                input_values = []
                for i in range(X.shape[1]):
                    input_values.append(st.number_input(f"Feature {i+1}", value=float(X[0,i])))
                
                if st.button("Predict", use_container_width=True):
                    input_array = np.array([input_values])
                    pred = st.session_state.model.predict(input_array, verbose=0)
                    predicted_class = np.argmax(pred)
                    confidence = np.max(pred)
                    
                    st.markdown(f"""
                    <div style="text-align:center;margin:1rem 0">
                        <div style="font-size:1.2rem;color:#333">Predicted Class</div>
                        <div style="font-size:2.5rem;font-weight:bold;color:#1E88E5">
                            {predicted_class}
                        </div>
                        <div style="color:#333">Confidence: {confidence:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    neural_network_page()