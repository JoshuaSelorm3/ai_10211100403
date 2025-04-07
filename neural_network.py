# neural_network.py
# Name: [Your Name]
# Index Number: [Your Index Number]

import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def neural_network_page():
    st.title("Neural Network Problem")
    st.write("Upload a CSV file for classification or use MNIST dataset.")

    # Dataset choice
    dataset_option = st.radio("Choose Dataset", ["MNIST (Default)", "Upload CSV"])
    
    if dataset_option == "MNIST (Default)":
        # Load MNIST
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        X = x_train.reshape(-1, 28*28) / 255.0
        y = tf.keras.utils.to_categorical(y_train, 10)
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:", data.head())
            target_col = st.selectbox("Select Target Column", data.columns)
            X = data.drop(columns=[target_col]).values
            y = pd.get_dummies(data[target_col]).values  # One-hot encode
        else:
            st.warning("Please upload a CSV file.")
            return

    # Hyperparameters
    epochs = st.slider("Epochs", 1, 10, 5)
    learning_rate = st.number_input("Learning Rate", 0.001, 0.1, 0.01)

    # Build and train model
    if st.button("Train Model"):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(y.shape[1], activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        history = model.fit(X, y, epochs=epochs, validation_split=0.2, batch_size=32, verbose=0)
        
        # Visualize training progress
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.legend()
        st.pyplot(fig)

    # Custom Prediction
    st.subheader("Make a Prediction")
    if dataset_option == "MNIST (Default)":
        sample_idx = st.slider("Select Test Sample", 0, len(x_test)-1, 0)
        sample = x_test[sample_idx].reshape(1, -1) / 255.0
        if st.button("Predict"):
            pred = model.predict(sample)
            st.write(f"Predicted Class: {np.argmax(pred)}")
            st.image(x_test[sample_idx], width=100)
    else:
        custom_input = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(X.shape[1])]
        if st.button("Predict"):
            input_array = np.array([custom_input])
            pred = model.predict(input_array)
            st.write(f"Predicted Class: {np.argmax(pred)}")

if __name__ == "__main__":
    neural_network_page()