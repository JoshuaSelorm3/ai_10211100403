# AI_10211100403 - IntelliHub

## Student Information
**Name:** Joshua Jerry Selorm Yegbe  
**Index Number:** 10211100403

## Project Description
IntelliHub is a comprehensive Streamlit-based application that explores and solves diverse machine learning and AI problems. Developed for the CE4143/CS4241/IT4230 Introduction to Artificial Intelligence course at Academic City University, the application provides interactive interfaces for regression, clustering, neural networks, and a large language model implementation using Retrieval-Augmented Generation (RAG).

### Features

#### Regression Analysis
- Upload custom regression datasets (CSV)
- Interactive model configuration with adjustable test size
- Performance metrics (MAE, R² Score)
- Feature coefficient visualization
- Custom prediction generation with feature impact visualization
- Data correlations and visualization

#### Clustering Analysis
- K-Means clustering with interactive cluster selection (2-10 clusters)
- 2D/3D scatter plot visualizations
- Downloadable clustered datasets
- Cluster centroid visualization

#### Neural Network Explorer
- Support for MNIST dataset and custom CSV uploads
- Adjustable hyperparameters (epochs, learning rate, batch size)
- Real-time training progress visualization
- Custom predictions with confidence scores
- Architecture: Feedforward neural network with 128 neurons (ReLU), 64 neurons (ReLU), output layer (softmax)

#### VoteWise Analytics (LLM RAG System)
- Question-answering system for Ghana election data (1992-2020)
- RAG implementation with Gemini-Pro
- Adjustable context chunk retrieval
- Source data transparency
- Interactive visualizations of election trends
- Performance evaluation metrics

## Architecture
The application is structured with modular components:

- **app.py**: Main application entry point with navigation
- **regression.py**: Regression task implementation
- **clustering.py**: Clustering task implementation
- **neural_network.py**: Neural network task implementation
- **rag_interface.py**: LLM RAG interface
- Supporting modules:
  - **data_processor.py**: Handles data loading and preprocessing
  - **embedding.py**: Manages text embeddings and vector storage
  - **retriever.py**: Retrieves relevant data chunks for LLM queries
  - **generator.py**: Integrates the LLM for answer generation
  - **evaluation.py**: Evaluates LLM responses
  - **visualization.py**: Generates visualizations for election data

## Setup Instructions

### Local Setup
1. Clone the repository:
 git clone https://github.com/JoshuaSelorm3/ai_10211100403
 cd ai_10211100403
2. Install dependencies:
 pip install -r requirements.txt

4. Run the application:
 streamlit run app.py

### Requirements
- Python 3.8+
- streamlit
- pandas, numpy
- scikit-learn
- tensorflow
- plotly, matplotlib
- faiss-cpu
- sentence-transformers
- langchain, langchain_google_genai
- google-generative-ai

## Deployed Application
(https://ai10211100403-mb7hv99f8ex4cotctmrzfk.streamlit.app/)

## Documentation
For full documentation including:
- Detailed usage instructions
- LLM architecture and methodology
- Performance evaluation
- Model comparisons

Please refer to the complete project documentation in the repository.

## Course Information
- **Course**: CE4143/CS4241/IT4230 - Introduction to Artificial Intelligence
- **Institution**: Academic City University
- **Semester**: Second Semester, 2024/2025

## GitHub Repository
Private repository at `ai_10211100403` with collaborator access granted to godwin.danso@acity.edu.gh
