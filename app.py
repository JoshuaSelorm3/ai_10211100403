# Name: Joshua Jerry selrom Yegbe
# Index Number: 10211100403


import streamlit as st
from regression import regression_page
from clustering import clustering_page
from neural_network import neural_network_page

# Sidebar Navigation
st.sidebar.title("AI Project Navigation")
page = st.sidebar.selectbox(
    "Choose a Section", 
    ["Home", "Regression", "Clustering", "Neural Network", "LLM Q&A"]
)

# Page Routing Logic
if page == "Home":
    st.title("Welcome to My AI Project")

    st.markdown("""
    ## Explore Various AI Tasks
    Welcome to the AI Project! Choose a section from the sidebar to dive into different AI tasks.
    
    ### Here's a brief overview of each section:
    - **Home**: A warm welcome and overview of the project.
    - **Regression**: Dive into regression models and understand their application in AI.
    - **Clustering**: Explore clustering techniques and how they are used in unsupervised learning.
    - **Neural Network**: Learn about neural networks and their application in deep learning.
    - **LLM Q&A**: A Q&A section powered by Language Models to explore various topics.
    """)

    st.write("Feel free to explore the AI tasks in the sidebar.")

    if st.button("Start Exploring"):
        st.success("Great! Let's dive into AI!")

elif page == "Regression":
    regression_page()

elif page == "Clustering":
    clustering_page()

elif page == "Neural Network":
    neural_network_page()

elif page == "LLM Q&A":
    st.title("LLM Q&A")
    st.write("LLM section coming soon!")
