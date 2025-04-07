import streamlit as st

# Sidebar
st.sidebar.title("AI Project")
page = st.sidebar.selectbox(
    "Choose a section", 
    ["Home", "Regression", "Clustering", "Neural Network", "LLM Q&A"]
)

# Content based on the selected page
if page == "Home":
    # Main title
    st.title("Welcome to My AI Project")
    
    # Description for the home page
    st.markdown("""
    ## Explore Various AI Tasks
    Welcome to the AI Project! Choose a section from the sidebar to dive into different AI tasks.
    Here's a brief overview of each section:
    
    - **Home**: A warm welcome and overview of the project.
    - **Regression**: Dive into regression models and understand their application in AI.
    - **Clustering**: Explore clustering techniques and how they are used in unsupervised learning.
    - **Neural Network**: Learn about neural networks and their application in deep learning.
    - **LLM Q&A**: A Q&A section powered by Language Models to explore various topics.
    
    Please select one of the options to get started!
    """)
    
    # Add a friendly note
    st.write("Feel free to explore the AI tasks in the sidebar.")
    
    # Optional: Add a button to encourage interaction
    if st.button("Start Exploring"):
        st.write("Great! Let's dive into AI!")
