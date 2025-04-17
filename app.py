# Name:  Joshua Jerry Selorm Yegbe

# Index Number: 10211100403



import streamlit as st
from regression import regression_page
from clustering import clustering_page
from neural_network import neural_network_page
from rag_interface import ghana_election_rag_app
import base64
import os

# Custom functions for styling and images
def add_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def set_custom_style():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #333;
        margin-top: 1rem;
        text-align: center;
    }
    .card {
        padding: 2rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .question-section {
        padding: 2rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .answer-container {
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.1) 0%, rgba(13, 71, 161, 0.1) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 6px solid #1E88E5;
        transition: all 0.3s ease;
    }
    .answer-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    .context-chunk {
        background: rgba(30, 136, 229, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #42A5F5;
        font-size: 0.95rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
        text-align: left;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button:active {
        background: linear-gradient(135deg, #0D47A1 0%, #0D47A1 100%);
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }
    .active-nav {
        background: linear-gradient(135deg, #0D47A1 0%, #0D47A1 100%) !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
        color: #E3F2FD !important;
    }
    .content-area {
        padding: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    /* Logo styling */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 1.5rem;
    }
    .logo-title {
        color: #1565C0;
        font-size: 2.6rem;
        margin: 0;
        font-weight: 700;
        line-height: 1;
        font-family: 'Roboto', sans-serif;
        letter-spacing: 0.5px;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px 16px 0 0;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(30, 136, 229, 0.3) !important;
        color: #1E88E5 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def display_section_icon(icon_name, section_name, description):
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f'<i class="material-icons" style="font-size:3rem;color:#1E88E5">{icon_name}</i>', unsafe_allow_html=True)
        with col2:
            st.markdown(f"### {section_name}")
            st.write(description)

# Configure page
st.set_page_config(
    page_title="IntelliHub | Advanced Multi-Task Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
set_custom_style()

# Add a subtle background pattern
add_bg_from_url("https://img.freepik.com/free-vector/white-abstract-background_23-2148810113.jpg")

# Load Material Design icons and fonts
st.markdown("""
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Initialize session state for page
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Home"

# Sidebar for navigation
with st.sidebar:
    # Logo in sidebar
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="logo-title">IntelliHub</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation buttons
    pages = [
        "🏠 Home",
        "📊 Regression",
        "🔄 Clustering",
        "🧠 Neural Network",
        "🗳️ VoteWise Analytics"
    ]
    
    for page_option in pages:
        button_key = f"btn_{page_option.replace(' ', '_')}"
        is_active = st.session_state.page == page_option
        
        # Use custom CSS to highlight active page
        if is_active:
            st.markdown(
                f"""
                <style>
                button[kind="secondary"][data-testid="{button_key}"] {{
                    background: linear-gradient(135deg, #0D47A1 0%, #0D47A1 100%) !important;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
                    color: #E3F2FD !important;
                }}
                button[kind="secondary"][data-testid="{button_key}"] span {{
                    color: #E3F2FD !important;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
            
        if st.button(page_option, key=button_key):
            st.session_state.page = page_option
            st.rerun()

# Content area
with st.container():
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    
    # Page Routing Logic
    page = st.session_state.page
    
    if page == "🏠 Home":
        st.markdown('<h1 class="main-header">Welcome to IntelliHub</h1>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:1.2rem;text-align:center;">
            An advanced platform to explore and interact with various AI techniques and models.
        </p>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;justify-content:center;margin:2rem 0">
            <img src="https://img.freepik.com/free-vector/ai-technology-brain-background-vector-digital-transformation-concept_53876-117812.jpg" width="500">
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<h2 class="sub-header">Explore AI Capabilities</h2>', unsafe_allow_html=True)
        
        # Sequential cards
        st.markdown('<div class="card">', unsafe_allow_html=True)
        display_section_icon("trending_up", "Regression Analysis", 
                            "Analyze relationships between variables and make predictions using linear regression models.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        display_section_icon("device_hub", "Neural Networks", 
                            "Explore deep learning with neural networks for pattern recognition and classification tasks.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        display_section_icon("bubble_chart", "Clustering", 
                            "Discover patterns and group similar data points using unsupervised learning algorithms.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        display_section_icon("question_answer", "VoteWise Analytics", 
                            "Dive into election data with advanced retrieval-augmented generation for insightful analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="text-align:center;margin-top:2rem">', unsafe_allow_html=True)
        if st.button("Begin Your AI Journey", key="begin_journey"):
            st.balloons()
            st.success("Let's explore the fascinating world of AI!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "📊 Regression":
        st.markdown('<h1 class="main-header">Regression Analysis</h1>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        regression_page()
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "🔄 Clustering":
        st.markdown('<h1 class="main-header">Clustering Analysis</h1>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        clustering_page()
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "🧠 Neural Network":
        st.markdown('<h1 class="main-header">Neural Network Explorer</h1>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        neural_network_page()
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "🗳️ VoteWise Analytics":
        st.markdown('<h1 class="main-header">VoteWise Analytics</h1>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        ghana_election_rag_app()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)