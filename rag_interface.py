# Name:  Joshua Jerry Selorm Yegbe

# Index Number: 10211100403


import pandas as pd
import os
import streamlit as st
from data_processor import GhanaElectionDataProcessor
from embedding import TextEmbedder
from retriever import ElectionDataRetriever
from generator import GeminiGenerator
from evaluation import RagEvaluator
from visualization import ElectionDataVisualizer
import plotly.express as px
import plotly.graph_objects as go

def rag_system(data_path=None, vector_store_dir="./vector_store", 
              model_id="gemini-1.5-pro", 
              use_saved_vectors=False):
    processor = GhanaElectionDataProcessor()
    embedder = TextEmbedder()
    retriever = ElectionDataRetriever(embedder)
    generator = GeminiGenerator(model_name=model_id)
    evaluator = RagEvaluator()

    if data_path:
        raw_data = processor.load_data(data_path)
        processed_data = processor.preprocess_data()
        chunks = processor.create_text_chunks()
        visualizer = ElectionDataVisualizer(processed_data)

        if use_saved_vectors and os.path.exists(vector_store_dir):
            retriever.setup_from_saved(vector_store_dir)
        else:
            retriever.setup_from_chunks(chunks)
            embedder.save_vector_store(vector_store_dir)
    else:
        if use_saved_vectors and os.path.exists(vector_store_dir):
            retriever.setup_from_saved(vector_store_dir)
            visualizer = ElectionDataVisualizer()
        else:
            raise ValueError("Either data_path or use_saved_vectors with existing vectors must be provided")

    generator.load_model()
    generator.setup_pipeline()
    chain = generator.create_rag_chain()

    return {
        "processor": processor,
        "retriever": retriever,
        "generator": generator,
        "evaluator": evaluator,
        "visualizer": visualizer,
        "chain": chain
    }

def ask_question(rag_components, question, k=5):
    retriever = rag_components["retriever"]
    generator = rag_components["generator"]
    chain = rag_components["chain"]
    evaluator = rag_components["evaluator"]

    retrieved_chunks = retriever.retrieve(question, k=k)
    context = retriever.format_for_llm(retrieved_chunks)
    answer = generator.generate_answer(chain, context, question)
    evaluation = evaluator.evaluate_response(question, context, answer)

    return answer, retrieved_chunks, evaluation

def ghana_election_rag_app():
    # Apply custom styling
    st.markdown("""
    <style>
    .rag-header {
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
    .context-chunk {
        background: rgba(30, 136, 229, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .tab-content {
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 0 0 16px 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-top: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 16px 16px 0 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.18);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(30, 136, 229, 0.2) !important;
        color: #1E88E5 !important;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(30, 136, 229, 0.1) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #1E88E5 !important;
    }
    .answer-container {
        background: rgba(30, 136, 229, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with icon and gradient text
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;gap:15px;margin-bottom:1.5rem">
        <i class="material-icons" style="font-size:2.5rem;background:linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent">how_to_vote</i>
        <h1 class="rag-header">Ghana Election RAG System</h1>
    </div>
    <p style="text-align:center;color:#666;margin-bottom:2rem;font-size:1.1rem">
        Ask questions about Ghana election data using Retrieval-Augmented Generation
    </p>
    """, unsafe_allow_html=True)

    # Configuration in main screen
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1E88E5;margin-bottom:1rem;">Configuration</h3>', unsafe_allow_html=True)
    
    if 'rag_components' not in st.session_state:
        st.session_state.rag_components = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'question_history' not in st.session_state:
        st.session_state.question_history = []
    
    data_option = st.radio("Data Source:", ["Upload CSV", "Use Saved Vectors"])
    
    if data_option == "Upload CSV" and not st.session_state.data_loaded:
        uploaded_file = st.file_uploader("Upload Ghana Election Data CSV", type=['csv'])
        if uploaded_file:
            with st.spinner("Processing data and building vector store..."):
                try:
                    with open("temp_data.csv", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.rag_components = rag_system(
                        data_path="temp_data.csv",
                        use_saved_vectors=False
                    )
                    st.session_state.data_loaded = True
                    st.success("Data processed and RAG system initialized!")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
    
    elif data_option == "Use Saved Vectors" and not st.session_state.data_loaded:
        vector_dir = st.text_input("Vector Store Directory", value="./vector_store")
        if st.button("Load Vectors"):
            if os.path.exists(vector_dir):
                with st.spinner("Loading saved vector store..."):
                    try:
                        st.session_state.rag_components = rag_system(
                            data_path=None,
                            vector_store_dir=vector_dir,
                            use_saved_vectors=True
                        )
                        st.session_state.data_loaded = True
                        st.success("Vector store loaded and RAG system initialized!")
                    except Exception as e:
                        st.error(f"Error loading vectors: {str(e)}")
            else:
                st.error(f"Vector store directory {vector_dir} not found!")
    
    if st.session_state.data_loaded and st.button("Reset System"):
        st.session_state.rag_components = None
        st.session_state.data_loaded = False
        st.session_state.question_history = []
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.data_loaded and st.session_state.rag_components:
        tab1, tab2, tab3 = st.tabs(["🗳️ Ask Questions", "📊 Visualizations", "📈 Evaluation"])

        with tab1:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown("<h3 style='color:#1E88E5;margin-bottom:1.5rem'>Ask About Ghana Elections</h3>", unsafe_allow_html=True)
            
            question = st.text_input("Enter your question about Ghana elections:", placeholder="E.g., What were the main issues in the 2020 Ghana election?")
            
            col1, col2 = st.columns(2)
            with col1:
                k_value = st.slider("Number of context chunks", 1, 10, 5, help="How many document chunks to retrieve for context")
            with col2:
                submit_button = st.button("Ask Question", use_container_width=True, type="primary")
            
            if submit_button and question:
                with st.spinner("Searching election data and generating answer..."):
                    answer, chunks, eval_result = ask_question(
                        st.session_state.rag_components, 
                        question,
                        k=k_value
                    )
                
                st.session_state.question_history.append({
                    "question": question,
                    "answer": answer,
                    "chunks": chunks,
                    "evaluation": eval_result
                })
                
                st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                st.markdown("<h4 style='color:#1E88E5;margin-bottom:0.5rem'>Answer:</h4>", unsafe_allow_html=True)
                st.write(answer)
                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("View Retrieved Context", expanded=False):
                    for i, chunk in enumerate(chunks):
                        st.markdown(f'<div class="context-chunk">', unsafe_allow_html=True)
                        st.markdown(f"**Context {i+1}** (Relevance: {chunk['score']:.4f})")
                        st.write(chunk['text'])
                        st.markdown('</div>', unsafe_allow_html=True)

            if st.session_state.question_history:
                st.markdown("<h4 style='color:#1E88E5;margin-top:2rem;margin-bottom:1rem'>Recent Questions</h4>", unsafe_allow_html=True)
                for i, item in enumerate(reversed(st.session_state.question_history[-5:])):
                    with st.expander(f"Q: {item['question']}"):
                        st.markdown("**Answer:**")
                        st.write(item['answer'])
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown("<h3 style='color:#1E88E5;margin-bottom:1.5rem'>Election Data Visualizations</h3>", unsafe_allow_html=True)
            
            if hasattr(st.session_state.rag_components["visualizer"], "data") and \
               st.session_state.rag_components["visualizer"].data is not None:

                visualizer = st.session_state.rag_components["visualizer"]

                st.markdown("<h4 style='color:#1E88E5;margin-bottom:1rem'>Top Parties by Votes</h4>", unsafe_allow_html=True)
                party_fig = visualizer.plot_party_votes(top_n=5)
                if party_fig:
                    st.plotly_chart(party_fig, use_container_width=True)

                st.markdown("<h4 style='color:#1E88E5;margin-top:2rem;margin-bottom:1rem'>Vote Distribution by Region</h4>", unsafe_allow_html=True)
                region_fig = visualizer.plot_regional_distribution()
                if region_fig:
                    st.plotly_chart(region_fig, use_container_width=True)

                st.markdown("<h4 style='color:#1E88E5;margin-top:2rem;margin-bottom:1rem'>Party Comparison by Region</h4>", unsafe_allow_html=True)
                party_region_fig = visualizer.plot_party_comparison_by_region()
                if party_region_fig:
                    st.plotly_chart(party_region_fig, use_container_width=True)

                st.markdown("<h4 style='color:#1E88E5;margin-top:2rem;margin-bottom:1rem'>Voter Turnout by Region</h4>", unsafe_allow_html=True)
                turnout_fig = visualizer.plot_voter_turnout()
                if turnout_fig:
                    st.plotly_chart(turnout_fig, use_container_width=True)
            else:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("""
                <div style="text-align:center;padding:2rem">
                    <i class="material-icons" style="font-size:3rem;color:#1E88E5">info</i>
                    <h4 style="color:#1E88E5;margin-top:1rem">No Visualization Data</h4>
                    <p style="color:#666">Please upload a CSV file with election data to enable visualizations</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown("<h3 style='color:#1E88E5;margin-bottom:1.5rem'>System Evaluation Metrics</h3>", unsafe_allow_html=True)
            
            if st.session_state.question_history:
                evaluator = st.session_state.rag_components["evaluator"]
                summary = evaluator.generate_summary()

                st.markdown("<h4 style='color:#1E88E5;margin-bottom:1rem'>Average Performance</h4>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{summary['average_metrics'].get('precision', 0):.2f}</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{summary['average_metrics'].get('recall', 0):.2f}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{summary['average_metrics'].get('f1_score', 0):.2f}</div>
                        <div class="metric-label">F1 Score</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<h4 style='color:#1E88E5;margin-top:2rem;margin-bottom:1rem'>Recent Evaluations</h4>", unsafe_allow_html=True)
                for i, item in enumerate(reversed(st.session_state.question_history[-5:])):
                    with st.expander(f"Question: {item['question']}"):
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Precision", f"{item['evaluation']['metrics'].get('precision', 0):.2f}")
                        with cols[1]:
                            st.metric("Recall", f"{item['evaluation']['metrics'].get('recall', 0):.2f}")
                        with cols[2]:
                            st.metric("F1 Score", f"{item['evaluation']['metrics'].get('f1_score', 0):.2f}")
            else:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("""
                <div style="text-align:center;padding:2rem">
                    <i class="material-icons" style="font-size:3rem;color:#1E88E5">question_answer</i>
                    <h4 style="color:#1E88E5;margin-top:1rem">No Evaluations Yet</h4>
                    <p style="color:#666">Ask some questions to see evaluation metrics</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)