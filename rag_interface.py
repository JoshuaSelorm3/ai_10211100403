# Name:  Joshua Jerry Selorm Yegbe

# Index Number: 10211100403


import pandas as pd
import os
import streamlit as st
import requests
from io import StringIO
import tempfile
from data_processor import GhanaElectionDataProcessor
from embedding import TextEmbedder
from retriever import ElectionDataRetriever
from generator import GeminiGenerator
from evaluation import RagEvaluator
from visualization import ElectionDataVisualizer

# Set Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDPN4UAAazu3-fDEQ6Mkvwbc_ZNtw8j4Wc"

def rag_system(vector_store_dir="./vector_store", model_id="gemini-1.5-pro", use_saved_vectors=False):
    processor = GhanaElectionDataProcessor()
    embedder = TextEmbedder()
    retriever = ElectionDataRetriever(embedder)
    generator = GeminiGenerator(model_name=model_id)
    evaluator = RagEvaluator()

    # Preload the CSV file from Google Drive
    csv_url = "https://drive.google.com/uc?export=download&id=1oIhHeepzjAX_piQ2FBtxcj0eUq-dNsWn"
    try:
        response = requests.get(csv_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        raw_data = processor.load_data(temp_file_path)
        os.unlink(temp_file_path)
    except requests.HTTPError as e:
        st.error(f"Failed to load CSV from Google Drive: {str(e)}. Please ensure the file is shared publicly ('Anyone with the link').")
        return None
    except requests.RequestException as e:
        st.error(f"Network error while fetching CSV: {str(e)}. Please check your internet connection or the URL.")
        return None
    except pd.errors.ParserError:
        st.error("Invalid CSV format. Please ensure the URL points to a valid CSV file.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

    processed_data = processor.preprocess_data()
    chunks = processor.create_text_chunks()
    visualizer = ElectionDataVisualizer(processed_data)

    if use_saved_vectors and os.path.exists(vector_store_dir):
        retriever.setup_from_saved(vector_store_dir)
    else:
        retriever.setup_from_chunks(chunks)
        embedder.save_vector_store(vector_store_dir)

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
    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;gap:15px;margin-bottom:1.5rem">
        <i class="material-icons" style="font-size:2.5rem;color:#1E88E5">how_to_vote</i>
        <h1 class="rag-header">Ghana Election Insights</h1>
    </div>
    <p style="text-align:center;color:#666;margin-bottom:2rem;font-size:1.1rem">
        Explore Ghana's election data with intelligent insights powered by AI
    </p>
    """, unsafe_allow_html=True)

    # Initialize RAG system
    if 'rag_components' not in st.session_state:
        with st.spinner("Fetching and initializing election data..."):
            st.session_state.rag_components = rag_system()
            if st.session_state.rag_components:
                st.success("Election data loaded successfully from Google Drive!")
            else:
                return

    # Tabs for navigation
    tab1, tab2, tab3 = st.tabs(["🗳️ Ask Questions", "📊 Visualizations", "📈 Evaluation"])

    with tab1:
        st.markdown('<div class="question-section">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Ask About Ghana Elections</h3>", unsafe_allow_html=True)
        
        question = st.text_input(
            "Your Question:",
            placeholder="E.g., Who won the most votes in 2020 in Ashanti Region?",
            help="Ask specific questions about Ghana election results from 1992 to 2020."
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            k_value = st.slider("Number of context chunks", 1, 10, 5, help="Adjust how many data points to consider for the answer")
        with col2:
            submit_button = st.button("Get Answer", use_container_width=True, type="primary")
        
        if submit_button and question:
            with st.spinner("Analyzing election data..."):
                answer, chunks, eval_result = ask_question(
                    st.session_state.rag_components,
                    question,
                    k=k_value
                )
            
            if 'question_history' not in st.session_state:
                st.session_state.question_history = []
            
            st.session_state.question_history.append({
                "question": question,
                "answer": answer,
                "chunks": chunks,
                "evaluation": eval_result
            })
            
            st.markdown('<div class="answer-container">', unsafe_allow_html=True)
            st.markdown("<h4 style='color:#1E88E5;margin-bottom:0.5rem'>Answer:</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:1.1rem;line-height:1.6'>{answer}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("View Source Data", expanded=False):
                for i, chunk in enumerate(chunks):
                    st.markdown(f'<div class="context-chunk">', unsafe_allow_html=True)
                    st.markdown(f"**Source {i+1}** (Relevance: {chunk['score']:.4f})")
                    st.write(chunk['text'])
                    st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.get('question_history'):
            st.markdown("<h4 style='color:#1E88E5;margin-top:2rem'>Recent Questions</h4>", unsafe_allow_html=True)
            for item in reversed(st.session_state.question_history[-3:]):
                with st.expander(f"Q: {item['question']}"):
                    st.markdown("**Answer:**")
                    st.write(item['answer'])
                    st.markdown("**Sources Used:**")
                    for i, chunk in enumerate(item['chunks'][:2]):
                        st.write(f"{i+1}. {chunk['text']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        visualizer = st.session_state.rag_components["visualizer"]
        st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Election Visualizations</h3>", unsafe_allow_html=True)
        
        if visualizer.data is not None:
            st.markdown("**Top Parties by Votes**")
            fig = visualizer.plot_party_votes()
            if fig:
                st.pyplot(fig)
            
            st.markdown("**Regional Vote Distribution**")
            fig = visualizer.plot_regional_distribution()
            if fig:
                st.pyplot(fig)
            
            st.markdown("**Party Comparison by Region**")
            fig = visualizer.plot_party_comparison_by_region()
            if fig:
                st.pyplot(fig)
            
            st.markdown("**Voter Turnout**")
            fig = visualizer.plot_voter_turnout()
            if fig:
                st.pyplot(fig)
        else:
            st.warning("No visualization data available.")

    with tab3:
        st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Performance Metrics</h3>", unsafe_allow_html=True)
        
        if st.session_state.get('question_history'):
            evaluator = st.session_state.rag_components["evaluator"]
            summary = evaluator.generate_summary()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['average_metrics'].get('context_relevance', 0):.2f}</div>
                    <div class="metric-label">Context Relevance</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['average_metrics'].get('response_completeness', 0):.2f}</div>
                    <div class="metric-label">Response Completeness</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['average_metrics'].get('response_conciseness', 0):.2f}</div>
                    <div class="metric-label">Response Conciseness</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Ask questions to see evaluation metrics.")
