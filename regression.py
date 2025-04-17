# Name:  Joshua Jerry Selorm Yegbe

# Index Number: 10211100403


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def regression_page():
    # Apply custom styling for the regression page
    st.markdown("""
    <style>
    .regression-header {
        font-size: 2rem;
        color: #1E88E5;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
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
    .feature-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.18);
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
    </style>
    """, unsafe_allow_html=True)

    # Header with icon and gradient text
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;gap:15px;margin-bottom:1.5rem">
        <i class="material-icons" style="font-size:2.5rem;background:linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent">trending_up</i>
        <h1 class="regression-header">Linear Regression Analysis</h1>
    </div>
    <p style="text-align:center;color:#666;margin-bottom:2rem;font-size:1.1rem">
        Upload your dataset to perform regression analysis with beautiful visualizations and insights
    </p>
    """, unsafe_allow_html=True)

    # File upload section with visual enhancements
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    
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
        # Load and display data with improved styling
        data = pd.read_csv(uploaded_file)
        
        # Add tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "📈 Model Training", "🔮 Predictions"])
        
        with tab1:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown("<h3 style='color:#1E88E5;margin-bottom:1rem'>Dataset Preview</h3>", unsafe_allow_html=True)
            st.dataframe(data.head(), use_container_width=True)
            
            # Display statistics
            st.markdown("<h3 style='color:#1E88E5;margin-top:2rem;margin-bottom:1rem'>Data Statistics</h3>", unsafe_allow_html=True)
            st.dataframe(data.describe(), use_container_width=True)
            
            # Show data visualization
            if len(data.columns) > 1:
                st.markdown("<h3 style='color:#1E88E5;margin-top:2rem;margin-bottom:1rem'>Feature Correlations</h3>", unsafe_allow_html=True)
                corr = data.corr()
                fig = px.imshow(corr, 
                              text_auto=True, 
                              aspect="auto",
                              color_continuous_scale='blues',
                              template='plotly_white')
                fig.update_layout(height=500,
                                margin=dict(l=0, r=0, t=0, b=0),
                                font=dict(size=12))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            # Target column selection with improved UI
            st.markdown("<h3 style='color:#1E88E5;margin-bottom:1.5rem'>Model Configuration</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                target_col = st.selectbox("Select Target Variable", data.columns, key="target_select")
            with col2:
                test_size = st.slider("Test Set Size (%)", 10, 40, 20, key="test_size_slider")
            
            feature_cols = [col for col in data.columns if col != target_col]
            
            # Allow selecting specific features
            selected_features = st.multiselect("Select Features for Model Training", 
                                             feature_cols, 
                                             default=feature_cols,
                                             key="feature_multiselect")
            
            if not selected_features:
                st.warning("Please select at least one feature for training.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
                
            # Preprocessing with progress indicator
            st.markdown("<h3 style='color:#1E88E5;margin-top:2rem;margin-bottom:1rem'>Data Preprocessing</h3>", unsafe_allow_html=True)
            
            with st.status("Preprocessing data...", expanded=True) as status:
                st.write("Removing missing values...")
                data = data.dropna()
                st.write("Selecting features and target...")
                X = data[selected_features]
                y = data[target_col]
                status.update(label="Preprocessing complete!", state="complete", expanded=False)
            
            # Train Linear Regression with better visual feedback
            st.markdown("<h3 style='color:#1E88E5;margin-top:2rem;margin-bottom:1rem'>Model Training</h3>", unsafe_allow_html=True)
            
            with st.spinner("Training linear regression model..."):
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                # Calculate metrics
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
            
            # Display metrics in nice cards
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{mae:.4f}</div>
                    <div class="metric-label">Mean Absolute Error</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{r2:.4f}</div>
                    <div class="metric-label">R² Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature importance
            st.markdown("<h3 style='color:#1E88E5;margin-top:2rem;margin-bottom:1rem'>Feature Coefficients</h3>", unsafe_allow_html=True)
            
            coef_df = pd.DataFrame({
                'Feature': selected_features,
                'Coefficient': model.coef_
            })
            coef_df = coef_df.sort_values('Coefficient', ascending=False)
            
            fig = px.bar(coef_df, 
                        x='Feature', 
                        y='Coefficient', 
                        color='Coefficient', 
                        color_continuous_scale='blues',
                        template='plotly_white')
            fig.update_layout(xaxis_title='Feature', 
                            yaxis_title='Coefficient Value',
                            margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualization with Plotly
            st.markdown("<h3 style='color:#1E88E5;margin-top:2rem;margin-bottom:1rem'>Prediction Performance</h3>", unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y, 
                y=y_pred, 
                mode='markers', 
                marker=dict(color='#1E88E5', size=8, opacity=0.7),
                name='Predictions'
            ))
            
            # Add identity line
            fig.add_trace(go.Scatter(
                x=[y.min(), y.max()], 
                y=[y.min(), y.max()],
                mode='lines', 
                line=dict(color='red', dash='dash', width=2),
                name='Ideal'
            ))
            
            fig.update_layout(
                template='plotly_white',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                height=500,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            # Custom Prediction with improved UI
            st.markdown("<h3 style='color:#1E88E5;margin-bottom:1.5rem'>Make a Prediction</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <p style="color:#666;margin-bottom:1.5rem">
                Enter values for each feature to get a prediction from the trained model.
            </p>
            """, unsafe_allow_html=True)
            
            # Create a cleaner UI for inputs
            custom_input = {}
            
            # Display inputs in two columns for better layout
            cols = st.columns(2)
            
            for i, col in enumerate(selected_features):
                col_idx = i % 2
                with cols[col_idx]:
                    # Get min/max/mean for better default values
                    min_val = float(data[col].min())
                    max_val = float(data[col].max())
                    mean_val = float(data[col].mean())
                    
                    custom_input[col] = st.number_input(
                        f"{col}", 
                        value=mean_val,
                        min_value=min_val,
                        max_value=max_val,
                        step=(max_val - min_val)/100,
                        help=f"Range: {min_val:.2f} to {max_val:.2f}, Avg: {mean_val:.2f}",
                        key=f"pred_input_{col}"
                    )
            
            # Prediction button with styling
            predict_clicked = st.button("Generate Prediction", 
                                       use_container_width=True,
                                       type="primary",
                                       key="predict_button")
            
            if predict_clicked:
                with st.spinner("Calculating prediction..."):
                    input_df = pd.DataFrame([custom_input])
                    pred = model.predict(input_df)[0]
                
                # Display the prediction with a nice animation
                st.markdown("""
                <div style="text-align:center;margin:2rem 0;padding:2rem;background:rgba(30,136,229,0.1);border-radius:16px;border-left:4px solid #1E88E5">
                    <div style="font-size:1.2rem;margin-bottom:10px;color:#666">Predicted</div>
                    <div style="font-size:2.5rem;font-weight:bold;background:linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
                        {pred:.4f}
                    </div>
                    <div style="font-size:1rem;color:#666;margin-top:5px">{target_col}</div>
                </div>
                """.format(pred=pred, target_col=target_col), unsafe_allow_html=True)
                
                # Show explanation
                st.markdown("<h4 style='color:#1E88E5;margin-top:2rem;margin-bottom:1rem'>How we calculated this:</h4>", unsafe_allow_html=True)
                
                explanation = pd.DataFrame({
                    'Feature': selected_features,
                    'Value': [custom_input[col] for col in selected_features],
                    'Coefficient': model.coef_,
                    'Impact': [custom_input[col] * coef for col, coef in zip(selected_features, model.coef_)]
                })
                
                explanation['Impact'] = explanation['Value'] * explanation['Coefficient']
                explanation['Percent'] = (explanation['Impact'] / explanation['Impact'].sum() * 100).abs()
                
                # Create an explanatory chart
                fig = px.bar(explanation, 
                            x='Feature', 
                            y='Impact', 
                            color='Impact', 
                            color_continuous_scale='blues',
                            template='plotly_white',
                            title='Feature Contribution to Prediction')
                fig.update_layout(xaxis_title='Feature', 
                                yaxis_title='Impact on Prediction',
                                margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    regression_page()