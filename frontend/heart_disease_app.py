"""
Streamlit App for Heart Disease Prediction with AI Chat Assistant
Run with: streamlit run frontend/heart_disease_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Import from backend
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.predict_heart_disease import (
    load_model_and_scaler,
    predict_heart_disease,
    batch_predict,
    FEATURE_NAMES
)
from backend.llm_service import (
    initialize_chat_model,
    get_ai_response
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)



def initialize_session_state():
    """Initialize session state for chat"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'prediction_context' not in st.session_state:
        st.session_state.prediction_context = None

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .healthy {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .disease {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-very-low { background-color: #20c997; color: white; }
    .risk-low { background-color: #28a745; color: white; }
    .risk-medium { background-color: #ffc107; color: black; }
    .risk-high { background-color: #fd7e14; color: white; }
    .risk-very-high { background-color: #dc3545; color: white; }
    .stButton>button {
        width: 100%;
        background-color: #e74c3c;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #c0392b;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    
    /* Chat-specific styles */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize session state
    initialize_session_state()
    
    # Initialize chat model
    chat_model = initialize_chat_model()
    
    # Header
    st.markdown('<h1 class="main-header">🫀 Heart Disease Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Cardiovascular Risk Assessment with Chat Assistant</p>', 
                unsafe_allow_html=True)
    
    # Load model from backend
    model, scaler, feature_info, metrics = load_model_and_scaler()
    
    if model is None:
        st.error("❌ Failed to load model. Please ensure model files are in 'backend/heart_disease_model' folder.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=100)
        st.title("About")
        st.info("""
        This application uses a **Gradient Boosting** machine learning model 
        to predict the likelihood of heart disease based on patient health parameters.
        
        **Features:**
        - Heart disease prediction
        - AI Chat assistant for medical queries
        - Batch predictions
        - Model insights
        
        **Model Accuracy:** ~91%
        """)
        
        st.title("Instructions")
        st.markdown("""
        1. Enter patient information
        2. Click **Predict** button
        3. View prediction results
        4. Ask questions to AI assistant
        5. Review recommendations
        """)
        
        # Add chat status indicator
        if chat_model:
            st.success("🤖 AI Assistant: Online")
        else:
            st.warning("🤖 AI Assistant: Offline")
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "💬 AI Assistant", "📊 Model Info", "📋 Batch Prediction"])
    
    # ========================================================================
    # TAB 1: SINGLE PREDICTION
    # ========================================================================
    
    with tab1:
        st.header("Patient Information")
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            age = st.slider("Age (years)", min_value=1, max_value=120, value=55, help="Patient's age in years")
            
            gender = st.selectbox(
                "Gender",
                options=[0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male",
                help="Patient's biological gender"
            )
        
        with col2:
            st.subheader("Vital Signs")
            restingBP = st.slider(
                "Resting Blood Pressure (mm Hg)",
                min_value=50, max_value=250, value=140,
                help="Resting blood pressure on admission"
            )
            
            maxheartrate = st.slider(
                "Maximum Heart Rate",
                min_value=50, max_value=250, value=150,
                help="Maximum heart rate achieved during exercise"
            )
            
            serumcholestrol = st.slider(
                "Serum Cholesterol (mg/dl)",
                min_value=100, max_value=600, value=260,
                help="Serum cholesterol level"
            )
        
        with col3:
            st.subheader("Blood Tests")
            fastingbloodsugar = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Is fasting blood sugar greater than 120 mg/dl?"
            )
            
            oldpeak = st.slider(
                "ST Depression (Oldpeak)",
                min_value=0.0, max_value=10.0, value=1.5, step=0.1,
                help="ST depression induced by exercise relative to rest"
            )
        
        st.markdown("---")
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.subheader("Symptoms")
            chestpain = st.selectbox(
                "Chest Pain Type",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "Typical Angina",
                    1: "Atypical Angina",
                    2: "Non-anginal Pain",
                    3: "Asymptomatic"
                }[x],
                help="Type of chest pain experienced"
            )
            
            exerciseangia = st.selectbox(
                "Exercise Induced Angina",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Does exercise induce angina?"
            )
        
        with col5:
            st.subheader("ECG Results")
            restingrelectro = st.selectbox(
                "Resting ECG Results",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "Normal",
                    1: "ST-T Wave Abnormality",
                    2: "Left Ventricular Hypertrophy"
                }[x],
                help="Resting electrocardiographic results"
            )
            
            slope = st.selectbox(
                "Slope of Peak Exercise ST",
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "Upsloping",
                    1: "Flat",
                    2: "Downsloping"
                }[x],
                help="Slope of the peak exercise ST segment"
            )
            
            noofmajorvessels = st.selectbox(
                "Number of Major Vessels",
                options=[0, 1, 2, 3, 4],
                help="Number of major vessels colored by fluoroscopy (0-4)"
            )
        
        st.markdown("---")
        
        # Predict button
        col_predict, col_chat = st.columns([2, 1])
        
        with col_predict:
            predict_button = st.button("🔍 Predict Heart Disease Risk", use_container_width=True)
        
        with col_chat:
            if predict_button:
                st.info("💬 Check the AI Assistant tab to discuss your results!")
        
        if predict_button:
            # Collect patient data
            patient_data = {
                "age": age,
                "gender": gender,
                "chestpain": chestpain,
                "restingBP": restingBP,
                "serumcholestrol": serumcholestrol,
                "fastingbloodsugar": fastingbloodsugar,
                "restingrelectro": restingrelectro,
                "maxheartrate": maxheartrate,
                "exerciseangia": exerciseangia,
                "oldpeak": oldpeak,
                "slope": slope,
                "noofmajorvessels": noofmajorvessels
            }
            
            # Make prediction using backend
            with st.spinner("Analyzing patient data..."):
                result = predict_heart_disease(model, scaler, patient_data)
            
            # Store in session state for chat context
            st.session_state.prediction_context = result
            
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            # Result columns
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                # Prediction result box
                if result["prediction"] == 0:
                    st.success(f"### ✅ {result['prediction_label']}")
                else:
                    st.error(f"### ⚠️ {result['prediction_label']}")
            
            with res_col2:
                # Risk level
                st.markdown(f"""
                <div style="background-color: {result['risk_color']}; color: white; 
                            padding: 1rem; border-radius: 10px; text-align: center;">
                    <h3 style="margin: 0;">Risk Level</h3>
                    <h2 style="margin: 0;">{result['risk_level']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with res_col3:
                # Confidence
                confidence = result['probability_disease'] if result['prediction'] == 1 else result['probability_no_disease']
                st.metric("Confidence", f"{confidence:.1%}")
            
            st.markdown("---")
            
            # Probability visualization
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['probability_disease'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Disease Probability", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': result['risk_color']},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'steps': [
                            {'range': [0, 20], 'color': '#d4edda'},
                            {'range': [20, 40], 'color': '#d4edda'},
                            {'range': [40, 60], 'color': '#fff3cd'},
                            {'range': [60, 80], 'color': '#f8d7da'},
                            {'range': [80, 100], 'color': '#f5c6cb'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with prob_col2:
                # Pie chart
                fig2 = px.pie(
                    values=[result['probability_no_disease'], result['probability_disease']],
                    names=['No Disease', 'Disease'],
                    color_discrete_sequence=['#28a745', '#dc3545'],
                    title="Probability Distribution"
                )
                fig2.update_traces(textposition='inside', textinfo='percent+label')
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Recommendation
            st.markdown("---")
            st.subheader("📋 Recommendation")
            
            if result['probability_disease'] >= 0.6:
                st.error(result['recommendation'])
            elif result['probability_disease'] >= 0.4:
                st.warning(result['recommendation'])
            else:
                st.success(result['recommendation'])
            
            # Add button to navigate to chat
            st.info("💬 Have questions about your results? Visit the **AI Assistant** tab to chat with our medical AI!")
            
            # Patient data summary
            with st.expander("📄 View Patient Data Summary"):
                st.json(patient_data)
    
    # ========================================================================
    # TAB 2: AI CHAT ASSISTANT
    # ========================================================================
    
    with tab2:
        st.header("💬 AI Medical Assistant")
        
        if not chat_model:
            st.warning("⚠️ Chat feature is not available. Please configure your Hugging Face API token.")
            st.info("""
            To enable the chat assistant:
            1. Create a `.env` file in your project directory
            2. Add your Hugging Face API token: `HUGGINGFACEHUB_API_TOKEN=your_token_here`
            3. Restart the application
            """)
        else:
            # Display context if prediction was made
            if st.session_state.prediction_context:
                context = st.session_state.prediction_context
                st.info(f"""
                📊 **Current Prediction Context:**
                - Result: {context['prediction_label']}
                - Risk Level: {context['risk_level']}
                - Probability: {context['probability_disease']:.1%}
                """)
                
                # Quick question buttons
                st.subheader("Quick Questions")
                quick_questions = [
                    "What does my prediction result mean?",
                    "What lifestyle changes can help reduce heart disease risk?",
                    "What are the symptoms of heart disease?",
                    "Should I see a doctor based on my results?",
                    "What do the different risk factors mean?"
                ]
                
                cols = st.columns(2)
                for idx, question in enumerate(quick_questions):
                    with cols[idx % 2]:
                        if st.button(question, key=f"quick_{idx}"):
                            st.session_state.chat_history.append({"role": "user", "content": question})
                            with st.spinner("AI is thinking..."):
                                response = get_ai_response(chat_model, question, st.session_state.prediction_context)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Chat interface
            st.subheader("Chat with AI Assistant")
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.chat_message("user").write(message["content"])
                    else:
                        st.chat_message("assistant").write(message["content"])
            
            # Chat input
            user_input = st.chat_input("Ask me anything about heart health...")
            
            if user_input:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Get AI response
                with st.spinner("AI is thinking..."):
                    response = get_ai_response(chat_model, user_input, st.session_state.prediction_context)
                
                # Add AI response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Rerun to update chat display
                st.rerun()
            
            # Clear chat button
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            # Medical disclaimer
            st.warning("""
            ⚠️ **Medical Disclaimer**: This AI assistant provides general health information only. 
            It is not a substitute for professional medical advice, diagnosis, or treatment. 
            Always consult with qualified healthcare providers for medical concerns.
            """)
    
    # ========================================================================
    # TAB 3: MODEL INFO
    # ========================================================================
    
    with tab3:
        st.header("📊 Model Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.subheader("Model Details")
            st.markdown("""
            | Parameter | Value |
            |-----------|-------|
            | **Algorithm** | Gradient Boosting Classifier |
            | **Number of Estimators** | 100 |
            | **Max Depth** | 5 |
            | **Learning Rate** | 0.1 |
            """)
        
        with info_col2:
            if metrics:
                st.subheader("Performance Metrics")
                
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric("Accuracy", f"{metrics.get('test_accuracy', 0):.2%}")
                    st.metric("Precision", f"{metrics.get('test_precision', 0):.2%}")
                    st.metric("Recall", f"{metrics.get('test_recall', 0):.2%}")
                
                with metric_col2:
                    st.metric("F1 Score", f"{metrics.get('test_f1_score', 0):.2%}")
                    st.metric("ROC-AUC", f"{metrics.get('test_roc_auc', 0):.2%}")
        
        # Feature importance
        st.subheader("Feature Importance")
        
        if feature_info and 'feature_importance' in feature_info:
            importance_df = pd.DataFrame(feature_info['feature_importance'])
            importance_df = importance_df.sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance",
                color='Importance',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance data not available.")
        
        # Feature descriptions
        st.subheader("Feature Descriptions")
        
        feature_desc = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Description': [
                'Age in years',
                'Gender (0: Female, 1: Male)',
                'Chest pain type (0-3)',
                'Resting blood pressure (mm Hg)',
                'Serum cholesterol (mg/dl)',
                'Fasting blood sugar > 120 mg/dl',
                'Resting ECG results (0-2)',
                'Maximum heart rate achieved',
                'Exercise induced angina (0: No, 1: Yes)',
                'ST depression induced by exercise',
                'Slope of peak exercise ST segment',
                'Number of major vessels (0-4)'
            ],
            'Range': [
                '1-120', '0-1', '0-3', '50-250', '100-600',
                '0-1', '0-2', '50-250', '0-1', '0.0-10.0', '0-2', '0-4'
            ]
        })
        
        st.dataframe(feature_desc, use_container_width=True)
    
    # ========================================================================
    # TAB 4: BATCH PREDICTION
    # ========================================================================
    
    with tab4:
        st.header("📋 Batch Prediction")
        st.info("Upload a CSV file with patient data to get predictions for multiple patients at once.")
        
        # Template download
        st.subheader("1️⃣ Download Template")
        
        template_df = pd.DataFrame({
            'age': [55, 45, 60],
            'gender': [1, 0, 1],
            'chestpain': [2, 1, 3],
            'restingBP': [140, 120, 160],
            'serumcholestrol': [260, 220, 300],
            'fastingbloodsugar': [0, 0, 1],
            'restingrelectro': [1, 0, 2],
            'maxheartrate': [150, 170, 130],
            'exerciseangia': [0, 0, 1],
            'oldpeak': [1.5, 0.5, 2.5],
            'slope': [1, 1, 2],
            'noofmajorvessels': [0, 0, 2]
        })
        
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="patient_template.csv",
            mime="text/csv"
        )
        
        # File upload
        st.subheader("2️⃣ Upload Patient Data")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read CSV
                patients_df = pd.read_csv(uploaded_file)
                
                st.subheader("3️⃣ Preview Data")
                st.dataframe(patients_df.head(), use_container_width=True)
                st.info(f"Loaded {len(patients_df)} patients")
                
                # Predict button
                if st.button("🔍 Predict for All Patients", use_container_width=True):
                    
                    with st.spinner("Processing predictions..."):
                        results = batch_predict(model, scaler, patients_df)
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame(results)
                    
                    st.subheader("4️⃣ Prediction Results")
                    
                    # Summary metrics
                    total = len(results_df)
                    disease_count = results_df['prediction'].sum()
                    no_disease_count = total - disease_count
                    
                    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                    
                    with sum_col1:
                        st.metric("Total Patients", total)
                    with sum_col2:
                        st.metric("Disease Detected", int(disease_count))
                    with sum_col3:
                        st.metric("No Disease", int(no_disease_count))
                    with sum_col4:
                        st.metric("Disease Rate", f"{disease_count/total:.1%}")
                    
                    # Results table
                    display_df = results_df[['patient_id', 'prediction_label', 'probability_disease', 'risk_level', 'recommendation']]
                    display_df.columns = ['Patient ID', 'Prediction', 'Disease Probability', 'Risk Level', 'Recommendation']
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Visualization
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Pie chart
                        fig = px.pie(
                            values=[no_disease_count, disease_count],
                            names=['No Disease', 'Disease'],
                            color_discrete_sequence=['#28a745', '#dc3545'],
                            title="Prediction Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_col2:
                        # Risk level distribution
                        risk_counts = results_df['risk_level'].value_counts()
                        fig = px.bar(
                            x=risk_counts.index,
                            y=risk_counts.values,
                            title="Risk Level Distribution",
                            color=risk_counts.index,
                            color_discrete_map={
                                'Very Low': '#20c997',
                                'Low': '#28a745',
                                'Medium': '#ffc107',
                                'High': '#fd7e14',
                                'Very High': '#dc3545'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    st.subheader("5️⃣ Download Results")
                    
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_results,
                        file_name="heart_disease_predictions.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🫀 Heart Disease Prediction App with AI Assistant | Built with Streamlit & Machine Learning</p>
        <p>Powered by Mistral-7B AI Model 🤖</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
