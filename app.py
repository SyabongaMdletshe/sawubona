import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils import load_and_process_data, train_model

# Set page configuration - FULL WIDTH
st.set_page_config(
    page_title="Healthcare Demand Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .team-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 15px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title with custom styling
st.markdown('<h1 class="main-header">🏥 Healthcare Service Demand Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2000/2000866.png", width=100)
    st.title("Navigation")
    page = st.radio("Select Page", ["🏠 Home", "📊 Data Overview", "📈 Data Analysis", "🤖 Model Training", "🔮 Predictions", "👥 Team Details"])
    
    st.markdown("---")
    st.markdown("### About This Project")
    st.info("""
    This dashboard predicts healthcare service demand 
    for NHI planning in South Africa using machine learning.
    """)

# Load data
@st.cache_data
def load_data():
    return load_and_process_data("data/")

# Home Page
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Healthcare Demand Prediction Dashboard
        
        This interactive dashboard provides insights and predictions about healthcare service demand
        in South Africa, supporting the planning and implementation of the National Health Insurance (NHI) program.
        
        ### Key Features:
        - 📊 **Data Exploration**: Explore the comprehensive healthcare dataset
        - 📈 **Visual Analytics**: Interactive charts and visualizations
        - 🤖 **Machine Learning**: Predictive modeling using Random Forest
        - 🔮 **Demand Prediction**: Forecast healthcare service needs
        - 👥 **Team Collaboration**: Learn about our development team
        
        ### Project Background
        The NHI aims to provide equitable access to healthcare for all South Africans. 
        Accurate forecasting of future demand for services is crucial for resource allocation 
        like doctors, nurses, medication, and facility space.
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2785/2785819.png", width=250)
        st.markdown("""
        **Dataset Features:**
        - Demographic information
        - Health status predictors
        - Historical service usage
        - Geographic data
        - Insurance coverage
        """)

# Data Overview Page
elif page == "📊 Data Overview":
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    data = load_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Individuals", f"{data.shape[0]:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Number of Features", data.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Zero Visits", f"{len(data[data['total_visits'] == 0]):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("One+ Visits", f"{len(data[data['total_visits'] > 0]):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data sample and description
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Sample Data")
        st.dataframe(data.head(10), height=300)
    
    with col2:
        st.subheader("Data Description")
        st.write("The dataset contains comprehensive health and demographic information:")
        st.write("- Personal demographics (age, gender, employment)")
        st.write("- Household information (income, size, location)")
        st.write("- Health conditions (HIV, TB, diabetes, hypertension, asthma)")
        st.write("- Insurance coverage details")
        st.write("- Geographic distribution")
    
    # Target variable distribution
    st.subheader("Distribution of Healthcare Visits")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data['total_visits'], kde=True, bins=30, ax=ax, color='#3498db')
    ax.set_title('Distribution of Total Number of Visits per Person', fontsize=14, fontweight='bold')
    ax.set_xlabel('Total Visits')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Data Analysis Page
elif page == "📈 Data Analysis":
    st.markdown('<h2 class="sub-header">Data Analysis</h2>', unsafe_allow_html=True)
    
    data = load_data()
    
    # Age vs Visits
    st.subheader("Age vs. Number of Healthcare Visits")
    fig = px.scatter(data, x='age', y='total_visits', 
                     title='Relationship Between Age and Healthcare Visits',
                     labels={'age': 'Age', 'total_visits': 'Total Visits'},
                     color_discrete_sequence=['#3498db'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Health conditions analysis
    st.subheader("Health Conditions Impact on Visits")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        condition = st.selectbox("Select Health Condition", 
                               ['hiv', 'tb', 'diabetes', 'hypertension', 'asthma'])
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=data[condition], y=data['total_visits'], ax=ax, palette='Set2')
        ax.set_title(f'Healthcare Visits: With vs. Without {condition.capitalize()}', fontweight='bold')
        ax.set_xlabel(f'Has {condition.capitalize()} (0=No, 1=Yes)')
        ax.set_ylabel('Total Visits')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Between Features', fontweight='bold')
    st.pyplot(fig)

# Model Training Page
elif page == "🤖 Model Training":
    st.markdown('<h2 class="sub-header">Machine Learning Model</h2>', unsafe_allow_html=True)
    
    data = load_data()
    
    st.markdown("""
    <div class="info-box">
    We're using a <b>Random Forest Regressor</b> to predict healthcare service demand based on demographic
    and health factors. This ensemble learning method combines multiple decision trees for accurate predictions.
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner('Training the model... This may take a few minutes'):
            model, X_train, X_test, y_train, y_test, y_pred, mae, mse, r2 = train_model(data)
        
        st.markdown("""
        <div class="success-box">
        <h3>✅ Model trained successfully!</h3>
        The machine learning model has been trained and evaluated on the healthcare dataset.
        </div>
        """, unsafe_allow_html=True)
        
        # Display metrics in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Mean Absolute Error", f"{mae:.2f}", help="Average absolute error between predictions and actual values")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Mean Squared Error", f"{mse:.2f}", help="Average squared error - punishes larger errors more")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("R² Score", f"{r2:.4f}", help="Proportion of variance explained - closer to 1 is better")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Actual vs Predicted plot
        st.subheader("Actual vs Predicted Values")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color='#3498db')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Actual Visits')
        ax.set_ylabel('Predicted Visits')
        ax.set_title('Model Performance: Actual vs Predicted Healthcare Visits', fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax, palette='viridis')
        ax.set_title('Top 10 Most Important Features for Prediction', fontweight='bold')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        st.pyplot(fig)

# Predictions Page
elif page == "🔮 Predictions":
    st.markdown('<h2 class="sub-header">Healthcare Demand Predictor</h2>', unsafe_allow_html=True)
    
    data = load_data()
    
    st.markdown("""
    <div class="info-box">
    Adjust the parameters below to predict healthcare service demand for different demographic profiles.
    The model will estimate the expected number of healthcare visits based on these factors.
    </div>
    """, unsafe_allow_html=True)
    
    # Create input widgets for key features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographic Information")
        age = st.slider("Age", int(data['age'].min()), int(data['age'].max()), 35)
        is_employed = st.selectbox("Employment Status", [0, 1], format_func=lambda x: "Unemployed" if x == 0 else "Employed")
        household_size = st.slider("Household Size", int(data['household_size'].min()), 
                                  int(data['household_size'].max()), 3)
    
    with col2:
        st.subheader("Socioeconomic Factors")
        monthly_income = st.slider("Monthly Income (ZAR)", 
                                  float(data['monthly_income_zar'].min()), 
                                  float(data['monthly_income_zar'].max()), 
                                  float(data['monthly_income_zar'].median()))
        is_urban = st.selectbox("Urban/Rural", [0, 1], format_func=lambda x: "Rural" if x == 0 else "Urban")
        has_medical_scheme = st.selectbox("Has Medical Scheme", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    with col3:
        st.subheader("Health Conditions")
        hiv = st.selectbox("HIV", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    if st.button("🔍 Predict Healthcare Visits", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'is_employed': [is_employed],
            'household_size': [household_size],
            'monthly_income_zar': [monthly_income],
            'is_urban': [is_urban],
            'has_medical_scheme': [has_medical_scheme],
            'hiv': [hiv],
            'diabetes': [diabetes],
            'hypertension': [hypertension],
            'asthma': [0],  # Default value
            'tb': [0],      # Default value
            'province_id_x': [data['province_id_x'].median()],
            'province_id_y': [data['province_id_y'].median()],
            'education_level': [data['education_level'].median()],
            'sex': [data['sex'].median()],
            'district_name': [data['district_name'].median()]
        })
        
        # Train model
        model, X_train, X_test, y_train, y_test, y_pred, mae, mse, r2 = train_model(data)
        
        # Make sure input_data has the same columns as X_train
        for col in X_train.columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        input_data = input_data[X_train.columns]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.markdown(f"""
        <div class="success-box">
        <h3>📋 Prediction Results</h3>
        <p style='font-size: 24px; font-weight: bold; color: #2c3e50;'>
        Predicted number of healthcare visits: <span style='color: #e74c3c;'>{prediction:.2f}</span>
        </p>
        """, unsafe_allow_html=True)
        
        # Interpretation
        if prediction < 1:
            interpretation = "Low healthcare service demand predicted."
            color = "#27ae60"
        elif prediction < 3:
            interpretation = "Moderate healthcare service demand predicted."
            color = "#f39c12"
        else:
            interpretation = "High healthcare service demand predicted."
            color = "#e74c3c"
        
        st.markdown(f"""
        <p style='font-size: 18px; font-weight: bold; color: {color};'>
        {interpretation}
        </p>
        </div>
        """, unsafe_allow_html=True)

# Team Details Page
elif page == "👥 Team Details":
    st.markdown('<h2 class="sub-header">Project Team</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>👥 Group Members</h3>
    This project was developed by our dedicated team for the Technical Programming 2 assessment.
    </div>
    """, unsafe_allow_html=True)
    
    # Team members in cards
    team_members = [
        {"name": "Mbambo AM", "id": "22305677", "role": "Data Processing"},
        {"name": "Mdletshe S", "id": "22317991", "role": "Machine Learning"},
        {"name": "Mngoma S", "id": "22438195", "role": "Visualization"},
        {"name": "Mnqayi LO", "id": "22341285", "role": "Dashboard Design"},
        {"name": "Mnqayi V", "id": "22444713", "role": "Data Analysis"},
        {"name": "Ntuli SB", "id": "22327734", "role": "Model Development"},
        {"name": "Qwabe SS", "id": "22361055", "role": "Project Coordination"}
    ]
    
    # Display team members in a grid
    cols = st.columns(2)
    for i, member in enumerate(team_members):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="team-card">
            <h3>{member['name']}</h3>
            <p><b>Student ID:</b> {member['id']}</p>
            <p><b>Role:</b> {member['role']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Project details
    st.markdown("""
    ## 📚 Project Information
    
    **Course:** Technical Programming 2 (TPRO200/TLPR200)  
    **Assessment:** Assessment 4  
    **Project:** Healthcare Service Demand Prediction  
    **Selected Task:** Task 1 - Predicting Healthcare Service Demand  
    
    ## 🎯 Project Objective
    
    The NHI aims to provide equitable access to healthcare for all South Africans. 
    A critical step in planning for this massive undertaking is accurately forecasting 
    future demand for services. By predicting how many people will need care, the government 
    can better allocate resources like doctors, nurses, medication, and facility space, 
    ensuring the system is efficient and effective from the start.
    
    ## 📊 Dataset
    
    **Dataset Name:** Comprehensive South African Health Simulation Dataset  
    **Note:** This is a simulated dataset created for academic purposes.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
<p>Technical Programming 2 (TPRO200/TLPR200) - Assessment 4 | Healthcare Demand Prediction Dashboard</p>
<p>© 2024 | Group Project</p>
</div>
""", unsafe_allow_html=True)