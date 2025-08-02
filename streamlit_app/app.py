"""
Streamlit Application for Employee Project Alignment Engine
A comprehensive web interface for employee-project matching using NLP
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocessing import DataPreprocessor
from src.feature_extraction import FeatureExtractor
from src.matching_engine import EmployeeProjectMatcher, load_trained_matcher
from src.utils import Config, Logger

# Page configuration
st.set_page_config(
    page_title="Employee Project Alignment Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = Config()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'pipeline_completed' not in st.session_state:
    st.session_state.pipeline_completed = False
if 'pipeline_stats' not in st.session_state:
    st.session_state.pipeline_stats = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Home & Overview"

def load_data():
    """Load data with caching"""
    try:
        config = st.session_state.config
        
        # Try to load processed data first
        df_emp = pd.read_csv(config.PROCESSED_DATA_DIR / "employee_experience_enhanced.csv")
        df_proj = pd.read_csv(config.PROCESSED_DATA_DIR / "client_projects_cleaned.csv")
        
        return df_emp, df_proj, True
    except FileNotFoundError:
        try:
            # Load raw data and process
            preprocessor = DataPreprocessor(config)
            df_emp, df_proj, _ = preprocessor.process_all_data()
            return df_emp, df_proj, True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None, False

def load_recommendations():
    """Load existing recommendations"""
    try:
        config = st.session_state.config
        recommendations = pd.read_csv(config.OUTPUT_DATA_DIR / "final_recommendations.csv")
        return recommendations
    except FileNotFoundError:
        return None

def create_overview_metrics(df_emp, df_proj, recommendations=None):
    """Create overview metrics display"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>👥 Total Employees</h3>
            <h2>{}</h2>
        </div>
        """.format(len(df_emp)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📋 Total Projects</h3>
            <h2>{}</h2>
        </div>
        """.format(len(df_proj)), unsafe_allow_html=True)
    
    with col3:
        departments = df_emp['Department'].nunique()
        st.markdown("""
        <div class="metric-card">
            <h3>🏢 Departments</h3>
            <h2>{}</h2>
        </div>
        """.format(departments), unsafe_allow_html=True)
    
    with col4:
        if recommendations is not None:
            total_recs = len(recommendations)
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Recommendations</h3>
                <h2>{}</h2>
            </div>
            """.format(total_recs), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Recommendations</h3>
                <h2>Not Generated</h2>
            </div>
            """, unsafe_allow_html=True)

def create_data_overview_charts(df_emp, df_proj):
    """Create data overview visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Employee Department Distribution")
        dept_counts = df_emp['Department'].value_counts()
        fig_dept = px.pie(
            values=dept_counts.values, 
            names=dept_counts.index,
            title="Employee Distribution by Department",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_dept.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_dept, use_container_width=True)
    
    with col2:
        st.subheader("📍 Employee Location Distribution")
        location_counts = df_emp['Location'].value_counts()
        fig_location = px.bar(
            x=location_counts.index, 
            y=location_counts.values,
            title="Employee Distribution by Location",
            color=location_counts.values,
            color_continuous_scale="viridis"
        )
        fig_location.update_layout(xaxis_title="Location", yaxis_title="Number of Employees")
        st.plotly_chart(fig_location, use_container_width=True)
    
    # Experience distribution
    st.subheader("📈 Employee Experience Distribution")
    fig_exp = px.histogram(
        df_emp, 
        x='Years_Experience', 
        nbins=20,
        title="Distribution of Years of Experience",
        color_discrete_sequence=['#3498db']
    )
    fig_exp.update_layout(xaxis_title="Years of Experience", yaxis_title="Count")
    st.plotly_chart(fig_exp, use_container_width=True)

def run_matching_pipeline():
    """Run the complete matching pipeline"""
    try:
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load data
            status_text.text("📂 Loading and preprocessing data...")
            progress_bar.progress(10)
            
            df_emp, df_proj, success = load_data()
            if not success:
                st.error("Failed to load data")
                return False
            
            progress_bar.progress(20)
            
            # Step 2: Feature extraction
            status_text.text("🔧 Extracting features...")
            feature_extractor = FeatureExtractor(st.session_state.config)
            features = feature_extractor.extract_all_features(df_emp, df_proj)
            progress_bar.progress(50)
            
            # Step 3: Matching
            status_text.text("🎯 Setting up matching engine...")
            matcher = EmployeeProjectMatcher(df_emp, st.session_state.config)
            matcher.add_experience_filter(min_years=3.0)
            matcher.add_similarity_threshold_filter(min_similarity=0.05)
            progress_bar.progress(60)
            
            # Step 4: Compute similarities
            status_text.text("🔬 Computing similarity matrices...")
            binary_similarity = matcher.compute_similarity_matrix(
                features['employee_binary'], features['project_binary'], 'cosine'
            )
            
            tfidf_similarity = matcher.compute_similarity_matrix(
                features['employee_tfidf'], features['project_tfidf'], 'cosine'
            )
            
            hybrid_similarity = matcher.compute_hybrid_similarity(
                binary_similarity, tfidf_similarity, 0.7, 0.3
            )
            progress_bar.progress(80)
            
            # Step 5: Generate recommendations
            status_text.text("📊 Generating recommendations...")
            recommendations_df, stats = matcher.generate_all_recommendations(hybrid_similarity)
            enhanced_recommendations = matcher.enhance_recommendations(recommendations_df, df_proj)
            progress_bar.progress(90)
            
            # Step 6: Save results
            status_text.text("💾 Saving results...")
            similarity_matrices = {
                'binary': binary_similarity,
                'tfidf': tfidf_similarity,
                'hybrid': hybrid_similarity
            }
            matcher.save_results(enhanced_recommendations, similarity_matrices)
            progress_bar.progress(100)
            
            # Update session state
            st.session_state.recommendations = enhanced_recommendations
            st.session_state.pipeline_completed = True
            st.session_state.pipeline_stats = {
                'total_recommendations': len(enhanced_recommendations),
                'unique_projects': enhanced_recommendations['Project_ID'].nunique(),
                'unique_employees': enhanced_recommendations['Employee_ID'].nunique(),
                'avg_similarity': enhanced_recommendations['Similarity_Score'].mean()
            }
            
            status_text.text("✅ Pipeline completed successfully!")
            progress_bar.empty()
            
            return True
            
    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        return False

def display_recommendations_analysis(recommendations):
    """Display comprehensive recommendations analysis"""
    st.header("📊 Recommendations Analysis")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Recommendations", len(recommendations))
    with col2:
        st.metric("Unique Projects", recommendations['Project_ID'].nunique())
    with col3:
        st.metric("Unique Employees", recommendations['Employee_ID'].nunique())
    with col4:
        avg_similarity = recommendations['Similarity_Score'].mean()
        st.metric("Avg Similarity Score", f"{avg_similarity:.3f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Similarity Score Distribution")
        fig_sim = px.histogram(
            recommendations, 
            x='Similarity_Score', 
            nbins=30,
            title="Distribution of Similarity Scores",
            color_discrete_sequence=['#e74c3c']
        )
        st.plotly_chart(fig_sim, use_container_width=True)
    
    with col2:
        st.subheader("🏢 Recommendations by Department")
        dept_recs = recommendations['Department'].value_counts().head(10)
        fig_dept_recs = px.bar(
            x=dept_recs.values, 
            y=dept_recs.index,
            orientation='h',
            title="Top 10 Departments in Recommendations",
            color=dept_recs.values,
            color_continuous_scale="plasma"
        )
        st.plotly_chart(fig_dept_recs, use_container_width=True)
    
    # Experience level analysis
    st.subheader("👨‍💼 Experience Level Analysis")
    exp_level_counts = recommendations['Experience_Level'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_exp_pie = px.pie(
            values=exp_level_counts.values,
            names=exp_level_counts.index,
            title="Recommendations by Experience Level"
        )
        st.plotly_chart(fig_exp_pie, use_container_width=True)
    
    with col2:
        # Location matching analysis
        location_match = recommendations['Location_Match'].value_counts()
        fig_location_match = px.bar(
            x=['Different Location', 'Same Location'],
            y=[location_match.get(False, 0), location_match.get(True, 0)],
            title="Location Matching Analysis",
            color=['Different Location', 'Same Location'],
            color_discrete_map={'Same Location': '#2ecc71', 'Different Location': '#e74c3c'}
        )
        st.plotly_chart(fig_location_match, use_container_width=True)
    
    # Top projects and employees
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Projects by Recommendations")
        top_projects = recommendations['Project_ID'].value_counts().head(10)
        st.bar_chart(top_projects)
    
    with col2:
        st.subheader("⭐ Top Employees by Recommendations")
        top_employees = recommendations['Employee_ID'].value_counts().head(10)
        st.bar_chart(top_employees)

def project_employee_search():
    """Interactive project-employee search interface"""
    st.header("🔍 Interactive Project-Employee Search")
    
    # Load data
    df_emp, df_proj, success = load_data()
    recommendations = load_recommendations()
    
    if not success or recommendations is None:
        st.warning("⚠️ Please run the matching pipeline first to enable search functionality.")
        return
    
    # Search interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Find Employees for Project")
        selected_project = st.selectbox(
            "Select a Project:",
            options=df_proj['Project_ID'].tolist(),
            key="project_search"
        )
        
        if selected_project:
            # Get project details
            project_info = df_proj[df_proj['Project_ID'] == selected_project].iloc[0]
            
            st.markdown(f"""
            <div class="info-box">
                <strong>Project Details:</strong><br>
                📋 <strong>Client:</strong> {project_info['Client_Name']}<br>
                📍 <strong>Location:</strong> {project_info['Location']}<br>
                🔧 <strong>Required Skills:</strong> {project_info['Required_Skills']}<br>
                📊 <strong>Status:</strong> {project_info['Status']}
            </div>
            """, unsafe_allow_html=True)
            
            # Get recommendations for this project
            project_recs = recommendations[
                recommendations['Project_ID'] == selected_project
            ].sort_values('Similarity_Score', ascending=False)
            
            if not project_recs.empty:
                st.subheader("🎖️ Top Recommended Employees")
                
                # Display top recommendations
                for idx, row in project_recs.head(5).iterrows():
                    similarity_color = "🟢" if row['Similarity_Score'] > 0.6 else "🟡" if row['Similarity_Score'] > 0.3 else "🔴"
                    location_icon = "📍" if row['Location_Match'] else "🌐"
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>#{row['Rank']} - {row['Employee_ID']}</strong> {similarity_color}<br>
                        🎯 <strong>Similarity Score:</strong> {row['Similarity_Score']:.3f}<br>
                        🏢 <strong>Department:</strong> {row['Department']}<br>
                        📅 <strong>Experience:</strong> {row['Years_Experience']:.1f} years ({row['Experience_Level']})<br>
                        {location_icon} <strong>Location:</strong> {row['Location_Employee']}<br>
                        💪 <strong>Strength:</strong> {row['Recommendation_Strength']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found for this project.")
    
    with col2:
        st.subheader("👤 Find Projects for Employee")
        selected_employee = st.selectbox(
            "Select an Employee:",
            options=df_emp['Employee_ID'].tolist(),
            key="employee_search"
        )
        
        if selected_employee:
            # Get employee details
            employee_info = df_emp[df_emp['Employee_ID'] == selected_employee].iloc[0]
            
            st.markdown(f"""
            <div class="info-box">
                <strong>Employee Details:</strong><br>
                👤 <strong>ID:</strong> {employee_info['Employee_ID']}<br>
                🏢 <strong>Department:</strong> {employee_info['Department']}<br>
                📍 <strong>Location:</strong> {employee_info['Location']}<br>
                📅 <strong>Experience:</strong> {employee_info['Years_Experience']:.1f} years<br>
                🛠️ <strong>Skills:</strong> {employee_info['Skills'][:100]}...
            </div>
            """, unsafe_allow_html=True)
            
            # Get projects for this employee
            employee_recs = recommendations[
                recommendations['Employee_ID'] == selected_employee
            ].sort_values('Similarity_Score', ascending=False)
            
            if not employee_recs.empty:
                st.subheader("🎯 Recommended Projects")
                
                # Display top project recommendations
                for idx, row in employee_recs.head(5).iterrows():
                    similarity_color = "🟢" if row['Similarity_Score'] > 0.6 else "🟡" if row['Similarity_Score'] > 0.3 else "🔴"
                    location_icon = "📍" if row['Location_Match'] else "🌐"
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>#{row['Rank']} - {row['Project_ID']}</strong> {similarity_color}<br>
                        🎯 <strong>Similarity Score:</strong> {row['Similarity_Score']:.3f}<br>
                        🏢 <strong>Client:</strong> {row['Client_Name']}<br>
                        {location_icon} <strong>Location:</strong> {row['Location_Project']}<br>
                        💪 <strong>Strength:</strong> {row['Recommendation_Strength']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No project recommendations found for this employee.")

def hr_employee_project_matcher():
    """HR functionality to input employee details and get project recommendations"""
    st.header("👔 HR Employee-Project Matcher")
    st.markdown("""
    <div class="info-box">
        <strong>HR Project Matching Tool</strong><br>
        Enter employee details below to find the most suitable projects based on their profile.
    </div>
    """, unsafe_allow_html=True)
    
    # Create input form
    with st.form("employee_details_form"):
        st.subheader("📝 Employee Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            department = st.selectbox(
                "🏢 Department",
                ["Data Science", "AI Research", "Software Engineering", "Web Development", 
                 "Mobile Development", "DevOps", "Quality Assurance", "Product Management",
                 "Business Analysis", "Consulting", "Finance", "Marketing", "Other"],
                help="Select the employee's department"
            )
            
            years_experience = st.number_input(
                "📅 Years of Experience",
                min_value=0.0,
                max_value=50.0,
                value=3.0,
                step=0.5,
                help="Enter the total years of professional experience"
            )
        
        with col2:
            location = st.selectbox(
                "🌍 Location",
                ["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Chennai", "Pune", 
                 "Kolkata", "Ahmedabad", "Remote", "Other"],
                help="Select the employee's preferred work location"
            )
            
            availability = st.selectbox(
                "⏰ Availability",
                ["Immediately Available", "Available in 1 week", "Available in 2 weeks", 
                 "Available in 1 month", "Available in 2 months"],
                help="When can the employee start a new project?"
            )
        
        st.subheader("💼 Experience & Skills")
        experience_text = st.text_area(
            "📝 Experience Description",
            placeholder="Describe the employee's skills, technologies, projects, and experience...\n\nExample:\n- Python programming for data analysis\n- Machine learning model development\n- SQL database management\n- Experience with TensorFlow and scikit-learn\n- Led 3 data science projects for healthcare clients",
            height=150,
            help="Provide detailed information about skills, technologies, and project experience"
        )
        
        # Additional preferences
        st.subheader("🎯 Project Preferences")
        col1, col2 = st.columns(2)
        
        with col1:
            project_type_preference = st.multiselect(
                "🎯 Preferred Project Types",
                ["Data Science", "Machine Learning", "Web Development", "Mobile Apps", 
                 "AI/ML Research", "Analytics", "Consulting", "Product Development"],
                help="Select preferred types of projects"
            )
        
        with col2:
            client_type_preference = st.selectbox(
                "🏢 Client Type Preference",
                ["No Preference", "Startup", "Enterprise", "Government", "Healthcare", 
                 "Finance", "Technology", "Education"],
                help="Preferred client industry or type"
            )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Find Matching Projects", type="primary")
    
    # Process the form when submitted
    if submitted:
        if not experience_text.strip():
            st.error("❌ Please provide experience description to get accurate recommendations.")
            return
        
        with st.spinner("🔍 Finding suitable projects..."):
            # Create temporary employee profile
            temp_employee = {
                'Employee_ID': 'TEMP_HR_001',
                'Department': department,
                'Location': location,
                'Years_Experience': years_experience,
                'Skills': experience_text,
                'Experience_Text': experience_text,
                'Availability': availability,
                'Project_Type_Preference': ', '.join(project_type_preference) if project_type_preference else '',
                'Client_Type_Preference': client_type_preference
            }
            
            # Find matching projects
            recommendations = find_projects_for_employee_profile(temp_employee)
            
            if recommendations is not None and len(recommendations) > 0:
                st.success(f"✅ Found {len(recommendations)} suitable projects!")
                
                # Display recommendations
                st.subheader("🎯 Recommended Projects")
                
                for idx, project in recommendations.iterrows():
                    with st.expander(f"🚀 {project['Project_Name']} - Score: {project['similarity_score']:.3f}", expanded=idx<3):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **🏢 Client:** {project['Client_Name']}<br>
                            **🌍 Location:** {project['Location']}<br>
                            **📋 Status:** {project['Status']}<br>
                            **💼 Required Skills:** {project['Required_Skills']}<br>
                            **🎯 Match Score:** {project['similarity_score']:.3f}/1.000
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Match strength indicator
                            if project['similarity_score'] >= 0.7:
                                strength = "🟢 Excellent Match"
                                color = "green"
                            elif project['similarity_score'] >= 0.5:
                                strength = "🟡 Good Match"
                                color = "orange"
                            else:
                                strength = "🟠 Moderate Match"
                                color = "red"
                            
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; background-color: {color}20; border-radius: 5px;">
                                <strong>{strength}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Why this project matches
                        st.markdown("**🔍 Why this project matches:**")
                        match_reasons = generate_match_reasons(temp_employee, project)
                        for reason in match_reasons:
                            st.markdown(f"• {reason}")
                
                # Export options
                st.subheader("📥 Export Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_data = recommendations.to_csv(index=False)
                    st.download_button(
                        "📄 Download CSV",
                        csv_data,
                        f"hr_project_recommendations_{temp_employee['Employee_ID']}.csv",
                        "text/csv"
                    )
                
                with col2:
                    # Create summary report
                    report = create_hr_matching_report(temp_employee, recommendations)
                    st.download_button(
                        "📋 Download Report",
                        report,
                        f"hr_matching_report_{temp_employee['Employee_ID']}.txt",
                        "text/plain"
                    )
            else:
                st.warning("⚠️ No suitable projects found. Try adjusting the experience description or preferences.")

def find_projects_for_employee_profile(employee_profile):
    """Find suitable projects for a given employee profile using the matching engine"""
    try:
        # Load project data
        config = st.session_state.config
        
        # Try processed data first, then raw data
        try:
            df_projects = pd.read_csv(config.PROCESSED_DATA_DIR / "client_projects_cleaned.csv")
        except FileNotFoundError:
            # Load and process raw data
            df_projects_raw = pd.read_csv(config.RAW_DATA_DIR / "client_projects.csv")
            preprocessor = DataPreprocessor(config)
            df_projects = preprocessor.clean_projects(df_projects_raw)
        
        # Convert employee profile to dataframe format
        employee_df = pd.DataFrame([employee_profile])
        
        # Initialize preprocessor and feature extractor
        preprocessor = DataPreprocessor(config)
        feature_extractor = FeatureExtractor(config)
        
        # Clean the employee data using the employee experience cleaning method
        # Since HR provides complete info in one record, we'll use this method
        employee_df = preprocessor.clean_employee_experience(employee_df)
        
        # Try to load existing project features or extract them
        try:
            # Load existing features if available
            with open(config.MODEL_DIR / "project_tfidf_matrix.pkl", 'rb') as f:
                project_tfidf = pickle.load(f)
            with open(config.MODEL_DIR / "project_binary_matrix.pkl", 'rb') as f:
                project_binary = pickle.load(f)
            with open(config.MODEL_DIR / "tfidf_vectorizer.pkl", 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            with open(config.MODEL_DIR / "skills_vocabulary.pkl", 'rb') as f:
                skills_vocabulary = pickle.load(f)
                
            # Extract features for the single employee using existing models
            emp_binary = feature_extractor._create_binary_matrix(
                employee_df, 'Skills', skills_vocabulary, 'Employee_ID'
            )
            
            # For TF-IDF, we need to transform using the existing vectorizer
            # and ensure the result matches the project feature dimensions
            emp_skills_text = employee_df['Skills'].fillna('').tolist()
            emp_tfidf_matrix = tfidf_vectorizer.transform(emp_skills_text)
            
            # Debug: Check if TF-IDF dimensions match what we expect
            expected_tfidf_cols = project_tfidf.shape[1]
            actual_tfidf_cols = emp_tfidf_matrix.shape[1]
            
            if actual_tfidf_cols != expected_tfidf_cols:
                st.warning(f"TF-IDF dimension mismatch: vectorizer produces {actual_tfidf_cols} features, but project has {expected_tfidf_cols}")
                # Create properly sized TF-IDF matrix
                if actual_tfidf_cols < expected_tfidf_cols:
                    # Pad with zeros if vectorizer produces fewer features
                    padded_matrix = np.zeros((1, expected_tfidf_cols))
                    padded_matrix[:, :actual_tfidf_cols] = emp_tfidf_matrix.toarray()
                    emp_tfidf = pd.DataFrame(
                        padded_matrix,
                        index=employee_df['Employee_ID'],
                        columns=project_tfidf.columns
                    )
                else:
                    # Truncate if vectorizer produces more features
                    truncated_matrix = emp_tfidf_matrix.toarray()[:, :expected_tfidf_cols]
                    emp_tfidf = pd.DataFrame(
                        truncated_matrix,
                        index=employee_df['Employee_ID'],
                        columns=project_tfidf.columns
                    )
            else:
                # Perfect match - use as is
                emp_tfidf = pd.DataFrame(
                    emp_tfidf_matrix.toarray(), 
                    index=employee_df['Employee_ID'],
                    columns=project_tfidf.columns
                )
            
            # For binary features, align with project features
            if emp_binary.shape[1] != project_binary.shape[1]:
                # Create aligned binary matrix
                aligned_binary = pd.DataFrame(
                    0, 
                    index=emp_binary.index, 
                    columns=project_binary.columns
                )
                # Fill in the matching columns
                for col in emp_binary.columns:
                    if col in aligned_binary.columns:
                        aligned_binary[col] = emp_binary[col]
                emp_binary = aligned_binary
                
        except FileNotFoundError:
            # If no existing features, extract features for both employee and projects
            st.warning("Pre-trained models not found. Extracting features from scratch...")
            features = feature_extractor.extract_all_features(employee_df, df_projects)
            emp_binary = features['employee_binary']
            emp_tfidf = features['employee_tfidf']
            project_binary = features['project_binary']
            project_tfidf = features['project_tfidf']
        
        # Initialize matcher
        matcher = EmployeeProjectMatcher(employee_df, config)
        
        # Verify feature dimensions match before computing similarities
        if emp_binary.shape[1] != project_binary.shape[1]:
            st.error(f"Binary feature dimension mismatch: {emp_binary.shape[1]} vs {project_binary.shape[1]}")
            return None
            
        if emp_tfidf.shape[1] != project_tfidf.shape[1]:
            st.error(f"TF-IDF feature dimension mismatch: {emp_tfidf.shape[1]} vs {project_tfidf.shape[1]}")
            return None
        
        # Compute similarities
        binary_similarity = matcher.compute_similarity_matrix(
            emp_binary, project_binary, 'cosine'
        )
        
        tfidf_similarity = matcher.compute_similarity_matrix(
            emp_tfidf, project_tfidf, 'cosine'
        )
        
        # Combine similarities (70% TF-IDF, 30% binary for better semantic matching)
        hybrid_similarity = matcher.compute_hybrid_similarity(
            binary_similarity, tfidf_similarity, 0.3, 0.7
        )
        
        # Get top project recommendations
        similarities = hybrid_similarity[0]  # Get similarities for our single employee
        
        # Create recommendations dataframe
        recommendations = []
        for proj_idx, similarity in enumerate(similarities):
            if similarity > 0.05:  # Minimum threshold
                project_info = df_projects.iloc[proj_idx].copy()
                project_info['similarity_score'] = similarity
                recommendations.append(project_info)
        
        if recommendations:
            recommendations_df = pd.DataFrame(recommendations)
            recommendations_df = recommendations_df.sort_values('similarity_score', ascending=False)
            return recommendations_df.head(10)  # Top 10 recommendations
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error finding projects: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return None

def generate_match_reasons(employee, project):
    """Generate reasons why an employee matches a project"""
    reasons = []
    
    # Check skill overlap
    emp_skills = employee['Skills'].lower().split()
    proj_skills = project['Required_Skills'].lower().split()
    
    common_skills = set(emp_skills) & set(proj_skills)
    if common_skills:
        reasons.append(f"Skills match: {', '.join(list(common_skills)[:3])}")
    
    # Check experience level
    if employee['Years_Experience'] >= 5:
        reasons.append("Senior experience level suitable for project complexity")
    elif employee['Years_Experience'] >= 2:
        reasons.append("Mid-level experience appropriate for project requirements")
    
    # Check location
    if employee['Location'] == project['Location']:
        reasons.append("Same location - no relocation required")
    elif employee['Location'] == "Remote":
        reasons.append("Remote work preference - flexible location")
    
    # Check department alignment
    dept_project_mapping = {
        'Data Science': ['analytics', 'data', 'science', 'ml', 'ai'],
        'AI Research': ['ai', 'machine learning', 'deep learning', 'nlp'],
        'Software Engineering': ['software', 'development', 'programming'],
        'Web Development': ['web', 'frontend', 'backend', 'javascript']
    }
    
    dept_keywords = dept_project_mapping.get(employee['Department'], [])
    proj_text = project['Required_Skills'].lower()
    
    if any(keyword in proj_text for keyword in dept_keywords):
        reasons.append(f"Department expertise aligns with project needs")
    
    if not reasons:
        reasons.append("General skill alignment based on experience description")
    
    return reasons

def create_hr_matching_report(employee, recommendations):
    """Create a detailed HR matching report"""
    report = f"""
EMPLOYEE PROJECT MATCHING REPORT
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

EMPLOYEE PROFILE:
================
Employee ID: {employee['Employee_ID']}
Department: {employee['Department']}
Location: {employee['Location']}
Years of Experience: {employee['Years_Experience']}
Availability: {employee['Availability']}

EXPERIENCE SUMMARY:
==================
{employee['Experience_Text'][:500]}...

PROJECT RECOMMENDATIONS:
========================
Total Projects Found: {len(recommendations)}

TOP 5 RECOMMENDATIONS:
"""
    
    for idx, project in recommendations.head(5).iterrows():
        report += f"""
{idx+1}. {project['Project_Name']}
   Client: {project['Client_Name']}
   Location: {project['Location']}
   Match Score: {project['similarity_score']:.3f}
   Required Skills: {project['Required_Skills']}
   Status: {project['Status']}
   
"""
    
    report += """
RECOMMENDATIONS:
================
- Consider top 3 projects for immediate assignment
- Schedule interviews with project managers
- Verify specific technical requirements
- Confirm availability dates with both parties

NOTES:
======
This report is generated automatically based on skill matching algorithms.
Please verify all recommendations with manual review and stakeholder input.
"""
    
    return report

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">🎯 Employee Project Alignment Engine</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Welcome to the Employee Project Alignment Engine!</strong><br>
        This advanced NLP-powered system helps match employees to projects based on skills, experience, and other factors.
        Use the sidebar to navigate between different features.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home & Overview", "⚙️ Run Matching Pipeline", "📊 View Recommendations", "🔍 Search & Explore", "👔 HR Employee-Project Matcher"],
        index=["🏠 Home & Overview", "⚙️ Run Matching Pipeline", "📊 View Recommendations", "🔍 Search & Explore", "👔 HR Employee-Project Matcher"].index(st.session_state.current_page) if st.session_state.current_page in ["🏠 Home & Overview", "⚙️ Run Matching Pipeline", "📊 View Recommendations", "🔍 Search & Explore", "👔 HR Employee-Project Matcher"] else 0
    )
    
    # Update current page in session state
    st.session_state.current_page = page
    
    # Add contextual sidebar information
    if page == "👔 HR Employee-Project Matcher":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💡 HR Matching Tips")
        st.sidebar.markdown("""
        **For best results:**
        - Provide detailed experience description
        - Include specific technologies and tools
        - Mention project types and domains
        - Specify years of experience accurately
        
        **The system will find projects that:**
        - Match technical skills
        - Align with experience level
        - Consider location preferences
        - Respect project requirements
        """)
    
    # Load basic data for all pages
    df_emp, df_proj, data_success = load_data()
    recommendations = load_recommendations()
    
    if page == "🏠 Home & Overview":
        st.header("📈 System Overview")
        
        if data_success:
            create_overview_metrics(df_emp, df_proj, recommendations)
            st.markdown("---")
            create_data_overview_charts(df_emp, df_proj)
        else:
            st.error("❌ Unable to load data. Please check your data files.")
    
    elif page == "⚙️ Run Matching Pipeline":
        st.header("🚀 Run Matching Pipeline")
        
        # Show success message if pipeline was just completed
        if st.session_state.pipeline_completed:
            st.success("✅ Matching pipeline completed successfully!")
            
            if st.session_state.pipeline_stats:
                stats = st.session_state.pipeline_stats
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📋 Total Recommendations", stats['total_recommendations'])
                with col2:
                    st.metric("🎯 Unique Projects", stats['unique_projects'])
                with col3:
                    st.metric("👥 Unique Employees", stats['unique_employees'])
                with col4:
                    st.metric("📊 Avg Similarity", f"{stats['avg_similarity']:.3f}")
            
            st.markdown("""
            <div class="success-box">
                <strong>🎉 Pipeline Results:</strong><br>
                ✅ Data preprocessing completed<br>
                ✅ Feature extraction completed<br>
                ✅ Similarity computation completed<br>
                ✅ Recommendations generated and saved<br>
                📁 Results saved to data/outputs/ folder
            </div>
            """, unsafe_allow_html=True)
            
            # Button to view recommendations
            if st.button("📊 View Recommendations", type="primary"):
                st.session_state.current_page = "📊 View Recommendations"
                st.rerun()
            
            # Button to reset and run again
            if st.button("🔄 Run Pipeline Again"):
                st.session_state.pipeline_completed = False
                st.session_state.pipeline_stats = None
                st.rerun()
        
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Important:</strong> Running the pipeline will process all data and generate new recommendations. 
                This process may take several minutes depending on your data size.
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Start Matching Pipeline", type="primary"):
                success = run_matching_pipeline()
                if success:
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Pipeline failed. Please check the logs for details.")
    
    elif page == "📊 View Recommendations":
        if recommendations is not None:
            display_recommendations_analysis(recommendations)
            
            # Download options
            st.header("📥 Download Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = recommendations.to_csv(index=False)
                st.download_button(
                    label="📄 Download Full Recommendations",
                    data=csv,
                    file_name="employee_project_recommendations.csv",
                    mime="text/csv"
                )
            
            with col2:
                top_candidates = recommendations[recommendations['Rank'] == 1]
                csv_top = top_candidates.to_csv(index=False)
                st.download_button(
                    label="🏆 Download Top Candidates",
                    data=csv_top,
                    file_name="top_candidates_per_project.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Create summary report
                summary = recommendations.groupby('Project_ID').agg({
                    'Employee_ID': 'count',
                    'Similarity_Score': 'mean'
                }).round(3)
                csv_summary = summary.to_csv()
                st.download_button(
                    label="📊 Download Project Summary",
                    data=csv_summary,
                    file_name="project_summary.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ No recommendations found. Please run the matching pipeline first.")
    
    elif page == "🔍 Search & Explore":
        project_employee_search()
    
    elif page == "👔 HR Employee-Project Matcher":
        hr_employee_project_matcher()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <strong>Employee Project Alignment Engine v1.0</strong><br>
        Powered by Advanced NLP & Machine Learning | Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()