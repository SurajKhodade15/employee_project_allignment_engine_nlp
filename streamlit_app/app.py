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
    with st.spinner("🚀 Running matching pipeline... This may take a few minutes."):
        try:
            # Load data
            df_emp, df_proj, success = load_data()
            if not success:
                st.error("Failed to load data")
                return False
            
            # Feature extraction
            feature_extractor = FeatureExtractor(st.session_state.config)
            features = feature_extractor.extract_all_features(df_emp, df_proj)
            
            # Matching
            matcher = EmployeeProjectMatcher(df_emp, st.session_state.config)
            matcher.add_experience_filter(min_years=3.0)
            matcher.add_similarity_threshold_filter(min_similarity=0.05)
            
            # Compute similarities
            binary_similarity = matcher.compute_similarity_matrix(
                features['employee_binary'], features['project_binary'], 'cosine'
            )
            
            tfidf_similarity = matcher.compute_similarity_matrix(
                features['employee_tfidf'], features['project_tfidf'], 'cosine'
            )
            
            hybrid_similarity = matcher.compute_hybrid_similarity(
                binary_similarity, tfidf_similarity, 0.7, 0.3
            )
            
            # Generate recommendations
            recommendations_df, stats = matcher.generate_all_recommendations(hybrid_similarity)
            enhanced_recommendations = matcher.enhance_recommendations(recommendations_df, df_proj)
            
            # Save results
            similarity_matrices = {
                'binary': binary_similarity,
                'tfidf': tfidf_similarity,
                'hybrid': hybrid_similarity
            }
            matcher.save_results(enhanced_recommendations, similarity_matrices)
            
            st.session_state.recommendations = enhanced_recommendations
            
            st.success("✅ Matching pipeline completed successfully!")
            st.balloons()
            
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
        ["🏠 Home & Overview", "⚙️ Run Matching Pipeline", "📊 View Recommendations", "🔍 Search & Explore"]
    )
    
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
        
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Important:</strong> Running the pipeline will process all data and generate new recommendations. 
            This process may take several minutes depending on your data size.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Matching Pipeline", type="primary"):
            success = run_matching_pipeline()
            if success:
                st.experimental_rerun()
    
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