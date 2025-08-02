"""
Matching engine module for Employee Project Alignment Engine
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional, Any
from .utils import Config, Logger, save_data, calculate_similarity_stats, categorize_experience


class EmployeeProjectMatcher:
    """Advanced matching engine with configurable business rules"""
    
    def __init__(self, df_emp: pd.DataFrame, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = Logger(__name__)
        self.df_emp = df_emp
        self.similarity_matrices = {}
        self.filters = []
        
    def compute_similarity_matrix(self, employee_matrix: pd.DataFrame, 
                                project_matrix: pd.DataFrame, method: str = 'cosine') -> pd.DataFrame:
        """Compute similarity matrix between employees and projects"""
        self.logger.info(f"Computing {method} similarity matrix...")
        
        if method == 'cosine':
            similarity_matrix = cosine_similarity(project_matrix.values, employee_matrix.values)
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
        
        # Create DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=project_matrix.index,
            columns=employee_matrix.index
        )
        
        # Calculate statistics
        stats = calculate_similarity_stats(similarity_df)
        self.logger.info(f"Similarity matrix: {similarity_df.shape}, Mean: {stats['mean']:.4f}")
        
        return similarity_df
    
    def compute_hybrid_similarity(self, binary_sim: pd.DataFrame, tfidf_sim: pd.DataFrame,
                                binary_weight: float = 0.7, tfidf_weight: float = 0.3) -> pd.DataFrame:
        """Combine multiple similarity matrices"""
        self.logger.info("Computing hybrid similarity matrix...")
        
        # Ensure matrices have same dimensions
        common_projects = set(binary_sim.index) & set(tfidf_sim.index)
        common_employees = set(binary_sim.columns) & set(tfidf_sim.columns)
        
        binary_aligned = binary_sim.loc[list(common_projects), list(common_employees)]
        tfidf_aligned = tfidf_sim.loc[list(common_projects), list(common_employees)]
        
        # Weighted combination
        hybrid_sim = (binary_weight * binary_aligned + tfidf_weight * tfidf_aligned)
        
        self.logger.info(f"Hybrid similarity matrix: {hybrid_sim.shape}")
        return hybrid_sim
    
    def add_experience_filter(self, min_years: float = None, max_years: float = None):
        """Add experience-based filter"""
        min_years = min_years or self.config.MIN_EXPERIENCE_YEARS
        
        def experience_filter(employee_ids):
            eligible = self.df_emp[
                (self.df_emp['Years_Experience'] >= min_years) &
                (self.df_emp['Years_Experience'] <= (max_years or float('inf')))
            ]['Employee_ID'].tolist()
            return [emp for emp in employee_ids if emp in eligible]
        
        self.filters.append(('experience', experience_filter))
        self.logger.info(f"Added experience filter: {min_years}+ years")
    
    def add_department_filter(self, preferred_departments: List[str] = None, 
                            excluded_departments: List[str] = None):
        """Add department-based filter"""
        def department_filter(employee_ids):
            if preferred_departments:
                eligible = self.df_emp[
                    self.df_emp['Department'].isin(preferred_departments)
                ]['Employee_ID'].tolist()
                employee_ids = [emp for emp in employee_ids if emp in eligible]
            
            if excluded_departments:
                excluded = self.df_emp[
                    self.df_emp['Department'].isin(excluded_departments)
                ]['Employee_ID'].tolist()
                employee_ids = [emp for emp in employee_ids if emp not in excluded]
            
            return employee_ids
        
        self.filters.append(('department', department_filter))
        self.logger.info("Added department filter")
    
    def add_location_filter(self, preferred_locations: List[str] = None,
                          excluded_locations: List[str] = None):
        """Add location-based filter"""
        def location_filter(employee_ids):
            if preferred_locations:
                eligible = self.df_emp[
                    self.df_emp['Location'].isin(preferred_locations)
                ]['Employee_ID'].tolist()
                employee_ids = [emp for emp in employee_ids if emp in eligible]
            
            if excluded_locations:
                excluded = self.df_emp[
                    self.df_emp['Location'].isin(excluded_locations)
                ]['Employee_ID'].tolist()
                employee_ids = [emp for emp in employee_ids if emp not in excluded]
            
            return employee_ids
        
        self.filters.append(('location', location_filter))
        self.logger.info("Added location filter")
    
    def add_similarity_threshold_filter(self, min_similarity: float = None):
        """Add minimum similarity threshold filter"""
        min_similarity = min_similarity or self.config.MIN_SIMILARITY_THRESHOLD
        
        def similarity_filter(employee_ids, project_id, similarity_matrix):
            if project_id not in similarity_matrix.index:
                return employee_ids
            
            project_similarities = similarity_matrix.loc[project_id]
            eligible = project_similarities[project_similarities >= min_similarity].index.tolist()
            return [emp for emp in employee_ids if emp in eligible]
        
        self.filters.append(('similarity_threshold', similarity_filter))
        self.logger.info(f"Added similarity threshold filter: {min_similarity}+")
    
    def apply_filters(self, employee_ids: List[str], project_id: str = None, 
                     similarity_matrix: pd.DataFrame = None) -> List[str]:
        """Apply all configured filters"""
        filtered_ids = employee_ids.copy()
        
        for filter_name, filter_func in self.filters:
            if filter_name == 'similarity_threshold' and project_id and similarity_matrix is not None:
                filtered_ids = filter_func(filtered_ids, project_id, similarity_matrix)
            else:
                filtered_ids = filter_func(filtered_ids)
        
        return filtered_ids
    
    def get_recommendations(self, project_id: str, similarity_matrix: pd.DataFrame,
                          top_n: int = None, apply_business_rules: bool = True) -> pd.Series:
        """Get top N employee recommendations for a project"""
        top_n = top_n or self.config.TOP_N_RECOMMENDATIONS
        
        if project_id not in similarity_matrix.index:
            self.logger.warning(f"Project {project_id} not found in similarity matrix")
            return pd.Series()
        
        # Get similarity scores for the project
        project_similarities = similarity_matrix.loc[project_id].sort_values(ascending=False)
        
        if apply_business_rules:
            # Apply filters
            eligible_employees = self.apply_filters(
                project_similarities.index.tolist(), 
                project_id, 
                similarity_matrix
            )
            project_similarities = project_similarities[eligible_employees]
        
        return project_similarities.head(top_n)
    
    def generate_all_recommendations(self, similarity_matrix: pd.DataFrame, 
                                   top_n: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate recommendations for all projects"""
        top_n = top_n or self.config.TOP_N_RECOMMENDATIONS
        self.logger.info(f"Generating recommendations for {len(similarity_matrix.index)} projects...")
        
        all_recommendations = []
        stats = {
            'projects_processed': 0,
            'total_recommendations': 0,
            'projects_with_no_candidates': 0,
            'avg_similarity_score': 0.0
        }
        
        for project_id in similarity_matrix.index:
            recommendations = self.get_recommendations(
                project_id, similarity_matrix, top_n, apply_business_rules=True
            )
            
            if len(recommendations) > 0:
                for rank, (emp_id, similarity_score) in enumerate(recommendations.items(), 1):
                    all_recommendations.append({
                        'Project_ID': project_id,
                        'Employee_ID': emp_id,
                        'Similarity_Score': similarity_score,
                        'Rank': rank
                    })
                
                stats['total_recommendations'] += len(recommendations)
                stats['avg_similarity_score'] += recommendations.mean()
            else:
                stats['projects_with_no_candidates'] += 1
            
            stats['projects_processed'] += 1
        
        # Calculate final stats
        if stats['projects_processed'] > 0:
            stats['avg_similarity_score'] /= stats['projects_processed']
        
        recommendations_df = pd.DataFrame(all_recommendations)
        
        self.logger.info(f"Generated {stats['total_recommendations']} recommendations")
        return recommendations_df, stats
    
    def enhance_recommendations(self, recommendations_df: pd.DataFrame, 
                              df_proj: pd.DataFrame) -> pd.DataFrame:
        """Enhance recommendations with metadata"""
        self.logger.info("Enhancing recommendations with metadata...")
        
        # Merge with employee data
        enhanced_df = recommendations_df.merge(
            self.df_emp[['Employee_ID', 'Department', 'Years_Experience', 'Location']], 
            on='Employee_ID', 
            how='left'
        )
        
        # Merge with project data
        enhanced_df = enhanced_df.merge(
            df_proj[['Project_ID', 'Client_Name', 'Location', 'Status', 'Required_Skills']], 
            on='Project_ID', 
            how='left',
            suffixes=('_Employee', '_Project')
        )
        
        # Add derived features
        enhanced_df['Location_Match'] = enhanced_df['Location_Employee'] == enhanced_df['Location_Project']
        enhanced_df['Experience_Level'] = enhanced_df['Years_Experience'].apply(categorize_experience)
        
        # Add similarity categories
        enhanced_df['Similarity_Category'] = pd.cut(
            enhanced_df['Similarity_Score'], 
            bins=[0, 0.3, 0.6, 0.8, 1.0], 
            labels=['Low', 'Medium', 'High', 'Excellent']
        )
        
        # Add recommendation strength
        enhanced_df['Recommendation_Strength'] = enhanced_df.apply(
            lambda row: 'Strong' if row['Similarity_Score'] > 0.6 and row['Location_Match'] 
            else 'Good' if row['Similarity_Score'] > 0.4 
            else 'Fair', axis=1
        )
        
        self.logger.info(f"Enhanced {len(enhanced_df)} recommendations")
        return enhanced_df
    
    def create_recommendation_report(self, enhanced_recommendations: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive recommendation report"""
        self.logger.info("Creating recommendation report...")
        
        report = {
            'total_recommendations': len(enhanced_recommendations),
            'unique_projects': enhanced_recommendations['Project_ID'].nunique(),
            'unique_employees': enhanced_recommendations['Employee_ID'].nunique(),
            'avg_similarity': enhanced_recommendations['Similarity_Score'].mean(),
            'similarity_range': {
                'min': enhanced_recommendations['Similarity_Score'].min(),
                'max': enhanced_recommendations['Similarity_Score'].max()
            },
            'location_match_rate': (enhanced_recommendations['Location_Match'].sum() / 
                                  len(enhanced_recommendations) * 100),
            'experience_distribution': enhanced_recommendations['Experience_Level'].value_counts().to_dict(),
            'department_distribution': enhanced_recommendations['Department'].value_counts().head(5).to_dict(),
            'top_projects': enhanced_recommendations['Project_ID'].value_counts().head(5).to_dict()
        }
        
        return report
    
    def save_results(self, enhanced_recommendations: pd.DataFrame, 
                    similarity_matrices: Dict[str, pd.DataFrame]):
        """Save all results and models"""
        self.logger.info("Saving results...")
        
        # Save recommendations in different formats
        save_data(enhanced_recommendations, 
                 self.config.OUTPUT_DATA_DIR / "final_recommendations.csv")
        
        # Save top candidates only
        top_candidates = enhanced_recommendations[enhanced_recommendations['Rank'] == 1]
        save_data(top_candidates, 
                 self.config.OUTPUT_DATA_DIR / "top_candidates_per_project.csv")
        
        # Save project summary
        project_summary = enhanced_recommendations.groupby('Project_ID').agg({
            'Employee_ID': 'count',
            'Similarity_Score': ['mean', 'max'],
            'Location_Match': lambda x: (x == True).sum()
        }).round(4)
        
        project_summary.columns = ['Total_Recommendations', 'Avg_Similarity', 
                                 'Max_Similarity', 'Same_Location_Count']
        save_data(project_summary.reset_index(), 
                 self.config.OUTPUT_DATA_DIR / "project_summary.csv")
        
        # Save similarity matrices
        for name, matrix in similarity_matrices.items():
            with open(self.config.MODEL_DIR / f"{name}_similarity_matrix.pkl", 'wb') as f:
                pickle.dump(matrix, f)
        
        # Save the matcher instance
        with open(self.config.MODEL_DIR / "trained_matcher.pkl", 'wb') as f:
            pickle.dump(self, f)
        
        self.logger.info("Results saved successfully!")


def load_trained_matcher(config: Optional[Config] = None) -> EmployeeProjectMatcher:
    """Load a previously trained matcher"""
    config = config or Config()
    
    try:
        with open(config.MODEL_DIR / "trained_matcher.pkl", 'rb') as f:
            matcher = pickle.load(f)
        return matcher
    except FileNotFoundError:
        raise FileNotFoundError("No trained matcher found. Please train a new matcher first.")