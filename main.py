"""
Main pipeline for Employee Project Alignment Engine
"""

from src.data_preprocessing import DataPreprocessor
from src.feature_extraction import FeatureExtractor
from src.matching_engine import EmployeeProjectMatcher
from src.utils import Config, Logger


def run_complete_pipeline():
    """Run the complete employee-project matching pipeline"""
    
    # Initialize configuration and logger
    config = Config()
    config.create_directories()
    logger = Logger("MainPipeline")
    
    logger.info("Starting Employee Project Alignment Engine Pipeline...")
    
    try:
        # Step 1: Data Preprocessing
        logger.info("Step 1: Data Preprocessing")
        preprocessor = DataPreprocessor(config)
        df_emp_enhanced, df_projects_clean, df_emp_master = preprocessor.process_all_data()
        
        # Step 2: Feature Extraction
        logger.info("Step 2: Feature Extraction")
        feature_extractor = FeatureExtractor(config)
        features = feature_extractor.extract_all_features(df_emp_enhanced, df_projects_clean)
        
        # Step 3: Similarity Computation and Matching
        logger.info("Step 3: Matching Engine")
        matcher = EmployeeProjectMatcher(df_emp_enhanced, config)
        
        # Configure business rules
        matcher.add_experience_filter(min_years=3.0)
        matcher.add_similarity_threshold_filter(min_similarity=0.05)
        
        # Compute similarity matrices
        binary_similarity = matcher.compute_similarity_matrix(
            features['employee_binary'], features['project_binary'], 'cosine'
        )
        
        tfidf_similarity = matcher.compute_similarity_matrix(
            features['employee_tfidf'], features['project_tfidf'], 'cosine'
        )
        
        hybrid_similarity = matcher.compute_hybrid_similarity(
            binary_similarity, tfidf_similarity, 
            config.BINARY_WEIGHT, config.TFIDF_WEIGHT
        )
        
        # Generate recommendations
        recommendations_df, stats = matcher.generate_all_recommendations(hybrid_similarity)
        enhanced_recommendations = matcher.enhance_recommendations(recommendations_df, df_projects_clean)
        
        # Create report
        report = matcher.create_recommendation_report(enhanced_recommendations)
        
        # Save results
        similarity_matrices = {
            'binary': binary_similarity,
            'tfidf': tfidf_similarity,
            'hybrid': hybrid_similarity
        }
        
        matcher.save_results(enhanced_recommendations, similarity_matrices)
        
        # Print summary
        logger.info("Pipeline completed successfully!")
        logger.info(f"Generated {report['total_recommendations']} recommendations")
        logger.info(f"Covering {report['unique_projects']} projects and {report['unique_employees']} employees")
        logger.info(f"Average similarity score: {report['avg_similarity']:.4f}")
        
        return {
            'recommendations': enhanced_recommendations,
            'similarity_matrices': similarity_matrices,
            'report': report,
            'matcher': matcher
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    results = run_complete_pipeline()
    print("✅ Employee Project Alignment Engine pipeline completed successfully!")
