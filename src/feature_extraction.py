"""
Feature extraction module for Employee Project Alignment Engine
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, Any, Optional
from .utils import Config, Logger


class FeatureExtractor:
    """Feature extraction and vectorization for employees and projects"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = Logger(__name__)
        self.tfidf_vectorizer = None
        self.label_encoders = {}
        self.scalers = {}
        
    def create_skills_vocabulary(self, df_emp: pd.DataFrame, df_proj: pd.DataFrame) -> set:
        """Create comprehensive skills vocabulary"""
        self.logger.info("Creating skills vocabulary...")
        
        # Combine all skills from employees and projects
        all_skills_text = []
        
        # Employee skills
        emp_skills = df_emp['Skills'].dropna().tolist()
        all_skills_text.extend(emp_skills)
        
        # Project skills
        proj_skills = df_proj['Required_Skills'].dropna().tolist()
        all_skills_text.extend(proj_skills)
        
        # Extract individual skills
        skills_set = set()
        for skills_text in all_skills_text:
            if isinstance(skills_text, str):
                # Split by common delimiters
                skills = skills_text.replace(',', ' ').replace(';', ' ').split()
                skills_set.update([skill.strip().lower() for skill in skills if skill.strip()])
        
        self.logger.info(f"Created vocabulary with {len(skills_set)} unique skills")
        return skills_set
    
    def create_binary_features(self, df_emp: pd.DataFrame, df_proj: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create binary skill matrices"""
        self.logger.info("Creating binary skill features...")
        
        # Create skills vocabulary
        skills_vocab = self.create_skills_vocabulary(df_emp, df_proj)
        
        # Create binary matrices
        emp_binary = self._create_binary_matrix(df_emp, 'Skills', skills_vocab, 'Employee_ID')
        proj_binary = self._create_binary_matrix(df_proj, 'Required_Skills', skills_vocab, 'Project_ID')
        
        self.logger.info(f"Binary matrices: Employees {emp_binary.shape}, Projects {proj_binary.shape}")
        return emp_binary, proj_binary
    
    def _create_binary_matrix(self, df: pd.DataFrame, skills_column: str, vocabulary: set, id_column: str) -> pd.DataFrame:
        """Helper method to create binary skill matrix"""
        binary_data = []
        
        for _, row in df.iterrows():
            skills_text = str(row[skills_column]).lower()
            binary_vector = []
            
            for skill in vocabulary:
                if skill in skills_text:
                    binary_vector.append(1)
                else:
                    binary_vector.append(0)
            
            binary_data.append(binary_vector)
        
        binary_df = pd.DataFrame(binary_data, columns=list(vocabulary))
        binary_df.index = df[id_column]
        
        return binary_df
    
    def create_tfidf_features(self, df_emp: pd.DataFrame, df_proj: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create TF-IDF features for skills"""
        self.logger.info("Creating TF-IDF features...")
        
        # Prepare text data
        emp_texts = df_emp['Skills'].fillna('').tolist()
        proj_texts = df_proj['Required_Skills'].fillna('').tolist()
        
        # Combine all texts for fitting
        all_texts = emp_texts + proj_texts
        
        # Initialize and fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.tfidf_vectorizer.fit(all_texts)
        
        # Transform employee and project texts
        emp_tfidf = self.tfidf_vectorizer.transform(emp_texts)
        proj_tfidf = self.tfidf_vectorizer.transform(proj_texts)
        
        # Convert to DataFrames
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        emp_tfidf_df = pd.DataFrame(
            emp_tfidf.toarray(),
            columns=feature_names,
            index=df_emp['Employee_ID']
        )
        
        proj_tfidf_df = pd.DataFrame(
            proj_tfidf.toarray(),
            columns=feature_names,
            index=df_proj['Project_ID']
        )
        
        self.logger.info(f"TF-IDF matrices: Employees {emp_tfidf_df.shape}, Projects {proj_tfidf_df.shape}")
        return emp_tfidf_df, proj_tfidf_df
    
    def create_basic_features(self, df_emp: pd.DataFrame) -> pd.DataFrame:
        """Create basic numerical features for employees"""
        self.logger.info("Creating basic numerical features...")
        
        df_features = df_emp[['Employee_ID', 'Years_Experience']].copy()
        
        # Encode categorical features
        categorical_cols = ['Department', 'Location', 'Experience_Level']
        
        for col in categorical_cols:
            if col in df_emp.columns:
                le = LabelEncoder()
                df_features[f'{col}_encoded'] = le.fit_transform(df_emp[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale numerical features
        scaler = StandardScaler()
        df_features['Years_Experience_scaled'] = scaler.fit_transform(df_features[['Years_Experience']])
        self.scalers['experience'] = scaler
        
        df_features.set_index('Employee_ID', inplace=True)
        
        self.logger.info(f"Basic features matrix: {df_features.shape}")
        return df_features
    
    def create_combined_features(self, basic_features: pd.DataFrame, binary_features: pd.DataFrame, 
                               tfidf_features: pd.DataFrame) -> pd.DataFrame:
        """Combine all feature types"""
        self.logger.info("Combining all features...")
        
        # Align indices
        common_employees = set(basic_features.index) & set(binary_features.index) & set(tfidf_features.index)
        
        basic_aligned = basic_features.loc[list(common_employees)]
        binary_aligned = binary_features.loc[list(common_employees)]
        tfidf_aligned = tfidf_features.loc[list(common_employees)]
        
        # Combine features
        combined_features = pd.concat([basic_aligned, binary_aligned, tfidf_aligned], axis=1)
        
        self.logger.info(f"Combined features matrix: {combined_features.shape}")
        return combined_features
    
    def extract_all_features(self, df_emp: pd.DataFrame, df_proj: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Extract all types of features"""
        self.logger.info("Starting feature extraction...")
        
        # Create different feature types
        emp_binary, proj_binary = self.create_binary_features(df_emp, df_proj)
        emp_tfidf, proj_tfidf = self.create_tfidf_features(df_emp, df_proj)
        emp_basic = self.create_basic_features(df_emp)
        emp_combined = self.create_combined_features(emp_basic, emp_binary, emp_tfidf)
        
        # Store skills vocabulary
        skills_vocab = self.create_skills_vocabulary(df_emp, df_proj)
        
        features = {
            'employee_binary': emp_binary,
            'project_binary': proj_binary,
            'employee_tfidf': emp_tfidf,
            'project_tfidf': proj_tfidf,
            'employee_basic': emp_basic,
            'employee_combined': emp_combined,
            'skills_vocabulary': skills_vocab
        }
        
        # Save features and models
        self.save_features_and_models(features)
        
        self.logger.info("Feature extraction completed successfully!")
        return features
    
    def save_features_and_models(self, features: Dict[str, Any]):
        """Save features and trained models"""
        self.logger.info("Saving features and models...")
        
        # Save feature matrices
        for name, feature_matrix in features.items():
            if name != 'skills_vocabulary':
                with open(self.config.MODEL_DIR / f"{name}.pkl", 'wb') as f:
                    pickle.dump(feature_matrix, f)
        
        # Save vocabulary
        with open(self.config.MODEL_DIR / "skills_vocabulary.pkl", 'wb') as f:
            pickle.dump(features['skills_vocabulary'], f)
        
        # Save vectorizer
        if self.tfidf_vectorizer:
            with open(self.config.MODEL_DIR / "tfidf_vectorizer.pkl", 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        self.logger.info("Features and models saved successfully!")
    
    def load_features(self) -> Dict[str, Any]:
        """Load previously saved features"""
        self.logger.info("Loading saved features...")
        
        features = {}
        feature_files = [
            'employee_binary', 'project_binary', 'employee_tfidf', 
            'project_tfidf', 'employee_basic', 'employee_combined', 'skills_vocabulary'
        ]
        
        for feature_name in feature_files:
            try:
                with open(self.config.MODEL_DIR / f"{feature_name}.pkl", 'rb') as f:
                    features[feature_name] = pickle.load(f)
            except FileNotFoundError:
                self.logger.warning(f"Feature file not found: {feature_name}.pkl")
        
        # Load vectorizer
        try:
            with open(self.config.MODEL_DIR / "tfidf_vectorizer.pkl", 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        except FileNotFoundError:
            self.logger.warning("TF-IDF vectorizer not found")
        
        self.logger.info("Features loaded successfully!")
        return features