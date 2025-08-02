"""
Utility functions and configuration for Employee Project Alignment Engine
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


class Config:
    """Configuration management for the application"""
    
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODEL_DIR = self.BASE_DIR / "model"
        self.REPORTS_DIR = self.BASE_DIR / "reports"
        
        # Data paths
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.OUTPUT_DATA_DIR = self.DATA_DIR / "outputs"
        
        # Model parameters
        self.MIN_EXPERIENCE_YEARS = 3.0
        self.MIN_SIMILARITY_THRESHOLD = 0.05
        self.TOP_N_RECOMMENDATIONS = 5
        
        # Similarity weights
        self.BINARY_WEIGHT = 0.7
        self.TFIDF_WEIGHT = 0.3
        
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.DATA_DIR, self.MODEL_DIR, self.REPORTS_DIR,
                         self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, self.OUTPUT_DATA_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


class Logger:
    """Simple logging utility"""
    
    def __init__(self, name: str = "EmployeeProjectAlignment", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data with error handling"""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {str(e)}")


def save_data(df: pd.DataFrame, file_path: str) -> bool:
    """Save DataFrame to CSV with error handling"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving data to {file_path}: {str(e)}")
        return False


def calculate_similarity_stats(similarity_matrix: pd.DataFrame) -> Dict[str, float]:
    """Calculate statistics for similarity matrix"""
    values = similarity_matrix.values.flatten()
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values))
    }


def categorize_experience(years: float) -> str:
    """Categorize experience level"""
    if years < 3:
        return 'Junior'
    elif years < 7:
        return 'Mid-Level'
    elif years < 12:
        return 'Senior'
    else:
        return 'Expert'


def format_skills_text(skills_text: str) -> str:
    """Clean and format skills text, extracting skills from experience descriptions"""
    if pd.isna(skills_text) or skills_text == '':
        return ''
    
    # Convert to string and basic cleaning
    skills_text = str(skills_text).strip()
    
    # Extract skills from experience text if it contains narrative
    # Look for patterns like "involving X, Y, and Z" or "expertise in A, B, C"
    import re
    
    # Pattern to extract skills from experience descriptions
    skill_patterns = [
        r'involving ([^.]+)',  # "involving X, Y, and Z"
        r'expertise in ([^.]+)',  # "expertise in A, B, C"
        r'experience with ([^.]+)',  # "experience with X, Y"
        r'worked on ([^.]+)',  # "worked on X, Y"
        r'skills include ([^.]+)',  # "skills include X, Y"
    ]
    
    extracted_skills = []
    for pattern in skill_patterns:
        matches = re.findall(pattern, skills_text, re.IGNORECASE)
        for match in matches:
            # Clean the matched text
            skills_part = match.strip()
            # Remove common words and phrases
            skills_part = re.sub(r'\b(projects?|solutions?|for|domain|and|the|a|an|in|on|with|using)\b', '', skills_part, flags=re.IGNORECASE)
            extracted_skills.append(skills_part)
    
    # If no patterns matched, use the whole text
    if not extracted_skills:
        extracted_skills = [skills_text]
    
    # Combine all extracted skills
    combined_skills = ' '.join(extracted_skills)
    
    # Basic text cleaning
    combined_skills = combined_skills.replace(',', ' ')
    combined_skills = combined_skills.replace(';', ' ')
    combined_skills = combined_skills.replace('.', ' ')
    combined_skills = ' '.join(combined_skills.split())  # Remove extra spaces
    
    return combined_skills.lower()