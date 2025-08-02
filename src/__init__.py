"""
Employee Project Alignment Engine
A comprehensive NLP-based system for matching employees to projects
"""

__version__ = "1.0.0"
__author__ = "Suraj Khodade"
__email__ = "suraj.khodade7@gmail.com"

from .data_preprocessing import DataPreprocessor
from .feature_extraction import FeatureExtractor
from .matching_engine import EmployeeProjectMatcher
from .utils import Config, Logger

__all__ = [
    'DataPreprocessor',
    'FeatureExtractor', 
    'EmployeeProjectMatcher',
    'Config',
    'Logger'
]
