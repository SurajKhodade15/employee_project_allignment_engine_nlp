"""
Data preprocessing module for Employee Project Alignment Engine
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Optional
from .utils import Config, Logger, load_data, save_data, format_skills_text


class DataPreprocessor:
    """Data preprocessing and cleaning operations"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = Logger(__name__)
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all raw data files"""
        self.logger.info("Loading raw data files...")
        
        try:
            df_emp_master = load_data(self.config.RAW_DATA_DIR / "employee_master.csv")
            df_emp_exp = load_data(self.config.RAW_DATA_DIR / "employee_experience.csv")
            df_projects = load_data(self.config.RAW_DATA_DIR / "client_projects.csv")
            
            self.logger.info(f"Loaded: {len(df_emp_master)} employees, {len(df_emp_exp)} experiences, {len(df_projects)} projects")
            return df_emp_master, df_emp_exp, df_projects
            
        except Exception as e:
            self.logger.error(f"Error loading raw data: {str(e)}")
            raise
    
    def clean_employee_master(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean employee master data"""
        self.logger.info("Cleaning employee master data...")
        
        df_clean = df.copy()
        
        # Handle missing values using proper pandas methods
        df_clean = df_clean.fillna({
            'Department': 'Unknown',
            'Location': 'Unknown',
            'Years_Experience': 0.0,
            'Current_Project_ID': ''
        })
        
        # Standardize department names
        dept_mapping = {
            'AI Research': 'AI Research',
            'Data Science': 'Data Science',
            'Software Development': 'Software Development',
            'Full Stack Dev': 'Software Development',
            'DevOps': 'DevOps',
            'QA': 'Quality Assurance',
            'Product Management': 'Product Management'
        }
        df_clean['Department'] = df_clean['Department'].map(dept_mapping).fillna(df_clean['Department'])
        
        # Standardize locations
        df_clean['Location'] = df_clean['Location'].str.title()
        
        # Handle years of experience
        df_clean['Years_Experience'] = pd.to_numeric(df_clean['Years_Experience'], errors='coerce')
        df_clean['Years_Experience'] = df_clean['Years_Experience'].fillna(0)
        
        self.logger.info(f"Cleaned employee master: {len(df_clean)} records")
        return df_clean
    
    def clean_employee_experience(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean employee experience data"""
        self.logger.info("Cleaning employee experience data...")
        
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.fillna({
            'Experience_Text': ''
        })
        
        # Clean experience text (which contains skills information)
        df_clean['Skills'] = df_clean['Experience_Text'].apply(format_skills_text)
        
        # Remove records with no experience/skills
        df_clean = df_clean[df_clean['Skills'] != '']
        
        self.logger.info(f"Cleaned employee experience: {len(df_clean)} records")
        return df_clean
    
    def clean_projects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean project data"""
        self.logger.info("Cleaning project data...")
        
        df_clean = df.copy()
        
        # Handle missing values using proper pandas methods
        df_clean = df_clean.fillna({
            'Client_Name': 'Unknown Client',
            'Location': 'Remote',
            'Status': 'Active',
            'Required_Skills': '',
            'Project_Description': ''
        })
        
        # Clean required skills
        df_clean['Required_Skills'] = df_clean['Required_Skills'].apply(format_skills_text)
        
        # Standardize locations
        df_clean['Location'] = df_clean['Location'].str.title()
        
        # Remove projects with no required skills
        df_clean = df_clean[df_clean['Required_Skills'] != '']
        
        self.logger.info(f"Cleaned projects: {len(df_clean)} records")
        return df_clean
    
    def enhance_employee_data(self, df_master: pd.DataFrame, df_experience: pd.DataFrame) -> pd.DataFrame:
        """Combine and enhance employee data"""
        self.logger.info("Enhancing employee data...")
        
        # Merge master and experience data
        df_enhanced = df_master.merge(df_experience, on='Employee_ID', how='inner')
        
        # Add derived features using Years_Experience from master data
        df_enhanced['Experience_Level'] = df_enhanced['Years_Experience'].apply(
            lambda x: 'Junior' if x < 3 else 'Mid-Level' if x < 7 else 'Senior' if x < 12 else 'Expert'
        )
        
        # Create a combined skills text for each employee (using Skills from experience data)
        df_enhanced['All_Skills'] = df_enhanced['Skills']
        
        # Add additional useful columns
        df_enhanced['Has_Current_Project'] = df_enhanced['Current_Project_ID'].notna() & (df_enhanced['Current_Project_ID'] != '')
        
        self.logger.info(f"Enhanced employee data: {len(df_enhanced)} records")
        return df_enhanced
    
    def process_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Complete data preprocessing pipeline"""
        self.logger.info("Starting complete data preprocessing...")
        
        # Load raw data
        df_emp_master, df_emp_exp, df_projects = self.load_raw_data()
        
        # Clean individual datasets
        df_emp_master_clean = self.clean_employee_master(df_emp_master)
        df_emp_exp_clean = self.clean_employee_experience(df_emp_exp)
        df_projects_clean = self.clean_projects(df_projects)
        
        # Enhance employee data
        df_emp_enhanced = self.enhance_employee_data(df_emp_master_clean, df_emp_exp_clean)
        
        # Save processed data
        self.save_processed_data(df_emp_master_clean, df_emp_enhanced, df_projects_clean)
        
        self.logger.info("Data preprocessing completed successfully!")
        return df_emp_enhanced, df_projects_clean, df_emp_master_clean
    
    def save_processed_data(self, df_emp_master: pd.DataFrame, df_emp_enhanced: pd.DataFrame, df_projects: pd.DataFrame):
        """Save processed data to files"""
        self.logger.info("Saving processed data...")
        
        save_data(df_emp_master, self.config.PROCESSED_DATA_DIR / "employee_master_cleaned.csv")
        save_data(df_emp_enhanced, self.config.PROCESSED_DATA_DIR / "employee_experience_enhanced.csv")
        save_data(df_projects, self.config.PROCESSED_DATA_DIR / "client_projects_cleaned.csv")
        
        self.logger.info("Processed data saved successfully!")