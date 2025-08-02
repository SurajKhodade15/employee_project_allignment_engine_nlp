
# Feature Matrices Usage Guide

## Available Pickle Files:

1. **employee_features_combined.pkl**: 
   - Complete employee feature matrix (binary + TF-IDF + attributes)
   - Use for: Comprehensive matching algorithms

2. **employee_binary_matrix.pkl**: 
   - Binary skill matrix for employees
   - Use for: Simple skill matching, Jaccard similarity

3. **employee_tfidf_matrix.pkl**: 
   - TF-IDF features from experience text
   - Use for: Semantic similarity, cosine similarity

4. **project_binary_matrix.pkl**: 
   - Binary skill matrix for projects
   - Use for: Direct skill requirement matching

5. **project_tfidf_matrix.pkl**: 
   - TF-IDF features from project descriptions
   - Use for: Semantic project analysis

6. **employee_features_basic.pkl**: 
   - Basic employee attributes (experience, department, location)
   - Use for: Filtering and business rules

7. **skills_vocabulary.pkl**: 
   - Master skills vocabulary list
   - Use for: Skill validation and new data processing

8. **tfidf_vectorizer.pkl**: 
   - Trained TF-IDF vectorizer
   - Use for: Processing new text data

## Loading Example:
```python
import pickle
with open('model/employee_features_combined.pkl', 'rb') as f:
    employee_features = pickle.load(f)
```
