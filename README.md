# 🎯 Employee Project Alignment Engine

A comprehensive NLP-powered system for intelligent employee-project matching based on skills, experience, and business requirements.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **Advanced NLP Processing**: Uses TF-IDF and binary vectorization for skill matching
- **Hybrid Similarity Engine**: Combines multiple similarity metrics for accurate recommendations
- **Business Rules Engine**: Configurable filters for experience, department, location, and similarity thresholds
- **Interactive Web Interface**: Rich Streamlit dashboard with visualizations and search capabilities
- **Modular Architecture**: Clean, maintainable codebase with separate modules for each functionality
- **Comprehensive Analytics**: Detailed reports and visualizations for decision-making
- **Export Capabilities**: Multiple export formats for recommendations and analysis

## 🏗️ Architecture

```
employee_project_allignment_engine_nlp/
├── 📁 src/                          # Core modules
│   ├── __init__.py                  # Package initialization
│   ├── data_preprocessing.py        # Data cleaning and preprocessing
│   ├── feature_extraction.py        # NLP feature extraction
│   ├── matching_engine.py           # Similarity computation and matching
│   └── utils.py                     # Utilities and configuration
├── 📁 streamlit_app/               # Web interface
│   ├── app.py                      # Main Streamlit application
│   └── requirements.txt            # Streamlit dependencies
├── 📁 data/                        # Data directories
│   ├── raw/                        # Original data files
│   ├── processed/                  # Cleaned data files
│   └── outputs/                    # Generated recommendations
├── 📁 model/                       # Trained models and matrices
├── 📁 notebooks/                   # Jupyter notebooks for development
├── 📁 reports/                     # Generated reports and visualizations
├── main.py                         # Main pipeline script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SurajKhodade15/employee_project_allignment_engine_nlp.git
   cd employee_project_allignment_engine_nlp
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   
   Place your CSV files in the `data/raw/` directory:
   - `employee_master.csv` - Employee basic information
   - `employee_experience.csv` - Employee skills and experience
   - `client_projects.csv` - Project requirements and details

### Running the Application

#### Option 1: Web Interface (Recommended)

Launch the Streamlit web application:

```bash
cd streamlit_app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

#### Option 2: Command Line Pipeline

Run the complete pipeline programmatically:

```bash
python main.py
```

## 📊 Data Format

### Employee Master (`employee_master.csv`)
```csv
Employee_ID,Department,Location
EMP001,Data Science,Mumbai
EMP002,AI Research,Bangalore
```

### Employee Experience (`employee_experience.csv`)
```csv
Employee_ID,Skills,Years_Experience
EMP001,"Python, Machine Learning, SQL",5.2
EMP002,"Deep Learning, Python, TensorFlow",3.8
```

### Client Projects (`client_projects.csv`)
```csv
Project_ID,Client_Name,Location,Status,Required_Skills
P301,TechCorp,Mumbai,Active,"Python, Data Analysis"
P302,DataInc,Bangalore,Active,"Machine Learning, AI"
```

## 🎯 How It Works

### 1. Data Preprocessing
- Cleans and standardizes employee and project data
- Handles missing values and data quality issues
- Creates enhanced datasets with derived features

### 2. Feature Extraction
- **Binary Features**: Exact skill matching using binary vectors
- **TF-IDF Features**: Semantic similarity using term frequency analysis
- **Basic Features**: Numerical features like experience, location encoding

### 3. Similarity Computation
- **Cosine Similarity**: Primary metric for comparing feature vectors
- **Hybrid Scoring**: Weighted combination of binary and TF-IDF similarities
- **Business Rules**: Filters based on experience, department, location

### 4. Recommendation Generation
- Generates top-N recommendations for each project
- Applies configurable business rules and filters
- Enhances results with metadata and quality indicators

## 🎨 Web Interface Features

### 🏠 Home & Overview
- System metrics and KPIs
- Data distribution visualizations
- Employee and project statistics

### ⚙️ Run Matching Pipeline
- One-click pipeline execution
- Progress tracking and status updates
- Error handling and feedback

### 📊 View Recommendations
- Comprehensive analytics dashboard
- Interactive charts and visualizations
- Download options for results

### 🔍 Search & Explore
- Project-to-employee search
- Employee-to-project search
- Detailed recommendation breakdowns

## ⚙️ Configuration

The system uses a centralized configuration class in `src/utils.py`:

```python
# Key configuration parameters
MIN_EXPERIENCE_YEARS = 3.0          # Minimum experience filter
MIN_SIMILARITY_THRESHOLD = 0.05     # Minimum similarity score
TOP_N_RECOMMENDATIONS = 5           # Number of recommendations per project
BINARY_WEIGHT = 0.7                 # Weight for binary similarity
TFIDF_WEIGHT = 0.3                  # Weight for TF-IDF similarity
```

## 📈 Business Rules

### Experience Filters
- Minimum years of experience requirement
- Experience level categorization (Junior, Mid-Level, Senior, Expert)

### Department Filters
- Preferred departments for specific projects
- Department exclusion rules

### Location Filters
- Same location preference
- Remote work considerations

### Similarity Thresholds
- Minimum similarity score requirements
- Quality-based filtering

## 🔧 Advanced Usage

### Custom Business Rules

```python
from src.matching_engine import EmployeeProjectMatcher

# Initialize matcher
matcher = EmployeeProjectMatcher(df_employees)

# Add custom filters
matcher.add_experience_filter(min_years=5.0, max_years=15.0)
matcher.add_department_filter(preferred_departments=['Data Science', 'AI Research'])
matcher.add_location_filter(excluded_locations=['Remote'])
matcher.add_similarity_threshold_filter(min_similarity=0.1)
```

### Batch Processing

```python
from main import run_complete_pipeline

# Run complete pipeline
results = run_complete_pipeline()

# Access results
recommendations = results['recommendations']
similarity_matrices = results['similarity_matrices']
report = results['report']
```

## 📊 Output Files

The system generates several output files:

- `final_recommendations.csv` - Complete recommendations with metadata
- `top_candidates_per_project.csv` - Top candidate for each project
- `project_summary.csv` - Aggregated project statistics
- `executive_summary_report.md` - Executive summary report

## 🧪 Testing and Validation

### Data Quality Checks
- Missing value handling
- Data type validation
- Consistency checks

### Model Validation
- Similarity score distributions
- Recommendation quality metrics
- Business rule compliance

### Performance Metrics
- Processing speed
- Memory usage
- Scalability testing

## 🚀 Performance Optimization

### For Large Datasets
- Batch processing capabilities
- Memory-efficient data handling
- Parallel processing options

### Caching
- Feature matrix caching
- Model persistence
- Result caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📋 TODO

- [ ] Add support for project priority weighting
- [ ] Implement real-time recommendation updates
- [ ] Add email notification system
- [ ] Create mobile-responsive UI
- [ ] Add multi-language support
- [ ] Implement user authentication
- [ ] Add API endpoints for integration

## 🐛 Troubleshooting

### Common Issues

1. **Data Loading Errors**
   - Check file paths and CSV format
   - Ensure all required columns are present
   - Verify data encoding (UTF-8 recommended)

2. **Memory Issues with Large Datasets**
   - Reduce batch size in configuration
   - Use data sampling for testing
   - Consider upgrading system RAM

3. **Streamlit App Not Loading**
   - Check port availability (8501)
   - Verify all dependencies are installed
   - Check Python path configuration

### Getting Help

- Check the [Issues](https://github.com/SurajKhodade15/employee_project_allignment_engine_nlp/issues) page
- Create a new issue with detailed problem description
- Include system information and error logs

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Suraj Khodade** - *Initial work* - [SurajKhodade15](https://github.com/SurajKhodade15)

## 🙏 Acknowledgments

- Streamlit team for the amazing web framework
- scikit-learn contributors for ML tools
- Plotly team for interactive visualizations
- Open source community for inspiration and tools

## 📞 Contact

- **Email**: suraj.khodade@example.com
- **LinkedIn**: [Suraj Khodade](https://linkedin.com/in/surajkhodade)
- **GitHub**: [SurajKhodade15](https://github.com/SurajKhodade15)

---

<div align="center">
  <strong>⭐ Star this repository if you find it helpful! ⭐</strong>
</div>
