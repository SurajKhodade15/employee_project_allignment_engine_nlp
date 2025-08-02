#!/usr/bin/env python3
"""
Quick launcher for Employee Project Alignment Engine
"""

import sys
import subprocess
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import plotly
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("📦 Please install requirements: pip install -r requirements.txt")
        return False

def launch_streamlit():
    """Launch the Streamlit application"""
    if not check_requirements():
        return
    
    print("🚀 Launching Employee Project Alignment Engine...")
    print("🌐 The application will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    
    # Change to streamlit_app directory
    streamlit_dir = Path(__file__).parent / "streamlit_app"
    os.chdir(streamlit_dir)
    
    # Launch Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")

def run_pipeline():
    """Run the complete matching pipeline"""
    if not check_requirements():
        return
    
    print("🔄 Running Employee Project Alignment Pipeline...")
    
    try:
        # Import and run the pipeline
        from main import run_complete_pipeline
        results = run_complete_pipeline()
        print("✅ Pipeline completed successfully!")
        return results
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        return None

def main():
    """Main launcher function"""
    print("🎯 Employee Project Alignment Engine Launcher")
    print("=" * 50)
    print("1. 🌐 Launch Web Interface (Recommended)")
    print("2. ⚙️ Run Pipeline Only")
    print("3. ❌ Exit")
    
    while True:
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == "1":
            launch_streamlit()
            break
        elif choice == "2":
            run_pipeline()
            break
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
