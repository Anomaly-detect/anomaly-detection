"""
Anomaly Detection Web Application - Main Entry Point
-----------------------------------------------------
Run the application using:
    streamlit run app.py
"""

import os
import sys

# Ensure the project root directory is in Python path for module resolution
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from website_interface.ui import main

if __name__ == "__main__":
    main()
else:
    main()
