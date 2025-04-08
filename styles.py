"""
Styles module for enhancing the UI of the CO2 Emissions Prediction app.
This file contains CSS styling and UI enhancement functions.
"""

import streamlit as st
import base64
import os
from pathlib import Path

# Define the colors and styles for the app
COLOR_PRIMARY = "#16A34A"  # Green color for primary elements
COLOR_SECONDARY = "#0F766E"  # Teal color for secondary elements
COLOR_BACKGROUND = "#F0FDF4"  # Light green background color
COLOR_TEXT = "#334155"  # Dark slate for text
COLOR_SIDEBAR = "#ECFDF5"  # Light mint green for sidebar
COLOR_WARNING = "#F97316"  # Orange for warnings
COLOR_DANGER = "#EF4444"  # Red for danger/severe levels

# CSS for custom styling
def load_css():
    """
    Apply custom CSS styling to the app.
    """
    st.markdown("""
    <style>
        /* Main content area */
        .stApp {
            background-color: #F8FAFC;
        }
        
        /* Headers */
        h1 {
            color: #16A34A;
            font-weight: 700;
            font-size: 2.5rem !important;
            margin-bottom: 1rem;
        }
        
        h2 {
            color: #0F766E;
            font-weight: 600;
            font-size: 1.8rem !important;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.3rem;
            border-bottom: 1px solid #E2E8F0;
        }
        
        h3 {
            color: #115E59;
            font-weight: 600;
            font-size: 1.4rem !important;
            margin-top: 1.2rem;
            margin-bottom: 0.8rem;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #ECFDF5;
        }
        
        /* Cards/containers */
        .stCard {
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background-color: white;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #16A34A;
            color: white;
            border-radius: 6px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #15803D;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Data visualization elements */
        .stPlotlyChart {
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            background-color: white;
            padding: 1rem;
        }
        
        /* Input elements */
        .stSelectbox, .stMultiSelect, .stSlider {
            margin-bottom: 1.2rem;
        }
        
        /* Text */
        p, li, .stMarkdown {
            color: #334155;
            font-size: 1rem;
            line-height: 1.6;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: #16A34A;
        }
        
        /* Warning colors for different threat levels */
        .success-text {
            color: #16A34A;
            font-weight: 600;
        }
        
        .warning-text {
            color: #F97316;
            font-weight: 600;
        }
        
        .danger-text {
            color: #EF4444;
            font-weight: 600;
        }
        
        .severe-text {
            color: #7F1D1D;
            font-weight: 600;
        }
        
        /* Custom cards */
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            text-align: center;
        }
        
        .metric-card h3 {
            margin-top: 0;
            font-size: 1.2rem !important;
            color: #334155;
        }
        
        .metric-card p {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0;
            color: #16A34A;
        }
        
        /* Recommendation cards */
        .recommendation-card {
            background-color: #F0FDF4;
            padding: 1rem;
            border-left: 4px solid #16A34A;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        .recommendation-card p {
            margin-bottom: 0;
        }
        
        /* Custom footer */
        .footer {
            margin-top: 3rem;
            text-align: center;
            padding: 1.5rem 0;
            border-top: 1px solid #E2E8F0;
            color: #64748B;
            font-size: 0.875rem;
        }
    </style>
    """, unsafe_allow_html=True)

def display_metric_card(title, value, unit="", icon="📊"):
    """
    Display a metric in a visually appealing card.
    
    Parameters:
    -----------
    title : str
        Title of the metric
    value : float or str
        Value to display
    unit : str, optional
        Unit of the metric
    icon : str, optional
        Emoji icon to display
    """
    st.markdown(f"""
    <div class="metric-card">
        <h3>{icon} {title}</h3>
        <p>{value}<span style="font-size: 1rem; font-weight: 400;"> {unit}</span></p>
    </div>
    """, unsafe_allow_html=True)

def display_recommendation(text):
    """
    Display a recommendation in a styled card.
    
    Parameters:
    -----------
    text : str
        Recommendation text
    """
    st.markdown(f"""
    <div class="recommendation-card">
        <p>🔍 {text}</p>
    </div>
    """, unsafe_allow_html=True)

def display_threat_level_label(level, category):
    """
    Display a threat level with appropriate styling.
    
    Parameters:
    -----------
    level : float
        Threat level (1-10)
    category : str
        Threat category ("Low", "Moderate", "High", "Severe")
    """
    color_class = ""
    if category == "Low":
        color_class = "success-text"
    elif category == "Moderate":
        color_class = "warning-text"
    elif category == "High":
        color_class = "danger-text"
    else:  # Severe
        color_class = "severe-text"
        
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1rem;">
        <span style="font-size: 3rem;" class="{color_class}">{level}</span>
        <br>
        <span style="font-size: 1.4rem;" class="{color_class}">{category}</span>
    </div>
    """, unsafe_allow_html=True)

def display_footer():
    """
    Display a footer with copyright and year information.
    """
    st.markdown("""
    <div class="footer">
        <p>© 2025 CO₂ Emissions Predictor | Created with ❤️ for environmental sustainability</p>
    </div>
    """, unsafe_allow_html=True)

def add_logo():
    """
    Add a logo to the sidebar
    """
    st.sidebar.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="margin-bottom: 0;">🌍 CO₂ Monitor</h2>
            <p style="margin-top: 0; font-size: 0.9rem; color: #64748B;">Environmental Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_images_directory():
    """
    Create an 'images' directory if it doesn't exist
    """
    if not os.path.exists('images'):
        os.makedirs('images')

def create_info_card(title, description, icon="ℹ️"):
    """
    Create an information card with icon and content.
    
    Parameters:
    -----------
    title : str
        Card title
    description : str
        Card description
    icon : str, optional
        Emoji icon to display
    """
    st.markdown(f"""
    <div style="background-color: white; padding: 1.5rem; border-radius: 8px; 
               box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin-bottom: 1rem;">
        <h3 style="display: flex; align-items: center; margin-top: 0;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
            {title}
        </h3>
        <p style="margin-bottom: 0;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

def local_css(file_name):
    """
    Load local CSS file.
    
    Parameters:
    -----------
    file_name : str
        Path to the CSS file
    """
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)