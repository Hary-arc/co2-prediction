"""
SVG Generator module for the CO2 Emissions Prediction app.
This file contains functions to generate SVG images for the app.
"""
import numpy as np

def generate_antenna_tower_svg(color="#16A34A", size=200):
    """
    Generate an SVG of a mobile antenna tower.
    
    Parameters:
    -----------
    color : str, optional
        Color of the tower
    size : int, optional
        Size of the SVG
        
    Returns:
    --------
    str
        SVG code as a string
    """
    svg = f'''
    <svg width="{size}" height="{size}" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
        <!-- Tower Base -->
        <rect x="135" y="260" width="30" height="20" fill="#555555" />
        
        <!-- Tower Body -->
        <rect x="145" y="90" width="10" height="170" fill="{color}" />
        
        <!-- Tower Supports -->
        <line x1="145" y1="110" x2="125" y2="130" stroke="{color}" stroke-width="5" />
        <line x1="155" y1="110" x2="175" y2="130" stroke="{color}" stroke-width="5" />
        <line x1="145" y1="150" x2="115" y2="180" stroke="{color}" stroke-width="5" />
        <line x1="155" y1="150" x2="185" y2="180" stroke="{color}" stroke-width="5" />
        <line x1="145" y1="190" x2="105" y2="230" stroke="{color}" stroke-width="5" />
        <line x1="155" y1="190" x2="195" y2="230" stroke="{color}" stroke-width="5" />
        
        <!-- Tower Top with Antenna -->
        <circle cx="150" cy="80" r="5" fill="{color}" />
        <rect x="148" y="40" width="4" height="40" fill="{color}" />
        
        <!-- Signal Waves -->
        <path d="M 160,45 Q 180,40 190,50" stroke="{color}" stroke-width="2" fill="none" />
        <path d="M 160,55 Q 190,45 210,60" stroke="{color}" stroke-width="2" fill="none" />
        <path d="M 160,65 Q 200,50 230,70" stroke="{color}" stroke-width="2" fill="none" />
        
        <!-- Tower Cross Sections -->
        <rect x="142" y="120" width="16" height="5" fill="{color}" />
        <rect x="142" y="160" width="16" height="5" fill="{color}" />
        <rect x="142" y="200" width="16" height="5" fill="{color}" />
        <rect x="142" y="240" width="16" height="5" fill="{color}" />
    </svg>
    '''
    return svg

def generate_eco_bulb_svg(color="#16A34A", size=200):
    """
    Generate an SVG of an eco-friendly light bulb with a leaf inside.
    
    Parameters:
    -----------
    color : str, optional
        Color of the bulb
    size : int, optional
        Size of the SVG
        
    Returns:
    --------
    str
        SVG code as a string
    """
    svg = f'''
    <svg width="{size}" height="{size}" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
        <!-- Bulb Base -->
        <rect x="130" y="230" width="40" height="10" rx="5" fill="#555555" />
        <rect x="135" y="220" width="30" height="10" rx="3" fill="#777777" />
        
        <!-- Bulb Screw -->
        <path d="M 135,200 L 135,220 L 165,220 L 165,200 C 165,200 175,210 175,190 C 175,170 155,190 150,190 C 145,190 125,170 125,190 C 125,210 135,200 135,200 Z" fill="#999999" />
        
        <!-- Light Bulb -->
        <ellipse cx="150" cy="140" rx="60" ry="70" fill="{color}" opacity="0.7" />
        
        <!-- Leaf Inside Bulb -->
        <path d="M 150,150 C 130,120 160,100 180,120 C 190,130 185,150 170,160 L 150,150 Z" fill="#10B981" />
        <path d="M 150,150 C 170,120 140,100 120,120 C 110,130 115,150 130,160 L 150,150 Z" fill="#10B981" />
        <line x1="150" y1="150" x2="150" y2="190" stroke="#10B981" stroke-width="3" />
        
        <!-- Light Rays -->
        <line x1="150" y1="50" x2="150" y2="20" stroke="{color}" stroke-width="2" />
        <line x1="110" y1="90" x2="80" y2="70" stroke="{color}" stroke-width="2" />
        <line x1="190" y1="90" x2="220" y2="70" stroke="{color}" stroke-width="2" />
        <line x1="90" y1="140" x2="60" y2="140" stroke="{color}" stroke-width="2" />
        <line x1="210" y1="140" x2="240" y2="140" stroke="{color}" stroke-width="2" />
    </svg>
    '''
    return svg

def generate_factory_svg(color="#FF4500", size=200):
    """
    Generate an SVG of a factory with emissions.
    
    Parameters:
    -----------
    color : str, optional
        Color of the emissions
    size : int, optional
        Size of the SVG
        
    Returns:
    --------
    str
        SVG code as a string
    """
    svg = f'''
    <svg width="{size}" height="{size}" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
        <!-- Factory Base -->
        <rect x="40" y="180" width="220" height="80" fill="#777777" />
        
        <!-- Factory Building -->
        <rect x="60" y="140" width="60" height="40" fill="#999999" />
        <rect x="140" y="120" width="60" height="60" fill="#999999" />
        <rect x="220" y="150" width="30" height="30" fill="#999999" />
        
        <!-- Windows -->
        <rect x="70" y="150" width="10" height="15" fill="#ADD8E6" />
        <rect x="90" y="150" width="10" height="15" fill="#ADD8E6" />
        <rect x="150" y="130" width="10" height="15" fill="#ADD8E6" />
        <rect x="170" y="130" width="10" height="15" fill="#ADD8E6" />
        <rect x="150" y="155" width="10" height="15" fill="#ADD8E6" />
        <rect x="170" y="155" width="10" height="15" fill="#ADD8E6" />
        
        <!-- Door -->
        <rect x="130" y="230" width="40" height="30" fill="#555555" />
        
        <!-- Smokestacks -->
        <rect x="80" y="100" width="15" height="40" fill="#555555" />
        <rect x="160" y="80" width="15" height="40" fill="#555555" />
        <rect x="230" y="120" width="10" height="30" fill="#555555" />
        
        <!-- Emissions -->
        <path d="M 85,100 C 75,90 95,80 85,70 C 75,60 95,50 85,40" stroke="{color}" stroke-width="5" fill="none" opacity="0.7" />
        <path d="M 167,80 C 157,70 177,60 167,50 C 157,40 177,30 167,20" stroke="{color}" stroke-width="5" fill="none" opacity="0.7" />
        <path d="M 235,120 C 245,110 225,100 235,90 C 245,80 225,70 235,60" stroke="{color}" stroke-width="5" fill="none" opacity="0.7" />
    </svg>
    '''
    return svg

def generate_earth_svg(size=200):
    """
    Generate an SVG of the Earth.
    
    Parameters:
    -----------
    size : int, optional
        Size of the SVG
        
    Returns:
    --------
    str
        SVG code as a string
    """
    svg = f'''
    <svg width="{size}" height="{size}" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
        <!-- Earth -->
        <circle cx="150" cy="150" r="100" fill="#1E88E5" />
        
        <!-- Continents -->
        <path d="M 100,90 C 120,85 130,100 150,95 C 170,90 180,100 200,95 C 210,90 220,100 210,110 C 200,120 180,115 170,125 C 160,135 170,145 160,155 C 150,165 130,160 120,170 C 110,180 120,200 110,210 C 100,220 80,215 70,205 C 60,195 70,185 60,175 C 50,165 70,155 65,145 C 60,135 80,125 75,115 C 70,105 90,95 100,90 Z" fill="#4CAF50" />
        <path d="M 200,170 C 210,165 220,180 230,175 C 240,170 250,180 245,190 C 240,200 220,195 210,205 C 200,215 220,225 210,235 C 200,245 180,235 190,225 C 200,215 190,205 200,195 C 210,185 190,175 200,170 Z" fill="#4CAF50" />
    </svg>
    '''
    return svg

def generate_gauge_svg(value, min_value=0, max_value=10, size=200, color_scheme="green_to_red"):
    """
    Generate an SVG gauge for visualizing threat levels.
    
    Parameters:
    -----------
    value : float
        Current value to display on the gauge
    min_value : float, optional
        Minimum value of the gauge
    max_value : float, optional
        Maximum value of the gauge
    size : int, optional
        Size of the SVG
    color_scheme : str, optional
        Color scheme for the gauge ('green_to_red', 'blue_to_red', etc.)
        
    Returns:
    --------
    str
        SVG code as a string
    """
    # Map value to degrees (0 to 180)
    value = max(min_value, min(value, max_value))  # Clamp value
    value_range = max_value - min_value
    angle = 180 * (value - min_value) / value_range

    # Choose color based on value and color scheme
    if color_scheme == "green_to_red":
        if value < (min_value + value_range * 0.3):
            color = "#10B981"  # Green
        elif value < (min_value + value_range * 0.6):
            color = "#F59E0B"  # Amber
        elif value < (min_value + value_range * 0.8):
            color = "#F97316"  # Orange
        else:
            color = "#EF4444"  # Red
    else:
        # Default blue gradient
        color = "#1E88E5"
        
    # Calculate needle endpoint
    center_x, center_y = 150, 150
    needle_length = 80
    needle_angle_rad = (180 - angle) * (3.14159 / 180)
    needle_x = center_x - needle_length * np.cos(needle_angle_rad)
    needle_y = center_y - needle_length * np.sin(needle_angle_rad)
    
    svg = f'''
    <svg width="{size}" height="{size}" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
        <!-- Gauge Background -->
        <path d="M 50,150 A 100,100 0 0 1 250,150" stroke="#E5E7EB" stroke-width="20" fill="none" />
        
        <!-- Gauge Value Arc -->
        <path d="M 50,150 A 100,100 0 0 1 {150 - 100 * np.cos((180 - angle) * (3.14159 / 180))},{150 - 100 * np.sin((180 - angle) * (3.14159 / 180))}" stroke="{color}" stroke-width="20" fill="none" />
        
        <!-- Gauge Needle -->
        <line x1="150" y1="150" x2="{needle_x}" y2="{needle_y}" stroke="#374151" stroke-width="3" />
        <circle cx="150" cy="150" r="10" fill="#374151" />
        
        <!-- Min Label -->
        <text x="50" y="180" font-family="Arial" font-size="14" fill="#6B7280">{min_value}</text>
        
        <!-- Max Label -->
        <text x="240" y="180" font-family="Arial" font-size="14" fill="#6B7280">{max_value}</text>
        
        <!-- Value Label -->
        <text x="150" y="210" font-family="Arial" font-size="20" fill="#374151" text-anchor="middle">{value}</text>
    </svg>
    '''
    return svg

# Remove the duplicate import