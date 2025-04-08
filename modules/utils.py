import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred)
    }
    return metrics

def calculate_threat_level(co2_value, base_threshold=1000, max_threshold=15000):
    """
    Calculate threat level on a scale of 1-10 based on CO2 emissions.
    
    Parameters:
    -----------
    co2_value : float
        CO2 emissions value
    base_threshold : float, optional
        Threshold for minimum threat level
    max_threshold : float, optional
        Threshold for maximum threat level
        
    Returns:
    --------
    float
        Threat level on a scale of 1-10
    """
    # Normalize the CO2 value to a 1-10 scale
    if co2_value <= base_threshold:
        threat_level = 1.0
    elif co2_value >= max_threshold:
        threat_level = 10.0
    else:
        # Linear mapping from base_threshold-max_threshold to 1-10
        normalized_value = (co2_value - base_threshold) / (max_threshold - base_threshold)
        threat_level = 1.0 + normalized_value * 9.0
    
    return threat_level

def get_emission_threshold(region_type='global', year=None):
    """
    Get CO2 emission thresholds for various threat levels.
    
    Parameters:
    -----------
    region_type : str, optional
        Type of region ('global', 'developed', 'developing')
    year : int, optional
        Year for which thresholds are needed
        
    Returns:
    --------
    dict
        Dictionary containing thresholds for different threat levels
    """
    # Base thresholds for different region types
    if region_type == 'developed':
        thresholds = {
            'low': 1000,
            'moderate': 5000,
            'high': 10000,
            'severe': 15000
        }
    elif region_type == 'developing':
        thresholds = {
            'low': 500,
            'moderate': 2500,
            'high': 7500,
            'severe': 12500
        }
    else:  # global
        thresholds = {
            'low': 750,
            'moderate': 3000,
            'high': 8000,
            'severe': 13000
        }
    
    # Adjust thresholds based on year if provided
    if year is not None:
        # Example: Assume thresholds decrease by 2% per year after 2020
        if year > 2020:
            adjustment_factor = 1.0 - 0.02 * (year - 2020)
            for level in thresholds:
                thresholds[level] *= adjustment_factor
    
    return thresholds

def get_recommendations(threat_level, emission_type='CO2'):
    """
    Get recommendations based on threat level.
    
    Parameters:
    -----------
    threat_level : float
        Threat level on a scale of 1-10
    emission_type : str, optional
        Type of emission ('CO2', 'CH4', etc.)
        
    Returns:
    --------
    list
        List of recommendation strings
    """
    recommendations = []
    
    if emission_type == 'CO2':
        if threat_level < 3:
            recommendations = [
                "Maintain current green practices and continue renewable energy integration",
                "Set more ambitious emission reduction targets for future planning",
                "Increase community awareness about sustainable energy practices",
                "Implement robust monitoring systems to track emissions over time"
            ]
        elif threat_level < 6:
            recommendations = [
                "Increase renewable energy sources to at least 50% of total power consumption",
                "Upgrade to energy-efficient infrastructure for mobile base stations",
                "Develop and strictly enforce emissions reduction policies",
                "Invest in green technology research and implementation",
                "Consider carbon offsetting programs for unavoidable emissions"
            ]
        elif threat_level < 8:
            recommendations = [
                "Accelerate urgent transition to renewable energy sources for all base stations",
                "Implement strict emissions regulations with penalties for non-compliance",
                "Consider carbon taxes or emissions trading systems to incentivize reductions",
                "Prioritize modernization of high-emission infrastructure",
                "Engage in international emissions reduction initiatives for industry standards",
                "Deploy energy storage solutions to increase renewable energy utilization"
            ]
        else:
            recommendations = [
                "Declare climate emergency and mobilize all available resources for mitigation",
                "Implement radical transformation of energy systems for telecom infrastructure",
                "Make emissions reduction the highest policy priority with executive oversight",
                "Seek international support and cooperation for advanced green technologies",
                "Address all emission sources simultaneously with comprehensive action plan",
                "Engage the public in emissions reduction efforts through awareness campaigns",
                "Implement continuous monitoring and adaptive management strategies",
                "Consider temporary restrictions on non-essential high-emission activities"
            ]
    elif emission_type == 'CH4':
        # Methane-specific recommendations
        if threat_level < 5:
            recommendations = [
                "Monitor methane leaks in natural gas infrastructure",
                "Implement regular inspection and maintenance schedules",
                "Upgrade to low-leak equipment and components"
            ]
        else:
            recommendations = [
                "Implement comprehensive methane leak detection and repair programs",
                "Replace high-leak equipment with zero-emission alternatives",
                "Consider methane capture and utilization technologies"
            ]
    else:
        # Generic recommendations
        if threat_level < 5:
            recommendations = [
                "Monitor emission levels regularly",
                "Implement basic emission control measures",
                "Develop environmental management plans"
            ]
        else:
            recommendations = [
                "Implement strict emission control technologies",
                "Consider facility upgrades or replacements",
                "Develop comprehensive environmental management systems"
            ]
    
    return recommendations

def convert_to_gwp(emissions, gas_type):
    """
    Convert emissions to Global Warming Potential (GWP) CO2 equivalent.
    
    Parameters:
    -----------
    emissions : float or array-like
        Emission values
    gas_type : str
        Type of gas ('CO2', 'CH4', 'CO', etc.)
        
    Returns:
    --------
    float or array-like
        CO2 equivalent emissions
    """
    # GWP values (100-year time horizon, based on IPCC AR5)
    gwp = {
        'CO2': 1,
        'CH4': 28,  # Methane
        'N2O': 265,  # Nitrous oxide
        'CO': 3,  # Carbon monoxide (approximate)
        'HFC-134a': 1300,  # Hydrofluorocarbon-134a
        'SF6': 23500  # Sulfur hexafluoride
    }
    
    if gas_type in gwp:
        return emissions * gwp[gas_type]
    else:
        # If gas type not found, return original emissions
        return emissions