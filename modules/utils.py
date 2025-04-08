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
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'R2': r2,
        'MAE': mae
    }

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
    # Ensure co2_value is not negative
    co2_value = max(0, co2_value)
    
    # Calculate raw threat level using a logarithmic scale
    # This ensures that small changes at low emissions are more significant than at high emissions
    if co2_value <= base_threshold:
        # Below base threshold, threat level is proportional to emission value
        threat_level = co2_value / base_threshold * 3  # Max 3 for values below base_threshold
    else:
        # Above base threshold, use logarithmic scaling
        log_factor = np.log10(co2_value / base_threshold) / np.log10(max_threshold / base_threshold)
        threat_level = 3 + (7 * min(1, log_factor))  # 3-10 range for values above base_threshold
    
    # Ensure threat level is between 1 and 10
    threat_level = max(1, min(10, threat_level))
    
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
    # Base thresholds for global average
    base_thresholds = {
        'low': 2000,        # Below 2000 kg/person is considered low
        'moderate': 5000,   # 2000-5000 kg/person is moderate
        'high': 10000,      # 5000-10000 kg/person is high
        'severe': 15000     # Above 10000 kg/person is severe
    }
    
    # Adjust thresholds based on region type
    if region_type == 'developed':
        # Developed countries have historically higher emissions
        factor = 1.2
    elif region_type == 'developing':
        # Developing countries have historically lower emissions
        factor = 0.8
    else:  # 'global'
        factor = 1.0
    
    # Adjust thresholds based on year (stricter thresholds for future years)
    if year and year > 2020:
        # Gradually reduce thresholds for future years
        year_factor = 1.0 - min(0.3, (year - 2020) * 0.02)  # Max 30% reduction
    else:
        year_factor = 1.0
    
    # Apply adjustments
    adjusted_thresholds = {
        level: threshold * factor * year_factor
        for level, threshold in base_thresholds.items()
    }
    
    return adjusted_thresholds

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
    # Base recommendations for CO2 emissions
    if emission_type == 'CO2':
        if threat_level < 3:  # Low
            recommendations = [
                "Maintain current sustainable practices",
                "Continue monitoring emissions",
                "Set targets for further reductions",
                "Explore renewable energy options for future implementation"
            ]
        elif threat_level < 6:  # Moderate
            recommendations = [
                "Increase renewable energy adoption",
                "Implement energy efficiency measures",
                "Develop emissions reduction policies",
                "Conduct regular energy audits",
                "Invest in green technology research"
            ]
        elif threat_level < 8:  # High
            recommendations = [
                "Accelerate transition to renewable energy",
                "Implement strict emissions regulations",
                "Consider carbon pricing mechanisms",
                "Modernize high-emission infrastructure",
                "Set aggressive emissions reduction targets",
                "Engage in international climate initiatives"
            ]
        else:  # Severe
            recommendations = [
                "Declare climate emergency",
                "Radically transform energy systems",
                "Prioritize emissions reduction in all policies",
                "Seek international support and cooperation",
                "Implement comprehensive emissions monitoring",
                "Engage the public in emissions reduction efforts",
                "Invest heavily in carbon capture technologies"
            ]
    else:
        # Generic recommendations for other emission types
        if threat_level < 5:
            recommendations = [
                "Monitor emissions regularly",
                "Implement standard mitigation measures",
                "Research sector-specific reduction strategies"
            ]
        else:
            recommendations = [
                "Implement aggressive emission control measures",
                "Adopt best available technologies",
                "Develop strict regulatory framework",
                "Collaborate with experts for tailored solutions"
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
    # GWP values for 100-year time horizon (IPCC Fifth Assessment Report)
    gwp_factors = {
        'CO2': 1,
        'CH4': 25,   # Methane is 25 times more potent than CO2
        'CO': 3,     # Approximate factor for carbon monoxide
        'N2O': 298,  # Nitrous oxide
        'HFC': 12400,  # Hydrofluorocarbons (average of common HFCs)
        'SF6': 22800  # Sulfur hexafluoride
    }
    
    # Get GWP factor for the gas type (default to 1 if not found)
    factor = gwp_factors.get(gas_type, 1)
    
    # Calculate CO2 equivalent
    co2_equivalent = emissions * factor
    
    return co2_equivalent
