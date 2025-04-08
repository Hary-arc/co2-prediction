import pandas as pd
import numpy as np

def get_country_emissions():
    """
    Get sample emissions data for different countries.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with country emissions data
    """
    # Sample data from the abstract
    data = {
        'Country': [
            'India', 'China', 'Brazil', 'Argentina', 'Russia', 
            'USA', 'Germany', 'France'
        ],
        'Per Capita CO₂ (kg/person)': [
            1850.12, 7680.45, 2950.78, 3215.40, 4102.32, 
            14512.90, 8420.67, 5896.78
        ],
        'Per Capita CH₄ (kg/person)': [
            22.35, 19.76, 24.89, 29.34, 27.50, 
            23.72, 15.98, 12.34
        ],
        'Per Capita CO (kg/person)': [
            10.45, 35.12, 18.67, 19.89, 21.56, 
            40.23, 33.45, 30.23
        ]
    }
    
    return pd.DataFrame(data)

def get_indian_states_emissions():
    """
    Get sample emissions data for Indian states.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with Indian states emissions data
    """
    # Sample data from the abstract
    data = {
        'State': [
            'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
            'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir'
        ],
        'CO₂ (kg/person)': [
            974.17, 405.90, 340.91, 179.01, 1963.88,
            2662.51, 1310.58, 1381.86, 784.16, 509.03
        ],
        'CO (kg/person)': [
            27.18, 17.43, 16.63, 8.83, 17.56,
            23.12, 24.01, 17.90, 16.98, 15.59
        ],
        'CH₄ (kg/person)': [
            16.97, 25.82, 21.29, 9.59, 22.37,
            7.62, 12.26, 21.57, 18.28, 14.42
        ]
    }
    
    return pd.DataFrame(data)

def get_emissions_time_series():
    """
    Get sample time series emissions data.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with time series emissions data
    """
    # Years
    years = list(range(2019, 2025))
    
    # Countries
    countries = ['China', 'India', 'USA', 'Brazil', 'Germany']
    
    data = []
    
    # Base values for 2019
    base_values = {
        'China': 7500,
        'India': 1750,
        'USA': 15000,
        'Brazil': 3000,
        'Germany': 8500
    }
    
    # Trend factors
    trend_factors = {
        'China': -0.02,  # Slight decrease
        'India': 0.04,   # Increase
        'USA': -0.03,    # Decrease
        'Germany': -0.04,  # Substantial decrease
        'Brazil': -0.01   # Slight decrease
    }
    
    # Generate data
    for country in countries:
        base = base_values[country]
        trend = trend_factors[country]
        
        for year in years:
            year_idx = year - 2019
            value = base * (1 + (year_idx * trend) + 0.01 * ((-1)**year_idx))  # Add slight oscillation
            
            data.append({
                'Country': country,
                'Year': year,
                'CO₂ Emissions (kg/person)': value
            })
    
    return pd.DataFrame(data)

def get_mobile_towers_emissions():
    """
    Get sample emissions data specific to mobile towers.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with mobile towers emissions data
    """
    # Regions
    regions = ['Latin America', 'China', 'India']
    
    # Years
    years = list(range(2019, 2025))
    
    # Emission types
    emission_types = ['Diesel Generators', 'Grid Electricity', 'Renewable Energy']
    
    data = []
    
    # Base values for 2019 (tons of CO2)
    base_values = {
        'Latin America': {
            'Diesel Generators': 1200000,
            'Grid Electricity': 800000,
            'Renewable Energy': 100000
        },
        'China': {
            'Diesel Generators': 3000000,
            'Grid Electricity': 5000000,
            'Renewable Energy': 500000
        },
        'India': {
            'Diesel Generators': 2500000,
            'Grid Electricity': 1500000,
            'Renewable Energy': 200000
        }
    }
    
    # Trend factors
    trend_factors = {
        'Latin America': {
            'Diesel Generators': -0.08,   # Decreasing
            'Grid Electricity': 0.02,     # Slight increase
            'Renewable Energy': 0.15      # Significant increase
        },
        'China': {
            'Diesel Generators': -0.05,   # Decreasing
            'Grid Electricity': 0.04,     # Increasing
            'Renewable Energy': 0.20      # Rapid increase
        },
        'India': {
            'Diesel Generators': -0.03,   # Slightly decreasing
            'Grid Electricity': 0.05,     # Increasing
            'Renewable Energy': 0.25      # Rapid increase
        }
    }
    
    # Generate data
    for region in regions:
        for emission_type in emission_types:
            base = base_values[region][emission_type]
            trend = trend_factors[region][emission_type]
            
            for year in years:
                year_idx = year - 2019
                value = base * (1 + (year_idx * trend) + 0.02 * np.random.randn())  # Add random noise
                
                data.append({
                    'Region': region,
                    'Year': year,
                    'Emission Source': emission_type,
                    'CO₂ Emissions (tons)': value
                })
    
    return pd.DataFrame(data)
