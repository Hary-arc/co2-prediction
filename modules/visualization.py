import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_emissions_trend(data, x_col, y_col, group_col=None, title=None):
    """
    Create a line plot of emissions trends over time.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    x_col : str
        Column name for x-axis (typically 'year')
    y_col : str
        Column name for y-axis (emissions)
    group_col : str, optional
        Column name for grouping (e.g., 'country')
    title : str, optional
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if group_col:
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col, 
            color=group_col,
            markers=True,
            title=title or f"{y_col} Trends by {group_col}"
        )
    else:
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col,
            markers=True,
            title=title or f"{y_col} Trends"
        )
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title=group_col if group_col else None
    )
    
    return fig

def plot_emissions_comparison(data, x_col, y_col, color_col=None, title=None):
    """
    Create a bar plot for emissions comparison across regions.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    x_col : str
        Column name for x-axis (typically region/country)
    y_col : str
        Column name for y-axis (emissions)
    color_col : str, optional
        Column name for color encoding
    title : str, optional
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if color_col:
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title or f"{y_col} Comparison by {x_col}"
        )
    else:
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            color=y_col,
            title=title or f"{y_col} Comparison by {x_col}"
        )
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return fig

def plot_emissions_map(data, region_col, value_col, title=None):
    """
    Create a choropleth map of emissions by region.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    region_col : str
        Column name for regions (country or state codes)
    value_col : str
        Column name for color values (emissions)
    title : str, optional
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # For simplicity, we'll create a world map
    # In a real application, you would need to ensure the region codes match the map format
    
    fig = px.choropleth(
        data,
        locations=region_col,
        color=value_col,
        hover_name=region_col,
        color_continuous_scale="Viridis",
        projection="natural earth",
        title=title or f"{value_col} by Region"
    )
    
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=value_col
        )
    )
    
    return fig

def plot_correlation_matrix(data, title=None):
    """
    Create a correlation matrix heatmap.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data (numeric columns only)
    title : str, optional
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        title=title or "Correlation Matrix"
    )
    
    fig.update_layout(
        width=700,
        height=700
    )
    
    return fig

def plot_feature_importance(importance_df, title=None):
    """
    Create a horizontal bar plot for feature importance.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame with 'Feature' and 'Importance' columns
    title : str, optional
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation='h',
        title=title or "Feature Importance"
    )
    
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature"
    )
    
    return fig

def plot_prediction_vs_actual(y_true, y_pred, title=None):
    """
    Create a scatter plot comparing actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    title : str, optional
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    # Calculate metrics
    r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
    
    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x='Actual',
        y='Predicted',
        title=title or f"Actual vs Predicted (R² = {r2:.4f})"
    )
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        )
    )
    
    fig.update_layout(
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values"
    )
    
    return fig

def plot_residuals(y_true, y_pred, title=None):
    """
    Create a plot of prediction residuals.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    title : str, optional
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Predicted': y_pred,
        'Residual': residuals
    })
    
    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x='Predicted',
        y='Residual',
        title=title or "Residual Plot"
    )
    
    # Add horizontal line at y=0
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red"
    )
    
    fig.update_layout(
        xaxis_title="Predicted Values",
        yaxis_title="Residuals"
    )
    
    return fig

def plot_uncertainty_predictions(x_values, predictions, lower_bounds, upper_bounds, x_label=None, y_label=None, title=None):
    """
    Create a plot of predictions with uncertainty intervals.
    
    Parameters:
    -----------
    x_values : array-like
        X-axis values (e.g., time points)
    predictions : array-like
        Predicted values
    lower_bounds : array-like
        Lower bounds of prediction intervals
    upper_bounds : array-like
        Upper bounds of prediction intervals
    x_label : str, optional
        X-axis label
    y_label : str, optional
        Y-axis label
    title : str, optional
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add predictions line
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=predictions,
            mode='lines+markers',
            name='Prediction',
            line=dict(color='royalblue')
        )
    )
    
    # Add uncertainty interval
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_values, x_values[::-1]]),
            y=np.concatenate([upper_bounds, lower_bounds[::-1]]),
            fill='toself',
            fillcolor='rgba(65, 105, 225, 0.2)',
            line=dict(color='rgba(65, 105, 225, 0)'),
            name='Uncertainty Interval'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title or "Predictions with Uncertainty Intervals",
        xaxis_title=x_label or "X",
        yaxis_title=y_label or "Prediction"
    )
    
    return fig
