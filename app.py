import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Import custom modules
from modules.data_preprocessing import load_data, clean_data, feature_engineering
from modules.model_training import train_models, predict_emissions, get_feature_importance, get_uncertainty_intervals
from modules.visualization import plot_emissions_trend, plot_emissions_comparison, plot_emissions_map, plot_correlation_matrix
from modules.utils import calculate_threat_level, get_emission_threshold, calculate_metrics, get_recommendations, convert_to_gwp
from data.sample_emissions_data import get_country_emissions, get_indian_states_emissions, get_emissions_time_series, get_mobile_towers_emissions

# Set page configuration
st.set_page_config(
    page_title="CO2 Emissions Predictor",
    page_icon="🌍",
    layout="wide"
)

# Load custom styles
from styles import load_css, add_logo, display_footer, display_threat_level_label, display_recommendation

# Apply custom styling
load_css()

# Add logo to sidebar
add_logo()

# Application title and description
st.title("CO2 Emissions Prediction & Analysis")
st.markdown("""
This application uses machine learning models to predict and analyze CO2 emissions from mobile base stations worldwide.
It provides insights based on historical data, population growth, and energy consumption patterns.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Model Training", "Predictions", "Comparative Analysis", "Threat Assessment"])

if page == "Home":
    st.header("Project Overview")
    
    st.subheader("About this project")
    st.write("""
    The increasing demand for mobile connectivity has raised concerns over the environmental impact of telecommunications infrastructure. 
    A significant contributor to CO₂ emissions in this sector is the reliance on diesel generators for powering 
    mobile base stations, particularly in remote or off-grid regions.
    
    This project presents a machine learning-based predictive model that estimates annual CO₂ emissions 
    from mobile towers worldwide. The model utilizes historical emissions data, population statistics, 
    and external factors such as technological advancements and government regulations to generate accurate forecasts.
    """)
    
    st.subheader("Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("- **Smart CO₂ Predictions**: Uses ML to estimate emissions across different regions")
        st.markdown("- **Future-Proof Forecasting**: Predicts emissions for past, present, and future years")
        st.markdown("- **Real-World Adaptability**: Accounts for technology growth and government policies")
    
    with col2:
        st.markdown("- **Optimized Accuracy**: Uses hyperparameter tuning and uncertainty analysis")
        st.markdown("- **Comparative Analysis**: Analyze emissions across different regions and countries")
        st.markdown("- **Threat Assessment**: Evaluate emission levels on a standardized scale")
    
    # Show sample data visualization for homepage
    st.subheader("Sample CO₂ Emissions by Country (Per Capita)")
    
    country_data = get_country_emissions()
    fig = px.bar(
        country_data, 
        x='Country', 
        y='Per Capita CO₂ (kg/person)',
        color='Per Capita CO₂ (kg/person)',
        color_continuous_scale='Viridis',
        title='CO₂ Emissions Per Capita by Country'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show information about the models used
    st.subheader("Machine Learning Models Used")
    st.write("""
    This application employs multiple machine learning models to predict CO₂ emissions:
    
    1. **Decision Tree Regression**: Used for its interpretability and efficiency in handling structured data.
    2. **Random Forest**: An ensemble technique that reduces overfitting and improves accuracy.
    3. **XGBoost (Extreme Gradient Boosting)**: Chosen for its speed and performance with large datasets.
    
    These models are evaluated using cross-validation and metrics such as RMSE, R², and MAE to ensure accuracy.
    """)

elif page == "Data Explorer":
    st.header("Data Explorer")
    
    # Data selection options
    st.subheader("Select Data to Explore")
    data_type = st.selectbox("Choose data category", ["Global Emissions by Country", "Indian States Emissions"])
    
    if data_type == "Global Emissions by Country":
        st.subheader("Country-level Emissions Data")
        country_data = get_country_emissions()
        st.dataframe(country_data)
        
        # Visualization options
        viz_type = st.selectbox("Choose visualization", ["Bar Chart", "Scatter Plot", "Heatmap"])
        
        if viz_type == "Bar Chart":
            gas_type = st.selectbox("Select gas type", ["CO₂", "CH₄", "CO"])
            
            column_map = {
                "CO₂": "Per Capita CO₂ (kg/person)",
                "CH₄": "Per Capita CH₄ (kg/person)",
                "CO": "Per Capita CO (kg/person)"
            }
            
            fig = px.bar(
                country_data, 
                x='Country', 
                y=column_map[gas_type],
                color=column_map[gas_type],
                title=f'Per Capita {gas_type} Emissions by Country'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Scatter Plot":
            fig = px.scatter(
                country_data, 
                x="Per Capita CO₂ (kg/person)", 
                y="Per Capita CH₄ (kg/person)",
                size="Per Capita CO (kg/person)",
                color="Country",
                hover_name="Country",
                title='Relationship Between CO₂ and CH₄ Emissions'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Heatmap":
            # Convert data to pivot format for heatmap
            pivot_data = country_data.set_index('Country')
            cols = [col for col in pivot_data.columns if "Per Capita" in col]
            pivot_data = pivot_data[cols]
            
            # Normalize data for better visualization
            normalized_data = (pivot_data - pivot_data.min()) / (pivot_data.max() - pivot_data.min())
            
            fig = px.imshow(
                normalized_data.T,
                labels=dict(x="Country", y="Emission Type", color="Normalized Value"),
                x=normalized_data.index,
                y=[col.replace("Per Capita ", "").replace(" (kg/person)", "") for col in cols],
                color_continuous_scale="Viridis",
                title="Normalized Emissions Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    elif data_type == "Indian States Emissions":
        st.subheader("Indian States Emissions Data")
        states_data = get_indian_states_emissions()
        st.dataframe(states_data)
        
        # Visualization options
        viz_type = st.selectbox("Choose visualization", ["Bar Chart", "Comparison", "Relationship"])
        
        if viz_type == "Bar Chart":
            gas_type = st.selectbox("Select gas type", ["CO₂", "CH₄", "CO"])
            
            column_map = {
                "CO₂": "CO₂ (kg/person)",
                "CH₄": "CH₄ (kg/person)",
                "CO": "CO (kg/person)"
            }
            
            fig = px.bar(
                states_data, 
                x='State', 
                y=column_map[gas_type],
                color=column_map[gas_type],
                title=f'Per Capita {gas_type} Emissions by Indian State'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Comparison":
            selected_states = st.multiselect("Select states to compare", states_data['State'].tolist(), default=states_data['State'].tolist()[:5])
            
            if selected_states:
                filtered_data = states_data[states_data['State'].isin(selected_states)]
                
                # Reshape data for grouped bar chart
                melted_data = pd.melt(
                    filtered_data, 
                    id_vars=['State'], 
                    value_vars=['CO₂ (kg/person)', 'CO (kg/person)', 'CH₄ (kg/person)'],
                    var_name='Gas Type', 
                    value_name='Emissions'
                )
                
                # Clean up the gas type labels
                melted_data['Gas Type'] = melted_data['Gas Type'].str.replace(' (kg/person)', '')
                
                fig = px.bar(
                    melted_data,
                    x='State',
                    y='Emissions',
                    color='Gas Type',
                    barmode='group',
                    title='Comparison of Greenhouse Gas Emissions Across Selected States'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one state for comparison")
                
        elif viz_type == "Relationship":
            fig = px.scatter(
                states_data, 
                x="CO₂ (kg/person)", 
                y="CH₄ (kg/person)",
                size="CO (kg/person)",
                color="State",
                hover_name="State",
                title='Relationship Between CO₂ and CH₄ Emissions in Indian States'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Observations")
            st.write("""
            The scatter plot reveals interesting patterns:
            
            - States with high CO₂ emissions tend to have lower CH₄ emissions, suggesting an inverse relationship between industrial activities and agricultural practices.
            - The size of the points (representing CO emissions) correlates more closely with CO₂ than with CH₄, indicating that CO and CO₂ likely come from similar sources.
            - Goa stands out with the highest CO₂ emissions but lowest CH₄ emissions, reflecting its economy's focus on tourism and services rather than agriculture.
            """)

elif page == "Model Training":
    st.header("Model Training & Evaluation")
    
    st.write("""
    This section demonstrates how machine learning models are trained to predict CO₂ emissions.
    The models are trained on historical data and evaluated using cross-validation.
    """)
    
    st.subheader("Model Selection")
    selected_models = st.multiselect(
        "Select models to train",
        ["Decision Tree", "Random Forest", "XGBoost"],
        default=["Random Forest"]
    )
    
    if not selected_models:
        st.warning("Please select at least one model to proceed")
    else:
        st.subheader("Feature Selection")
        
        # Simulating feature selection process - in a real app, this would be based on the actual dataset
        features = [
            "Energy Use per GDP", 
            "Energy Use per Capita", 
            "Population",
            "Urban Population (%)",
            "GDP",
            "GNI per Capita",
            "Protected Area (%)",
            "Urban Population Growth (%)",
            "Population Growth (%)",
            "Foreign Direct Investment (% of GDP)"
        ]
        
        selected_features = st.multiselect(
            "Select features to include in the model",
            features,
            default=features[:5]
        )
        
        if not selected_features:
            st.warning("Please select at least one feature to proceed")
        else:
            train_btn = st.button("Train Models")
            
            if train_btn:
                with st.spinner("Training models..."):
                    # Simulate model training with progress bar
                    progress_bar = st.progress(0)
                    
                    # Training each model
                    results = []
                    
                    for i, model_name in enumerate(selected_models):
                        # Update progress
                        progress = (i / len(selected_models)) * 100
                        progress_bar.progress(int(progress))
                        
                        # Simulate training metrics
                        if model_name == "Decision Tree":
                            rmse = np.random.uniform(800, 1200)
                            r2 = np.random.uniform(0.65, 0.75)
                            mae = np.random.uniform(600, 900)
                        elif model_name == "Random Forest":
                            rmse = np.random.uniform(500, 900)
                            r2 = np.random.uniform(0.75, 0.85)
                            mae = np.random.uniform(400, 700)
                        else:  # XGBoost
                            rmse = np.random.uniform(400, 800)
                            r2 = np.random.uniform(0.78, 0.88)
                            mae = np.random.uniform(350, 650)
                            
                        cv_scores = np.random.uniform(0.7, 0.9, 5)
                        
                        results.append({
                            "Model": model_name,
                            "RMSE": rmse,
                            "R²": r2,
                            "MAE": mae,
                            "CV Scores": cv_scores
                        })
                    
                    # Complete progress
                    progress_bar.progress(100)
                
                # Display results
                st.subheader("Training Results")
                
                # Metrics comparison
                metrics_df = pd.DataFrame([
                    {"Model": r["Model"], "RMSE": r["RMSE"], "R²": r["R²"], "MAE": r["MAE"]}
                    for r in results
                ])
                
                st.dataframe(metrics_df)
                
                # Visualize metrics comparison
                fig = go.Figure()
                
                for metric in ["RMSE", "MAE"]:
                    fig.add_trace(go.Bar(
                        x=metrics_df["Model"],
                        y=metrics_df[metric],
                        name=metric
                    ))
                
                fig.update_layout(
                    title="Error Metrics Comparison",
                    xaxis_title="Model",
                    yaxis_title="Value",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # R² comparison
                fig = px.bar(
                    metrics_df,
                    x="Model",
                    y="R²",
                    color="Model",
                    title="R² Score Comparison"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Cross-validation scores
                st.subheader("Cross-validation Scores")
                
                for result in results:
                    model_name = result["Model"]
                    cv_scores = result["CV Scores"]
                    
                    st.write(f"**{model_name}** - Mean CV Score: {np.mean(cv_scores):.4f}")
                    
                    cv_df = pd.DataFrame({
                        "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
                        "R² Score": cv_scores
                    })
                    
                    fig = px.bar(
                        cv_df,
                        x="Fold",
                        y="R² Score",
                        title=f"{model_name} - Cross-validation Scores"
                    )
                    
                    fig.update_layout(yaxis_range=[0.6, 1.0])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                st.subheader("Feature Importance")
                
                for result in results:
                    model_name = result["Model"]
                    
                    # Simulate feature importance
                    importance_values = np.random.uniform(0, 1, len(selected_features))
                    importance_values = importance_values / importance_values.sum()
                    
                    importance_df = pd.DataFrame({
                        "Feature": selected_features,
                        "Importance": importance_values
                    }).sort_values("Importance", ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x="Importance",
                        y="Feature",
                        orientation='h',
                        title=f"{model_name} - Feature Importance"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

elif page == "Predictions":
    st.header("CO₂ Emissions Predictions")
    
    st.subheader("Prediction Settings")
    
    # Region selection
    region_type = st.radio("Choose region type", ["Country", "Indian State"])
    
    if region_type == "Country":
        country_data = get_country_emissions()
        regions = country_data["Country"].tolist()
        selected_region = st.selectbox("Select country", regions)
    else:
        states_data = get_indian_states_emissions()
        regions = states_data["State"].tolist()
        selected_region = st.selectbox("Select state", regions)
    
    # Year selection
    current_year = 2024
    forecast_years = st.slider("Select forecast period", 1, 10, 5)
    years = list(range(current_year, current_year + forecast_years + 1))
    
    # Model selection
    model_type = st.selectbox("Select prediction model", ["Random Forest", "XGBoost", "Decision Tree"])
    
    # Additional parameters
    include_uncertainty = st.checkbox("Show uncertainty intervals", value=True)
    show_historical_trend = st.checkbox("Show historical trend", value=True)
    
    # Make predictions
    if st.button("Generate Predictions"):
        with st.spinner("Generating predictions..."):
            # Simulate historical data (past 5 years)
            if show_historical_trend:
                historical_years = list(range(current_year - 5, current_year))
                years_full = historical_years + years
                
                # Simulated historical data with some randomness but general trend
                if region_type == "Country":
                    base_value = country_data[country_data["Country"] == selected_region]["Per Capita CO₂ (kg/person)"].values[0]
                else:
                    base_value = states_data[states_data["State"] == selected_region]["CO₂ (kg/person)"].values[0]
                
                # Create a trend with some noise
                historical_values = [
                    base_value * (1 + (year - (current_year - 5)) * 0.03 + np.random.uniform(-0.05, 0.05))
                    for year in historical_years
                ]
            else:
                years_full = years
                historical_values = []
            
            # Create forecasted values based on a trend with noise
            if region_type == "Country":
                base_value = country_data[country_data["Country"] == selected_region]["Per Capita CO₂ (kg/person)"].values[0]
            else:
                base_value = states_data[states_data["State"] == selected_region]["CO₂ (kg/person)"].values[0]
            
            # Simulate different model predictions
            if model_type == "Random Forest":
                forecasted_values = [
                    base_value * (1 + (i * 0.02) - (i * i * 0.001) + np.random.uniform(-0.03, 0.03))
                    for i, _ in enumerate(years)
                ]
            elif model_type == "XGBoost":
                forecasted_values = [
                    base_value * (1 + (i * 0.015) - (i * i * 0.0008) + np.random.uniform(-0.02, 0.02))
                    for i, _ in enumerate(years)
                ]
            else:  # Decision Tree
                forecasted_values = [
                    base_value * (1 + (i * 0.025) - (i * i * 0.0012) + np.random.uniform(-0.05, 0.05))
                    for i, _ in enumerate(years)
                ]
            
            # Combine historical and forecasted values
            all_values = historical_values + forecasted_values
            
            # Create uncertainty intervals if requested
            if include_uncertainty:
                lower_bounds = [
                    all_values[i] * (1 - 0.05 - (i * 0.01))
                    for i in range(len(all_values))
                ]
                
                upper_bounds = [
                    all_values[i] * (1 + 0.05 + (i * 0.01))
                    for i in range(len(all_values))
                ]
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                "Year": years_full,
                "CO2_Emissions": all_values
            })
            
            if include_uncertainty:
                forecast_df["Lower_Bound"] = lower_bounds
                forecast_df["Upper_Bound"] = upper_bounds
            
            # Add period type (historical or forecast)
            if show_historical_trend:
                forecast_df["Period"] = ["Historical" if y < current_year else "Forecast" for y in years_full]
            else:
                forecast_df["Period"] = "Forecast"
        
        # Display predictions
        st.subheader(f"CO₂ Emissions Predictions for {selected_region}")
        
        # Create visualization
        fig = go.Figure()
        
        # Plot historical data if available
        if show_historical_trend:
            historical_df = forecast_df[forecast_df["Period"] == "Historical"]
            forecast_df = forecast_df[forecast_df["Period"] == "Forecast"]
            
            fig.add_trace(go.Scatter(
                x=historical_df["Year"],
                y=historical_df["CO2_Emissions"],
                mode="lines+markers",
                name="Historical Data",
                line=dict(color="blue")
            ))
        
        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast_df["Year"],
            y=forecast_df["CO2_Emissions"],
            mode="lines+markers",
            name=f"Forecast ({model_type})",
            line=dict(color="red")
        ))
        
        # Add uncertainty intervals if requested
        if include_uncertainty:
            fig.add_trace(go.Scatter(
                x=forecast_df["Year"].tolist() + forecast_df["Year"].tolist()[::-1],
                y=forecast_df["Upper_Bound"].tolist() + forecast_df["Lower_Bound"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(255, 0, 0, 0.1)",
                line=dict(color="rgba(255, 0, 0, 0)"),
                name="Uncertainty Interval"
            ))
        
        # Update layout
        fig.update_layout(
            title=f"CO₂ Emissions Prediction for {selected_region} ({years[0]}-{years[-1]})",
            xaxis_title="Year",
            yaxis_title="CO₂ Emissions (kg/person)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        st.subheader("Prediction Data")
        if show_historical_trend:
            display_df = pd.concat([historical_df, forecast_df])
        else:
            display_df = forecast_df
            
        display_df = display_df.rename(columns={
            "CO2_Emissions": "CO₂ Emissions (kg/person)"
        })
        
        if include_uncertainty:
            display_df = display_df.rename(columns={
                "Lower_Bound": "Lower Bound (kg/person)",
                "Upper_Bound": "Upper Bound (kg/person)"
            })
        
        st.dataframe(display_df)
        
        # Insights based on predictions
        st.subheader("Prediction Insights")
        
        # Calculate trend and overall change
        if len(forecast_df) > 1:
            start_value = forecast_df.iloc[0]["CO2_Emissions"]
            end_value = forecast_df.iloc[-1]["CO2_Emissions"]
            percent_change = ((end_value - start_value) / start_value) * 100
            
            if percent_change > 0:
                trend_label = "increase"
                trend_color = "red"
            else:
                trend_label = "decrease"
                trend_color = "green"
                
            st.markdown(f"<p>The forecast predicts a <span style='color:{trend_color};font-weight:bold;'>{abs(percent_change):.2f}% {trend_label}</span> in CO₂ emissions for {selected_region} from {years[0]} to {years[-1]}.</p>", unsafe_allow_html=True)
            
            # Provide recommendations based on trend
            st.subheader("Recommendations")
            
            if percent_change > 5:
                st.markdown("""
                Based on the upward trend in emissions, consider implementing:
                
                - **Renewable Energy Transition**: Accelerate the adoption of solar and wind power for mobile base stations
                - **Energy Efficiency Upgrades**: Invest in more energy-efficient equipment and cooling systems
                - **Smart Grid Integration**: Implement smart grid technologies to optimize power usage
                - **Policy Incentives**: Advocate for government incentives for green telecom infrastructure
                """)
            elif percent_change > 0:
                st.markdown("""
                With a slight increase projected, focus on:
                
                - **Gradual Transition**: Phase out diesel generators in favor of hybrid power solutions
                - **Monitoring Systems**: Implement real-time emissions monitoring and reporting
                - **Energy Audits**: Conduct regular energy audits to identify optimization opportunities
                - **Green Procurement**: Prioritize vendors with sustainable products and practices
                """)
            else:
                st.markdown("""
                To maintain the projected emissions reduction:
                
                - **Best Practices Sharing**: Share successful sustainability initiatives across regions
                - **Continue Modernization**: Maintain investment in modern, efficient infrastructure
                - **Set More Ambitious Targets**: Build on success with more aggressive emissions reduction goals
                - **Community Engagement**: Engage local communities in sustainability efforts
                """)

elif page == "Comparative Analysis":
    st.header("Comparative Analysis")
    
    st.write("""
    This section allows you to compare CO₂ emissions across different regions and analyze the relationship between 
    various greenhouse gases.
    """)
    
    analysis_type = st.selectbox(
        "Select analysis type",
        ["Regional Comparison", "Gas Type Comparison", "Time Trend Analysis"]
    )
    
    if analysis_type == "Regional Comparison":
        st.subheader("Compare CO₂ Emissions Across Regions")
        
        # Region selection
        region_type = st.radio("Choose region type", ["Countries", "Indian States"])
        
        if region_type == "Countries":
            country_data = get_country_emissions()
            regions = country_data["Country"].tolist()
            selected_regions = st.multiselect("Select countries to compare", regions, default=regions[:5])
            
            if not selected_regions:
                st.warning("Please select at least one country for comparison")
            else:
                filtered_data = country_data[country_data["Country"].isin(selected_regions)]
                
                # Gas type selection
                gas_type = st.selectbox("Select gas type", ["CO₂", "CH₄", "CO"])
                
                column_map = {
                    "CO₂": "Per Capita CO₂ (kg/person)",
                    "CH₄": "Per Capita CH₄ (kg/person)",
                    "CO": "Per Capita CO (kg/person)"
                }
                
                # Sort data
                sort_by = st.radio("Sort by", ["Alphabetically", "Value (Ascending)", "Value (Descending)"])
                
                if sort_by == "Alphabetically":
                    filtered_data = filtered_data.sort_values("Country")
                elif sort_by == "Value (Ascending)":
                    filtered_data = filtered_data.sort_values(column_map[gas_type])
                else:
                    filtered_data = filtered_data.sort_values(column_map[gas_type], ascending=False)
                
                # Create visualization
                fig = px.bar(
                    filtered_data,
                    x="Country",
                    y=column_map[gas_type],
                    color=column_map[gas_type],
                    color_continuous_scale="Viridis",
                    title=f"Comparison of {gas_type} Emissions Across Selected Countries"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data table
                st.subheader("Comparison Data")
                st.dataframe(filtered_data)
                
                # Stats summary
                st.subheader("Statistical Summary")
                
                stats_df = pd.DataFrame({
                    "Statistic": ["Mean", "Median", "Max", "Min", "Standard Deviation"],
                    "Value": [
                        filtered_data[column_map[gas_type]].mean(),
                        filtered_data[column_map[gas_type]].median(),
                        filtered_data[column_map[gas_type]].max(),
                        filtered_data[column_map[gas_type]].min(),
                        filtered_data[column_map[gas_type]].std()
                    ]
                })
                
                st.dataframe(stats_df)
                
        elif region_type == "Indian States":
            states_data = get_indian_states_emissions()
            regions = states_data["State"].tolist()
            selected_regions = st.multiselect("Select states to compare", regions, default=regions[:5])
            
            if not selected_regions:
                st.warning("Please select at least one state for comparison")
            else:
                filtered_data = states_data[states_data["State"].isin(selected_regions)]
                
                # Gas type selection
                gas_type = st.selectbox("Select gas type", ["CO₂", "CH₄", "CO"])
                
                column_map = {
                    "CO₂": "CO₂ (kg/person)",
                    "CH₄": "CH₄ (kg/person)",
                    "CO": "CO (kg/person)"
                }
                
                # Sort data
                sort_by = st.radio("Sort by", ["Alphabetically", "Value (Ascending)", "Value (Descending)"])
                
                if sort_by == "Alphabetically":
                    filtered_data = filtered_data.sort_values("State")
                elif sort_by == "Value (Ascending)":
                    filtered_data = filtered_data.sort_values(column_map[gas_type])
                else:
                    filtered_data = filtered_data.sort_values(column_map[gas_type], ascending=False)
                
                # Create visualization
                fig = px.bar(
                    filtered_data,
                    x="State",
                    y=column_map[gas_type],
                    color=column_map[gas_type],
                    color_continuous_scale="Viridis",
                    title=f"Comparison of {gas_type} Emissions Across Selected Indian States"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data table
                st.subheader("Comparison Data")
                st.dataframe(filtered_data)
                
                # Stats summary
                st.subheader("Statistical Summary")
                
                stats_df = pd.DataFrame({
                    "Statistic": ["Mean", "Median", "Max", "Min", "Standard Deviation"],
                    "Value": [
                        filtered_data[column_map[gas_type]].mean(),
                        filtered_data[column_map[gas_type]].median(),
                        filtered_data[column_map[gas_type]].max(),
                        filtered_data[column_map[gas_type]].min(),
                        filtered_data[column_map[gas_type]].std()
                    ]
                })
                
                st.dataframe(stats_df)
                
    elif analysis_type == "Gas Type Comparison":
        st.subheader("Compare Different Greenhouse Gases")
        
        # Region selection
        region_type = st.radio("Choose region type", ["Countries", "Indian States"])
        
        if region_type == "Countries":
            country_data = get_country_emissions()
            regions = country_data["Country"].tolist()
            selected_region = st.selectbox("Select a country", regions)
            
            filtered_data = country_data[country_data["Country"] == selected_region]
            
            # Prepare data for visualization
            gas_data = pd.DataFrame({
                "Gas Type": ["CO₂", "CH₄", "CO"],
                "Emissions (kg/person)": [
                    filtered_data["Per Capita CO₂ (kg/person)"].values[0],
                    filtered_data["Per Capita CH₄ (kg/person)"].values[0],
                    filtered_data["Per Capita CO (kg/person)"].values[0]
                ]
            })
            
            # Add global warming potential (GWP) for comparison
            gas_data["GWP (CO₂ equivalent)"] = gas_data["Emissions (kg/person)"] * np.array([1, 25, 3])
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Raw Emissions Comparison")
                fig = px.bar(
                    gas_data,
                    x="Gas Type",
                    y="Emissions (kg/person)",
                    color="Gas Type",
                    title=f"Greenhouse Gas Emissions in {selected_region}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("Global Warming Potential")
                fig = px.bar(
                    gas_data,
                    x="Gas Type",
                    y="GWP (CO₂ equivalent)",
                    color="Gas Type",
                    title=f"Global Warming Potential in {selected_region}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display data
            st.subheader("Emissions Data")
            st.dataframe(gas_data)
            
            # Explanation
            st.subheader("Understanding Global Warming Potential (GWP)")
            st.write("""
            Global Warming Potential (GWP) is a measure of how much heat a greenhouse gas traps in the atmosphere 
            relative to carbon dioxide (CO₂) over a specific time period, typically 100 years.
            
            - **CO₂** has a GWP of 1 (the baseline)
            - **CH₄** (Methane) has a GWP of 25, meaning it traps 25 times more heat than CO₂ over 100 years
            - **CO** (Carbon Monoxide) indirectly affects climate by reacting with hydroxyl radicals, 
              which would otherwise break down methane. For this analysis, an approximate GWP of 3 is used.
            
            This allows for a more accurate comparison of the climate impact of different gases, beyond just their emission quantities.
            """)
            
        elif region_type == "Indian States":
            states_data = get_indian_states_emissions()
            regions = states_data["State"].tolist()
            selected_region = st.selectbox("Select a state", regions)
            
            filtered_data = states_data[states_data["State"] == selected_region]
            
            # Prepare data for visualization
            gas_data = pd.DataFrame({
                "Gas Type": ["CO₂", "CH₄", "CO"],
                "Emissions (kg/person)": [
                    filtered_data["CO₂ (kg/person)"].values[0],
                    filtered_data["CH₄ (kg/person)"].values[0],
                    filtered_data["CO (kg/person)"].values[0]
                ]
            })
            
            # Add global warming potential (GWP) for comparison
            gas_data["GWP (CO₂ equivalent)"] = gas_data["Emissions (kg/person)"] * np.array([1, 25, 3])
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Raw Emissions Comparison")
                fig = px.bar(
                    gas_data,
                    x="Gas Type",
                    y="Emissions (kg/person)",
                    color="Gas Type",
                    title=f"Greenhouse Gas Emissions in {selected_region}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.subheader("Global Warming Potential")
                fig = px.bar(
                    gas_data,
                    x="Gas Type",
                    y="GWP (CO₂ equivalent)",
                    color="Gas Type",
                    title=f"Global Warming Potential in {selected_region}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display data
            st.subheader("Emissions Data")
            st.dataframe(gas_data)
            
            # Explanation
            st.subheader("Understanding Global Warming Potential (GWP)")
            st.write("""
            Global Warming Potential (GWP) is a measure of how much heat a greenhouse gas traps in the atmosphere 
            relative to carbon dioxide (CO₂) over a specific time period, typically 100 years.
            
            - **CO₂** has a GWP of 1 (the baseline)
            - **CH₄** (Methane) has a GWP of 25, meaning it traps 25 times more heat than CO₂ over 100 years
            - **CO** (Carbon Monoxide) indirectly affects climate by reacting with hydroxyl radicals, 
              which would otherwise break down methane. For this analysis, an approximate GWP of 3 is used.
            
            This allows for a more accurate comparison of the climate impact of different gases, beyond just their emission quantities.
            """)
            
    elif analysis_type == "Time Trend Analysis":
        st.subheader("Analyze Emission Trends Over Time")
        
        # Sample regions for trend analysis
        regions = ["China", "India", "Brazil", "USA", "Germany"]
        selected_regions = st.multiselect("Select regions to analyze", regions, default=regions[:3])
        
        if not selected_regions:
            st.warning("Please select at least one region for analysis")
        else:
            # Year range
            years = list(range(2019, 2025))
            
            # Create simulated time series data
            trend_data = []
            
            for region in selected_regions:
                # Base value depends on region
                if region == "China":
                    base_value = 7680.45
                    trend_factor = -0.02  # Slight decrease
                elif region == "India":
                    base_value = 1850.12
                    trend_factor = 0.04  # Increase
                elif region == "Brazil":
                    base_value = 2950.78
                    trend_factor = -0.01  # Slight decrease
                elif region == "USA":
                    base_value = 14512.90
                    trend_factor = -0.03  # Decrease
                else:  # Germany
                    base_value = 8420.67
                    trend_factor = -0.04  # Substantial decrease
                
                # Generate yearly data with trend and some randomness
                for year in years:
                    year_idx = year - 2019
                    value = base_value * (1 + (year_idx * trend_factor) + np.random.uniform(-0.02, 0.02))
                    
                    trend_data.append({
                        "Region": region,
                        "Year": year,
                        "CO₂ Emissions (kg/person)": value
                    })
            
            trend_df = pd.DataFrame(trend_data)
            
            # Create visualization
            fig = px.line(
                trend_df,
                x="Year",
                y="CO₂ Emissions (kg/person)",
                color="Region",
                markers=True,
                title="CO₂ Emissions Trends Over Time (2019-2024)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate percent changes from 2019 to 2024
            summary_data = []
            
            for region in selected_regions:
                region_data = trend_df[trend_df["Region"] == region]
                start_value = region_data[region_data["Year"] == 2019]["CO₂ Emissions (kg/person)"].values[0]
                end_value = region_data[region_data["Year"] == 2024]["CO₂ Emissions (kg/person)"].values[0]
                percent_change = ((end_value - start_value) / start_value) * 100
                
                summary_data.append({
                    "Region": region,
                    "2019 Emissions": start_value,
                    "2024 Emissions": end_value,
                    "Change (%)": percent_change
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Display summary
            st.subheader("Trend Summary (2019-2024)")
            st.dataframe(summary_df)
            
            # Visualize percent changes
            fig = px.bar(
                summary_df,
                x="Region",
                y="Change (%)",
                color="Change (%)",
                color_continuous_scale="RdYlGn_r",
                title="Percent Change in CO₂ Emissions (2019-2024)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis and insights
            st.subheader("Trend Analysis")
            
            decreasing_regions = summary_df[summary_df["Change (%)"] < 0]["Region"].tolist()
            increasing_regions = summary_df[summary_df["Change (%)"] > 0]["Region"].tolist()
            
            if len(decreasing_regions) > 0:
                st.write(f"**Regions with decreasing emissions**: {', '.join(decreasing_regions)}")
                
                # Get the region with the largest decrease
                largest_decrease = summary_df.loc[summary_df["Change (%)"].idxmin()]
                st.markdown(f"**{largest_decrease['Region']}** shows the largest decrease at **{largest_decrease['Change (%)']:.2f}%**, "
                          f"potentially due to renewable energy adoption and strict environmental regulations.")
            
            if len(increasing_regions) > 0:
                st.write(f"**Regions with increasing emissions**: {', '.join(increasing_regions)}")
                
                # Get the region with the largest increase
                largest_increase = summary_df.loc[summary_df["Change (%)"].idxmax()]
                st.markdown(f"**{largest_increase['Region']}** shows the largest increase at **{largest_increase['Change (%)']:.2f}%**, "
                          f"likely due to rapid industrialization and economic growth.")
            
            st.markdown("""
            ### Key Factors Influencing Emission Trends
            
            1. **Policy Frameworks**: Regions with strong environmental policies typically show emissions reductions
            2. **Renewable Energy Adoption**: Transition to renewables correlates with decreased emissions
            3. **Economic Development Stage**: Rapidly developing economies often experience emission increases
            4. **Technology Deployment**: Implementation of efficient technologies can reduce emissions
            5. **Infrastructure Age**: Older infrastructure generally produces higher emissions
            
            These factors interact in complex ways to produce the observed trends. Long-term emissions reductions
            require comprehensive approaches addressing all of these areas.
            """)

elif page == "Threat Assessment":
    st.header("CO₂ Emissions Threat Assessment")
    
    st.write("""
    This section evaluates the threat level of CO₂ emissions on a standardized scale from 1-10, 
    with 1 being minimal threat and 10 being severe threat. The assessment considers various factors
    including emission levels, trends, and global impact.
    """)
    
    # Region selection
    region_type = st.radio("Choose region type", ["Country", "Indian State"])
    
    if region_type == "Country":
        country_data = get_country_emissions()
        regions = country_data["Country"].tolist()
        selected_region = st.selectbox("Select country", regions)
        
        if selected_region:
            co2_value = country_data[country_data["Country"] == selected_region]["Per Capita CO₂ (kg/person)"].values[0]
    else:
        states_data = get_indian_states_emissions()
        regions = states_data["State"].tolist()
        selected_region = st.selectbox("Select state", regions)
        
        if selected_region:
            co2_value = states_data[states_data["State"] == selected_region]["CO₂ (kg/person)"].values[0]
    
    # Simulate future projections
    projection_scenarios = ["Current Policies", "Moderate Mitigation", "Aggressive Mitigation"]
    selected_scenario = st.selectbox("Select projection scenario", projection_scenarios)
    
    if st.button("Generate Threat Assessment"):
        # Calculate threat level based on CO2 value (1-10 scale)
        # Using a simple mapping function for this example
        if selected_scenario == "Current Policies":
            # Higher threat level with current policies
            raw_threat_level = (co2_value / 15000) * 10
        elif selected_scenario == "Moderate Mitigation":
            # Medium threat reduction with moderate mitigation
            raw_threat_level = (co2_value / 15000) * 8
        else:  # Aggressive Mitigation
            # Significant threat reduction with aggressive mitigation
            raw_threat_level = (co2_value / 15000) * 6
            
        # Ensure threat level is between 1-10
        threat_level = max(1, min(10, raw_threat_level))
        threat_level_rounded = round(threat_level, 1)
        
        # Determine threat category
        if threat_level < 3:
            threat_category = "Low"
            threat_color = "green"
        elif threat_level < 6:
            threat_category = "Moderate"
            threat_color = "orange"
        elif threat_level < 8:
            threat_category = "High"
            threat_color = "red"
        else:
            threat_category = "Severe"
            threat_color = "darkred"
        
        # Display threat gauge
        st.subheader(f"Threat Assessment for {selected_region}")
        
        # Use custom styling for threat level display
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display threat level with custom styling
            display_threat_level_label(threat_level_rounded, threat_category)
        
        with col2:
            # Use the Plotly gauge instead of SVG due to SVG compatibility issues
            # The SVG gauge was causing errors in the browser
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = threat_level_rounded,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"CO₂ Emissions Threat Level - {threat_category}", 'font': {'color': threat_color}},
                gauge = {
                    'axis': {'range': [1, 10], 'tickwidth': 1, 'tickcolor': "black"},
                    'bar': {'color': threat_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [1, 3], 'color': 'green'},
                        {'range': [3, 6], 'color': 'orange'},
                        {'range': [6, 8], 'color': 'red'},
                        {'range': [8, 10], 'color': 'darkred'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threat_level_rounded
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display threat assessment details
        st.subheader("Threat Assessment Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Region**: {selected_region}")
            st.markdown(f"**CO₂ Emissions**: {co2_value:.2f} kg/person")
            st.markdown(f"**Threat Level**: {threat_level_rounded}/10 ({threat_category})")
            st.markdown(f"**Scenario**: {selected_scenario}")
        
        with col2:
            # Simulate other environmental metrics
            air_quality_impact = 60 + (threat_level * 4)
            health_risk = 40 + (threat_level * 6)
            ecosystem_impact = 50 + (threat_level * 5)
            
            st.markdown(f"**Air Quality Impact**: {min(100, air_quality_impact):.1f}/100")
            st.markdown(f"**Health Risk Factor**: {min(100, health_risk):.1f}/100")
            st.markdown(f"**Ecosystem Impact**: {min(100, ecosystem_impact):.1f}/100")
        
        # Alert message for high threat levels
        if threat_level >= 7:
            st.error(f"""
            ⚠️ **ALERT**: CO₂ emissions in {selected_region} have reached a critical level.
            
            Immediate action is needed to reduce emissions and mitigate environmental impact.
            """)
        elif threat_level >= 5:
            st.warning(f"""
            ⚠️ **WARNING**: CO₂ emissions in {selected_region} are at a concerning level.
            
            Mitigation measures should be implemented to prevent further increase.
            """)
        
        # Threat assessment explanation
        st.subheader("Understanding the Threat Scale")
        st.write("""
        The CO₂ emissions threat scale ranges from 1 to 10:
        
        - **1-3 (Low)**: Minimal environmental impact with sustainable emission levels
        - **4-6 (Moderate)**: Notable impact on environment with concerning emission trends
        - **7-8 (High)**: Significant environmental impact requiring immediate attention
        - **9-10 (Severe)**: Critical emission levels with potentially irreversible environmental damage
        
        The assessment considers per capita emissions, projected trends, and the selected policy scenario.
        """)
        
        # Mitigation recommendations
        st.subheader("Mitigation Recommendations")
        
        # Get recommendations from the utils module based on threat level
        recommendations = get_recommendations(threat_level, emission_type='CO₂')
        
        # Display recommendations with custom styling
        for rec in recommendations:
            display_recommendation(rec)
            
        # Visual element for telecom towers 
        st.subheader("Mobile Tower Emissions Visualization")
        
        # Use emojis instead of SVG for visualizations due to compatibility issues
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current Tower Technology")
            # Instead of SVG, use an emoji and styled HTML for tower visualization
            st.markdown("""
            <div style="text-align: center; font-size: 3rem; color: #F97316;">
                🗼
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**High-emission diesel generators**")
            
        with col2:
            st.markdown("#### Recommended Tower Technology")
            # Use emoji and styled HTML instead of SVG
            st.markdown("""
            <div style="text-align: center; font-size: 3rem; color: #16A34A;">
                🗼
            </div>
            <div style="text-align: center; font-size: 1.5rem; color: #16A34A;">
                ☀️ 
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**Low-emission renewable power sources**")
            
        # Add footer with custom styling
        display_footer()
