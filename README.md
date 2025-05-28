# CO2 Emissions Predictor

A machine learning-based web application for predicting and analyzing CO2 emissions from mobile base stations worldwide.

## Features

- **Smart CO₂ Predictions**: Uses machine learning to estimate emissions across different regions
- **Future-Proof Forecasting**: Predicts emissions for past, present, and future years
- **Real-World Adaptability**: Accounts for technology growth and government policies
- **Optimized Accuracy**: Uses hyperparameter tuning and uncertainty analysis
- **Comparative Analysis**: Analyze emissions across different regions and countries
- **Threat Assessment**: Evaluate emission levels on a standardized scale

## Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost

## Live Demo

Visit the live application at: [CO2 Emissions Predictor](https://co2-prediction-majorproject-btechcse-2025sec-n.streamlit.app/)

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python -m streamlit run app.py
```

3. Open your web browser and navigate to `http://localhost:8501`

## Deployment on Streamlit Cloud

1. Fork this repository to your GitHub account

2. Visit [Streamlit Cloud](https://streamlit.io/cloud)

3. Sign in with your GitHub account

4. Click on "New app" and select this repository

5. Select the main branch and enter: `app.py` as the main file path

6. Click "Deploy"

The app will be automatically deployed and available at a public URL.

## Project Structure

- `app.py`: Main application file
- `data_preprocessing.py`: Data cleaning and feature engineering
- `model_training.py`: ML model training and prediction
- `visualization.py`: Data visualization functions
- `utils.py`: Utility functions
- `data/`: Sample data and data generation scripts
- `css/`: Custom styling
- `images/`: Project images and assets
- `requirements.txt`: Project dependencies
- `.streamlit/config.toml`: Streamlit configuration

## Models Used

1. **Decision Tree Regression**: For interpretability and efficiency
2. **Random Forest**: Ensemble technique for improved accuracy
3. **XGBoost**: High-performance gradient boosting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
