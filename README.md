# co2-prediction
📁 Project File Structure: co2_emission_predictor/

co2_emission_predictor/
│
├── app.py                            # Main Streamlit dashboard
├── config.py                         # Configuration variables (file paths, constants)
├── requirements.txt                  # Required Python packages
├── README.md                         # Project overview and instructions
│
├── data/
│   ├── raw/
│   │   └── base_station_data.csv     # Original dataset
│   ├── processed/
│   │   └── processed_data.csv        # Cleaned and feature-engineered data
│   └── models/
│       └── best_model.pkl            # Saved trained model
│
├── modules/
│   ├── data_loader.py                # Load and preprocess raw data
│   ├── feature_engineering.py        # Feature encoding, scaling, transformation
│   ├── model_trainer.py              # Train/test ML models and evaluate
│   ├── predictor.py                  # Load model and make predictions
│   └── visualizer.py                 # Generate plots for dashboard
│
├── notebooks/
│   └── exploratory_analysis.ipynb    # EDA and testing in Jupyter
│
└── utils/
    └── helpers.py                    # Utility functions (e.g., for logging, metrics)
