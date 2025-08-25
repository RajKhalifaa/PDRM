# PDRM Malaysia Asset & Facility Monitoring - ML Pipeline

## Overview

This project provides a complete Machine Learning pipeline for predicting asset conditions in the PDRM (Royal Malaysia Police) Asset & Facility Monitoring System. The system can predict whether an asset is in "Good", "Needs Action", or "Damaged" condition based on various features.

## Files Description

### Data Generation

- **`generate_data.py`** - Generates synthetic PDRM asset dataset with 10,000 records
- **`assets_data.csv`** - Generated synthetic dataset (10,000 rows, 28 columns)

### Machine Learning Pipeline

- **`train_ml_models.py`** - Main ML training script that trains and evaluates multiple models
- **`demo_predictions.py`** - Demonstration script showing how to use the trained model

### Generated Model Files

- **`best_pdrm_asset_condition_model.joblib`** - Trained best model (Logistic Regression)
- **`label_encoder.joblib`** - Label encoder for target variable conversion
- **`model_info.json`** - Model metadata and basic metrics
- **`evaluation_metrics.json`** - Comprehensive evaluation metrics for all models

## Dataset Features

### Asset Categories

1. **Weapons** (2,500 records) - Glock 19, M4 Carbine, Taser, etc.
2. **ICT Assets** (2,500 records) - Routers, servers, laptops, etc.
3. **Vehicles** (2,500 records) - Patrol cars, motorcycles, transport vehicles
4. **Devices** (2,500 records) - Body cameras, CCTV, drones, radios

### Key Features Used for Prediction

- **Common Features**: category, type, state, city, usage_hours, assigned_officer
- **Temporal Features**: asset_age_days, days_since_maintenance, days_to_maintenance
- **Category-specific Features**:
  - ammunition_count (weapons)
  - warranty_expired (ICT assets)
  - mileage_km (vehicles)

## Model Performance

### Trained Models

1. **Logistic Regression** ⭐ (Best Model)
   - F1-Score: 0.3493
   - Accuracy: 0.3495
2. **Random Forest**
   - F1-Score: 0.3471
   - Accuracy: 0.3470
3. **XGBoost Classifier**
   - F1-Score: 0.3349
   - Accuracy: 0.3350

### Model Selection Criteria

- Best model selected based on **weighted F1-Score**
- All models evaluated using accuracy, precision, recall, and F1-score
- Cross-validation with 80/20 train-test split

## Usage Instructions

### 1. Environment Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1

# Install required packages
pip install pandas numpy scikit-learn xgboost joblib
```

### 2. Generate Dataset

```powershell
python generate_data.py
```

### 3. Train ML Models

```powershell
python train_ml_models.py
```

### 4. Test Predictions

```powershell
python demo_predictions.py
```

## Using the Trained Model

### Loading the Model

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('best_pdrm_asset_condition_model.joblib')

# Load label encoder (if needed for XGBoost predictions)
label_encoder = joblib.load('label_encoder.joblib')
```

### Making Predictions

```python
# Prepare your data with the required features
# Features: category, type, state, city, usage_hours, assigned_officer,
#           mileage_km, asset_age_days, days_since_maintenance,
#           days_to_maintenance, has_ammunition_data,
#           ammunition_count_filled, warranty_expired

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

## Malaysian Context

### States Covered

Selangor, Johor, Penang, Sabah, Sarawak, Kuala Lumpur, Kedah, Pahang, Perak, Negeri Sembilan, Melaka, Terengganu, Kelantan, Putrajaya

### Realistic Features

- Malaysian license plates (e.g., WA 1234 A)
- Malay officer names (Ahmad Faiz, Noraini, Syafiq, etc.)
- Police units (Traffic Police, Criminal Investigation, etc.)
- Malaysian cities and states

## Model Limitations

### Current Performance Notes

- F1-Score of ~35% indicates room for improvement
- This is expected for synthetic data with random conditions
- Real-world data would likely show stronger patterns and better performance

### Potential Improvements

1. **Feature Engineering**: Add more domain-specific features
2. **Data Quality**: Use real historical maintenance data
3. **Advanced Models**: Try ensemble methods, neural networks
4. **Class Balancing**: Handle imbalanced target classes
5. **Temporal Features**: Add seasonal patterns, workload cycles

## Integration Ready

The trained model is ready for integration with:

- **Streamlit Dashboard** - For interactive web interface
- **REST API** - For system integration
- **Batch Processing** - For bulk predictions
- **Real-time Systems** - For live monitoring

## File Structure

```
PDRM 3/
├── generate_data.py                    # Dataset generation script
├── train_ml_models.py                  # ML training pipeline
├── demo_predictions.py                 # Prediction demo
├── assets_data.csv                     # Generated dataset
├── best_pdrm_asset_condition_model.joblib  # Trained model
├── label_encoder.joblib                # Label encoder
├── model_info.json                     # Model metadata
├── evaluation_metrics.json             # Evaluation metrics
└── README.md                           # This file
```

## Next Steps

1. **Dashboard Development** - Create Streamlit interface
2. **Model Improvement** - Experiment with feature engineering
3. **Real Data Integration** - Replace synthetic with real data
4. **Production Deployment** - Set up model serving infrastructure
5. **Monitoring Setup** - Implement model performance monitoring

---

**Author**: ML Training System  
**Date**: August 2025  
**Version**: 1.0.0
