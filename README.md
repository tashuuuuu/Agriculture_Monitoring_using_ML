# Weather Prediction Model

A machine learning system for agricultural weather prediction, providing temperature and humidity forecasts with precision metrics.

## Overview

This project uses historical weather data to build and evaluate multiple machine learning models for predicting temperature and humidity. The models are specifically designed to support agricultural decision-making by providing not just predictions but also precision metrics and confidence ranges.

## Features

- Data preprocessing and feature engineering
- Model comparison across multiple algorithms
- Hyperparameter tuning for optimal performance
- Precision-based evaluation metrics
- Visualization of predictions and model performance
- Feature importance analysis
- Future weather prediction with confidence ranges

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

## Installation

```bash
git clone https://github.com/your-username/Agricultural_Monitoring_usnig_ML.git
cd CapstoneProject
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
python weather_prediction.py --file your_weather_data.xlsx --output results
```

### Parameters

- `--file`: Path to your Excel weather data file
- `--output`: Directory to save output visualizations and models (optional)

### Input Data Format

The script expects an Excel file with the following columns:
- Date
- Temperature (Max, Avg, Min)
- Dew Point (Max, Avg, Min)
- Humidity (Max, Avg, Min)
- Wind Speed (Max, Avg, Min)
- Pressure (Max, Avg, Min)
- Precipitation Total

## Example Output

The script generates:
1. Model performance visualizations
2. Feature importance plots
3. Prediction vs actual comparisons
4. Saved ML models for temperature and humidity prediction

## Agricultural Applications

- Irrigation scheduling with confidence intervals
- Frost protection planning
- Optimal crop spraying timing
- Disease prevention based on humidity forecasts
- Risk assessment for planting and harvesting

## License

[MIT License](LICENSE)
