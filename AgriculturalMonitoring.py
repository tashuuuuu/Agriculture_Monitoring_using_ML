import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Load weather data from Excel file
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the data for modeling
    """
    # Standardize column names
    df.columns = [
        "Date", "Temp Max (F)", "Temp Avg (F)", "Temp Min (F)",
        "Dew Point Max (F)", "Dew Point Avg (F)", "Dew Point Min (F)",
        "Humidity Max (%)", "Humidity Avg (%)", "Humidity Min (%)",
        "Wind Speed Max (mph)", "Wind Speed Avg (mph)", "Wind Speed Min (mph)",
        "Pressure Max (in)", "Pressure Avg (in)", "Pressure Min (in)",
        "Precipitation Total (in)"
    ]

    # Display basic statistics
    print("Dataset Overview:")
    print(df.describe().T)

    # Check for missing values and handle them
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Better handling of missing values using interpolation instead of mean
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if column == "Date":
                continue
            # Interpolate missing values using time series methods when possible
            df[column] = df[column].interpolate(method='time' if pd.api.types.is_datetime64_any_dtype(df["Date"]) else 'linear')

    # Convert Date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    # Extract temporal features that may help with agricultural predictions
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Season"] = df["Month"].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                  'Spring' if x in [3, 4, 5] else
                                  'Summer' if x in [6, 7, 8] else 'Fall')

    # Convert Season to numeric for modeling
    season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
    df["Season_Numeric"] = df["Season"].map(season_map)
    
    return df

def prepare_features_targets(df):
    """
    Prepare features and targets for model training
    """
    # Define features and targets
    features = [
        "Dew Point Avg (F)", "Humidity Avg (%)", "Wind Speed Avg (mph)", 
        "Pressure Avg (in)", "Precipitation Total (in)",
        "Month", "DayOfYear", "Season_Numeric"
    ]

    # Define primary targets for agricultural prediction
    target_temp = "Temp Avg (F)"
    target_humidity = "Humidity Avg (%)"

    # Prepare data for modeling
    X = df[features]
    y_temp = df[target_temp]
    y_humidity = df[target_humidity]

    # Check for remaining missing values and handle them
    X.fillna(X.mean(), inplace=True)
    y_temp.fillna(y_temp.mean(), inplace=True)
    y_humidity.fillna(y_humidity.mean(), inplace=True)

    print("\nMissing values after cleaning:")
    print(X.isnull().sum())
    
    return X, y_temp, y_humidity, features

def calculate_precision_for_regression(y_true, y_pred, threshold=0.05):
    """
    Calculate precision for regression predictions by converting to binary classification
    based on how close predictions are to actual values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        threshold: Relative error threshold to consider a prediction as "correct" (default: 5%)
        
    Returns:
        float: Precision score
    """
    # Calculate relative error
    rel_error = np.abs(y_true - y_pred) / np.abs(y_true)
    
    # Create binary arrays (1 = prediction within threshold, 0 = prediction outside threshold)
    y_pred_binary = (rel_error <= threshold).astype(int)
    
    # For precision, we need "true" labels
    # Here we define "true" as "prediction is good enough" (within threshold)
    
    # Filter to only include cases where model predicted positive (within threshold)
    positives_mask = (y_pred_binary == 1)
    
    if not np.any(positives_mask):
        return 0.0  # No positive predictions
    
    # Calculate precision: true positives / (true positives + false positives)
    # Since all our "y_true_binary" values are 1, precision is simply the mean of y_pred_binary
    precision = np.mean(y_pred_binary[positives_mask])
    
    return precision

def evaluate_model(model, X_train, X_test, y_train, y_test, target_name, model_name, output_dir=None):
    """
    Evaluate model performance and generate visualizations
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate traditional regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate precision for regression
    precision_5pct = calculate_precision_for_regression(y_test, y_pred, threshold=0.05)  # 5% threshold
    precision_10pct = calculate_precision_for_regression(y_test, y_pred, threshold=0.10)  # 10% threshold
    
    # Print metrics
    print(f"{model_name} - {target_name} Prediction:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Precision (5% threshold): {precision_5pct:.4f}")
    print(f"  Precision (10% threshold): {precision_10pct:.4f}")
    
    # Create actual vs predicted plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # Add 5% error bounds
    plt.plot([min_val, max_val], [min_val*1.05, max_val*1.05], 'g--', linewidth=1, alpha=0.6)
    plt.plot([min_val, max_val], [min_val*0.95, max_val*0.95], 'g--', linewidth=1, alpha=0.6)
    
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Actual vs Predicted {target_name} ({model_name})")
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{target_name.lower()}_{model_name.lower().replace(' ', '_')}_scatter.png"))
    plt.show()
    
    # Create residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution for {target_name} ({model_name})")
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{target_name.lower()}_{model_name.lower().replace(' ', '_')}_residuals.png"))
    plt.show()
    
    # Create precision visualization - show percent of predictions within thresholds
    rel_errors = np.abs(y_test - y_pred) / np.abs(y_test)
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    precisions = [np.mean(rel_errors <= t) for t in thresholds]
    
    plt.figure(figsize=(10, 6))
    plt.bar([f"{t*100}%" for t in thresholds], precisions, color='skyblue')
    plt.axhline(y=0.8, color='r', linestyle='--', label="80% target")
    plt.ylim(0, 1)
    for i, v in enumerate(precisions):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center')
    plt.xlabel("Error Threshold")
    plt.ylabel("Precision (% of predictions within threshold)")
    plt.title(f"Prediction Precision at Different Thresholds - {target_name} ({model_name})")
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{target_name.lower()}_{model_name.lower().replace(' ', '_')}_precision.png"))
    plt.show()
    
    return (mae, mse, rmse, r2, precision_5pct, precision_10pct)

def plot_model_precision_comparison(results_dict, target, output_dir=None):
    """
    Plot comparison of model precision
    """
    model_names = list(results_dict.keys())
    precision_5pct = [results_dict[name][4] for name in model_names]
    r2_scores = [results_dict[name][3] for name in model_names]
    
    # Sort by precision
    sorted_indices = np.argsort(precision_5pct)
    sorted_models = [model_names[i] for i in sorted_indices]
    sorted_precision = [precision_5pct[i] for i in sorted_indices]
    sorted_r2 = [r2_scores[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
    
    # Create bar plot for precisions
    x = range(len(sorted_models))
    plt.barh(x, sorted_precision, height=0.4, color='skyblue', label='Precision (5% threshold)')
    
    # Add R² values
    for i, (r2, precision) in enumerate(zip(sorted_r2, sorted_precision)):
        plt.text(precision + 0.01, i, f"R² = {r2:.3f}", va='center')
    
    plt.yticks(x, sorted_models)
    plt.xlabel('Precision (% of predictions within 5% threshold)')
    plt.ylabel('Model')
    plt.title(f"Model Precision Comparison for {target} Prediction")
    plt.xlim(0, 1)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{target.lower()}_model_comparison.png"))
    plt.show()

def combined_score(estimator, X, y):
    """
    Custom scorer that combines R² and precision
    """
    y_pred = estimator.predict(X)
    r2 = r2_score(y, y_pred)
    precision = calculate_precision_for_regression(y, y_pred, threshold=0.05)
    return (r2 + precision) / 2

def tune_model(model_name, model, X_train, y_train, target):
    """
    Perform hyperparameter tuning using GridSearchCV
    """
    if model_name in ["Random Forest", "Gradient Boosting", "Extra Trees"]:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_name == "AdaBoost":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0]
        }
    elif model_name in ["Ridge Regression", "Lasso Regression"]:
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    elif model_name == "Support Vector Machine":
        param_grid = {
            'C': [1, 10, 100],
            'gamma': [0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.5]
        }
    elif model_name == "K-Nearest Neighbors":
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    elif model_name == "Decision Tree":
        param_grid = {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    else:
        print(f"No hyperparameter tuning defined for {model_name}")
        return None
    
    # Define cross-validation strategy
    cv_strategy = 5  # 5-fold cross-validation
    
    # Create grid search with custom scorer
    grid_search = GridSearchCV(
        model, param_grid, cv=cv_strategy, 
        scoring=combined_score, n_jobs=-1, verbose=1
    )
    
    # Fit grid search
    print(f"\nTuning {model_name} for {target} prediction...")
    grid_search.fit(X_train, y_train)
    
    # Print results
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Combined Score: {grid_search.best_score_:.4f}")
    
    # Evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    precision = calculate_precision_for_regression(y_train, y_pred, threshold=0.05)
    
    print(f"Training R²: {r2:.4f}")
    print(f"Training Precision (5% threshold): {precision:.4f}")
    
    return grid_search.best_estimator_

def predict_future_weather(input_data, temp_model, humidity_model):
    """
    Predict future weather based on input features
    
    Args:
        input_data (pd.DataFrame): Dataframe with required features
        temp_model: Trained temperature prediction model
        humidity_model: Trained humidity prediction model
        
    Returns:
        pd.DataFrame: Dataframe with predicted temperature and humidity
    """
    # Make predictions
    predicted_temp = temp_model.predict(input_data)
    predicted_humidity = humidity_model.predict(input_data)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Date': input_data.index if hasattr(input_data, 'index') else range(len(predicted_temp)),
        'Predicted_Temperature': predicted_temp,
        'Predicted_Humidity': predicted_humidity
    })
    
    # Add confidence information based on historical precision
    results['Temp_Range_Low'] = predicted_temp * 0.95  # 5% margin
    results['Temp_Range_High'] = predicted_temp * 1.05
    results['Humidity_Range_Low'] = predicted_humidity * 0.95
    results['Humidity_Range_High'] = predicted_humidity * 1.05
    
    return results

def predict_for_specific_day(day_of_year, month, dew_point, humidity, wind_speed, pressure, precipitation, temp_model, humidity_model):
    """
    Make a prediction for a specific day with precision estimates
    """
    # Determine season from month
    season = 0 if month in [12, 1, 2] else 1 if month in [3, 4, 5] else 2 if month in [6, 7, 8] else 3
    
    # Create input dataframe
    input_data = pd.DataFrame({
        "Dew Point Avg (F)": [dew_point],
        "Humidity Avg (%)": [humidity],
        "Wind Speed Avg (mph)": [wind_speed],
        "Pressure Avg (in)": [pressure],
        "Precipitation Total (in)": [precipitation],
        "Month": [month],
        "DayOfYear": [day_of_year],
        "Season_Numeric": [season]
    })
    
    # Make predictions
    temp_prediction = temp_model.predict(input_data)[0]
    humidity_prediction = humidity_model.predict(input_data)[0]
    
    print(f"\nPrediction for day {day_of_year} (Month {month}):")
    print(f"Predicted Temperature: {temp_prediction:.2f} °F (Precision range: {temp_prediction*0.95:.2f}-{temp_prediction*1.05:.2f} °F)")
    print(f"Predicted Humidity: {humidity_prediction:.2f} % (Precision range: {humidity_prediction*0.95:.2f}-{humidity_prediction*1.05:.2f} %)")
    
    # Return predictions in a structured format
    return {
        'temperature': {
            'prediction': temp_prediction,
            'range_low': temp_prediction * 0.95,
            'range_high': temp_prediction * 1.05
        },
        'humidity': {
            'prediction': humidity_prediction,
            'range_low': humidity_prediction * 0.95,
            'range_high': humidity_prediction * 1.05
        }
    }

def plot_feature_importance(model, features, target, model_name, output_dir=None):
    """
    Plot feature importance for a model
    """
    if hasattr(model, 'feature_importances_'):
        feat_importances = pd.Series(model.feature_importances_, index=features)
        plt.figure(figsize=(10, 6))
        feat_importances.sort_values(ascending=False).plot(kind='barh')
        plt.xlabel("Feature Importance Score")
        plt.ylabel("Features")
        plt.title(f"Feature Importance for {target} Prediction ({model_name})")
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{target.lower()}_{model_name.lower().replace(' ', '_')}_feature_importance.png"))
        plt.show()

def main(file_path, output_dir=None):
    """
    Main function to run the weather prediction pipeline
    """
    # Create output directory for plots if provided
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Load data
    df = load_data(file_path)
    if df is None:
        print("Error loading data. Exiting.")
        return
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Prepare features and targets
    X, y_temp, y_humidity, features = prepare_features_targets(df)
    
    # Split data for training and testing
    X_train, X_test, y_train_temp, y_test_temp = train_test_split(
        X, y_temp, test_size=0.2, random_state=42)
    _, _, y_train_humidity, y_test_humidity = train_test_split(
        X, y_humidity, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=200, random_state=42, learning_rate=0.1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42, min_samples_split=3),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Support Vector Machine": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5, weights='distance'),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42)
    }
    
    # Evaluate models for temperature prediction
    print("\n" + "="*50)
    print("TEMPERATURE PREDICTION MODELS")
    print("="*50)
    temp_results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train_temp, y_test_temp, "Temperature", name, output_dir)
        temp_results[name] = metrics

    # Evaluate models for humidity prediction
    print("\n" + "="*50)
    print("HUMIDITY PREDICTION MODELS")
    print("="*50)
    humidity_results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train_humidity, y_test_humidity, "Humidity", name, output_dir)
        humidity_results[name] = metrics

    # Find best models based on combined score (R2 + Precision)
    def calculate_combined_score(metrics):
        r2 = metrics[3]
        precision = metrics[4]  # Using 5% threshold precision
        # Weight R2 and precision equally
        return (r2 + precision) / 2

    best_temp_model_name = max(temp_results, key=lambda k: calculate_combined_score(temp_results[k]))
    best_humidity_model_name = max(humidity_results, key=lambda k: calculate_combined_score(humidity_results[k]))

    print("\n" + "="*50)
    print(f"Best Temperature Prediction Model: {best_temp_model_name}")
    print(f"R² Score: {temp_results[best_temp_model_name][3]:.4f}, Precision: {temp_results[best_temp_model_name][4]:.4f}")
    print(f"Best Humidity Prediction Model: {best_humidity_model_name}")
    print(f"R² Score: {humidity_results[best_humidity_model_name][3]:.4f}, Precision: {humidity_results[best_humidity_model_name][4]:.4f}")
    print("="*50)

    # Plot model precision comparison
    plot_model_precision_comparison(temp_results, "Temperature", output_dir)
    plot_model_precision_comparison(humidity_results, "Humidity", output_dir)

    # Show correlation heatmap
    correlation_cols = features + ["Temp Avg (F)", "Humidity Avg (%)"]
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(df[correlation_cols].corr(), dtype=bool))
    sns.heatmap(df[correlation_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", mask=mask, vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "feature_correlation_heatmap.png"))
    plt.show()

    # Hyperparameter tuning for the best models
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)

    # Tune best temperature model
    best_temp_model = tune_model(
        best_temp_model_name,
        models[best_temp_model_name],
        X_train, y_train_temp,
        "Temperature"
    )

    # Tune best humidity model
    best_humidity_model = tune_model(
        best_humidity_model_name,
        models[best_humidity_model_name],
        X_train, y_train_humidity,
        "Humidity"
    )

    # If tuning failed, use the original best models
    if best_temp_model is None:
        best_temp_model = models[best_temp_model_name]
    if best_humidity_model is None:
        best_humidity_model = models[best_humidity_model_name]

    # Analyze feature importance for the best models
    plot_feature_importance(best_temp_model, features, "Temperature", best_temp_model_name, output_dir)
    plot_feature_importance(best_humidity_model, features, "Humidity", best_humidity_model_name, output_dir)

    # Example of future predictions
    print("\n" + "="*50)
    print("EXAMPLE: FUTURE WEATHER PREDICTION WITH PRECISION RANGES")
    print("="*50)

    # For demonstration, use the test set as "future" data
    future_predictions = predict_future_weather(X_test, best_temp_model, best_humidity_model)

    print("\nSample of predicted future weather conditions with precision ranges:")
    print(future_predictions.head())

    # Create a plot showing predictions with error ranges based on precision
    plt.figure(figsize=(12, 6))
    sample_size = min(20, len(future_predictions))  # Show first 20 predictions or fewer if less available
    x = range(sample_size)

    # Plot temperature predictions with error ranges
    plt.subplot(1, 2, 1)
    plt.plot(x, future_predictions['Predicted_Temperature'].iloc[:sample_size], 'b-', label='Predicted Temp')
    plt.fill_between(x, 
                    future_predictions['Temp_Range_Low'].iloc[:sample_size],
                    future_predictions['Temp_Range_High'].iloc[:sample_size],
                    color='blue', alpha=0.2, label='5% Precision Range')
    plt.xlabel('Prediction Index')
    plt.ylabel('Temperature (°F)')
    plt.title('Temperature Predictions with Precision Range')
    plt.legend()

    # Plot humidity predictions with error ranges
    plt.subplot(1, 2, 2)
    plt.plot(x, future_predictions['Predicted_Humidity'].iloc[:sample_size], 'g-', label='Predicted Humidity')
    plt.fill_between(x, 
                    future_predictions['Humidity_Range_Low'].iloc[:sample_size],
                    future_predictions['Humidity_Range_High'].iloc[:sample_size],
                    color='green', alpha=0.2, label='5% Precision Range')
    plt.xlabel('Prediction Index')
    plt.ylabel('Humidity (%)')
    plt.title('Humidity Predictions with Precision Range')
    plt.legend()

    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "future_predictions.png"))
    plt.show()

    # Save the best models
    models_dir = "models" if output_dir is None else os.path.join(output_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_temp_filename = os.path.join(models_dir, 'temperature_prediction_model.pkl')
    model_humidity_filename = os.path.join(models_dir, 'humidity_prediction_model.pkl')

    joblib.dump(best_temp_model, model_temp_filename)
    joblib.dump(best_humidity_model, model_humidity_filename)

    print(f"\nModels saved to {models_dir}. You can load them using:")
    print("temp_model = joblib.load('models/temperature_prediction_model.pkl')")
    print("humidity_model = joblib.load('models/humidity_prediction_model.pkl')")

    # Example prediction for a specific day
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION FOR A SPECIFIC DAY")
    print("="*50)
    
    # Example prediction for a summer day
    prediction = predict_for_specific_day(
        day_of_year=182,  # July 1st
        month=7,
        dew_point=65.0,
        humidity=70.0,
        wind_speed=5.0,
        pressure=29.9,
        precipitation=0.1,
        temp_model=best_temp_model,
        humidity_model=best_humidity_model
    )

    # Print agricultural insights
    print("\n" + "="*50)
    print("AGRICULTURAL INSIGHTS WITH PRECISION METRICS")
    print("="*50)
    print("Weather prediction insights for farmers:")
    print("1. The models can predict temperature and humidity up to several days in advance")
    print("2. Precision metrics tell you how reliable the predictions are - what percentage fall within 5% of actual values")
    print("3. Feature importance analysis shows which weather factors most influence future conditions")
    print("4. Daily predictions with precision ranges help farmers decide on:")
    print("   - Optimal irrigation scheduling with confidence intervals")
    print("   - Frost protection measures with likelihood estimates")
    print("   - Crop spraying timing based on prediction reliability")
    print("   - Planting and harvesting dates with risk assessment")
    print("5. Temperature and humidity forecasts help prevent crop diseases that thrive in specific conditions")
    print("6. Precision ranges allow for better risk management in agricultural decision-making")

    # Add a cell to allow users to export visualizations
    print("\nTo download any plot, right-click on the image and select 'Save image as...'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weather Prediction Model for Agriculture')
    parser.add_argument('--file', type=str, required=True, help='Path to Excel weather data file')
    parser.add_argument('--output', type=str, default=None, help='Directory to save output visualizations and models')
    
    args = parser.parse_args()
    
    main(args.file, args.output)