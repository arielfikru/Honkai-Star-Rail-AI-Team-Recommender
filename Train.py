from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
from Dataset_Preprocessing import prepare_dataset
import pandas as pd

def train_model():
    X, y, _ = prepare_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    
    # Save model
    joblib.dump(regressor, 'team_recommender_model.joblib')
    print("Model saved as 'team_recommender_model.joblib'")
    
    # Display feature importance
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': regressor.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nTop 10 important features:")
    print(feature_importance.head(10))
    
    # Display some predictions for verification
    print("\nSample Predictions:")
    num_samples = min(5, len(y_test))
    for i in range(num_samples):
        print(f"Actual: {y_test[i]:.4f}, Predicted: {y_pred[i]:.4f}")
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

if __name__ == "__main__":
    train_model()